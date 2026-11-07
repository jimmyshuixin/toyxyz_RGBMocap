"""HTD-Refine motion-optimization pipeline."""

from __future__ import annotations

from pathlib import Path

import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

from .body_model import create_body_model
from .config import OPTIMIZATION_CONFIG
from .forward_kinematics import BodyForwardKinematics
from .geometry import apply_T_on_points, get_new_transl, solve_pnp
from .losses import make_scale_weights, make_smpl_weights
from .optimizers import optimize_camera_scale, optimize_smpl
from .postprocess import apply_ik_postprocess
from .camera_alignment import static_camera_from_motion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMPL2COCO17_PATH = PROJECT_ROOT / "utils/body_model/smpl_coco17_J_regressor.pt"
SMPLX2SMPL_PATH = PROJECT_ROOT / "utils/body_model/smplx2smpl_sparse.pt"


def _clone_tree(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in params.items()}


def _load_body_model(model_name: str, device: torch.device):
    if model_name == "smplx":
        return create_body_model("smplx_full_pose").to(device)
    if model_name == "smpl":
        return create_body_model("smpl").to(device)
    raise ValueError(f"unsupported model {model_name!r}")


def _world_params_from_incam(
    pred_smpl_params_incam: dict[str, torch.Tensor],
    T_w2c: torch.Tensor,
    smpl_model,
) -> dict[str, torch.Tensor]:
    params = _clone_tree(pred_smpl_params_incam)
    R_smpl2cam = axis_angle_to_matrix(pred_smpl_params_incam["global_orient"])
    T_c2w = torch.linalg.inv(T_w2c)
    R_smpl2world = T_c2w[:, :3, :3] @ R_smpl2cam
    params["global_orient"] = matrix_to_axis_angle(R_smpl2world)
    offset = smpl_model.get_skeleton(params["betas"][0])[0]
    params["transl"] = get_new_transl(T_c2w, pred_smpl_params_incam["transl"], offset)
    return params


def _initial_world_params(
    pred_smpl_params_incam: dict[str, torch.Tensor],
    pred_smpl_params_global: dict[str, torch.Tensor] | None,
    T_w2c: torch.Tensor,
    smpl_model,
) -> dict[str, torch.Tensor]:
    if pred_smpl_params_global is not None:
        return _clone_tree(pred_smpl_params_global)
    return _world_params_from_incam(pred_smpl_params_incam, T_w2c, smpl_model)


def _coco17_joints_cam(
    smpl_params_world: dict[str, torch.Tensor],
    T_w2c: torch.Tensor,
    smpl_model,
    model_name: str,
) -> torch.Tensor:
    device = T_w2c.device
    coco17_regressor = torch.load(SMPL2COCO17_PATH, weights_only=True).to(device)
    smpl_output = smpl_model(**smpl_params_world)
    vertices_cam = apply_T_on_points(smpl_output.vertices, T_w2c)
    if model_name == "smplx":
        smplx2smpl = torch.load(SMPLX2SMPL_PATH, weights_only=True).to(device)
        vertices_cam = torch.stack([torch.matmul(smplx2smpl, vertices) for vertices in vertices_cam], dim=0)
    return torch.einsum("jv,lvi->lji", coco17_regressor, vertices_cam)


def _static_camera_from_first_frame(
    pred_smpl_params_incam: dict[str, torch.Tensor],
    pred_smpl_params_global: dict[str, torch.Tensor] | None,
    tgt_kp2d: torch.Tensor,
    K: torch.Tensor,
    model_name: str,
) -> torch.Tensor:
    device = tgt_kp2d.device
    model_type = "smpl_coco17" if model_name == "smpl" else "smplx_coco17"
    smpl_coco17 = create_body_model(model_type).to(device)
    source = pred_smpl_params_global if pred_smpl_params_global is not None else pred_smpl_params_incam
    first_params = {key: value[:1] for key, value in source.items()}
    joints = smpl_coco17(**first_params)
    T_w2c, _, _ = solve_pnp(j3d=joints[:1], j2d=tgt_kp2d[:1][..., :2], K=K[:1])
    return T_w2c[:1].repeat(source["global_orient"].shape[0], 1, 1)


def _print_refinement_weights(weights) -> None:
    print(f"jitter: {weights.jitter}")
    print(f"vel_3d_mse: {weights.vel_3d_mse}")
    print(f"accel_3d_mse: {weights.accel_3d_mse}")
    print(f"kp2d_mse: {weights.kp2d_mse}")
    print(f"reg: {weights.reg}")


def _apply_ik_postprocess(
    smpl_params_world: dict[str, torch.Tensor],
    tgt_vel_3d: torch.Tensor,
    model_name: str,
    vel_thr: float | None = None,
) -> dict[str, torch.Tensor]:
    cfg = OPTIMIZATION_CONFIG.postprocess
    effective_vel_thr = cfg.vel_thr if vel_thr is None else float(vel_thr)
    if effective_vel_thr <= 0:
        raise ValueError(f"vel_thr must be positive, got {effective_vel_thr}")
    tgt_vel = tgt_vel_3d.norm(dim=-1)
    static_vel = tgt_vel[:, list(cfg.joint_ids)]
    static_conf_4 = torch.zeros_like(static_vel)
    static_index_vel = static_vel < effective_vel_thr
    static_conf_4[static_index_vel] = 1.0 - (static_vel[static_index_vel] / effective_vel_thr)
    static_conf_4 = static_conf_4.clamp(min=0.0, max=1.0)
    tgt_static_conf = torch.cat([static_conf_4, static_conf_4[[-1]]], dim=0)[None]

    print(f"ik postprocess vel_thr={effective_vel_thr}, displacement_thr={cfg.displacement_thr} m")
    forward_kinematics = BodyForwardKinematics().to(tgt_vel_3d.device)
    pp_smpl_keys = ["transl", "global_orient", "body_pose", "betas"]
    ik_batch = {
        "static_conf_logits": tgt_static_conf,
        "pred_smpl_params_global": {
            key: value.clone()[None] for key, value in smpl_params_world.items() if key in pp_smpl_keys
        },
    }
    if model_name == "smplx":
        smpl_params_world["body_pose"] = apply_ik_postprocess(
            ik_batch,
            forward_kinematics,
            displacement_thr=cfg.displacement_thr,
        )[0]
    else:
        hand_pose = ik_batch["pred_smpl_params_global"]["body_pose"][0, :, 63:]
        ik_batch["pred_smpl_params_global"]["body_pose"] = ik_batch["pred_smpl_params_global"]["body_pose"][:, :, :63]
        body_pose = apply_ik_postprocess(
            ik_batch,
            forward_kinematics,
            displacement_thr=cfg.displacement_thr,
        )[0]
        smpl_params_world["body_pose"] = torch.cat([body_pose, hand_pose], dim=-1)
    return smpl_params_world


def opt_vel(
    tgt_vel_3d: torch.Tensor,
    tgt_accel_3d: torch.Tensor,
    tgt_kp2d: torch.Tensor,
    K: torch.Tensor,
    T_w2c_tgt: torch.Tensor,
    pred_smpl_params_incam: dict[str, torch.Tensor],
    pred_smpl_params_global: dict[str, torch.Tensor] | None = None,
    model: str = "smplx",
    vid: str | Path | None = None,
    pp: bool = True,
    static: bool = False,
    mask: torch.Tensor | None = None,
    tgt_conf: torch.Tensor | None = None,
    vel_thr: float | None = None,
    preserve_input_camera: bool = False,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    del vid
    device = tgt_kp2d.device
    smpl_model = _load_body_model(model, device)
    if preserve_input_camera and not static:
        raise ValueError('Preserving the input static camera requires static mode')
    if static and preserve_input_camera:
        if pred_smpl_params_global is None:
            raise ValueError('GVHMR world-space motion is required for camera alignment')
        offset = smpl_model.get_skeleton(pred_smpl_params_global['betas'][0])[0]
        T_w2c = static_camera_from_motion(pred_smpl_params_incam, pred_smpl_params_global, offset)
        scale = torch.tensor(1.0, device=device)
    elif static:
        T_w2c = _static_camera_from_first_frame(
            pred_smpl_params_incam, pred_smpl_params_global, tgt_kp2d, K, model
        )
        scale = torch.tensor(1.0, device=device)
    else:
        aligned_params = _initial_world_params(
            pred_smpl_params_incam,
            pred_smpl_params_global,
            T_w2c_tgt,
            smpl_model,
        )
        alignment_weights = make_smpl_weights(OPTIMIZATION_CONFIG.camera_alignment)
        aligned_params = optimize_smpl(
            smpl_params=aligned_params,
            tgt_vel_3d=tgt_vel_3d,
            tgt_accel_3d=tgt_accel_3d,
            tgt_kp2d=tgt_kp2d,
            model_name=model,
            K=K,
            T_w2c=T_w2c_tgt,
            weights=alignment_weights,
            stage=OPTIMIZATION_CONFIG.camera_alignment,
            frame_rate=OPTIMIZATION_CONFIG.frame_rate,
            mask=mask,
            tgt_conf=tgt_conf,
        )

        j3d_cam = _coco17_joints_cam(aligned_params, T_w2c_tgt, smpl_model, model)
        scale_weights = make_scale_weights(OPTIMIZATION_CONFIG.scale_calibration)
        T_w2c, scale = optimize_camera_scale(
            tgt_vel=tgt_vel_3d.norm(dim=-1),
            tgt_accel=tgt_accel_3d.norm(dim=-1),
            j3d_cam=j3d_cam,
            T_w2c=T_w2c_tgt,
            weights=scale_weights,
            config=OPTIMIZATION_CONFIG.scale_calibration,
            frame_rate=OPTIMIZATION_CONFIG.frame_rate,
            mask=mask,
            tgt_conf=tgt_conf,
        )
    print(f"scale={scale}")

    smpl_params_world = _initial_world_params(pred_smpl_params_incam, pred_smpl_params_global, T_w2c, smpl_model)
    refinement_weights = make_smpl_weights(OPTIMIZATION_CONFIG.motion_refinement)
    _print_refinement_weights(refinement_weights)
    smpl_params_world = optimize_smpl(
        smpl_params=smpl_params_world,
        tgt_vel_3d=tgt_vel_3d,
        tgt_accel_3d=tgt_accel_3d,
        tgt_kp2d=tgt_kp2d,
        model_name=model,
        K=K,
        T_w2c=T_w2c,
        weights=refinement_weights,
        stage=OPTIMIZATION_CONFIG.motion_refinement,
        frame_rate=OPTIMIZATION_CONFIG.frame_rate,
        mask=mask,
        tgt_conf=tgt_conf,
    )
    # Optimization needs gradients; its subsequent IK is inference only. Drop
    # the trainable leaves before whole-sequence FK/CCD can retain a new graph.
    smpl_params_world = {
        key: value.detach().to(device).requires_grad_(False)
        for key, value in smpl_params_world.items()
    }
    if pp:
        with torch.no_grad():
            smpl_params_world = _apply_ik_postprocess(smpl_params_world, tgt_vel_3d, model, vel_thr=vel_thr)
    return smpl_params_world, T_w2c, scale


__all__ = ["opt_vel"]
