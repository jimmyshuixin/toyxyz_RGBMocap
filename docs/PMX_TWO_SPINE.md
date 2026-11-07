# Native PMX two-spine targets

This fork includes native two-spine PMX support alongside the long-video ExMR
adaptation, integrated with upstream `75cda26`. The original patch targeted
`ef05127a0dbddb3f3ac672a0722dddeae42ea651`; no standalone patch installation is
needed in this fork. See [Fork integration and validation](FORK_INTEGRATION.md)
for current checks; the Blender results below are historical patch records.

## Supported target structure

The existing PMX body mapping expects UpperBody (`上半身`), UpperBody1
(`上半身1`) and UpperBody2 (`上半身2`). Some PMX models, including the historically tested
Firefly 3.0 model, have UpperBody and UpperBody2 without UpperBody1.

An explicit two-spine profile is selected only when all other required body
bones resolve uniquely, UpperBody1 alone is absent, and:

- UpperBody2 is a direct child of UpperBody.
- The neck and both shoulders have UpperBody2 as their nearest mapped ancestor.
  Native intermediate shoulder controls are allowed.
- Actual mapped body bones share one root. Unrelated control trees may coexist.
- The actual UpperBody-to-UpperBody2 segment is nonzero. Existing actual limb
  length checks still apply.

The importer uses retained MMD names as well as the established Blender `.L` and
`.R` names. Ambiguous identities, other missing bones, unsupported parent chains,
and reserved internal names produce explicit errors. This is not a general
manual bone mapper or a promise to support every PMX skeleton.

## Representation

Source motion remains in the existing 22-slot representation. The PMX target
records a versioned `body_profile` in `pmx-rig.json`:

- `id`: `pmx_two_spine_v1`, `pmx_three_spine_v1`, or `humanoid_22_v1`.
- `slot_bones`: 22 actual Blender names, with JSON null only at virtual slot 6.
- `virtual_slots`: `[6]` for a two-spine PMX target, otherwise `[]`.

`pmx-canonical.npz` keeps 22 Unicode slot names and numeric arrays; it never
requires pickle/object arrays. The two-spine slot 6 has the reserved internal
label `__rgb_mocap_virtual_spine1__` and exactly the same rest position as slot 9.
Its outgoing edge has zero length. No Blender/PMX bone is created for this slot,
no mesh weights are assigned to it, and no animation track is exported for it.

With a co-located slot, rotating the internal middle joint cannot introduce a
physical bend pivot absent from the actual target. The real chest uses slot 9's
final world orientation. Baking computes its local rotation relative to its
actual parent and native rest axes. Thus the source middle-spine rotation can
contribute to the final chest orientation without creating an unexportable bone.
This does not make a two-segment skin capable of every three-segment deformation.

Contact geometry retains 22 weight columns, with the virtual column strictly
zero. Existing PMX D/EX foot aliases remain. The solver explicitly skips the
declared virtual edge during direction alignment. Canonical and rig metadata
must agree before contact solving, geometry registration, or baking.

Three-spine targets retain the original computation. Older imported characters
with 22 real names and no profile remain readable. A virtual name without its
explicit profile is rejected with a request to import again.

## MMD workflow

1. Use this fork with both adaptations already integrated, and start RGB MoCap.
2. Keep the original PMX and its original texture paths together.
3. Use **File > Import Character...** to create a fresh character import.
4. Use **Retarget > [source person]** after the source motion is ready.
5. Review a short bend, twist, side bend, and contact motion first.
6. Use **Export Character... > VMD** and load the motion on the same original
   PMX in MMD. Existing characters and failed import folders are not migrated.

The existing RGB VMD writer is retained, including native bone names,
additional-transform compensation, root/waist handling, 30 fps sampling and IK
disable tracks for baked FK contacts. This patch does not generate facial motion
or hair/clothing physics. Actual appearance still requires the model's textures.

## Reproducible checks

The following tests use synthetic data and need Python, NumPy and SciPy. The
combined long-video suite additionally requires Torch. No PMX or licensed body
model is included in the patch.

```text
python -X utf8 -m unittest discover -s tests -p "test_pmx_two_spine*.py" -v
```

For Blender integration, use a separately obtained PMX and the application's
Blender 4.5 runtime. First run the real character import into a new validation
folder, then run the fixture generator and Blender check:

```text
blender -b --factory-startup --disable-autoexec --python-exit-code 1 --python mocap/retarget/blender_entry.py -- import VALIDATION_FOLDER MODEL.pmx
python -X utf8 tests/pmx_two_spine_motion_fixture.py --folder VALIDATION_FOLDER
blender -b --factory-startup --disable-autoexec --python-exit-code 1 --python tests/blender_pmx_two_spine_integration.py -- --folder VALIDATION_FOLDER --pmx MODEL.pmx
```

Use absolute paths and quote paths containing spaces. The generator writes only
inside the supplied validation folder and supplies controlled three-spine motion
to the real contact pipeline. The Blender check exercises actual baking,
viewport generation, RGB VMD export and reimport into a fresh native PMX rig.
Synthetic motion and a Blender round trip do not establish AI capture quality,
Windows GUI compatibility, native MMD playback, or a complete six-minute GPU run.
Do not point these developer checks at an existing user project/character folder.

## Historical patch validation (2026-11-04; not rerun during fork integration)

The combined synthetic CPU suite passed 31 tests, including the earlier
long-video regressions. Two separately supplied PMX models were imported with
the actual app importer and Blender 4.5.0 on Linux. Temporary placeholder textures
were used; PMX bytes were unchanged. Each model then completed the real contact
pipeline on 61 controlled frames at 30 fps, with bend/twist/side-bend source
motion and overlapping left/right foot-contact intervals. Hand contacts were
inactive; finger motion was not part of this controlled source.

| Check | Firefly 3.0 (two-spine) | Castorice 3 (three-spine) |
| --- | ---: | ---: |
| Import, preview, contact solve, bake, RGB VMD export and fresh PMX/VMD reimport | PASS | PASS |
| Canonical to Blender mapped-joint position maximum | 4.85e-7 m | 4.11e-7 m |
| VMD round-trip pose matrix component maximum | 8.34e-7 | 8.05e-7 |
| VMD round-trip evaluated vertex position maximum, 5 sampled frames | 8.36e-7 m | 6.45e-7 m |
| Native bone names/parents/bind data and base mesh/weights preserved | PASS | PASS |

All 61 frames were checked for mapped bone poses. Evaluated meshes were compared
at frames 1, 16, 31, 46 and 61. Firefly's exported torso tracks are UpperBody and
UpperBody2, with no virtual or UpperBody1 track. Castorice retains all three torso
tracks. Castorice's pre-existing canonical fields, contact geometry and rest
preview arrays also matched its pre-patch import exactly.

These are round-trip and representation checks, not measurements of motion
capture accuracy or guarantees for other PMX structures. The patch preserves the
earlier ExMR adaptation; `LONG_VIDEO_EXMR.md` describes that earlier component,
while this document describes the additional target-mapping behavior.
