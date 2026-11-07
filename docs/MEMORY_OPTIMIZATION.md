# Long-video memory optimization

These changes preserve model weights, inference precision, image preprocessing,
all working frames and the complete temporal context. They do not enable longer
clips by changing the existing model limits or by joining independently inferred
segments. Resource Safety and the existing OOM recovery remain enabled.

## Changed allocation paths

- **ExMR SimpleVO:** process sampled images as an iterator instead of retaining
  every decoded image. Pixel conversions, frame indices, SIFT/matching order,
  random seeds, pose accumulation and interpolation are unchanged. Close the
  underlying reader after success, exceptions and cancellation.
- **ExMR RoPE:** share identical FP32-angle cos/sin tables within one model.
  There is no process-global cache. Device/dtype moves clear derived tables;
  angle buffers that have undergone a non-FP32 conversion keep private per-layer
  cache histories, including after conversion back to FP32, because the upstream
  long-sequence table is generated in FP32. Checkpoint keys are unchanged.
- **Character source preparation:** reuse the static NPZ mesh asset and preserve
  its original values. Body retargeting and Blender baking have been restored to
  upstream `75cda265`; the fork's bounded foot/palm surface evaluation and
  single-read bake optimization are withdrawn. Whole-take geometry allocations
  and repeated bake archive reads therefore follow the original behavior.
- **HTD refinement:** detach optimized parameters before IK and run only that
  postprocessing under `torch.no_grad()`. The actual optimization retains gradients.

Memory use is still not independent of video duration: input features, model
activations, skeleton trajectories, global optimization and Blender animation
curves retain sequence-dependent storage. Fixed batches alone cannot establish a
whole-application RAM/VRAM guarantee.

## Quality checks

The tests compare the retained optimized allocation paths with original equations,
including frame-window boundaries, long/short cache reuse, precision/device
migration, cancellation cleanup, all eligible surface vertices and the HTD
optimization/postprocessing gradient boundary. `test_memory_retarget.py` now checks
the restored whole-take geometry and contact calculations, without asserting
bounded allocation. They use synthetic inputs without
licensed model assets.

```text
runtime_exmr\python.exe -X utf8 -B -m unittest discover -s tests -p "test_memory_exmr.py" -v
runtime\python.exe -X utf8 -B -m unittest discover -s tests -p "test_memory_retarget.py" -v
runtime\python.exe -X utf8 -B -m unittest discover -s tests -p "test_memory_htd.py" -v
runtime_exmr\python.exe -X utf8 -B -m unittest discover -s tests -p "test_longvideo_*.py" -v
runtime\python.exe -X utf8 -B -m unittest discover -s tests -p "test_pmx_upstream*.py" -v
```

ExMR fixed-feature motion and camera rotations are checked for exact tensor/array
equality. Retarget surface contraction changes BLAS matrix dimensions; unchanged
float64 equations can therefore differ in the last bits, with small propagated
differences in the nonlinear solver. Retarget validation must report position and
rotation differences, contact masks, iteration/unreachable counts and contact
errors separately. Do not hide this distinction behind a relaxed generic
`allclose` result or call roundoff-affected retarget output bit-exact.

For the target-video comparison, the checks were specified before running its
retarget solve: maximum joint-position difference 0.1 mm, maximum joint-rotation
difference 0.02 degrees, unchanged contact masks/iteration counts, and at most
0.01 mm worsening in contact diagnostics. These are regression limits, not a
measurement of capture accuracy against ground truth.

Private validation inputs, original/optimized motion artifacts, source snapshot,
commands and sampled process measurements are retained outside the repository.
Do not include videos, licensed geometry, personal filenames or those artifacts
in a public source package. Validation results must distinguish synthetic
checks, component measurements and complete real-video runs.

## Historical component validation on 2026-11-07

These measurements predate restoration of upstream character retargeting.
The contact-surface allocation savings below describe the withdrawn optimization;
they must not be presented as current retarget memory use.

The frozen comparison source was commit `3cfc5f2`. Tests used the installed pinned
Windows runtimes; no packages or assets were upgraded. The machine has an NVIDIA
GeForce RTX 5080 Laptop GPU with 16,303 MiB VRAM and 31.37 GiB physical RAM.

| Check | Original | Optimized | Output comparison |
| --- | ---: | ---: | --- |
| SimpleVO, first 301 working frames of the supplied 4K video, sampled process RSS | 1,659.08 MiB | 1,529.34 MiB | All 301 camera rotations exactly equal |
| Synthetic foot geometry, 10,800 frames, 500 vertices per foot, traced peak allocation | 805.23 MiB | 7.08 MiB | Probes and minimum heights exactly equal |
| Synthetic foot geometry, 257 frames, 13,594 vertices per foot, traced peak allocation | 526.77 MiB | 30.37 MiB | Probe maximum difference 5.55e-17 m; heights exactly equal |

The geometry measurements use `tracemalloc` after creating their synthetic inputs;
they are component allocation peaks, not whole-process RAM. SimpleVO ran alongside
the separate GPU job, so its timings do not establish a speed improvement. The
full-video capture uses fixed-camera mode; moving-camera validation covers only
the stated 301-frame prefix.

HTD's real optimization/FK/CCD code with a synthetic body model produced exactly
equal parameters for 60- and 240-frame cases. The postprocessing retained zero
backward tensors after this change. This is a CPU component test, not a real-video
HTD or GVHMR accuracy benchmark.

## Historical supplied-video validation on 2026-11-07

The ExMR comparison remains a record of the retained inference changes. The
retarget comparison below concerns the former bounded geometry implementation,
not a fresh test of the restored upstream PMX behavior.

The source was a 257.466625-second, 3840x2160 video at approximately 60 fps
(15,448 source frames). Both versions used the application's unchanged 30 fps
preparation: 7,724 working frames at 3840x2160. ExMR used fixed-camera mode,
global smoothing, Automatic resource mode and Resource Safety enabled. No extra
frame skipping, resolution reduction, precision change or temporal split was used.

The original source performed the complete tracking and image-feature extraction.
Two fresh processes then ran original and optimized full-sequence ExMR from the
same immutable features. Optimized correction and export were also run afresh.
All **120** strict comparisons passed with exact equality: ExMR/native/canonical
and corrected tensors, original-run repeatability, joint/contact caches, rig
frames/joints/reference/geometry, and both 22-joint and 52-bone BVH bytes. The
initial result does not generate a contact NPZ; the comparison explicitly records
that absence and requires the corrected contact NPZ.

| Fresh full-sequence ExMR process | Original | Optimized |
| --- | ---: | ---: |
| PyTorch peak allocated VRAM | 0.941389 GiB | 0.840495 GiB |
| PyTorch peak reserved VRAM | 1.060547 GiB | 0.912109 GiB |
| Sampled process-tree peak RSS | 2.100822 GiB | 2.252602 GiB |
| Sampled process-tree private bytes | 4.036144 GiB | 4.013329 GiB |
| Process wall time, including startup/loading | 23.078 s | 21.485 s |

Peak allocated VRAM decreased by 103.315 MiB (10.72%). RSS did not decrease in
this core comparison. These single paired runs do not establish a general speedup
or lower whole-application RAM. The unchanged image stages were not repeated in
the optimized run; the original complete image/core worker peaked at 11.93 GiB
RSS and 20.47 GiB private bytes. Its largest PyTorch allocated peak was 5.68 GiB
in unchanged Sapiens2, so the 10.72% core saving is not a whole-capture reduction.
Process-tree sampling was every 0.5 s and can miss
short peaks. Whole-board `nvidia-smi` measurements are recorded separately because
they include the desktop and other processes.

The isolated baseline initially lacked the installed native contact-correction
extension. Connecting that same existing binary and resuming correction/export
resolved the test-fixture failure; no installation or dependency was changed.

The same complete 7,724-frame corrected motion was then prepared and run through
the original and optimized CPU character-retarget solver with one real PMX asset
(799/829 eligible foot vertices). This used Python 3.10.22, NumPy 1.23.5 and SciPy
1.14.1, with OMP/OpenBLAS/MKL thread settings of one. All **166** NPZ members were
bit-exact; joint/root position and world/local rotation differences were zero.
Contact masks, unreachable masks, iteration counts and every contact diagnostic
were unchanged, and all predefined regression limits passed. Fixed-bone-length
error was 5.28e-12 m in both versions.

The retarget worker's sampled peak RSS was 2.642 GiB before and 2.638 GiB after;
wall time was 132.792 s and 135.674 s. This establishes neither a significant
whole-solver RAM reduction nor a speedup for this asset: other complete-sequence
arrays dominate its peak. The bounded surface-allocation benefit is demonstrated
separately in the component table. An additional 2,994-frame cached-motion test
under NumPy 2.3.5 was not bit-exact (maximum joint difference 0.03291 mm and local
rotation difference 0.008759 degrees), illustrating the BLAS rounding boundary.

Exact target-video agreement means no output regression for the tested path; it
does not remove existing capture/retarget errors. Both versions retained the same
contact outliers, including a left-foot maximum horizontal residual of 440.852 mm
and right-foot residual of 117.151 mm. No ground-truth motion accuracy, Blender
bake, interactive playback or MMD loading was tested here. Source-level checks and
52 focused regression tests passed across the installed runtimes.
