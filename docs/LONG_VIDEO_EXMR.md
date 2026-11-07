# Six-minute ExMR source adaptation

This fork includes the long-video adaptation, originally prepared against
`ef05127a0dbddb3f3ac672a0722dddeae42ea651`, integrated with upstream `75cda26`.
No separate patch installation is required. It remains an experimental source
adaptation, not a claim of a completed six-minute CUDA/MMD run. See
[Fork integration and validation](FORK_INTEGRATION.md) for the current checks.

## Behavior

A six-minute take at the application's 30 fps working rate contains 10,800
frames. ExMR has no corresponding hard frame-count limit. Its body, foot and
hand encoders use a native 120-frame attention window, but the original mask
builder represents that window as an entire L-by-L matrix.

For evaluation without gradients, sequences longer than 256 frames now use
256-query batches. Each batch reads every key allowed by the original window.
The largest key span for the default 120-frame window is 375 frames. Masks are
built for one batch at a time, rather than for the entire video. This bounds
each attention call, not the memory use of the whole application.

RoPE is still evaluated using the original complete-sequence positions, before
query batching. All transformer layers, shape averaging, root integration,
contact processing and smoothing retain their existing complete-take behavior.
No video segments are concatenated and no frames are removed. Training and
gradient-enabled calls retain the upstream implementation. The encoder's
optional `query_chunk_size=None` selects the original implementation for
developer comparisons. Model parameter names and checkpoint shapes are unchanged.

The ExMR motion receipt hashes both modified attention sources, and the vendor
provenance records the adaptation. Since the application also hashes the backend
and provenance for earlier stages, installing this patch can require their
recomputation. Existing project data is not rewritten by installing the patch.

## MMD workflow

Use ExMR and retain one continuous tracked take. Prepare the selected source,
import the PMX character, retarget it and export a character VMD as usual. The
existing exporter writes 30 fps frames 0 through 10,799 for a 10,800-frame take.
The patch does not join separate takes caused by loss of person visibility.

Retarget preparation also reuses static NPZ arrays instead of unpacking the
same mesh data every frame or the same motion matrices for every bone. The
retarget mathematics, character mapping and VMD writer remain unchanged.

Keep Resource Safety enabled. The existing image batches, on-disk caches,
out-of-memory recovery and cancellation behavior remain in place. A long video
still needs time and disk space for feature extraction, preview and Blender
baking; this patch does not establish a specific processing-time or VRAM promise.
Other backends' existing frame limits are not changed.

## Validation boundary

The accompanying tests exercise attention visibility and numerical equivalence,
long-sequence bounded calls, and NPZ reuse with synthetic inputs. These tests do
not require licensed body models or checkpoint files. The original validation
used Linux, Python 3.12 and CPU Torch 2.8.0. The integrated fork's 14 long-video
tests were rerun on Windows with Python 3.12.10 and Torch 2.8.0+cu128 using CPU
synthetic inputs; that does not constitute CUDA inference validation.

Real-video pose quality, Windows GPU peak memory, interactive preview, a complete
Blender bake and loading the resulting VMD into MMD must still be checked on the
target machine. Source modification cannot guarantee faithful capture where a
person is occluded, leaves the image, or the video cuts between shots.

Run the packaged tests from the source root using the installed ExMR interpreter,
or another development interpreter with Torch 2.8.0, NumPy and SciPy:

```text
runtime_exmr\python.exe -X utf8 -m unittest discover -s tests -p "test_longvideo_*.py" -v
```

The historical standalone installers target earlier source hashes and should
not be applied to this integrated fork. Keep any local rollback backups until
they are no longer needed. No model files are included. Original RGB MoCap and ExMR license terms
continue to apply; the ExMR derivative license restricts commercial use.
