# Video face capture (MediaPipe)

RGB MoCap can capture expressions from a local video and combine them with a
character's body/finger animation. Enable **Capture Face** in the normal motion
settings, then click **Reconstruct 3D**. When you retarget a compatible character,
its source take's facial track is included automatically. New captures use native
MediaPipe expression scores, adaptive smoothing and bounded missing-face estimates.

## Integrated motion-capture checkbox

- **Capture Face** defaults off and is saved with the project/model settings.
  It currently supports **ExMR, FlowHMR and GVHMR**. Select the source person(s)
  and trim as usual; the facial pass follows the body/correction/export stages
  and uses each actor's unchanged prepared CFR video and tracking boxes.
- Once enabled, native expression capture, smoothing and gap estimation are on
  by default. The advanced manual dialog offers independent smoothing and gap
  estimation switches; the main workflow still needs only the one checkbox.
- The face region follows the upper portion of each actor's tracked box.
  Missing body detections and regions containing multiple detected faces are
  blocked: expressions become neutral and temporal history cannot cross that
  boundary. Ordinary face-detector misses within an observed actor's region can
  instead use the limited estimates described below. This helps avoid assigning
  expressions to another actor but cannot repair an identity switch in the body tracker.
- **Prepare Preview**, **Apply Correction** and export do not launch new facial
  inference. Reconstruct can reuse a matching facial result; body steps retain
  their existing cache checks. The checkbox is independent of body precision,
  model weights and body result identities.
- As with ordinary body reconstruction, run **Retarget** to refresh an already
  baked character. Its per-take facial track is picked up automatically, without
  the separate Capture/Apply dialog. Manually assigned character face tracks
  take priority over automatic source tracks.
- No detectable face produces a visible warning while keeping completed body
  motion. Characters without supported facial shape keys retain body animation
  with a warning in automatic mode. Manual application still reports an error.
- The checkbox takes effect on the next **Reconstruct 3D** run. Retarget afterward
  to update the character; toggling it alone does not erase saved animations or
  face generations, or change an already generated source take.
- Automatic capture is disabled for MAMMA, PromptHMR and PEAR because their
  existing actor/camera paths are not connected to this face association yet.
  The manual video workflow below remains available outside PEAR live mode.

Automatic results are stored in each actor/take's `face/generation-*/` folder,
with a `face_capture.json` pointer bound to video/tracking/timing/options and
algorithm versions. Saved project copies retain relative references. A stale,
disabled or unsuccessful pointer is not consumed by later retargeting.

After upgrading from the original five-expression capture, run **Reconstruct 3D**
with the checkbox enabled, then **Retarget**, to generate and apply the new track.
The changed face algorithm invalidates old automatic face-cache pointers without
changing body-inference cache identities. Previously saved five-expression tracks
and baked animations remain readable; old generations are not deleted.

The optional runtime is installed using `Setup_Face.cmd`, or **File > Capture
Face... > Set Up Face Capture**. Setup reports Ready after validating the runtime;
missing runtime requirements are checked before starting a reconstruction with
this option enabled.
An already installed, Ready face runtime supports the native expression model;
this upgrade does not require reinstalling it or changing body-model packages.

## Advanced manual video workflow

Open **File > Capture Face...** or right-click a character and choose
**Capture Face...** for a separate face video, a manually selected region or
facial-only VMD. No body reconstruction is required for this route; a character
without body animation can receive facial animation in its rest pose.


1. Run `Setup_Face.cmd`, or use **Set Up Face Capture** in the dialog. This optional
   setup needs the base `install.cmd` Python runtime and an internet connection.
   It installs a separate, pinned CPU MediaPipe runtime and the official face
   landmarker model. It does not change the body/hand/GUI environments.
2. Open the source video/project and choose the desired timeline trim. Import
   and retarget the character as usual when combining with body motion.
3. Open **Capture Face...**, check the video, start/end seconds, and character.
   For a full-body shot with a small face, use **Select Face Region...** and drag
   a box around the head. Include its movement across the selected interval;
   the fixed detector region does not crop or rewrite the original video.
   **Video begins on timeline** is normally zero; saved trimmed videos retain
   their original timeline offset automatically. For an independently trimmed
   face video, enter its first frame's time on the body timeline.
4. Leave **Smooth expressions** and **Estimate brief detection gaps** enabled
   for the default continuity processing. They are independent switches; their
   exact behavior is described below. **Swap left/right expressions** defaults
   off, preserving MediaPipe's anatomical sides. Enable it for mirrored footage.
5. Click **Capture Face**, then **Apply to Character**. Processing and character
   baking run outside the GUI thread. The old character animation stays usable
   if capture or baking fails or is cancelled. A successful apply creates a new
   animation generation without changing body inference or contact results.
6. Review playback, then **Export Character...**. PMX VMD includes body and facial
   morph keys. FBX with mesh includes shape-key animation; animation-only FBX
   cannot carry mesh shape keys. Source skeleton FBX/BVH exports do not contain
   facial geometry or expressions.

**Save facial motion only** retains a local track; **Export Face VMD...** writes
a facial-only VMD starting at frame 0, converting native scores into common MMD
morphs. This conversion is an approximation; it is not a file containing all
native channel names. Its timing is relative to the captured interval. Applying
inside RGB MoCap instead uses absolute source-video time,
including take offsets. An attached track is reapplied when the character is
retargeted or refreshed after contact correction. Project copies include the
relative face track and animation data.

## Continuity and missing detections

These rules operate in seconds, independently of capture frame rate. They are
conservative motion estimates, not recovered measurements of an invisible face.

- A missing run of at most **0.20 seconds**, with measured expressions on both
  sides, is interpolated between those observations. This offline pass can use
  the later observation; it is not a live-stream prediction system.
- A longer gap or missing tail uses a damped recent trend for up to **0.20
  seconds**, then fades toward neutral, reaching zero at **0.80 seconds** after
  the last observation. Trend speed is limited to 1.5 coefficient units/second
  and the extrapolated change to 0.12. It does not freeze a smile or open mouth
  indefinitely and does not borrow a distant future expression.
- Predicted eyelid closure is more conservative: it never extrapolates toward a
  more closed eye, starts relaxing after **0.04 seconds**, and its prediction
  target is neutral by **0.25 seconds**. There is no generated periodic blinking.
  Eye blinks use faster filtering than slower expressions to limit smearing.
- Smoothing reduces jitter and eases reacquisition over **0.12 seconds**.
  Disabling **Smooth expressions** also disables that recovery easing but leaves
  gap estimation available. Disabling **Estimate brief detection gaps** makes
  every missing frame neutral and resets smoothing history there.
- No prior observation means no extrapolation: leading missing frames stay
  neutral, and an entirely undetected clip is rejected. In automatic capture,
  an absent actor or ambiguous multiple faces form an immediate neutral boundary;
  interpolation and extrapolation never cross it.

Review estimated portions, especially occlusions, fast expressions and changes
of view. Small, blurred or hidden faces still contain insufficient information;
more continuous animation does not establish that the estimated expression is
correct. Detection coverage and temporal support are not accuracy measurements.

## Expressions, models and limitations

- MediaPipe's **52 native scores** include `_neutral`, which is preserved in
  the track but never drives a shape key. The remaining names describe eyes,
  brows, cheeks, jaw and lips. This model does not output `tongueOut`; it is not
  a complete substitute for every ARKit rig or a phoneme recognizer. Head/body
  rotations continue to come from body capture. Named eye-look channels can
  drive compatible shape keys; this pass does not bake eye-bone rotations.
- Compatible ARKit-style shape keys receive their named scores directly, with
  explicit supported aliases. MMD/VRM mappings use only shapes that the character
  actually contains. Blink, smile, brow and approximate vowel-shape combinations
  cannot recreate facial geometry that is absent from a model. The MMD vowel
  mapping is geometric expression approximation, not speech recognition.
- New PMX imports load facial morphs. Older imports made before this integration
  omitted them: reimport the original PMX if applying reports missing morphs.
  Supported named shape-key aliases also cover some VRM/FBX models. This is not
  universal automatic expression mapping; missing channels are reported by the
  character worker. Models need actual compatible shape keys.
- Manual capture selects one face. In scenes with multiple people, crop the input to
  the intended face; the manual dialog does not use body actor tracking. Face
  visibility, resolution, occlusion and view angle affect the result. Short ordinary
  detector misses can be estimated, but this is not face-identity tracking.
- Input uses OpenCV's constant-rate frame clock. Manual capture outputs 30 fps;
  integrated capture follows the prepared body video's timeline. Prepare
  variable-rate phone recordings as a constant-frame-rate video before capture
  when precise synchronization matters. This integration does not record a
  webcam, simulate face muscles, or generate hair/clothing physics.
  Each face pass is limited to 54,000 samples (30 minutes at 30 fps).
- Original five-channel reze-mipo tracks remain supported without converting
  their saved coefficients. Their historical optional eye rotations remain
  inspection data; new native captures store expression scores instead.

## Implementation and provenance

New capture uses the native [MediaPipe FaceLandmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker?tab=Python)
with expression output enabled. `blendshapes.py` validates the 52 category names
and applies explicit side swapping; `temporal.py` implements the continuity rules
above. The native expression model is not a port of reze-mipo's geometric solver.

The historical geometric solver and scalar smoothing in `mocap/face/solver.py`
were ported from
[AmyangXYZ/reze-mipo](https://github.com/AmyangXYZ/reze-mipo/tree/234b51553f44d2578606a91e1aecc7267b964854).
The pinned source paths, hashes and adaptation notes are in
`third_party/reze-mipo/PROVENANCE.json`. The port retains GPL-3.0 terms and its
license; the host's MIT license does not relicense this derived code. That solver
is retained for historical compatibility and regression checks, and is not called
by new capture. See
[third-party notices](../THIRD_PARTY_NOTICES.md).

`installer/face-runtime-lock.json` records exact wheels, SHA-256 hashes and the
official task model URL/hash. `tools/setup_face.py` verifies downloads and runs a
real CPU VIDEO-mode landmarker operation with blendshape output enabled before
registering readiness in `.setup/face-runtime.json`. Source video frames and face
tracks stay on this PC.
The setup downloads packages/model files; capture does not upload media.

Manual capture artifacts are `characters/face-<id>/face.npz` and `face.json`;
automatic captures use the per-take generation folders described above. Metadata
records source timing, actual detection coverage, estimated-frame counts, options,
algorithm hashes and runtime versions. Native `face.npz` stores without pickle:

- `morph_names` and processed `weights` for animation, plus `fps` and
  `start_seconds` for absolute timeline alignment.
- `raw_weights`: observed model scores in the selected anatomical side order;
  missing frames contain zeros. They are preserved before temporal processing.
- `detected`: real model observations only. `interpolated` marks two-sided gap
  filling; `predicted` marks one-sided prediction/fading. These three masks are
  mutually exclusive. Neither kind of estimate increases the detection count.
- `blocked`: identity/actor visibility boundaries that remain neutral.
- `reliability`: a heuristic temporal support value in [0, 1], with observed=1
  and unsupported/blocked=0. It is not calibrated detector confidence or accuracy.

A unique output directory is published only after completion.
Character application uses separate animation generations and bounded sparse
morph preview data. No existing project output is overwritten by capture.

Relevant modules: `mocap/face_ui.py`, `mocap/face/`,
`mocap/retarget/face_track.py`, `mocap/retarget/blender_face.py`.
Run the targeted checks with `runtime\python.exe -X utf8 -B -m unittest discover
-s tests -p "test_face_*.py" -v`; Blender integration scripts document their own
commands. Real footage, generated motion and character assets remain untracked.

## Native expression and continuity validation (2026-11-07)

- 83 targeted tests passed, including native category validation, side swapping,
  atomic capture, actor isolation, queue/settings behavior, legacy compatibility,
  shape mapping, provenance masks and independent VMD parsing. A 15-to-30 fps
  check verifies resampling before nonlinear MMD conversion, matching character
  application rather than changing the expression between export paths.
- The temporal module's 13 behavioral tests passed, covering short interpolation,
  limited extrapolation, finite neutral return, 30/60/120 fps timing consistency,
  reacquisition, stationary-noise suppression, retained fast blinks, no invented
  blink closure, actor barriers, absent observations and preservation of inputs.
  These are synthetic contract checks, not a facial-estimation accuracy benchmark.
- On the same private three-second full-body sample, native coefficients matched
  an independent call to the installed MediaPipe SDK within `1e-6`. Actual
  detections remained 43/90 frames. There were 23 predicted/fading frames and one
  ambiguous multi-face frame, which remained neutral and blocked propagation.
- Across detection-state transitions in that sample, coefficient RMS jumps were
  `0.227985` with missing frames zero-filled and `0.103468` after processing.
  Artificially hiding four otherwise detected frames gave coefficient MAE
  `0.018019` against the unhidden native scores, compared with `0.106603` for
  zero filling. This checks continuity and withheld coefficients, not facial
  expression ground-truth accuracy or general quality across videos.
- Real Qt subprocess checks completed both the manual capture dialog and the
  automatic reconstruction face stage. They loaded the 52-channel results and
  displayed observation/estimate counts without conflating them.
- Synthetic character integration verified 51 deforming native shape channels,
  12 mapped MMD channels, unchanged moving-body animation, sparse preview,
  native mesh FBX reimport and both merged/standalone VMD. Actual OpenGL readback
  differed from reference vertices by at most `7.45058e-9` app-coordinate units
  in this fixture. Source video and previous projects were preserved; fresh
  body-model inference was not rerun.
- A previously imported real PMX (28,407 vertices, 798 bones, 66 shape keys) was
  copied to a new private generation and accepted 11 mapped native expressions.
  Existing bone matrices at checked frames were unchanged; mesh FBX reimport and
  VMD retained the expected channel values. Sparse preview versus Blender had
  maximum vertex difference `3.57628e-7`, and actual OpenGL readback `2.38419e-7`.
  Original asset hashes were unchanged. That PMX source had rest-body animation;
  moving-body preservation was independently checked with the synthetic rig.
- A separate Blender replacement check upgraded an old managed `ワ` animation
  to a new `にやり` mapping. Obsolete managed curves were removed from the new
  generation, while unrelated artist shape animation, body curves and all old
  source files remained unchanged. Driver/absolute-shape failures and optional
  missing-shape behavior were also checked.

## Historical validation: initial five-channel integration (2026-11-07)

The following records describe the original reze-mipo five-expression capture
and its checkbox integration. They are retained as evidence for that version;
they do not measure the new native expression model or its temporal estimates.

- 54 targeted tests passed, covering analytic expressions, left/right conversion,
  crop-coordinate restoration, timing/offsets, missing-face handling, runtime
  guards, cancellation at process startup, GUI completion, transactional character
  failure/adoption, an independent VMD binary parse, checkbox persistence and queue
  isolation, per-actor/take routing, automatic track validity and skip behavior.
- The integrated GUI QProcess stage completed actual inference on a private copy
  of the three-second sample below, using the original actor tracking and dynamic
  crops. It detected 43/90 frames and published a valid automatic track. This
  checked the facial stage and queue wiring; fresh body-model inference was not
  rerun. A separate Blender check verified that optional facial application to a
  body-only rig leaves its animation file byte-identical, while explicit manual
  application and incompatible facial drivers still report errors.
- A local 3840×2160, 25 fps full-body clip was sampled over three seconds at
  30 fps. Full-frame detection found no face and correctly published no track.
  A manually chosen head region found a face in 43 of 90 samples; the complete
  worker took approximately 2.97 seconds on this PC. This is a single input's
  detection coverage and timing, not an accuracy or speed guarantee.
- Real MediaPipe inference and the GUI capture/region-selection flow completed.
  Chinese/Japanese model, request and output paths passed actual inference.
- The resulting 90-frame track was applied to a real PMX in a new test generation.
  All five VMD channels matched the saved coefficients; mesh FBX reimport retained
  all five animated shape keys at the checked frames. Synthetic moving-body
  checks independently verified unchanged body motion. No new body inference was
  needed for these checks, and source projects/assets were preserved.
- Sparse preview was compared with Blender evaluated vertices and actual OpenGL
  readback. The largest measured GPU difference was `2.38419e-7` app-coordinate
  units in the checked PMX frames. Existing PMX regression checks also passed.
- The pre-existing `test_longvideo_retarget.py` suite still has three errors in
  unchanged `mocap/retarget/source.py:55`: the installed SciPy rotation adapter
  rejects a read-only memmap. That independent baseline issue was not changed
  by this face integration. A clean Windows reinstall and manual MMD playback
  were not tested in this pass.
