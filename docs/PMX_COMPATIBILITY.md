# PMX compatibility and quality checks

PMX import supports the native `上半身 → 上半身1 → 上半身2` and
`上半身 → 上半身2 → 上半身3` three-spine conventions, and true two-spine
characters. The numbered convention is accepted only after checking the native
torso, neck and shoulder ancestry. Intermediate control bones remain native;
the importer does not rename or insert physical bones. Props and accessories
without a humanoid skeleton are rejected with an explicit diagnostic.

Missing image references are reported in the job log and `import.json` rather
than discarding a usable character because a shared MMD toon image is absent.
Valid images are packed individually. Missing artwork is not reconstructed;
obtain the correct image from the model provider if its appearance is needed.

The simplified viewport reads MMD base-color textures, not sphere/toon lighting
maps, and multiplies them by the material's diffuse color and opacity. Alpha
cutouts are supported; this is not complete MMD shading, transparent-surface
sorting, material-morph evaluation, or hair/clothing simulation.

Preview textures retain their source dimensions when they fit the existing
256 MiB/image and 512 MiB/character RGBA8 budgets. Above those budgets, the
largest image is halved repeatedly with area averaging until it fits. Only the
preview cache changes: packed Blender images and source model files retain their
original dimensions. The log, import metadata and texture manifest record each
changed image and both sizes. Large pixel transfers use temporary disk mappings;
the texture budget is not a hard limit on the Blender process's total RAM or on
GPU mipmap storage. Cache version 2 rebuilds on first display and keeps old caches.

Reimport characters created with an incorrect spine mapping, then retarget them
again. Existing project assets and completed animations are not automatically
rewritten. Retarget again to obtain updated helper-bone preview/export behavior.
Native append-driven skin helpers and accessory IK chains are included in
preview and FBX sampling. VMD disables the body IK chains actually muted during
baking and retains active shoe/accessory IK. A mapped bone beneath an unsupported
animated append-transform control is rejected during retargeting with the control
names; use the standard model variant. Static shoulder-cancel controls and the
explicitly supported waist-cancel path remain available. Static import alone
does not establish that a complex auxiliary rig can be retargeted.

Active append transforms that reference their own bone are rejected before
Blender creates a cyclic dependency graph. Repair an explicit copy of the model
and use that same repaired PMX for subsequent VMD playback. The developer tool
`tools/repair_pmx_self_grants.py` accepts `--input` and a new `--output-directory`;
it removes only the self-grant flags/payloads, preserves other source bytes,
copies available local referenced textures and writes a repair receipt. It
never overwrites the original or an existing output directory. It does not
repair arbitrary rig cycles or recover missing images. For byte-preserving
relocation, this tool requires referenced textures to lie inside the model's
directory and rejects external texture references before creating an output.

For repeatable local import checks, run the installed Python with
`tools/validate_pmx_catalog.py --model-root <directory> --output <new-directory>`.
Repeat `--model-root` for multiple libraries. The audit uses actual bundled
Blender imports and records model hashes, source snapshots, per-model logs,
geometry checks, material/image diagnostics and timings. Licensed model-derived
outputs belong outside the public checkout. `tests/blender_pmx_motion_audit.py`
checks prepared motion against evaluated Blender skinning and a fresh PMX/VMD
round trip; synthetic and previously captured inputs must be labeled separately.
These tests do not establish capture accuracy against ground truth or native
MikuMikuDance rendering equivalence.

## Local validation, 2026-11-07

The local catalog contained 29 PMX files: 23 humanoid variants and six props.
Full import improved from five original humanoids to 19 original humanoids;
four additional models required explicit, separately saved self-grant repairs.
Eighteen originals and all four repaired copies passed the 45-frame CPU pipeline,
Blender bake, viewport skinning and fresh PMX/VMD round trip. A complex auxiliary
variant was explicitly rejected during retargeting; its standard variant passed.
Across five sampled mesh frames per accepted model, maximum viewport and VMD
vertex differences were 1.013 and 1.561 micrometers respectively. These are
internal consistency measurements, not capture-accuracy estimates.

Three ordinary rigs additionally passed 300-frame previously captured motion,
and another 120-frame capture passed a mesh-bearing FBX round trip. Non-contact
sole-proxy penetration of approximately 5–8 mm remained in the captured sample.
The targeted suites passed 76 tests, including existing face and memory tests;
real OpenGL readback verified material opacity and color multiplication.
All 29 original PMX hashes were unchanged. No new AI inference or native
MikuMikuDance playback was performed. Licensed assets and detailed local records
were kept outside the repository.
