# PMX two-spine extension: retired

PMX retargeting now follows the original toyxyz_RGBMocap implementation at
commit `75cda26531b47f41c75d4aa4a82c4f135daa6eea`. It requires 22 real mapped
bones, including `上半身`, `上半身1` and `上半身2`. The former virtual Spine1 and
alternate `上半身 → 上半身2 → 上半身3` mappings are no longer supported.

Existing compatible three-spine imports remain usable. Cached extended mappings
are rejected before source preparation; neither the cache nor the original PMX
is migrated or rewritten. Reimport a model supported by the original mapping,
then use **Retarget > [source person]**. Old generated animations remain saved;
retarget and export again to obtain the restored behavior. Body inference does
not need to be repeated.

Use the current mapping/cache checks with:

```text
runtime\python.exe -X utf8 -m unittest discover -s tests -p "test_pmx_upstream*.py" -v
```

`tests/pmx_two_spine_motion_fixture.py` and
`tests/blender_pmx_two_spine_integration.py` retain historical filenames, but
now accept only the original three-spine PMX mapping. Their outputs belong in a
new private validation folder, never an existing user project. See
[PMX compatibility](PMX_COMPATIBILITY.md) for current behavior and limits.

The following records describe the removed extension. They do not establish
support or validation for the restored implementation.

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
