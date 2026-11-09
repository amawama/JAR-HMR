JAR-HMR downtown_crossStreets_00 / V9 5-group montage - PIL float-coordinate fix + retry-safe group images
========================================================================================================

Upload ZIP to /root/data1/experiments/Human3R/wao_hmr/tmp_install/

RUN (GPU; uses official 896L + JAR-HMR):
  source /root/data1/miniconda3/etc/profile.d/conda.sh
  conda activate human3r
  cd /root/data1/models/Human3R
  python -B -u /root/data1/experiments/Human3R/wao_hmr/tmp_install/jar_hmr_crossstreets5_centerpairs_nofloor_horizontal_v9.zip --stage run --cleanup-temp

Quick CPU-only compositing from saved pairs (NO GPU inference):
  python -B -u /root/data1/experiments/Human3R/wao_hmr/tmp_install/jar_hmr_crossstreets5_centerpairs_nofloor_horizontal_v9.zip --stage compose

CPU-only smoke test:
  python -B -u /root/data1/experiments/Human3R/wao_hmr/tmp_install/jar_hmr_crossstreets5_centerpairs_nofloor_horizontal_v9.zip --stage self-test

Input: downtown_crossStreets_00 planned frames [70,90,110,130,150]; each frame contains
2 center-selected meshes: pale yellow + pale purple. JAR-HMR pred geometry and relative
3D placement from the existing pipeline. One horizontal camera elevation=0, azimuth=65,
fixed V5 soft shading, no floor/grid/shadows, white backdrop. 3x actual mesh rasterization
relative to V5. Per-group bbox blank gap=220px; across groups=520px; dynamic no-crop output.

Changes vs failed V8:
  - Pillow paste coordinates are always converted to integers, including float CLI settings.
  - Five rendered group PNGs and five small JSON sidecars are saved to group_images/ BEFORE
    final montage composition. Thus a later stitching failure can be recovered with --stage
    compose and new --group-gap-px/--outer-margin-px, without repeating GPU inference.
  - The output remains a single pure mesh montage (no source RGB strip and no ground).
  - If original V8 was run with --cleanup-temp, it has no saved NPZ or pair PNGs; one GPU
    re-run is unavoidable. V9 never falsely claims that V8 assets are present.

Output folder:
  /root/data1/experiments/Human3R/wao_hmr/results_safe/paper_ablation_official_v3/downtown_crossStreets_00_two_person_montage_v9/
    downtown_crossStreets_00_f00070_00090_00110_00130_00150_JAR_HMR_two_center_5groups_sidefront25_nofloor.png
    render_manifest.json
    group_images/group_00_f00070.png ... group_04_f00150.png
    group_images/group_00_f00070.json ... group_04_f00150.json

No edits to official code. No network downloads. --cleanup-temp only removes this run's
temporary GPU captures, never the five successfully rendered group PNGs.
