# JAR-HMR: Joint-guided and Root-anchored Human Mesh Refinement

JAR-HMR is a human mesh refinement framework built on top of [Human3R](https://github.com/fanegg/Human3R). It uses external 2D keypoints and initial 3D body structure to construct bounded joint corrections (JCR), realizes them through a learned joint-to-SMPL pose mapper (Learned J2S), and preserves the original spatial root anchor using analytic translation compensation (RootComp).

This README focuses on **reproducing the inference/evaluation environment and preparing the required model weights**. Human3R is the frozen backbone; JAR-HMR uses additional project-specific checkpoints and 2D observations.

## 1. Environment

### Tested setup

The project was developed and evaluated using the following environment:

| Item | Version / configuration |
| --- | --- |
| Operating system | Linux (GPU server) |
| GPU | NVIDIA GeForce RTX 4090, 24 GB |
| NVIDIA driver | 535.104.05 (recorded test environment) |
| Python | 3.11 |
| PyTorch | 2.4.1 + cu121 |
| torchvision | 0.19.1 + cu121 |
| CUDA toolkit | 12.1 |
| Conda environment | `human3r` |
| Headless execution | `HUMAN3R_NO_VIEWER=1` |

> These are the **recorded research environment versions**, not a complete dependency lockfile. The [upstream Human3R installation guide](https://github.com/fanegg/Human3R#installation) currently documents a CUDA 12.4 installation example. Choose a PyTorch/CUDA combination compatible with your driver and build the CUDA extensions against it. Other packages should follow the upstream `requirements.txt` and any project-provided dependency files.

### Install Human3R

```bash
git clone https://github.com/fanegg/Human3R.git
cd Human3R

conda create -n human3r python=3.11 cmake -y
conda activate human3r

# Reproduce the PyTorch version used in our experiments.
python -m pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121

# Install the upstream dependencies. Recheck torch/torchvision afterwards
# if the dependency resolver changes them.
python -m pip install -r requirements.txt
python -m pip install accelerate evo open3d

# Compile the RoPE CUDA extension required by Human3R.
cd src/croco/models/curope
python setup.py build_ext --inplace
cd ../../../../

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"
```

Human3R also documents optional/environment-dependent setup for `gsplat`, compiler packages, and `llvm-openmp`. Consult the [official installation instructions](https://github.com/fanegg/Human3R) if any compiled dependency or extension is missing. In the recorded server environment, the 3DPW annotation preparation additionally required working `chumpy` and `evo` installations.

## 2. Model Weights

### 2.1 Human3R pretrained weights (official downloads)

**Official weights page:** https://huggingface.co/faneggg/human3r/tree/main  
**Official code repository:** https://github.com/fanegg/Human3R

| Checkpoint |描述| Usage |
| --- | --- | --- |
| [`human3r_896L.pth`](https://huggingface.co/faneggg/human3r/blob/main/human3r_896L.pth) | Human3R ViT-L / 896 | **Main backbone for JAR-HMR official experiments** |
| [`human3r_672S.pth`](https://huggingface.co/faneggg/human3r/blob/main/human3r_672S.pth) | Human3R ViT-S / 672 | Optional lightweight baseline |
| [`human3r_672B.pth`](https://huggingface.co/faneggg/human3r/blob/main/human3r_672B.pth) | Human3R ViT-B / 672 | Optional backbone |
| [`human3r_672L.pth`](https://huggingface.co/faneggg/human3r/blob/main/human3r_672L.pth) | Human3R ViT-L / 672 | Optional backbone |

Download the main checkpoint into the **upstream Human3R `src/` directory**:

```bash
# Run inside the Human3R repository.
python -m pip install -U huggingface_hub
hf download faneggg/human3r human3r_896L.pth --local-dir ./src

# Optional, smaller Human3R checkpoint:
hf download faneggg/human3r human3r_672S.pth --local-dir ./src
```

The upstream documentation also uses `huggingface-cli download` for the same files; use that command if you have an older compatible Hugging Face CLI. On servers without internet access, download the checkpoints on another machine and transfer them to `Human3R/src/`.

Expected locations:

```text
Human3R/
└── src/
    ├── human3r_896L.pth
    └── human3r_672S.pth          # optional
```

### 2.2 JAR-HMR checkpoints and auxiliary observations

In addition to Human3R, the JAR-HMR implementation expects the following **project-specific** files:

| File | Purpose | Default path relative to `JAR_HMR_ROOT` |
| --- | --- | --- |
| JCR checkpoint (`best.pt`) | Bounded 3D joint correction model | `training/t1_jcr/model_multiseq_magnitude_calibrated/best.pt` |
| Learned J2S checkpoint | Learned joint-to-SMPL pose proposal | `assets/learned_bounded_pose/joint_to_smpl_learned_bounded.pt` |
| `e4_2d_observations.pkl` | Cached external 2D keypoints / detections; **not a model weight** | `assets/e4_2d_observations.pkl` |

These files are **not included in the official Human3R weights**. Their distribution/download locations must be provided separately by the JAR-HMR project maintainer; this README does not claim that they are publicly hosted.

Relevant environment variables:

```bash
export WAO_HMR_STAGE=joint_to_smpl_bounded
export WAO_J2S_SOLVER=learned_bounded
export WAO_JCR_CKPT="$JAR_HMR_ROOT/training/t1_jcr/model_multiseq_magnitude_calibrated/best.pt"
export WAO_J2S_LEARNED_CKPT="$JAR_HMR_ROOT/assets/learned_bounded_pose/joint_to_smpl_learned_bounded.pt"
export WAO_JCR_OBS="$JAR_HMR_ROOT/assets/e4_2d_observations.pkl"
```

**Please do not commit pretrained `.pth`/`.pt` files, third-party body-model assets, the 3DPW dataset, or large cached observations to the source repository without checking the respective licenses.**

## 3. SMPL / SMPL-X and Dataset Preparation

Human3R requires licensed SMPL/SMPL-X body-model assets and auxiliary conversion/regression files. Follow the upstream registration and download procedure:

```bash
# From Human3R/; registration or acceptance of the body-model licenses may be required.
bash scripts/fetch_smplx.sh
```

The tested setup includes:

```text
Human3R/src/models/smpl/J_regressor_h36m.npy
Human3R/src/models/smplx/smplx2smpl.pkl
```

### 3.1 Dataset overview and reproduction scope

Human3R's [official evaluation guide](https://github.com/fanegg/Human3R/blob/master/docs/eval.md) covers **3DPW, EMDB, and RICH**. These benchmarks serve different purposes:

| Dataset / split | Main evaluation task |访问| Status in this README |
| --- | --- | --- | --- |
| **3DPW (test)** | Camera-coordinate human mesh reconstruction | Official dataset download, subject to terms | **JAR-HMR full-test evaluation performed** |
| **EMDB1** | Camera-relative / local human mesh reconstruction | **Institutional-email application and approval required** | JAR-HMR evaluation conducted |
| **EMDB2** | Global/world-coordinate human motion recovery | **Same approved EMDB access** | JAR-HMR evaluation conducted |
| **RICH (test)** | Human reconstruction and motion with known scene/camera geometry | Registration at official RICH site | JAR-HMR evaluation conducted |

### 3.2 3DPW (main reported experiment)

Download 3DPW from its [official website](https://virtualhumans.mpi-inf.mpg.de/3DPW/), subject to its license. Follow the [Human3R evaluation instructions](https://github.com/fanegg/Human3R/blob/master/docs/eval.md):

```bash
# Run from Human3R/ after configuring the dataset and body-model paths.
PYTHONPATH=src:. python -m eval.dataset.prepare_3dpw "create_annots()"
```

Expected annotation file: `Human3R/eval/global_human/annots/3dpw_test.pkl`. Our recorded full test protocol uses **24 sequences, 24,547 frames, and 35,080 evaluated human instances**. Verify that local dataset paths and the official test annotations match the chosen installation; the research server stores 3DPW under `/root/data1/datasets/3DPW`.

### 3.3 EMDB — official application, terms of use, and evaluation splits

**Official project and dataset information:** https://eth-ait.github.io/emdb/  
**Official access request and license:** https://emdb.ait.ethz.ch/  
**Official dataset/code repository:** https://github.com/eth-ait/emdb

EMDB (*The Electromagnetic Database of Global 3D Human Pose and Shape in the Wild*, Kaufmann et al., ICCV 2023) contains synchronized video, reconstructed body pose/shape, and global trajectory information. The EMDB authors provide access **only after an application is approved**. An automated public download command for the original EMDB data is **not** provided by this project.

**How to obtain EMDB:**

1. Open the [official EMDB registration page](https://emdb.ait.ethz.ch/).
2. Apply using a **valid institutional email address**; the official terms state that requests from non-institutional accounts are rejected.
3. Describe the intended academic use specifically, including the reconstruction task and evaluation splits, and accept the official license terms.
4. After approval, follow the **download instructions delivered by the EMDB organizers**. Retain the dataset locally; do not upload it to this source repository.

For example, an appropriate *intended usage* description **when it accurately reflects the applicant's project** is:

> We plan to use EMDB exclusively for non-commercial academic research to evaluate monocular RGB-based 3D human mesh reconstruction and global human motion recovery. We will compare camera-relative accuracy on EMDB1 and world-coordinate trajectory consistency on EMDB2, using authorized dataset access and the published evaluation protocols. We will not attempt participant identification or redistribute any EMDB data or derived dataset assets, and we will cite the EMDB publication.

**EMDB terms that users must respect (summarized; the official license takes precedence):**

- **Non-commercial academic use only.** Commercial use, including training AI systems for commercial applications, is prohibited.
- **No redistribution.** Do not publish the original data, converted/derived dataset parameters, or any dataset content in public repositories or third-party mirrors.
- **Privacy obligations.** Do not attempt participant re-identification, identity recognition, or prohibited surveillance/misleading uses.
- **Citation is mandatory** for publications using EMDB; see [Kaufmann et al., ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Kaufmann_EMDB_The_Electromagnetic_Database_of_Global_3D_Human_Pose_and_ICCV_2023_paper.html).
- Dataset access and continued use are governed by the current [official terms](https://emdb.ait.ethz.ch/), including possible updates or withdrawal of access.

**Preparing the evaluation data:** Human3R explicitly follows [GVHMR's dataset preparation](https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md) for EMDB. Once original EMDB access has been authorized, obtain the appropriate GVHMR evaluation support annotations (e.g., its `EMDB/hmr4d_support/` package) according to GVHMR's instructions, and configure Human3R's local EMDB data paths. The support annotations **do not replace** the license requirement to obtain the original EMDB data from its official source.

**Official Human3R entry points (Human3R baseline):**

```bash
# From Human3R/, after approved access, full dataset preparation,
# and setting the checkpoint path in the upstream launch scripts.
# run.sh evaluates both 3DPW and EMDB1.
CUDA_VISIBLE_DEVICES=0 bash eval/global_human/run.sh

# EMDB2: global/world-coordinate evaluation.
CUDA_VISIBLE_DEVICES=0 bash eval/global_human/run_emdb2.sh
```

The upstream shell scripts may default to `src/human3r.pth`; **set their weight path to `src/human3r_896L.pth`** (or pass the correct path directly to `launch.py`) before running. These commands evaluate the **upstream Human3R baseline**, not JAR-HMR, unless the JAR-HMR evaluation hook and EMDB-specific external observations are also installed and checked. Record EMDB1 and EMDB2 results separately. The full dataset and all derived resources must remain outside the public GitHub repository.

### 3.4 RICH — official registration and Human3R annotations

**Official project, registration, and download portal:** https://rich.is.tue.mpg.de/  
**Official paper:** [Huang et al., *Capturing and Inferring Dense Full-Body Human-Scene Contact*, CVPR 2022](https://rich.is.tue.mpg.de/)

RICH (*Real scenes, Interaction, Contact and Humans*) provides multi-view images, human body ground truth, scene geometry, and body-scene contact annotations. It is suitable for assessing reconstruction in complex indoor/outdoor scenes, with real contact and occlusion.

1. Register/sign in at the [official RICH download portal](https://rich.is.tue.mpg.de/), accept its applicable license, and obtain the **test-set images and annotations needed for the selected benchmark**. Do not redistribute the licensed dataset files.
2. Follow the [Human3R evaluation guide](https://github.com/fanegg/Human3R/blob/master/docs/eval.md), which uses GVHMR-style RICH annotations. In particular, prepare these two files in `Human3R/eval/global_human/annots/RICH/`:
   - [`rich_test_labels.pt` — evaluation SMPL annotations](https://drive.google.com/file/d/17fbG1IsN6DfF_KwYWR2FbNZFKvs9waVb/view?usp=drive_link) (linked by Human3R/GVHMR).
   - [`cam2params.pt` — camera ground-truth parameters](https://github.com/zju3dv/GVHMR/blob/088caff492aa38c2d82cea363b78a3c65a83118f/hmr4d/dataset/rich/resource/cam2params.pt) (linked by Human3R/GVHMR).
3. Configure the RICH image/data paths used by the Human3R dataset loader, and ensure the evaluation split and camera geometry agree with those annotations.

```text
Human3R/
└── eval/global_human/annots/RICH/
    ├── rich_test_labels.pt
    └── cam2params.pt
```

**Official Human3R entry point (Human3R baseline):**

```bash
# From Human3R/ after preparing the images, annotations, and body models.
# Update the upstream script's weight path to human3r_896L.pth as needed.
CUDA_VISIBLE_DEVICES=0 bash eval/global_human/run_rich.sh
```

As with EMDB, an actual **JAR-HMR-on-RICH** evaluation requires a RICH-specific 2D observation cache and verification that the JCR/Learned J2S matching and camera projection operate correctly on RICH.

## 4. Full 3DPW Evaluation

The JAR-HMR experiment integrates its refinement hook into the Human3R evaluation flow. **An unmodified upstream `launch.py` evaluates Human3R alone, not JAR-HMR.** Before using the following example, ensure that the JAR-HMR method files and the corresponding evaluation integration/patch from this project have been installed and verified.

Example paths from the tested research server:

```bash
export HUMAN3R_ROOT=/root/data1/models/Human3R
export JAR_HMR_ROOT=/root/data1/experiments/Human3R/wao_hmr

conda activate human3r
export HUMAN3R_NO_VIEWER=1
export PYOPENGL_PLATFORM=egl
export WAO_HMR_STAGE=joint_to_smpl_bounded
export WAO_J2S_SOLVER=learned_bounded
export WAO_JCR_CKPT="$JAR_HMR_ROOT/training/t1_jcr/model_multiseq_magnitude_calibrated/best.pt"
export WAO_J2S_LEARNED_CKPT="$JAR_HMR_ROOT/assets/learned_bounded_pose/joint_to_smpl_learned_bounded.pt"
export WAO_JCR_OBS="$JAR_HMR_ROOT/assets/e4_2d_observations.pkl"

cd "$HUMAN3R_ROOT"
export PYTHONPATH=src:.

accelerate launch \
  --num_processes 1 --num_machines 1 \
  --mixed_precision no --dynamo_backend no \
  --main_process_port 29851 \
  eval/global_human/launch.py \
  --weights ./src/human3r_896L.pth \
  --output_dir "$JAR_HMR_ROOT/results_safe/paper_ablation_official_v3/full/jar_hmr_full" \
  --eval_dataset 3dpw \
  --size 512 \
  --reset_interval 100 \
  --use_ttt3r
```

The output directory above is illustrative; choose a new, non-conflicting folder for each experiment. Report the official `MPJPE`, `PA-MPJPE`, `PVE`, `PA-PVE`, `Metric-MPJPE`, and `Root-Error` values together with the exact checkpoint versions and evaluation settings. A successful process exit alone is insufficient to verify complete dataset coverage.

## 5. Reproducibility Notes

- Keep the Human3R backbone frozen for JAR-HMR inference and document the exact Human3R source revision used.
- Store the exact JCR and Learned J2S checkpoint identities (e.g., SHA256), and the version of the external 2D observation cache.
- Ensure every full experiment uses the **same 24 official 3DPW test sequences** and the same matching/evaluation protocol.
- Do not infer global/world accuracy from zero-filled metrics when world-coordinate evaluation is disabled; use the metrics actually computed by the chosen official evaluator.
- The pretrained Human3R checkpoints, SMPL/SMPL-X assets, 3DPW, EMDB, and RICH are subject to their respective original licenses. Do not redistribute third-party data or converted annotations without explicit permission. Cite [Human3R](https://fanegg.github.io/Human3R/), [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/), [EMDB](https://eth-ait.github.io/emdb/), and/or [RICH](https://rich.is.tue.mpg.de/) whenever they are used.

## References

- [Human3R — official repository](https://github.com/fanegg/Human3R)
- [Human3R — official pretrained checkpoints](https://huggingface.co/faneggg/human3r/tree/main)
- [Human3R — official evaluation instructions](https://github.com/fanegg/Human3R/blob/master/docs/eval.md)
- [3DPW — official dataset](https://virtualhumans.mpi-inf.mpg.de/3DPW/)
- [EMDB — official dataset website, registration and licensing](https://eth-ait.github.io/emdb/) · [access form / terms](https://emdb.ait.ethz.ch/) · [paper (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Kaufmann_EMDB_The_Electromagnetic_Database_of_Global_3D_Human_Pose_and_ICCV_2023_paper.html)
- [RICH — official dataset and registration portal](https://rich.is.tue.mpg.de/) · [paper (CVPR 2022)](https://rich.is.tue.mpg.de/)
- [GVHMR — dataset support / preprocessing instructions](https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md)
