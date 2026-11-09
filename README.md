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

| Checkpoint | Description | Usage |
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

Human3R requires licensed SMPL/SMPL-X assets, including auxiliary conversion/regression files. Follow the upstream model download procedure and any required registration:

```bash
# From Human3R/; may require registering for SMPL/SMPL-X access.
bash scripts/fetch_smplx.sh
```

The tested setup includes:

```text
Human3R/src/models/smpl/J_regressor_h36m.npy
Human3R/src/models/smplx/smplx2smpl.pkl
```

For full 3DPW evaluation, download 3DPW from its [official website](https://virtualhumans.mpi-inf.mpg.de/3DPW/), subject to its license. Prepare the test annotations according to [Human3R's official evaluation guide](https://github.com/fanegg/Human3R/blob/master/docs/eval.md):

```bash
# Run from Human3R/ after setting the dataset and body-model paths.
PYTHONPATH=src:. python -m eval.dataset.prepare_3dpw "create_annots()"
```

Expected result: `Human3R/eval/global_human/annots/3dpw_test.pkl`. The recorded full test protocol uses **24 sequences, 24,547 frames, and 35,080 human instances**. Ensure that the dataset paths in the local Human3R configuration match your installation; our server's test data were stored under `/root/data1/datasets/3DPW`.

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
- The pretrained Human3R checkpoints, SMPL/SMPL-X assets, and 3DPW data are distributed under their respective original licenses. Credit and cite the [Human3R paper](https://fanegg.github.io/Human3R/) and the [3DPW dataset](https://virtualhumans.mpi-inf.mpg.de/3DPW/) when appropriate.

## References

- [Human3R — official repository](https://github.com/fanegg/Human3R)
- [Human3R — official pretrained checkpoints](https://huggingface.co/faneggg/human3r/tree/main)
- [Human3R — official evaluation instructions](https://github.com/fanegg/Human3R/blob/master/docs/eval.md)
- [3DPW — official dataset](https://virtualhumans.mpi-inf.mpg.de/3DPW/)
