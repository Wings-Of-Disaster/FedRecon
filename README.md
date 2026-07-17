# FedRecon: Missing Modality Reconstruction in Heterogeneous Distributed Environments

This is the official implementation of **FedRecon: Missing Modality Reconstruction in Heterogeneous Distributed Environments**.

This release provides the CUB Image-Captions cross-modal generation evaluation code for FedRecon. We provide scripts to evaluate a trained FedRecon mapping checkpoint and visualize cross-modal generation results on CUB.

We include the following upstream repositories as references:

- [mmvaeplus](https://github.com/epalu/mmvaeplus): MVAE+ backbone for multimodal latent representations.
- [fed-multimodal](https://github.com/usc-sail/fed-multimodal): heterogeneous federated multimodal learning framework.


## Environment Setup

Clone this repository with submodules:

```bash
git clone --recursive https://github.com/Wings-Of-Disaster/FedRecon.git
```

If you have already cloned the repository, initialize the reference repositories with:

```bash
git submodule update --init --recursive
```

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

You also need the CUB Image-Captions data prepared in the MMVAE+ format.

## Released Files

```text
cub_eval/
  evaluate_cub_mapping.py       # CUB cross-modal generation evaluation
  visualize_cub_retrieval.py    # CUB cross-modal generation visualization
external/
  mmvaeplus/                    # Reference MVAE+ repository
  fed-multimodal/               # Reference federated multimodal repository
checkpoints/
  mmvae_cub_model_50.rar        # Trained MMVAE+ CUB backbone checkpoint
  mmvae_cub_args.rar            # MMVAE+ CUB checkpoint arguments
  fedrecon_cub_mapping.pt       # Trained FedRecon CUB mapping checkpoint
requirements.txt
```

We provide a trained FedRecon mapping checkpoint for reference. The checkpoint contains two MLP mappings:

- `T_0_1`: image latent -> caption latent
- `T_1_0`: caption latent -> image latent

In CUB Image-Captions:

- `m0` is the image modality.
- `m1` is the caption modality.
- `m0 -> m1` generates a caption-side latent from an image latent, then evaluates it with nearest-neighbor matching.
- `m1 -> m0` generates an image-side latent from a caption latent, then evaluates it with nearest-neighbor matching.

## CUB Evaluation

Run quantitative cross-modal generation evaluation with:

```bash
python cub_eval/evaluate_cub_mapping.py \
  --mmvae-src external/mmvaeplus/src \
  --data-root /path/to/mmvaeplus/src/data \
  --model-ckpt checkpoints/mmvae_cub_model_50.rar \
  --args-path checkpoints/mmvae_cub_args.rar \
  --mapping-ckpt checkpoints/fedrecon_cub_mapping.pt \
  --split test \
  --output-json results/cub_test_metrics.json
```

The script reports both the no-mapping baseline and the FedRecon mapping result. It generates the target-modality latent and evaluates it by nearest-neighbor retrieval against ground-truth CUB samples. It reports Top@1 to Top@10, median rank, and mean rank for both directions.

## Visualization

Generate qualitative cross-modal generation examples with:

```bash
python cub_eval/visualize_cub_retrieval.py \
  --mmvae-src external/mmvaeplus/src \
  --data-root /path/to/mmvaeplus/src/data \
  --model-ckpt checkpoints/mmvae_cub_model_50.rar \
  --args-path checkpoints/mmvae_cub_args.rar \
  --mapping-ckpt checkpoints/fedrecon_cub_mapping.pt \
  --split test \
  --num-queries 8 \
  --top-k 5 \
  --out-dir visualizations/cub_test
```

The script saves image-to-caption generation panels, caption-to-image generation grids, and a `retrieval_examples.json` file with nearest-neighbor matches.

## Notes

Each CUB image has ten captions. For image-to-caption evaluation, any caption belonging to the query image is counted as correct. For caption-to-image evaluation, the paired image is counted as correct.
