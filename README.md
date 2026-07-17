# FedRecon: Missing Modality Reconstruction in Heterogeneous Distributed Environments

This repository provides the official release of the CUB Image-Captions evaluation utilities for **FedRecon: Missing Modality Reconstruction in Heterogeneous Distributed Environments**.

FedRecon studies missing-modality reconstruction and cross-modal retrieval under heterogeneous multimodal settings. In this release, we provide a compact evaluation package for checking a trained FedRecon mapping checkpoint on the CUB Image-Captions benchmark.

## Release Scope

This repository includes:

- CUB image-to-caption and caption-to-image retrieval evaluation.
- Top@1 to Top@10, median rank, and mean rank reporting.
- Retrieval visualization scripts for qualitative inspection.
- Path-agnostic command-line interfaces for local checkpoints and datasets.

The full training pipeline is not included in this release because it depends on experiment-specific infrastructure and third-party training frameworks that are maintained separately. We release the CUB evaluation code to make checkpoint evaluation and qualitative analysis straightforward and reproducible.

For the underlying MVAE+ model implementation, please refer to the [mmvaeplus](https://github.com/epalu/mmvaeplus) repository. For heterogeneous federated multimodal experiments, please refer to the [fed-multimodal](https://github.com/usc-sail/fed-multimodal) framework.

## Environment Setup

Install the lightweight evaluation dependencies with:

```bash
pip install -r requirements.txt
```

The scripts also require a local MMVAE+ source tree and local CUB Image-Captions data prepared in the format expected by MMVAE+.

## CUB Evaluation

Evaluate a trained FedRecon mapping checkpoint on CUB Image-Captions:

```bash
python cub_eval/evaluate_cub_mapping.py \
  --mmvae-src /path/to/mmvaeplus/src \
  --data-root /path/to/mmvaeplus/src/data \
  --model-ckpt /path/to/model_50.rar \
  --args-path /path/to/args.rar \
  --mapping-ckpt /path/to/mapping_checkpoint.pt \
  --split test \
  --output-json results/cub_test_metrics.json
```

In the CUB setup used here:

- `m0` denotes the image modality.
- `m1` denotes the caption modality.
- `m0 -> m1` evaluates image-to-caption retrieval.
- `m1 -> m0` evaluates caption-to-image retrieval.

Each CUB image has ten captions. For image-to-caption retrieval, any caption belonging to the query image is counted as correct. For caption-to-image retrieval, the paired image is counted as correct.

## Qualitative Visualization

Generate retrieval examples for inspection:

```bash
python cub_eval/visualize_cub_retrieval.py \
  --mmvae-src /path/to/mmvaeplus/src \
  --data-root /path/to/mmvaeplus/src/data \
  --model-ckpt /path/to/model_50.rar \
  --args-path /path/to/args.rar \
  --mapping-ckpt /path/to/mapping_checkpoint.pt \
  --split test \
  --num-queries 8 \
  --top-k 5 \
  --out-dir visualizations/cub_test
```

The visualization script writes image-to-caption panels, caption-to-image grids, and a `retrieval_examples.json` summary file.

## Repository Layout

```text
cub_eval/
  evaluate_cub_mapping.py       # Quantitative CUB retrieval evaluation
  visualize_cub_retrieval.py    # Qualitative CUB retrieval visualization
requirements.txt                # Evaluation dependencies
```
