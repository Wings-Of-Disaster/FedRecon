# FedRecon: Missing Modality Reconstruction in Heterogeneous Distributed Environments

This is the official implementation of **FedRecon: Missing Modality Reconstruction in Heterogeneous Distributed Environments**.

This release provides the CUB Image-Captions evaluation code for FedRecon. We provide scripts to evaluate a trained FedRecon mapping checkpoint and visualize cross-modal retrieval results on CUB.

For the MVAE+ backbone, please refer to the [mmvaeplus](https://github.com/epalu/mmvaeplus) repository. For heterogeneous federated multimodal experiments, please refer to the [fed-multimodal](https://github.com/usc-sail/fed-multimodal) framework.

## Environment Setup

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

You also need a local MMVAE+ source tree and the CUB Image-Captions data prepared in the MMVAE+ format.

## Released Files

```text
cub_eval/
  evaluate_cub_mapping.py       # CUB retrieval evaluation
  visualize_cub_retrieval.py    # CUB retrieval visualization
requirements.txt
```

We also provide a trained FedRecon mapping checkpoint for reference. The checkpoint contains two MLP mappings:

- `T_0_1`: image latent -> caption latent
- `T_1_0`: caption latent -> image latent

In CUB Image-Captions:

- `m0` is the image modality.
- `m1` is the caption modality.
- `m0 -> m1` is image-to-caption retrieval.
- `m1 -> m0` is caption-to-image retrieval.

## CUB Evaluation

Run quantitative retrieval evaluation with:

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

The script reports Top@1 to Top@10, median rank, and mean rank for both retrieval directions.

## Visualization

Generate qualitative retrieval examples with:

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

The script saves image-to-caption panels, caption-to-image retrieval grids, and a `retrieval_examples.json` file.

## Notes

Each CUB image has ten captions. For image-to-caption retrieval, any caption belonging to the query image is counted as correct. For caption-to-image retrieval, the paired image is counted as correct.
