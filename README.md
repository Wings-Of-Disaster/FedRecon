# FedRecon CUB Evaluation Utilities

This repository contains lightweight evaluation utilities for CUB Image-Captions retrieval experiments with a trained MMVAE+ checkpoint and a learned FedRecon mapping checkpoint.

The scripts are path-agnostic: pass local paths with command-line arguments. No cluster-specific paths, usernames, logs, or private data are stored in this repository.

## Modalities

- `m0`: image modality
- `m1`: caption modality
- `m0 -> m1`: image-to-caption retrieval
- `m1 -> m0`: caption-to-image retrieval

## Expected Files

You need local copies of:

- MMVAE+ source tree, passed via `--mmvae-src`
- CUB data root used by MMVAE+, passed via `--data-root`
- MMVAE+ model checkpoint, passed via `--model-ckpt`
- MMVAE+ args file, passed via `--args-path`
- mapping checkpoint with `T_0_1` and `T_1_0`, passed via `--mapping-ckpt`

The mapping checkpoint expected by the current experiment is the all-data MLP mapping checkpoint, e.g. `mapping_cub_ratio_alldata_best.pt`.

## Evaluate Retrieval

```bash
python cub_eval/evaluate_cub_mapping.py \
  --mmvae-src /path/to/mmvaeplus/src \
  --data-root /path/to/mmvaeplus/src/data \
  --model-ckpt /path/to/model_50.rar \
  --args-path /path/to/args.rar \
  --mapping-ckpt /path/to/mapping_cub_ratio_alldata_best.pt \
  --split test \
  --output-json results/cub_test_metrics.json
```

The script reports top@1..10, median rank, and mean rank for both retrieval directions.

## Visualize Retrieval

```bash
python cub_eval/visualize_cub_retrieval.py \
  --mmvae-src /path/to/mmvaeplus/src \
  --data-root /path/to/mmvaeplus/src/data \
  --model-ckpt /path/to/model_50.rar \
  --args-path /path/to/args.rar \
  --mapping-ckpt /path/to/mapping_cub_ratio_alldata_best.pt \
  --split test \
  --num-queries 8 \
  --top-k 5 \
  --out-dir visualizations/cub_test
```

This creates:

- image-to-caption text panels
- caption-to-image retrieval grids
- a `retrieval_examples.json` file containing ranks and retrieved indices

## Notes

The CUB Image-Captions setup uses ten captions per image. For image-to-caption retrieval, any of the ten captions belonging to the query image is counted as a correct hit. For caption-to-image retrieval, the exact paired image is counted as correct.
