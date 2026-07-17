#!/usr/bin/env python3
"""Evaluate CUB Image-Captions retrieval with a learned mapping checkpoint."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MappingMLP(nn.Module):
    def __init__(self, dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mmvae-src', required=True, help='Path to mmvaeplus/src')
    parser.add_argument('--data-root', required=True, help='Path passed to MMVAE+ args.datadir')
    parser.add_argument('--model-ckpt', required=True, help='MMVAE+ model checkpoint, e.g. model_50.rar')
    parser.add_argument('--args-path', required=True, help='MMVAE+ args checkpoint, e.g. args.rar')
    parser.add_argument('--mapping-ckpt', required=True, help='Mapping checkpoint with T_0_1 and T_1_0')
    parser.add_argument('--split', choices=['train', 'test'], default='test')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-json', default=None)
    return parser.parse_args()


def setup_imports(mmvae_src):
    sys.path.insert(0, str(Path(mmvae_src).resolve()))
    import models  # noqa: WPS433
    from utils import unpack_data_cubIC  # noqa: WPS433
    return models, unpack_data_cubIC


def load_model(args, models):
    saved_args = torch.load(args.args_path, map_location='cpu', weights_only=False)
    saved_args.cuda = args.device.startswith('cuda')
    saved_args.no_cuda = not saved_args.cuda
    saved_args.datadir = args.data_root
    model = models.MMVAEplus_CUB_Image_Captions(saved_args).to(args.device)
    state = torch.load(args.model_ckpt, map_location=args.device, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, saved_args


def collect_latents(model, split, batch_size, device, unpack_data_cubIC):
    train_loader, test_loader = model.getDataLoaders(batch_size, shuffle=False, device=device)
    loader = train_loader if split == 'train' else test_loader
    z0s, z1s = [], []
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            data = unpack_data_cubIC(data_t, device=device)
            mu0, _ = model.vaes[0].enc(data[0])
            mu1, _ = model.vaes[1].enc(data[1])
            z0s.append(mu0.cpu())
            z1s.append(mu1.cpu())
            if batch_idx % 50 == 0:
                print(f'collect {split} batch {batch_idx}', flush=True)
    z0 = torch.cat(z0s, dim=0).float()
    z1 = torch.cat(z1s, dim=0).float()
    image_ids = torch.arange(z0.size(0), dtype=torch.long) // 10
    unique_ids = torch.unique(image_ids)
    z0_img = z0[unique_ids * 10]
    print(f'{split} captions={z1.size(0)} images={z0_img.size(0)}')
    return {'z0_img': z0_img, 'z1': z1, 'image_ids': image_ids, 'unique_ids': unique_ids}


def load_mapping(path, dim, hidden_dim, device):
    maps = nn.ModuleDict({
        'T_0_1': MappingMLP(dim, hidden_dim),
        'T_1_0': MappingMLP(dim, hidden_dim),
    }).to(device)
    state = torch.load(path, map_location=device, weights_only=False)
    maps.load_state_dict(state)
    maps.eval()
    return maps


def ranks_image(sim, query_ids, target_ids):
    ranks = []
    q_np = query_ids.cpu().numpy()
    t_np = target_ids.cpu().numpy()
    for row_idx in range(sim.size(0)):
        order = torch.argsort(sim[row_idx], descending=True).cpu().numpy()
        hit = np.where(t_np[order] == q_np[row_idx])[0]
        ranks.append(int(hit[0]))
    return np.asarray(ranks)


def summarize(ranks):
    top = {f'top{k}': float((ranks < k).mean() * 100.0) for k in range(1, 11)}
    top['median_rank'] = int(np.median(ranks))
    top['mean_rank'] = float(np.mean(ranks))
    return top


def print_metrics(name, metrics):
    top = ' '.join(f'top{k}={metrics[f"top{k}"]:.2f}' for k in range(1, 11))
    print(f'{name} {top}')
    print(f'{name} median={metrics["median_rank"]} mean={metrics["mean_rank"]:.1f}')


def evaluate(data, maps, device):
    z0_img = data['z0_img']
    z1 = data['z1']
    unique = data['unique_ids']
    image_ids = data['image_ids']
    with torch.no_grad():
        mapped_img = maps['T_0_1'](z0_img.to(device)).cpu()
        sim01 = F.normalize(mapped_img, dim=-1) @ F.normalize(z1, dim=-1).T
        ranks01 = ranks_image(sim01, unique, image_ids)

        mapped_cap = maps['T_1_0'](z1.to(device)).cpu()
        sim10 = F.normalize(mapped_cap, dim=-1) @ F.normalize(z0_img, dim=-1).T
        ranks10 = ranks_image(sim10, image_ids, unique)

    m01 = summarize(ranks01)
    m10 = summarize(ranks10)
    avg = {f'top{k}': (m01[f'top{k}'] + m10[f'top{k}']) / 2.0 for k in range(1, 11)}
    return {'m0_to_m1': m01, 'm1_to_m0': m10, 'avg': avg}


def main():
    args = parse_args()
    models, unpack_data_cubIC = setup_imports(args.mmvae_src)
    model, saved_args = load_model(args, models)
    dim = saved_args.latent_dim_w + saved_args.latent_dim_z
    data = collect_latents(model, args.split, args.batch_size, args.device, unpack_data_cubIC)
    maps = load_mapping(args.mapping_ckpt, dim, args.hidden_dim, args.device)
    result = evaluate(data, maps, args.device)

    print_metrics('m0->m1', result['m0_to_m1'])
    print_metrics('m1->m0', result['m1_to_m0'])
    avg_line = ' '.join(f'top{k}={result["avg"][f"top{k}"]:.2f}' for k in range(1, 11))
    print(f'avg {avg_line}')

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'split': args.split,
            'num_images': int(data['z0_img'].size(0)),
            'num_captions': int(data['z1'].size(0)),
            'metrics': result,
        }
        out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(f'wrote {out}')


if __name__ == '__main__':
    main()
