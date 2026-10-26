#!/usr/bin/env python3
"""Visualize CUB retrieval examples for a learned mapping checkpoint."""
import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


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
    parser.add_argument('--model-ckpt', required=True)
    parser.add_argument('--args-path', required=True)
    parser.add_argument('--mapping-ckpt', required=True)
    parser.add_argument('--split', choices=['train', 'test'], default='test')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--num-queries', type=int, default=8)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out-dir', default='visualizations/cub_retrieval')
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
    model.load_state_dict(torch.load(args.model_ckpt, map_location=args.device, weights_only=False), strict=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, saved_args


def get_image_dataset(model, split, batch_size, device):
    train_loader, test_loader = model.vaes[0].getDataLoaders(batch_size, shuffle=False, device=device)
    return train_loader.dataset if split == 'train' else test_loader.dataset


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
    return {'z0_img': z0_img, 'z1': z1, 'image_ids': image_ids, 'unique_ids': unique_ids}


def load_mapping(path, dim, hidden_dim, device):
    maps = nn.ModuleDict({
        'T_0_1': MappingMLP(dim, hidden_dim),
        'T_1_0': MappingMLP(dim, hidden_dim),
    }).to(device)
    maps.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    maps.eval()
    return maps


def caption_text(caption_dataset, cap_idx):
    data = caption_dataset.data[str(int(cap_idx))]
    if 'raw' in data:
        return data['raw']
    if 'text' in data:
        return data['text']
    if hasattr(caption_dataset, '_to_string'):
        item, _ = caption_dataset[int(cap_idx)]
        arr = item.detach().cpu().numpy() if torch.is_tensor(item) else np.asarray(item)
        return caption_dataset._to_string(arr)
    return f'caption #{cap_idx}'


def get_caption_dataset(model, split, batch_size, device):
    train_loader, test_loader = model.vaes[1].getDataLoaders(batch_size, shuffle=False, device=device)
    return train_loader.dataset if split == 'train' else test_loader.dataset


def rank_position(order, target_ids, query_id):
    hits = np.where(target_ids[order] == query_id)[0]
    return int(hits[0]) if len(hits) else None


def save_caption_panel(path, query_image_path, retrieved, title):
    fig = plt.figure(figsize=(11, 5))
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.7])
    ax_img = fig.add_subplot(grid[0, 0])
    ax_txt = fig.add_subplot(grid[0, 1])
    ax_img.imshow(Image.open(query_image_path).convert('RGB'))
    ax_img.set_title('query image')
    ax_img.axis('off')
    ax_txt.axis('off')
    lines = [title, '']
    for row in retrieved:
        marker = 'OK' if row['correct'] else ' '
        lines.append(f"{marker} #{row['rank']} image_id={row['image_id']}")
        lines.append(row['caption'][:220])
        lines.append('')
    ax_txt.text(0, 1, '\n'.join(lines), va='top', fontsize=9, family='monospace')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_image_grid(path, query_caption, retrieved, title):
    n = len(retrieved)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.8))
    if n == 1:
        axes = [axes]
    for ax, row in zip(axes, retrieved):
        ax.imshow(Image.open(row['image_path']).convert('RGB'))
        ok = 'correct' if row['correct'] else 'wrong'
        ax.set_title(f"#{row['rank']} {ok}\nid={row['image_id']}", fontsize=9)
        ax.axis('off')
    fig.suptitle(title + '\n' + query_caption[:180], fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models, unpack_data_cubIC = setup_imports(args.mmvae_src)
    model, saved_args = load_model(args, models)
    dim = saved_args.latent_dim_w + saved_args.latent_dim_z
    data = collect_latents(model, args.split, args.batch_size, args.device, unpack_data_cubIC)
    maps = load_mapping(args.mapping_ckpt, dim, args.hidden_dim, args.device)
    image_dataset = get_image_dataset(model, args.split, args.batch_size, args.device)
    caption_dataset = get_caption_dataset(model, args.split, args.batch_size, args.device)

    with torch.no_grad():
        mapped_img = maps['T_0_1'](data['z0_img'].to(args.device)).cpu()
        sim01 = F.normalize(mapped_img, dim=-1) @ F.normalize(data['z1'], dim=-1).T
        mapped_cap = maps['T_1_0'](data['z1'].to(args.device)).cpu()
        sim10 = F.normalize(mapped_cap, dim=-1) @ F.normalize(data['z0_img'], dim=-1).T

    num_images = data['z0_img'].size(0)
    num_captions = data['z1'].size(0)
    image_query_ids = random.sample(range(num_images), min(args.num_queries, num_images))
    caption_query_ids = random.sample(range(num_captions), min(args.num_queries, num_captions))

    examples = {'image_to_caption': [], 'caption_to_image': []}
    target_caption_image_ids = data['image_ids'].numpy()
    target_image_ids = data['unique_ids'].numpy()

    for image_id in image_query_ids:
        order = torch.argsort(sim01[image_id], descending=True).cpu().numpy()
        rank = rank_position(order, target_caption_image_ids, image_id)
        rows = []
        for rank_idx, cap_idx in enumerate(order[:args.top_k], start=1):
            rows.append({
                'rank': rank_idx,
                'caption_index': int(cap_idx),
                'image_id': int(target_caption_image_ids[cap_idx]),
                'correct': bool(target_caption_image_ids[cap_idx] == image_id),
                'caption': caption_text(caption_dataset, cap_idx),
            })
        img_path = image_dataset.samples[image_id][0]
        save_caption_panel(
            out_dir / f'image_to_caption_{image_id:05d}.png',
            img_path,
            rows,
            f'image query id={image_id}, first correct rank={rank}',
        )
        examples['image_to_caption'].append({'query_image_id': image_id, 'rank': rank, 'query_image_path': img_path, 'retrieved': rows})

    for cap_idx in caption_query_ids:
        query_image_id = int(target_caption_image_ids[cap_idx])
        order = torch.argsort(sim10[cap_idx], descending=True).cpu().numpy()
        rank = rank_position(order, target_image_ids, query_image_id)
        rows = []
        for rank_idx, image_id in enumerate(order[:args.top_k], start=1):
            rows.append({
                'rank': rank_idx,
                'image_id': int(image_id),
                'correct': bool(image_id == query_image_id),
                'image_path': image_dataset.samples[int(image_id)][0],
            })
        cap = caption_text(caption_dataset, cap_idx)
        save_image_grid(
            out_dir / f'caption_to_image_{cap_idx:05d}.png',
            cap,
            rows,
            f'caption query index={cap_idx}, target image id={query_image_id}, first correct rank={rank}',
        )
        examples['caption_to_image'].append({'query_caption_index': int(cap_idx), 'target_image_id': query_image_id, 'rank': rank, 'caption': cap, 'retrieved': rows})

    (out_dir / 'retrieval_examples.json').write_text(json.dumps(examples, indent=2), encoding='utf-8')
    print(f'wrote visualizations to {out_dir}')


if __name__ == '__main__':
    main()
