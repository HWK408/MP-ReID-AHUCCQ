import argparse
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import cfg
from datasets.curriculum import domain_from_view
from datasets.mmmp import MMMP
from model.make_model_uniprompt import load_clip_to_cpu


class TrainImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, pid, camid, viewid = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, index, img_path, pid, camid, viewid


def norm_path(path):
    return os.path.normcase(os.path.normpath(path))


def build_transform(cfg):
    return T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])


def extract_features(cfg, samples, batch_size, device):
    h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
    w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
    clip_model = load_clip_to_cpu(cfg.MODEL.NAME, h_resolution, w_resolution, cfg.MODEL.STRIDE_SIZE[0])
    image_encoder = clip_model.visual.to(device).eval()

    dataset = TrainImageDataset(samples, build_transform(cfg))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS)

    features = [None] * len(samples)
    with torch.no_grad():
        for images, indices, _, _, _, _ in loader:
            images = images.to(device)
            visual_output = image_encoder(images)
            image_features_proj = visual_output[2]
            if cfg.MODEL.NAME == "RN50":
                batch_features = image_features_proj[0]
            else:
                batch_features = image_features_proj[:, 0]
            batch_features = F.normalize(batch_features.float(), dim=-1).cpu()
            for idx, feat in zip(indices.tolist(), batch_features):
                features[idx] = feat

    return torch.stack(features, dim=0)


def compute_scores(samples, features):
    by_pid = defaultdict(list)
    for idx, (_, pid, _, _) in enumerate(samples):
        by_pid[pid].append(idx)

    scores = torch.zeros(len(samples), dtype=torch.float32)
    for indices in by_pid.values():
        pid_features = features[indices]
        sim = pid_features @ pid_features.t()
        domains = [domain_from_view(samples[idx][3]) for idx in indices]

        for local_idx, global_idx in enumerate(indices):
            candidate_indices = [
                j for j, domain in enumerate(domains)
                if j != local_idx and domain != domains[local_idx]
            ]
            if candidate_indices:
                scores[global_idx] = sim[local_idx, candidate_indices].max()
            else:
                scores[global_idx] = 0.0

    return scores.tolist()


def main():
    parser = argparse.ArgumentParser(description="Build MMMP curriculum scores from training-set CLIP similarities.")
    parser.add_argument("--config_file", default="configs/ours/uav_ir_cctv_ir.yml", type=str)
    parser.add_argument("--output", default="", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.freeze()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dataset = MMMP(root=cfg.DATASETS.ROOT_DIR, exp_setting=cfg.DATASETS.EXP_SETTING)
    samples = dataset.train

    features = extract_features(cfg, samples, args.batch_size, device)
    sample_scores = compute_scores(samples, features)

    records = []
    scores = {}
    for sample, score in zip(samples, sample_scores):
        img_path, pid, camid, viewid = sample
        key = norm_path(img_path)
        scores[key] = float(score)
        records.append({
            "img_path": key,
            "pid": int(pid),
            "camid": int(camid),
            "viewid": int(viewid),
            "domain": domain_from_view(viewid),
            "score": float(score),
        })

    output = args.output
    if not output:
        output = os.path.join("curriculum_cache", "{}_clip_scores.pkl".format(cfg.DATASETS.EXP_SETTING))
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    with open(output, "wb") as f:
        pickle.dump({
            "meta": {
                "exp_setting": cfg.DATASETS.EXP_SETTING,
                "root_dir": cfg.DATASETS.ROOT_DIR,
                "model": cfg.MODEL.NAME,
                "num_samples": len(samples),
            },
            "scores": scores,
            "records": records,
        }, f)

    print("Saved curriculum scores to {}".format(output))


if __name__ == "__main__":
    main()
