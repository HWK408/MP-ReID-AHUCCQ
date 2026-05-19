import math
import os
import pickle
from collections import defaultdict


def domain_from_view(viewid):
    """Map MMMP camera/view ids to coarse modality-platform domains."""
    viewid = int(viewid)
    if 0 <= viewid <= 5:
        return "cctv_rgb"
    if 6 <= viewid <= 11:
        return "cctv_ir"
    if viewid == 12:
        return "uav_rgb"
    if viewid == 13:
        return "uav_ir"
    return "unknown"


def _norm_path(path):
    return os.path.normcase(os.path.normpath(path))


def _load_score_map(score_file):
    if not score_file:
        raise ValueError("DATASETS.CURRICULUM.SCORE_FILE must be set for easy/medium curriculum phases.")
    if not os.path.isfile(score_file):
        raise FileNotFoundError(f"Curriculum score file not found: {score_file}")

    with open(score_file, "rb") as f:
        data = pickle.load(f)

    raw_scores = data.get("scores", data) if isinstance(data, dict) else None
    if not isinstance(raw_scores, dict):
        raise ValueError("Curriculum score file must contain a dict or a dict with a 'scores' field.")

    return {_norm_path(path): float(score) for path, score in raw_scores.items()}


def _ratio_for_phase(curriculum_cfg, phase):
    if phase == "easy":
        return float(curriculum_cfg.EASY_RATIO)
    if phase == "medium":
        return float(curriculum_cfg.MEDIUM_RATIO)
    if phase == "full":
        return 1.0
    raise ValueError(f"Unsupported curriculum phase: {phase}")


def filter_train_data(train_data, cfg, phase=None):
    curriculum_cfg = getattr(cfg.DATASETS, "CURRICULUM", None)
    if curriculum_cfg is None or not curriculum_cfg.ENABLED:
        return list(train_data), {
            "enabled": False,
            "phase": "full",
            "original": len(train_data),
            "filtered": len(train_data),
        }

    phase = phase or curriculum_cfg.PHASE
    ratio = _ratio_for_phase(curriculum_cfg, phase)
    if phase == "full" or ratio >= 1.0:
        return list(train_data), {
            "enabled": True,
            "phase": "full",
            "original": len(train_data),
            "filtered": len(train_data),
        }

    score_map = _load_score_map(curriculum_cfg.SCORE_FILE)
    min_keep = int(getattr(curriculum_cfg, "MIN_SAMPLES_PER_ID_DOMAIN", 1))
    grouped = defaultdict(list)
    missing_scores = 0

    for sample in train_data:
        img_path, pid, camid, viewid = sample
        score = score_map.get(_norm_path(img_path))
        if score is None:
            missing_scores += 1
            score = float("-inf")
        grouped[(pid, domain_from_view(viewid))].append((score, sample))

    if missing_scores == len(train_data):
        raise ValueError(
            "None of the training image paths matched DATASETS.CURRICULUM.SCORE_FILE. "
            "Please regenerate the score file with the same DATASETS.ROOT_DIR."
        )

    selected = set()
    for items in grouped.values():
        keep = max(min_keep, int(math.ceil(len(items) * ratio)))
        keep = min(len(items), keep)
        items = sorted(items, key=lambda x: x[0], reverse=True)
        for _, sample in items[:keep]:
            selected.add(sample)

    filtered = [sample for sample in train_data if sample in selected]
    return filtered, {
        "enabled": True,
        "phase": phase,
        "ratio": ratio,
        "original": len(train_data),
        "filtered": len(filtered),
        "missing_scores": missing_scores,
    }
