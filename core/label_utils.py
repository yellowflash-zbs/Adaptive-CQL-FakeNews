# coding: utf-8
"""Dataset label helpers and reward shaping for fact-checking experiments."""

LABEL_NAMES = {
    "RAWFC": ["false", "true", "half"],
    "LIAR-RAW": ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"],
}

ORDINAL_LABELS = {
    "LIAR-RAW": ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"],
}


def normalize_label(label):
    return str(label).strip().lower().replace("_", "-")


def label_options(dataset_name):
    return LABEL_NAMES[dataset_name]


def parse_label_to_int(label, dataset_name):
    label = normalize_label(label)
    if dataset_name == "LIAR-RAW":
        if "pants" in label or "fire" in label:
            return 0
        if "barely" in label:
            return 2
        if "half" in label:
            return 3
        if "mostly" in label:
            return 4
        if "false" in label or "fake" in label:
            return 1
        if "true" in label or "real" in label:
            return 5
        return -1

    if "half" in label:
        return 2
    if "true" in label or "real" in label:
        return 1
    if "false" in label or "fake" in label:
        return 0
    return -1


def int_to_label(label_id, dataset_name):
    names = label_options(dataset_name)
    if 0 <= int(label_id) < len(names):
        return names[int(label_id)]
    return "unknown"


def verdict_reward(prediction, gold_label, dataset_name):
    """Reward final verdicts while allowing ordinal partial credit on LIAR-RAW."""
    pred_id = parse_label_to_int(prediction, dataset_name)
    gold_id = parse_label_to_int(gold_label, dataset_name)
    if pred_id < 0 or gold_id < 0:
        return -1.0
    if pred_id == gold_id:
        return 2.0
    if dataset_name in ORDINAL_LABELS:
        distance = abs(pred_id - gold_id)
        if distance == 1:
            return 0.75
        if distance == 2:
            return -0.25
        return -1.25
    return -1.0


def evidence_quality_reward(stance_labels):
    """Small dense reward for useful, non-noisy evidence composition."""
    if not stance_labels:
        return 0.0
    stances = [normalize_label(s) for s in stance_labels]
    useful = sum(1 for s in stances if s in {"support", "refute"})
    neutral = sum(1 for s in stances if s == "neutral")
    irrelevant = sum(1 for s in stances if s in {"irrelevant", "error"})
    has_both_sides = int("support" in stances and "refute" in stances)
    return 0.35 * useful + 0.1 * neutral + 0.4 * has_both_sides - 0.3 * irrelevant


def noise_penalty(selected_evidence):
    """Penalize empty, very short, or repeated evidence bundles."""
    if not selected_evidence:
        return 0.6
    normalized = [" ".join(str(s).lower().split()) for s in selected_evidence]
    duplicate_ratio = 1.0 - (len(set(normalized)) / max(1, len(normalized)))
    short_ratio = sum(1 for s in normalized if len(s.split()) < 5) / max(1, len(normalized))
    return duplicate_ratio + 0.5 * short_ratio
