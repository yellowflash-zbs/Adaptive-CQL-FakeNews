# coding: utf-8
"""Evidence selection and bundle construction utilities."""

import numpy as np

from core.feature_utils import build_state_vector


ACTION_NAMES = [
    "claim_only",
    "cosine_k3",
    "cosine_k5",
    "cql_mmr_k5_l0.5",
    "cql_mmr_k3_l0.8",
    "cql_top1",
    "defense_support_refute",
]


def clean_spaced_text(text):
    text = str(text).strip()
    if "   " in text:
        words = text.split("   ")
        cleaned_words = [word.replace(" ", "") for word in words]
        return " ".join(cleaned_words)
    return text


def deduplicate_candidates(candidate_vectors, candidate_sentences, min_chars=10):
    vectors, sentences, seen = [], [], set()
    for vec, text in zip(candidate_vectors, candidate_sentences):
        clean_text = clean_spaced_text(text)
        key = " ".join(clean_text.lower().split())
        if len(key) < min_chars or key in seen:
            continue
        seen.add(key)
        vectors.append(vec)
        sentences.append(clean_text)
    return vectors, sentences


def _safe_vec(vec, vector_dim=768):
    arr = np.array(vec, dtype=np.float32).flatten()[:vector_dim]
    if len(arr) < vector_dim:
        arr = np.pad(arr, (0, vector_dim - len(arr)))
    return arr


def cosine_scores(claim_vec, cand_vecs, vector_dim=768):
    if not cand_vecs:
        return np.array([], dtype=np.float32)
    c_vec = _safe_vec(claim_vec, vector_dim)
    c_norm = np.linalg.norm(c_vec)
    scores = []
    for cand_vec in cand_vecs:
        v = _safe_vec(cand_vec, vector_dim)
        v_norm = np.linalg.norm(v)
        if c_norm == 0 or v_norm == 0:
            scores.append(-1.0)
        else:
            scores.append(float(np.dot(c_vec, v) / (c_norm * v_norm)))
    return np.array(scores, dtype=np.float32)


def topk_indices(scores, k):
    if len(scores) == 0 or k <= 0:
        return []
    k = min(k, len(scores))
    return np.argsort(scores)[-k:][::-1].tolist()


def random_indices(num_candidates, k, rng):
    if num_candidates == 0:
        return []
    indices = np.arange(num_candidates)
    rng.shuffle(indices)
    return indices[: min(k, num_candidates)].tolist()


def rl_action_scores(policy, claim_vec, cand_vecs):
    state_vec = build_state_vector(claim_vec, cand_vecs)
    action, _ = policy.get_action(state_vec)
    return np.array(action[: len(cand_vecs)], dtype=np.float32)


def mmr_indices(action_scores, cand_vecs, k=5, lambda_mmr=0.5, vector_dim=768):
    num_cands = len(cand_vecs)
    if num_cands == 0:
        return []
    k = min(k, num_cands)
    unselected = list(range(num_cands))
    selected = []

    first_idx = unselected[int(np.argmax(action_scores))]
    selected.append(first_idx)
    unselected.remove(first_idx)

    for _ in range(k - 1):
        if not unselected:
            break
        best_score, best_idx = -float("inf"), None
        for idx in unselected:
            candidate_vec = _safe_vec(cand_vecs[idx], vector_dim)
            candidate_norm = np.linalg.norm(candidate_vec)
            max_sim = 0.0
            if candidate_norm > 0:
                for selected_idx in selected:
                    selected_vec = _safe_vec(cand_vecs[selected_idx], vector_dim)
                    selected_norm = np.linalg.norm(selected_vec)
                    if selected_norm > 0:
                        sim = float(np.dot(candidate_vec, selected_vec) / (candidate_norm * selected_norm))
                        max_sim = max(max_sim, sim)
            score = lambda_mmr * float(action_scores[idx]) - (1.0 - lambda_mmr) * max_sim
            if score > best_score:
                best_score, best_idx = score, idx
        selected.append(best_idx)
        unselected.remove(best_idx)
    return selected


def _ordered_unique(indices, limit):
    output = []
    for idx in indices:
        if idx not in output:
            output.append(idx)
        if len(output) >= limit:
            break
    return output


def build_evidence_bundles(claim_vec, cand_vecs, action_scores=None):
    cos_scores = cosine_scores(claim_vec, cand_vecs)
    if action_scores is None:
        action_scores = cos_scores if len(cos_scores) else np.array([], dtype=np.float32)

    cosine_k3 = topk_indices(cos_scores, 3)
    cosine_k5 = topk_indices(cos_scores, 5)
    cql_mmr_k5 = mmr_indices(action_scores, cand_vecs, k=5, lambda_mmr=0.5)
    cql_mmr_k3 = mmr_indices(action_scores, cand_vecs, k=3, lambda_mmr=0.8)
    cql_top1 = topk_indices(action_scores, 1)

    # Proxy for a support/refute bundle before stance labels exist: combine strong RL
    # evidence with high lexical relevance, then let the structured judge split stances.
    defense_indices = _ordered_unique(cql_mmr_k3 + cosine_k3 + cql_mmr_k5, 5)

    return {
        "claim_only": [],
        "cosine_k3": cosine_k3,
        "cosine_k5": cosine_k5,
        "cql_mmr_k5_l0.5": cql_mmr_k5,
        "cql_mmr_k3_l0.8": cql_mmr_k3,
        "cql_top1": cql_top1,
        "defense_support_refute": defense_indices,
    }


def bundle_state_features(claim_vec, cand_vecs, action_scores=None, candidate_sentences=None):
    cos_scores = cosine_scores(claim_vec, cand_vecs)
    if action_scores is None or len(action_scores) == 0:
        action_scores = np.zeros(len(cand_vecs), dtype=np.float32)
    action_scores = np.array(action_scores, dtype=np.float32)
    candidate_sentences = candidate_sentences or []
    lengths = np.array([len(clean_spaced_text(text).split()) for text in candidate_sentences], dtype=np.float32)
    num_candidates = float(len(cand_vecs))

    def stats(values):
        values = np.array(values, dtype=np.float32)
        if len(values) == 0:
            return [0.0, 0.0, 0.0, 0.0]
        return [float(values.mean()), float(values.std()), float(values.max()), float(values.min())]

    return np.array(
        [
            num_candidates,
            float(num_candidates < 3),
            float(num_candidates < 5),
            *stats(cos_scores),
            *stats(action_scores),
            *stats(lengths),
        ],
        dtype=np.float32,
    )
