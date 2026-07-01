# coding: utf-8
"""Evaluate evidence-routing modes with a structured LLM judge.

用法示例：
    python evaluate.py --dataset RAWFC --mode current_cql --limit 20
    python evaluate.py --dataset LIAR-RAW --mode bundle_rl --split val

注意：需要先设置环境变量 DEEPSEEK_API_KEY。
"""

import argparse
import json
import os
import random
import sys
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from core.bundle_policy import choose_bundle_action, load_bundle_policy
from core.evidence_selection import (
    ACTION_NAMES,
    build_evidence_bundles,
    bundle_state_features,
    clean_spaced_text,
    deduplicate_candidates,
    random_indices,
    rl_action_scores,
    topk_indices,
    cosine_scores,
    mmr_indices,
)
from core.json_stream import iter_json_array
from core.label_utils import int_to_label, label_options, parse_label_to_int
from core.llm_judge import PROMPT_VERSION, judge_stances, judge_verdict
from core.logger import setup_logger


SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_feature_items(dataset_name, split_name):
    path = os.path.join("datasets", dataset_name, f"rl_offline_buffer_{split_name}_features.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到特征文件: {path}")
    return path


def select_indices_for_mode(mode, claim_vec, cand_vecs, cand_sentences, rng, cql_policy=None, bundle_policy=None, bundle_ckpt=None):
    cos_scores = cosine_scores(claim_vec, cand_vecs)

    if mode == "random":
        return random_indices(len(cand_vecs), 5, rng), "random"
    if mode == "cosine":
        return topk_indices(cos_scores, 5), "cosine_k5"
    if mode == "claim_only":
        return [], "claim_only"

    action_scores = None
    if mode in {"current_cql", "defense_judge", "bundle_rl"}:
        if cql_policy is None:
            raise RuntimeError(f"{mode} 需要 CQL 句子级策略权重。")
        action_scores = rl_action_scores(cql_policy, claim_vec, cand_vecs)

    if mode == "current_cql":
        return mmr_indices(action_scores, cand_vecs, k=5, lambda_mmr=0.5), "cql_mmr_k5_l0.5"
    if mode == "defense_judge":
        return mmr_indices(action_scores, cand_vecs, k=5, lambda_mmr=0.5), "defense_support_refute"
    if mode == "bundle_rl":
        if bundle_policy is None or bundle_ckpt is None:
            raise RuntimeError("bundle_rl 需要先运行 scripts/train_bundle_policy.py 生成 bundle policy。")
        state_features = bundle_state_features(
            claim_vec,
            cand_vecs,
            action_scores=action_scores,
            candidate_sentences=cand_sentences,
        )
        action_name = choose_bundle_action(bundle_policy, bundle_ckpt, state_features)
        bundles = build_evidence_bundles(claim_vec, cand_vecs, action_scores=action_scores)
        return bundles[action_name], action_name

    raise ValueError(f"未知 mode: {mode}")


def evaluate_results(y_true, y_pred, dataset_name):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    labels = list(range(len(label_options(dataset_name))))
    target_names = [int_to_label(i, dataset_name) for i in labels]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )

    print("\n" + "=" * 80)
    print("最终指标")
    print("=" * 80)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")
    print("\n预测分布:", dict(Counter(y_pred)))
    print("\n混淆矩阵 labels=", target_names)
    print(matrix)
    print("\n分类报告")
    print(report)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "macro_f1": macro_f1,
        "confusion_matrix": matrix.tolist(),
        "target_names": target_names,
    }


def main():
    parser = argparse.ArgumentParser(description="多模式事实核查评估脚本")
    parser.add_argument("--dataset", choices=["RAWFC", "LIAR-RAW"], default="RAWFC")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--mode",
        choices=["random", "cosine", "current_cql", "bundle_rl", "defense_judge", "claim_only"],
        default="current_cql",
    )
    parser.add_argument("--limit", type=int, default=0, help="调试用：只评估前 N 条，0 表示全量")
    parser.add_argument("--prompt-version", default=PROMPT_VERSION)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bundle-policy-suffix", default="", help="bundle policy checkpoint 后缀，例如 debug2")
    parser.add_argument("--dry-run-selection", action="store_true", help="只检查证据选择流程，不调用 DeepSeek")
    args = parser.parse_args()

    if not args.dry_run_selection and not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("请先设置环境变量 DEEPSEEK_API_KEY，再运行 evaluate.py")

    set_seed(args.seed)
    setup_logger(log_dir="logs", prefix=f"evaluate_{args.dataset}_{args.split}_{args.mode}")

    print("实验配置")
    print(f"dataset={args.dataset}")
    print(f"split={args.split}")
    print(f"mode={args.mode}")
    print(f"seed={args.seed}")
    print(f"prompt_version={args.prompt_version}")
    print(f"dry_run_selection={args.dry_run_selection}")
    print("RAWFC best baseline: kl_scale=0.01, Top-K=5, lambda=0.5, temperature=0")
    print("LIAR-RAW current fallback: Top-K=3, lambda=0.8 for noisy evidence pools")

    cql_policy = None
    if args.mode in {"current_cql", "defense_judge", "bundle_rl"}:
        from core.cql_agent import load_adaptive_cql_policy

        cql_weight = os.path.join("checkpoints", f"{args.dataset}_adaptive_cql_policy_epoch_100.pth")
        if not os.path.exists(cql_weight):
            raise FileNotFoundError(f"找不到 CQL 权重: {cql_weight}")
        cql_policy = load_adaptive_cql_policy(weight_path=cql_weight)

    bundle_policy, bundle_ckpt = None, None
    if args.mode == "bundle_rl":
        suffix = f"_{args.bundle_policy_suffix}" if args.bundle_policy_suffix else ""
        bundle_weight = os.path.join("checkpoints", f"{args.dataset}_bundle_cql_policy{suffix}.pth")
        if not os.path.exists(bundle_weight):
            raise FileNotFoundError(f"找不到 bundle policy 权重: {bundle_weight}")
        bundle_policy, bundle_ckpt = load_bundle_policy(bundle_weight)

    feature_path = load_feature_items(args.dataset, args.split)
    print(f"feature_path={feature_path}")
    items = iter_json_array(feature_path, limit=args.limit)

    rng = np.random.default_rng(args.seed)
    y_true, y_pred = [], []
    case_studies = []

    for item in tqdm(items, total=args.limit or None, desc=f"Evaluating {args.mode}"):
        cand_vecs, cand_sentences = deduplicate_candidates(
            item["candidate_vectors"],
            item["candidate_sentences"],
        )
        if not cand_vecs:
            continue

        gold_label = item["ground_truth_label"]
        gold_id = parse_label_to_int(gold_label, args.dataset)
        if gold_id < 0:
            continue

        claim_vec = item["claim_vector"]
        claim_text = item.get("claim_text", item.get("claim", "UNKNOWN CLAIM"))
        selected_indices, selected_action = select_indices_for_mode(
            args.mode,
            claim_vec,
            cand_vecs,
            cand_sentences,
            rng,
            cql_policy=cql_policy,
            bundle_policy=bundle_policy,
            bundle_ckpt=bundle_ckpt,
        )
        selected_evidence = [cand_sentences[idx] for idx in selected_indices]

        if args.dry_run_selection:
            case_studies.append(
                {
                    "claim_index": item["claim_index"],
                    "claim_text": clean_spaced_text(claim_text),
                    "ground_truth_label": gold_label,
                    "selected_action": selected_action,
                    "selected_indices": selected_indices,
                    "selected_evidence": selected_evidence,
                }
            )
            continue

        stance_labels = []
        if args.mode in {"defense_judge", "bundle_rl"} and selected_evidence:
            stance_labels = judge_stances(claim_text, selected_evidence, args.dataset)

        result = judge_verdict(
            claim_text,
            selected_evidence,
            args.dataset,
            stance_labels=stance_labels,
            prompt_version=args.prompt_version,
        )
        pred_id = parse_label_to_int(result.get("prediction", "unknown"), args.dataset)
        if pred_id < 0:
            pred_id = 0

        y_true.append(gold_id)
        y_pred.append(pred_id)
        case_studies.append(
            {
                "claim_index": item["claim_index"],
                "claim_text": clean_spaced_text(claim_text),
                "ground_truth_label": gold_label,
                "selected_action": selected_action,
                "prediction": result.get("prediction", "unknown"),
                "stance_labels": stance_labels,
                "selected_evidence": selected_evidence,
                "reasoning": result.get("step_by_step_analysis", ""),
            }
        )

    if args.dry_run_selection:
        metrics = {
            "dry_run_selection": True,
            "num_cases": len(case_studies),
            "selected_action_distribution": dict(Counter(c["selected_action"] for c in case_studies)),
        }
        print("\nDry run completed. No DeepSeek calls were made.")
        print("selected_action_distribution:", metrics["selected_action_distribution"])
    else:
        metrics = evaluate_results(y_true, y_pred, args.dataset)

    dry_suffix = "_dryrun" if args.dry_run_selection else ""
    output_path = os.path.join("logs", f"case_study_{args.dataset}_{args.split}_{args.mode}{dry_suffix}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "metrics": metrics,
                "case_studies": case_studies,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\ncase study 已保存: {output_path}")


if __name__ == "__main__":
    main()
