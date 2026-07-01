# coding: utf-8
"""Generate bundle-level offline RL candidates.

This script does not train a model. It prepares one JSONL row for each
claim/action pair so the bundle policy can later learn which evidence route
works best for each claim.
"""

import argparse
import json
import os
import sys

from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.evidence_selection import (
    ACTION_NAMES,
    build_evidence_bundles,
    bundle_state_features,
    deduplicate_candidates,
    rl_action_scores,
)
from core.label_utils import evidence_quality_reward, noise_penalty, verdict_reward
from core.llm_judge import PROMPT_VERSION, judge_stances, judge_verdict


def load_items(dataset_name, split_name):
    path = os.path.join(project_root, "datasets", dataset_name, f"rl_offline_buffer_{split_name}_features.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到特征文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="生成 bundle-level offline RL 候选证据包")
    parser.add_argument("--dataset", choices=["RAWFC", "LIAR-RAW"], default="RAWFC")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--limit", type=int, default=0, help="调试用：只处理前 N 条，0 表示全量")
    parser.add_argument("--skip-llm", action="store_true", help="只生成证据包，不调用 DeepSeek 打分")
    parser.add_argument("--prompt-version", default=PROMPT_VERSION)
    args = parser.parse_args()

    weight_path = os.path.join(project_root, "checkpoints", f"{args.dataset}_adaptive_cql_policy_epoch_100.pth")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"找不到 CQL 权重: {weight_path}")

    from core.cql_agent import load_adaptive_cql_policy

    policy = load_adaptive_cql_policy(weight_path=weight_path)
    items = load_items(args.dataset, args.split)
    if args.limit > 0:
        items = items[: args.limit]

    output_path = os.path.join(
        project_root,
        "datasets",
        args.dataset,
        f"evidence_bundle_candidates_{args.split}.jsonl",
    )

    print(f"dataset={args.dataset} split={args.split} items={len(items)}")
    print(f"actions={ACTION_NAMES}")
    print(f"output={output_path}")
    print(f"skip_llm={args.skip_llm} prompt_version={args.prompt_version}")

    with open(output_path, "w", encoding="utf-8") as f_out:
        for item in tqdm(items, desc="Generating evidence bundles"):
            cand_vecs, cand_sentences = deduplicate_candidates(
                item["candidate_vectors"],
                item["candidate_sentences"],
            )
            if not cand_vecs:
                continue

            claim_vec = item["claim_vector"]
            claim_text = item.get("claim_text", item.get("claim", "UNKNOWN CLAIM"))
            gold_label = item["ground_truth_label"]
            action_scores = rl_action_scores(policy, claim_vec, cand_vecs)
            state_features = bundle_state_features(
                claim_vec,
                cand_vecs,
                action_scores=action_scores,
                candidate_sentences=cand_sentences,
            ).tolist()
            bundles = build_evidence_bundles(claim_vec, cand_vecs, action_scores=action_scores)

            for action_name in ACTION_NAMES:
                selected_indices = bundles[action_name]
                selected_evidence = [cand_sentences[idx] for idx in selected_indices]

                stance_labels = []
                llm_result = {"prediction": "unknown", "step_by_step_analysis": "LLM skipped"}
                if not args.skip_llm:
                    stance_labels = judge_stances(claim_text, selected_evidence, args.dataset)
                    llm_result = judge_verdict(
                        claim_text,
                        selected_evidence,
                        args.dataset,
                        stance_labels=stance_labels,
                        prompt_version=args.prompt_version,
                    )

                final_reward = verdict_reward(llm_result.get("prediction", "unknown"), gold_label, args.dataset)
                dense_reward = evidence_quality_reward(stance_labels)
                penalty = noise_penalty(selected_evidence)
                reward_total = final_reward + dense_reward - penalty

                record = {
                    "claim_index": item["claim_index"],
                    "dataset": args.dataset,
                    "split": args.split,
                    "gold_label": gold_label,
                    "state_features": state_features,
                    "action_name": action_name,
                    "selected_indices": selected_indices,
                    "selected_evidence": selected_evidence,
                    "stance_labels": stance_labels,
                    "llm_prediction": llm_result.get("prediction", "unknown"),
                    "llm_reasoning": llm_result.get("step_by_step_analysis", ""),
                    "final_verdict_reward": final_reward,
                    "evidence_quality_reward": dense_reward,
                    "noise_penalty": penalty,
                    "reward_total": reward_total,
                    "prompt_version": args.prompt_version,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()


if __name__ == "__main__":
    main()
