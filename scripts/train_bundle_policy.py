# coding: utf-8
"""Train a discrete bundle-level offline RL controller.

The objective is a conservative Q-learning style contextual bandit loss:
fit observed bundle rewards while penalizing high Q values for unobserved
actions. This is intentionally small and practical for the current project.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.bundle_policy import DiscreteCQLPolicy
from core.evidence_selection import ACTION_NAMES


def load_bundle_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"候选证据包文件为空: {path}")
    return records


def records_to_tensors(records):
    action_to_id = {name: idx for idx, name in enumerate(ACTION_NAMES)}
    states, actions, rewards = [], [], []
    for record in records:
        action_name = record["action_name"]
        if action_name not in action_to_id:
            continue
        states.append(record["state_features"])
        actions.append(action_to_id[action_name])
        rewards.append(float(record["reward_total"]))

    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int64)
    rewards = np.array(rewards, dtype=np.float32)
    mean = states.mean(axis=0)
    std = states.std(axis=0)
    std[std < 1e-6] = 1.0
    states = (states - mean) / std
    return states, actions, rewards, mean, std


def evaluate_policy(model, records, mean, std, device):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["claim_index"]].append(record)

    chosen_rewards = []
    action_counts = defaultdict(int)
    for claim_records in grouped.values():
        state = np.array(claim_records[0]["state_features"], dtype=np.float32)
        state = (state - mean) / std
        tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_id = int(torch.argmax(model(tensor), dim=1).item())
        action_name = ACTION_NAMES[action_id]
        action_counts[action_name] += 1
        reward_by_action = {r["action_name"]: float(r["reward_total"]) for r in claim_records}
        chosen_rewards.append(reward_by_action.get(action_name, min(reward_by_action.values())))

    return float(np.mean(chosen_rewards)), dict(action_counts)


def main():
    parser = argparse.ArgumentParser(description="训练 bundle-level 离线 RL/CQL 控制器")
    parser.add_argument("--dataset", choices=["RAWFC", "LIAR-RAW"], default="RAWFC")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cql-alpha", type=float, default=0.2)
    args = parser.parse_args()

    input_path = os.path.join(
        project_root,
        "datasets",
        args.dataset,
        f"evidence_bundle_candidates_{args.split}.jsonl",
    )
    records = load_bundle_records(input_path)
    states, actions, rewards, mean, std = records_to_tensors(records)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiscreteCQLPolicy(states.shape[1], len(ACTION_NAMES), hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    dataset = TensorDataset(
        torch.tensor(states, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.long),
        torch.tensor(rewards, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        losses = []
        for batch_states, batch_actions, batch_rewards in loader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            batch_rewards = batch_rewards.to(device)

            q_values = model(batch_states)
            chosen_q = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
            bellman_loss = F.mse_loss(chosen_q, batch_rewards)
            cql_loss = torch.logsumexp(q_values, dim=1).mean() - chosen_q.mean()
            loss = bellman_loss + args.cql_alpha * cql_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_reward, action_counts = evaluate_policy(model, records, mean, std, device)
            print(
                f"epoch={epoch + 1:03d} loss={np.mean(losses):.4f} "
                f"offline_avg_reward={avg_reward:.4f} actions={action_counts}"
            )

    checkpoint_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    output_path = os.path.join(checkpoint_dir, f"{args.dataset}_bundle_cql_policy.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_dim": states.shape[1],
            "hidden_size": args.hidden_size,
            "action_names": ACTION_NAMES,
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "dataset": args.dataset,
            "split": args.split,
            "cql_alpha": args.cql_alpha,
        },
        output_path,
    )
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
