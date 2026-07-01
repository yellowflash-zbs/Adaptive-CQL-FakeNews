# coding: utf-8
"""Discrete offline RL policy for choosing evidence bundle actions."""

import json

import numpy as np
import torch
from torch import nn

from core.evidence_selection import ACTION_NAMES


class DiscreteCQLPolicy(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

    def forward(self, states):
        return self.net(states)


def load_bundle_policy(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    policy = DiscreteCQLPolicy(
        state_dim=checkpoint["state_dim"],
        num_actions=len(checkpoint["action_names"]),
        hidden_size=checkpoint.get("hidden_size", 128),
    )
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.to(device)
    policy.eval()
    return policy, checkpoint


def choose_bundle_action(policy, checkpoint, state_features, device="cpu"):
    action_names = checkpoint.get("action_names", ACTION_NAMES)
    mean = np.array(checkpoint["feature_mean"], dtype=np.float32)
    std = np.array(checkpoint["feature_std"], dtype=np.float32)
    features = (np.array(state_features, dtype=np.float32) - mean) / std
    tensor = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action_id = int(torch.argmax(policy(tensor), dim=1).item())
    return action_names[action_id]


def save_policy_metadata(path, metadata):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
