# coding: utf-8
"""Project-specific Adaptive CQL trainer.

Only the single-step offline CQL logic used by Adaptive-CQL-FakeNews is kept.
All unrelated rlkit algorithms and environments are intentionally removed.
"""

from collections import OrderedDict
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

import core.simple_rl as ptu


class AdaptiveCQLTrainer:
    def __init__(
        self,
        env,
        policy,
        qf1,
        qf2,
        bc_model_path,
        kl_scale=0.5,
        target_qf1=None,
        target_qf2=None,
        discount=0.99,
        reward_scale=1.0,
        policy_lr=1e-3,
        qf_lr=1e-3,
        optimizer_class=optim.Adam,
        soft_target_tau=1e-2,
        automatic_entropy_tuning=True,
        target_entropy=None,
        bc_warm_start=0,
        num_qs=2,
        min_q_version=2,
        temp=1.0,
        pbrs_u=10.0,
        pbrs_g="softplus",
        max_q_backup=False,
        deterministic_backup=True,
        num_random=10,
        with_lagrange=False,
        lagrange_thresh=0.0,
    ):
        self.env = env
        self.max_action = float(env.action_space.high[0])
        self.min_action = float(env.action_space.low[0])
        self.policy = policy
        self.behavior_policy = copy.deepcopy(policy)
        self.behavior_policy.load_state_dict(torch.load(bc_model_path, map_location=ptu.device))
        self.behavior_policy.to(ptu.device)
        self.behavior_policy.eval()
        for param in self.behavior_policy.parameters():
            param.requires_grad = False

        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.discount = discount
        self.reward_scale = reward_scale
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.bc_warm_start = bc_warm_start
        self.num_qs = num_qs
        self.min_q_version = min_q_version
        self.temp = temp
        self.pbrs_c = kl_scale
        self.pbrs_u = pbrs_u
        self.pbrs_g = pbrs_g
        self.max_q_backup = max_q_backup
        self.deterministic_backup = deterministic_backup
        self.num_random = num_random
        self.with_lagrange = with_lagrange
        self.eval_statistics = OrderedDict()
        self._need_to_update_eval_statistics = True
        self._n_train_steps_total = 0

        if target_entropy is None:
            target_entropy = -np.prod(self.env.action_space.shape).item()
        self.target_entropy = target_entropy
        self.log_alpha = ptu.zeros(1, requires_grad=True)
        self.alpha_optimizer = optimizer_class([self.log_alpha], lr=policy_lr)

        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            self.log_alpha_prime = ptu.zeros(1, requires_grad=True)
            self.alpha_prime_optimizer = optimizer_class([self.log_alpha_prime], lr=qf_lr)

        self.qf_criterion = nn.MSELoss()
        self.policy_optimizer = optimizer_class(self.policy.parameters(), lr=policy_lr)
        self.qf1_optimizer = optimizer_class(self.qf1.parameters(), lr=qf_lr)
        self.qf2_optimizer = optimizer_class(self.qf2.parameters(), lr=qf_lr)

    def _get_tensor_values(self, obs, actions, network):
        action_size = actions.shape[0]
        obs_size = obs.shape[0]
        num_repeat = int(action_size / obs_size)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs_size * num_repeat, obs.shape[1])
        preds = network(obs_temp, actions)
        return preds.view(obs_size, num_repeat, 1)

    def _get_policy_actions(self, obs, num_actions=1, network=None):
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
        actions_dist = network(obs_temp)
        normal_mean = actions_dist.normal_mean
        normal_std = actions_dist.stddev
        new_obs_actions, log_pi = actions_dist.rsample_and_logprob()
        new_obs_actions = new_obs_actions.clamp(self.min_action - 1e-6, self.max_action + 1e-6)
        return new_obs_actions, normal_mean, normal_std, log_pi.view(log_pi.shape[0], 1)

    def train_from_torch(self, batch):
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]

        new_obs_actions, policy_mean, policy_std, log_pi = self._get_policy_actions(obs, network=self.policy)
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        alpha = self.log_alpha.exp()

        q_new_actions = torch.min(self.qf1(obs, new_obs_actions), self.qf2(obs, new_obs_actions))
        policy_loss = (alpha * log_pi - q_new_actions).mean()
        if self._n_train_steps_total + 1 < self.bc_warm_start:
            policy_loss = (alpha * log_pi - self.policy.logprob(actions, policy_mean, policy_std)).mean()

        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        new_next_actions, _, _, new_next_log_pi = self._get_policy_actions(next_obs, network=self.policy)

        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        )
        if not self.deterministic_backup:
            target_q_values = target_q_values - alpha * new_next_log_pi

        q_target = self.reward_scale * rewards + (1.0 - terminals.float()) * self.discount * target_q_values
        q_target = q_target.detach()
        qf1_loss = self.qf_criterion(q1_pred, q_target)
        qf2_loss = self.qf_criterion(q2_pred, q_target)

        random_actions = torch.FloatTensor(q2_pred.shape[0] * self.num_random, actions.shape[-1]).uniform_(-1, 1)
        random_actions = random_actions.to(ptu.device)
        curr_actions, _, _, curr_log_pis = self._get_policy_actions(obs, num_actions=self.num_random, network=self.policy)
        curr_log_pis = curr_log_pis.view(q2_pred.shape[0], self.num_random, 1)
        next_curr_actions, _, _, next_curr_log_pis = self._get_policy_actions(
            next_obs,
            num_actions=self.num_random,
            network=self.policy,
        )
        next_curr_log_pis = next_curr_log_pis.view(q2_pred.shape[0], self.num_random, 1)

        q1_rand = self._get_tensor_values(obs, random_actions, network=self.qf1)
        q2_rand = self._get_tensor_values(obs, random_actions, network=self.qf2)
        q1_curr_actions = self._get_tensor_values(obs, curr_actions.detach(), network=self.qf1)
        q2_curr_actions = self._get_tensor_values(obs, curr_actions.detach(), network=self.qf2)
        q1_next_actions = self._get_tensor_values(obs, next_curr_actions.detach(), network=self.qf1)
        q2_next_actions = self._get_tensor_values(obs, next_curr_actions.detach(), network=self.qf2)

        cat_q1 = torch.cat([q1_rand, q1_pred.unsqueeze(1), q1_next_actions, q1_curr_actions], 1)
        cat_q2 = torch.cat([q2_rand, q2_pred.unsqueeze(1), q2_next_actions, q2_curr_actions], 1)
        std_q1 = torch.std(cat_q1, dim=1)
        std_q2 = torch.std(cat_q2, dim=1)

        with torch.no_grad():
            q_variance = ((std_q1 + std_q2) / 2.0).unsqueeze(1)
            variance_clipped = torch.clamp(q_variance, min=0.0, max=self.pbrs_u)
            b_k = self.pbrs_c * F.softplus(variance_clipped) if self.pbrs_g == "softplus" else self.pbrs_c * variance_clipped

        cql_q1_penalty = torch.clamp(torch.logsumexp(cat_q1 / self.temp, dim=1, keepdim=True) * self.temp - q1_pred, min=0.0)
        cql_q2_penalty = torch.clamp(torch.logsumexp(cat_q2 / self.temp, dim=1, keepdim=True) * self.temp - q2_pred, min=0.0)
        min_qf1_loss = (cql_q1_penalty * b_k).mean()
        min_qf2_loss = (cql_q2_penalty * b_k).mean()
        qf1_loss = qf1_loss + min_qf1_loss
        qf2_loss = qf2_loss + min_qf2_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=False)
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=True)
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward(retain_graph=False)
        self.qf2_optimizer.step()

        ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
        ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics["QF1 Loss"] = float(ptu.get_numpy(qf1_loss))
            self.eval_statistics["QF2 Loss"] = float(ptu.get_numpy(qf2_loss))
            self.eval_statistics["min QF1 Loss"] = float(ptu.get_numpy(min_qf1_loss))
            self.eval_statistics["min QF2 Loss"] = float(ptu.get_numpy(min_qf2_loss))
            self.eval_statistics["PBRS b_k(s) Mean"] = float(np.mean(ptu.get_numpy(b_k)))
            self.eval_statistics["PBRS b_k(s) Max"] = float(np.max(ptu.get_numpy(b_k)))
            self.eval_statistics["PBRS Q-Variance"] = float(np.mean(ptu.get_numpy(variance_clipped)))
            self.eval_statistics["Policy Loss"] = float(ptu.get_numpy(policy_loss))
            self.eval_statistics["Alpha"] = float(alpha.item())
            self.eval_statistics["Alpha Loss"] = float(alpha_loss.item())

        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [self.policy, self.qf1, self.qf2, self.target_qf1, self.target_qf2, self.behavior_policy]
