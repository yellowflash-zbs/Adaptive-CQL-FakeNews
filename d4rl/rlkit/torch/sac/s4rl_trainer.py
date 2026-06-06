from rlkit.torch.torch_rl_algorithm import TorchTrainer
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from collections import OrderedDict

MEAN_MIN = -9.0
MEAN_MAX = 9.0

class S4RLTrainer(TorchTrainer):
    def __init__(self,
                 env,
                 policy,
                 qf1,
                 qf2,
                 target_qf1=None,
                 target_qf2=None,
                 vf=None,
                 discount=0.99,
                 reward_scale=1.0,
                 policy_lr=1e-3,
                 qf_lr=1e-3,
                 optimizer_class=optim.Adam,
                 soft_target_tau=1e-2,
                 # SAC
                 automatic_entropy_tuning=True,
                 target_entropy=None,
                 bc_warm_start=0,
                 num_qs=2,
                 total_training_steps=1E6,
                 # CQL
                 min_q_version=3,
                 temp=1.0,
                 min_q_weight=1.0,
                 ## sort of backup
                 max_q_backup=False,
                 deterministic_backup=True,
                 num_random=10,
                 with_lagrange=False,
                 lagrange_thresh=0.0,
                 ### augmentation type
                 s4rl=dict(type='normal', params=0.0003)
                 ):
        super().__init__()
        self.env = env
        self.obs_lower_bound = float(self.env.observation_space.low[0])
        self.obs_upper_bound = float(self.env.observation_space.high[0])
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.vf = vf
        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._need_to_update_eval_statistics = True

        # setting of loss and optimizer
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.alpha=0.2 # constant alpha if not automatic tuning
        if self.automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()

            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )
            self.alpha = self.log_alpha.exp()

        self.with_lagrange = with_lagrange
        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            self.log_alpha_prime = ptu.zeros(1, requires_grad=True)
            self.alpha_prime_optimizer = optimizer_class(
                [self.log_alpha_prime],
                lr=qf_lr,
            )

        self.soft_target_tau = soft_target_tau

        self.bc_warm_start = bc_warm_start
        self.num_qs = num_qs
        self.num_random = num_random
        self.total_training_steps = total_training_steps
        self._n_train_steps_total = 0
        self.min_q_version = min_q_version
        self.min_q_weight = min_q_weight
        self.temp = temp
        self.max_q_backup = max_q_backup
        self.deterministic_backup = deterministic_backup

        self.s4rl_type = s4rl['type']
        self.s4rl_params = s4rl['params']

    def augment_states(self, states):
        if self.s4rl_type == 'normal':
            noise = torch.normal(0, self.s4rl_params, size=states.shape).to(ptu.device)
            aug_states = states + noise
        elif self.s4rl_type == 'uniform':
            # noise = (self.s4rl_params[1] - self.s4rl_params[0]) * torch.rand(size=states.shape).to(ptu.device) + self.s4rl_params[0]
            noise = torch.FloatTensor(size=states.shape).uniform_(-self.s4rl_params, self.s4rl_params).to(ptu.device)
            aug_states = states + noise
        elif self.s4rl_type == 'adv':
            # todo: adversarial augment
            # aug_states = states
            pass
        else:
            raise ValueError('Not support the augmentation:{}'.format(self.s4rl_type))

        return aug_states.clamp(self.obs_lower_bound, self.obs_upper_bound)

    def _get_tensor_values(self, obs, actions, network=None):
        action_size = actions.shape[0]
        obs_size= obs.shape[0]
        num_repeat = int (action_size / obs_size)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        preds = network(obs_temp, actions)
        preds = preds.view(obs.shape[0], num_repeat, 1)
        return preds

    def _get_policy_actions(self, obs, num_actions=1, network=None):
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
        actions_dist = network(obs_temp)
        normal_mean = actions_dist.normal_mean # no tanh()
        normal_std = actions_dist.stddev
        new_obs_actions, log_pi = actions_dist.rsample_and_logprob() # (bs*num_actions, act_dim), (bs*num_actions,)

        return new_obs_actions, normal_mean, normal_std, log_pi.view(log_pi.shape[0], 1)

    def train_from_torch(self, batch):

        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Critic Loss
        """
        with torch.no_grad():
            next_state_action, _, _, next_state_log_pi = self._get_policy_actions(next_obs, network=self.policy)

            if not self.max_q_backup:
                if self.num_qs == 1:
                    target_q_values = self.target_qf1(self.augment_states(next_obs), next_state_action)
                else:
                    target_q_values = torch.min(
                        self.target_qf1(self.augment_states(next_obs), next_state_action),
                        self.target_qf2(self.augment_states(next_obs), next_state_action),
                    )

                if not self.deterministic_backup:
                    target_q_values = target_q_values - self.alpha * next_state_log_pi
            else:
                """when using max q backup"""
                next_actions_temp, _, _, _ = self._get_policy_actions(next_obs, num_actions=10, network=self.policy)
                target_qf1_values = self._get_tensor_values(self.augment_states(next_obs), next_actions_temp, network=self.target_qf1).max(1)[0].view(-1, 1)
                target_qf2_values = self._get_tensor_values(self.augment_states(next_obs), next_actions_temp, network=self.target_qf2).max(1)[0].view(-1, 1)
                target_q_values = torch.min(target_qf1_values, target_qf2_values)


            q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values

        q1_pred = self.qf1(self.augment_states(obs), actions)
        if self.num_qs > 1:
            q2_pred = self.qf2(self.augment_states(obs), actions)


        qf1_loss = self.qf_criterion(q1_pred, q_target)
        if self.num_qs > 1:
            qf2_loss = self.qf_criterion(q2_pred, q_target)

        # add CQL loss
        random_actions_tensor = torch.FloatTensor(q2_pred.shape[0] * self.num_random, actions.shape[-1]).uniform_(-1, 1).to(ptu.device)
        curr_actions_tensor, _, _, curr_log_pis = self._get_policy_actions(obs, num_actions=self.num_random, network=self.policy)
        curr_log_pis = curr_log_pis.view(q2_pred.shape[0], self.num_random, 1)
        next_curr_actions_tensor, _, _, next_curr_log_pis = self._get_policy_actions(next_obs, num_actions=self.num_random, network=self.policy)
        next_curr_log_pis = next_curr_log_pis.view(q2_pred.shape[0], self.num_random, 1)

        q1_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.qf1)
        q2_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.qf2)
        q1_curr_actions = self._get_tensor_values(obs, curr_actions_tensor.detach(), network=self.qf1)
        q2_curr_actions = self._get_tensor_values(obs, curr_actions_tensor.detach(), network=self.qf2)
        q1_next_actions = self._get_tensor_values(obs, next_curr_actions_tensor.detach(), network=self.qf1)
        q2_next_actions = self._get_tensor_values(obs, next_curr_actions_tensor.detach(), network=self.qf2)

        if self.min_q_version == 3:
            # importance sammpled version
            random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
            cat_q1 = torch.cat(
                [q1_rand - random_density, q1_next_actions - next_curr_log_pis.detach(), q1_curr_actions - curr_log_pis.detach()], 1
            )
            cat_q2 = torch.cat(
                [q2_rand - random_density, q2_next_actions - next_curr_log_pis.detach(), q2_curr_actions - curr_log_pis.detach()], 1
            )
            std_q1 = torch.std(cat_q1, dim=1)
            std_q2 = torch.std(cat_q2, dim=1)
        else:
            cat_q1 = torch.cat(
                [q1_rand, q1_pred.unsqueeze(1), q1_next_actions, q1_curr_actions], 1
            )
            cat_q2 = torch.cat(
                [q2_rand, q2_pred.unsqueeze(1), q2_next_actions, q2_curr_actions], 1
            )
            std_q1 = torch.std(cat_q1, dim=1)
            std_q2 = torch.std(cat_q2, dim=1)

        min_qf1_loss = torch.logsumexp(cat_q1 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
        min_qf2_loss = torch.logsumexp(cat_q2 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp

        """Subtract the log likelihood of data"""
        min_qf1_loss = min_qf1_loss - q1_pred.mean() * self.min_q_weight
        min_qf2_loss = min_qf2_loss - q2_pred.mean() * self.min_q_weight

        if self.with_lagrange:
            alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
            min_qf1_loss = alpha_prime * (min_qf1_loss - self.target_action_gap)
            min_qf2_loss = alpha_prime * (min_qf2_loss - self.target_action_gap)

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-min_qf1_loss - min_qf2_loss)*0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()

        qf1_loss = qf1_loss + min_qf1_loss
        qf2_loss = qf2_loss + min_qf2_loss

        """
        Update Critic
        """
        # Update the Q-functions iff
        if self.num_qs == 1:
            self.qf1_optimizer.zero_grad()
            qf1_loss.backward()
            self.qf1_optimizer.step()
        else:
            total_qf_loss = qf1_loss + qf2_loss
            self.qf1_optimizer.zero_grad()
            self.qf2_optimizer.zero_grad()
            total_qf_loss.backward()
            self.qf1_optimizer.step()
            self.qf2_optimizer.step()

        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_normal_mean, policy_normal_std, log_pi = self._get_policy_actions(obs,
                                                                                                  network=self.policy)

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        if self.num_qs == 1:
            q_new_actions = self.qf1(obs, new_obs_actions)
        else:
            q_new_actions = torch.min(
                self.qf1(obs, new_obs_actions),
                self.qf2(obs, new_obs_actions),
            )

        policy_loss = (self.alpha*log_pi - q_new_actions).mean()

        if self._n_train_steps_total + 1 < self.bc_warm_start:
            """
            For the initial few epochs, try doing behaivoral cloning, if needed
            conventionally, there's not much difference in performance with having 20k 
            gradient steps here, or not having it
            """
            policy_normal_mean = torch.clamp(policy_normal_mean, MEAN_MIN, MEAN_MAX)
            policy_log_prob = self.policy.logprob(actions, policy_normal_mean, policy_normal_std) # no gradient flow
            policy_loss = (self.alpha * log_pi - policy_log_prob).mean()

        """
        Update Policy
        """
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        ptu.soft_update_from_to(
            self.qf1, self.target_qf1, self.soft_target_tau
        )
        if self.num_qs > 1:
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['min QF1 Loss'] = np.mean(ptu.get_numpy(min_qf1_loss))
            if self.num_qs > 1:
                self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
                self.eval_statistics['min QF2 Loss'] = np.mean(ptu.get_numpy(min_qf2_loss))

            self.eval_statistics['Std QF1 values'] = np.mean(ptu.get_numpy(std_q1))
            self.eval_statistics['Std QF2 values'] = np.mean(ptu.get_numpy(std_q2))
            self.eval_statistics.update(create_stats_ordered_dict(
                'QF1 in-distribution values',
                ptu.get_numpy(q1_curr_actions),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'QF2 in-distribution values',
                ptu.get_numpy(q2_curr_actions),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'QF1 random values',
                ptu.get_numpy(q1_rand),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'QF2 random values',
                ptu.get_numpy(q2_rand),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'QF1 next_actions values',
                ptu.get_numpy(q1_next_actions),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'QF2 next_actions values',
                ptu.get_numpy(q2_next_actions),
            ))

            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            if self.num_qs > 1:
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q2 Predictions',
                    ptu.get_numpy(q2_pred),
                ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_normal_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy std',
                ptu.get_numpy(policy_normal_std),
            ))

            if self.automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = self.alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()

            if self.with_lagrange:
                self.eval_statistics['Alpha_prime'] = alpha_prime.item()
                self.eval_statistics['min_q1_loss'] = ptu.get_numpy(min_qf1_loss).mean()
                self.eval_statistics['min_q2_loss'] = ptu.get_numpy(min_qf2_loss).mean()
                self.eval_statistics['threshold action gap'] = self.target_action_gap
                self.eval_statistics['alpha prime loss'] = alpha_prime_loss.item()

        self._n_train_steps_total += 1

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        nets = [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]
        return nets

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
        )


