import copy
import torch as th
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.mixers.qsim_vdn import VDNMixer
from modules.mixers.qsim_mix import QMixer
from modules.autoencoder.ae_state import autoencoder
import torch.nn.functional as F
import numpy as np


class QsimQmixLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.mac = mac
        self.logger = logger
        
        self.input_shape = self._get_input_shape(scheme)

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = Adam(params=self.params, lr=args.lr)

        self.ae = autoencoder(args, self.input_shape)
        self.ae_params = list(self.ae.parameters())
        self.ae_optimiser = Adam(params=self.ae_params,  lr=args.lr)

        self.target_mac = copy.deepcopy(mac)

        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(1,), device=device)
        if self.args.standardise_rewards:
            rew_shape = (1,) if self.args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        
        onehot_action = batch["actions_onehot"]
        state = batch["state"]
        obs_inputs = self._build_inputs(batch)

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # ------------QSIM specific code start------------
        # Get current observations and actions
        current_obs = obs_inputs[:, :-1].detach()
        current_actions = onehot_action[:, :-1]
        current_state = state[:, :-1].unsqueeze(-2).repeat(1, 1, self.args.n_agents, 1).detach()
        current_obs_state = th.cat([current_obs, current_state], dim=-1)

        # Predict next observations
        predicted_next_obs = self.ae.predict_next_obs(current_obs_state, current_actions)

        next_obs = obs_inputs[:, 1:].detach()
        next_state = state[:, 1:].unsqueeze(-2).repeat(1, 1, self.args.n_agents, 1).detach()
        next_obs_state = th.cat([next_obs, next_state], dim=-1)
        
        # Calculate MSE loss
        prediction_mse_loss = (predicted_next_obs - next_obs).pow(2)
        prediction_loss = (prediction_mse_loss * mask.unsqueeze(-1).expand_as(prediction_mse_loss)).sum() / mask.unsqueeze(-1).expand_as(prediction_mse_loss).sum()

        # Optimize AE
        ae_loss = prediction_loss
        self.ae_optimiser.zero_grad()
        ae_loss.backward()
        grad_norm_ae = th.nn.utils.clip_grad_norm_(self.ae_params, self.args.grad_norm_clip)
        self.ae_optimiser.step()

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals_agents = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        target_mac_out = th.stack(target_mac_out[1:], dim=1)
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            with th.no_grad():
                node_features = self.ae.get_embedding(next_obs_state)
                max_actions_embedding = th.gather(node_features, -2, cur_max_actions.flatten(0, 1).unsqueeze(-1).repeat(1, 1, 1,node_features.size(-1)))
                similar = th.nn.functional.cosine_similarity(node_features, max_actions_embedding, dim=-1)
                similar[batch['avail_actions'][:, 1:].flatten(0, 1) == 0] = -999999
                similar_threshold = self.args.similar_threshold
                topn = self.args.n_actions
                k_softmax = self.args.k_softmax
                similar[similar < similar_threshold] = -999999
                idx = similar.argsort(dim=-1, descending=True)
                idx = idx.view(*actions.size()[:2], self.args.n_agents, self.args.n_actions)
                idx = idx[:, :, :, : topn]
                top_similar = th.gather(similar.view(*actions.size()[:2], self.args.n_agents, self.args.n_actions), dim=-1, index=idx)
                weight = F.softmax(top_similar * k_softmax, dim=-1)
                weight = weight.view(weight.size(0), weight.size(1), -1)

                # construct the near-greedy joint action space
                modified_idx = idx.clone()
                modified_idx = modified_idx.unsqueeze(-3).unsqueeze(-3).repeat(1, 1, self.args.n_agents, topn, 1, 1)
                for i in range(idx.size(-2)):
                    for j in range(idx.size(-1)):
                        modified_idx[:, :, i, j, i, 0] = idx[:, :, i, j]
                modified_idx = modified_idx[:, :, :, :, :, 0]
                modified_idx = modified_idx.reshape([modified_idx.size(0), modified_idx.size(1), -1, self.args.n_agents, 1])
            target_max_qvals_agents = th.gather(target_mac_out.unsqueeze(2).repeat(1, 1, modified_idx.size(2), 1, 1), dim=-1, index=modified_idx)
        else:
            target_max_qvals_agents = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals_agents, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals_agents.squeeze(-1).permute(0, 1, 3, 2), batch["state"][:, 1:])

        if self.args.standardise_returns:
            target_max_qvals = target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean
        
        if self.args.common_reward:
             rewards = rewards.view(-1, rewards.size(1), 1)

        # Calculate the TD target candidates
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()
        
        # Calculate the weighted TD target
        target_mask = mask.expand_as(targets) * th.gather(batch['avail_actions'][:, 1:].unsqueeze(2).repeat(1, 1, modified_idx.size(2), 1, 1), dim=-1, index=modified_idx).cumprod(dim=-2)[:, :, :, -1, 0].detach() * weight.to(chosen_action_qvals.device)
        targets = (targets * target_mask).sum(dim=-1, keepdim=True) / (target_mask.sum(dim=-1, keepdim=True) + 1e-8)

        # ------------QSIM specific code over------------

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        # Td-error
        td_error = chosen_action_qvals - targets.detach()
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        loss = (masked_td_error**2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
        
        if (
            self.args.target_update_interval_or_tau > 1
            and (self.training_steps - self.last_target_update_step)
            / self.args.target_update_interval_or_tau
            >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("loss_ae", ae_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item() / mask_elems, t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / mask_elems, t_env)
            self.log_stats_t = t_env

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.ae.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.args.n_agents
        return input_shape

    def _build_inputs(self, batch):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, :batch.max_seq_length])
        if self.args.obs_last_action:
            inputs.append(th.cat([
                th.zeros_like(batch["actions_onehot"][:, 0]).unsqueeze(1), batch["actions_onehot"][:, :batch.max_seq_length-1]
            ], dim=1))
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.args.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, batch.max_seq_length, -1, -1))
        inputs = th.cat(inputs, dim=-1)
        return inputs