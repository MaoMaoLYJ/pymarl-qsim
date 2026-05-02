import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.qsim_vdn import VDNMixer
from modules.mixers.qsim_mix import QMixer
import torch as th
from torch.optim import RMSprop, Adam
from modules.autoencoder.ae_state import autoencoder
import torch.nn.functional as F
import json
import numpy as np
import os
import matplotlib.pyplot as plt


class QsimQmixLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
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

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        
        self.ae = autoencoder(args, self.input_shape)
        
        self.ae_params = list(self.ae.parameters())
        self.ae_optimiser = Adam(params=self.ae_params,  lr=args.lr)

        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        # --- MODIFICATION: Prepare path for saving estimation_error ---
        self.estimation_error_filename = f"{self.args.name}_{self.args.env_args['map_name']}_{self.args.seed}.txt"
        self.save_dir = "estimation_error"
        os.makedirs(self.save_dir, exist_ok=True)
        self.estimation_error_filepath = os.path.join(self.save_dir, self.estimation_error_filename)
        # ---

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        onehot_action = batch["actions_onehot"]
        state = batch["state"]
        obs_inputs = self._build_inputs(batch)

        # --- MODIFICATION: Calculate True Discounted Returns ---
        with th.no_grad():
            full_rewards = batch["reward"]
            full_terminated = batch["terminated"].float()
            true_discounted_returns = th.zeros_like(full_rewards)
            true_discounted_returns[:, -1] = full_rewards[:, -1]
            for t in range(batch.max_seq_length - 2, -1, -1):
                true_discounted_returns[:, t] = full_rewards[:, t] + self.args.gamma * true_discounted_returns[:, t+1] * (1 - full_terminated[:, t])
            true_discounted_returns_for_log = true_discounted_returns[:, :-1]
        # ---

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time


        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

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
        grad_norm = th.nn.utils.clip_grad_norm_(self.ae_params, self.args.grad_norm_clip)
        self.ae_optimiser.step()


        # Max over target Q-Values
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
            target_max_qvals = th.gather(target_mac_out.unsqueeze(2).repeat(1, 1, modified_idx.size(2), 1, 1), dim=-1, index=modified_idx)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]


        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals.squeeze(-1).permute(0, 1, 3, 2), batch["state"][:, 1:])

        # Calculate the TD target candidates
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        
        # Calculate the weighted TD target
        target_mask = mask.expand_as(targets) * th.gather(batch['avail_actions'][:, 1:].unsqueeze(2).repeat(1, 1, modified_idx.size(2), 1, 1), dim=-1, index=modified_idx).cumprod(dim=-2)[:, :, :, -1, 0].detach() * weight.to(chosen_action_qvals.device)
        targets = (targets * target_mask).sum(dim=-1, keepdim=True) / (target_mask.sum(dim=-1, keepdim=True) + 1e-8)
        
        # ------------QSIM specific code over------------

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()


        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("loss_ae", ae_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            # --- MODIFICATION: Fix logging and add estimation_error ---
            q_taken_mean_val = (chosen_action_qvals * mask).sum().item() / mask_elems
            self.logger.log_stat("q_taken_mean", q_taken_mean_val, t_env)

            target_mean_val = (targets * mask).sum().item() / mask_elems
            self.logger.log_stat("target_mean", target_mean_val, t_env)

            true_return_mean_val = (true_discounted_returns_for_log * mask).sum().item() / mask_elems
            self.logger.log_stat("true_return_mean", true_return_mean_val, t_env) # Still log for info.json

            # Calculate and log the estimation error
            estimation_error_val = q_taken_mean_val - true_return_mean_val
            self.logger.log_stat("estimation_error", estimation_error_val, t_env)
            
            # Append the estimation_error to our dedicated text file
            with open(self.estimation_error_filepath, "a") as f:
                f.write(f"{estimation_error_val}\n")
            # ---
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

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
