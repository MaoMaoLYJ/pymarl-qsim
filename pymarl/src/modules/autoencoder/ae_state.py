import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class autoencoder(nn.Module):
    def __init__(self, args, input_shape):
        super(autoencoder, self).__init__()
        self.args = args
        self.input_dim = input_shape
        self.hidden_dim = args.ae_hidden_dim
        self.n_agents = args.n_agents
        self.n_actions = self.args.n_actions
        self.state_dim = int(np.prod(args.state_shape))

        self.obs_state_encoder = nn.Sequential(
            nn.Linear(self.input_dim + self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(self.n_actions, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.merge_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.prediction_network = nn.Sequential(
            nn.Linear(self.hidden_dim * self.n_agents, self.hidden_dim * self.n_agents),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * self.n_agents,  self.input_dim * self.n_agents)
        )

    def predict_next_obs(self, obs_state, actions):
        obs_state_embedding = self.obs_state_encoder(obs_state)
        act_embedding = self.action_encoder(actions)

        # Merge and predict next observation
        merge = torch.cat([obs_state_embedding, act_embedding], dim=-1)
        node_features = self.merge_encoder(merge)
        node_features = node_features.reshape(node_features.size(0), node_features.size(1), -1)
        predicted_next_obs = self.prediction_network(node_features)

        predicted_next_obs = predicted_next_obs.reshape(predicted_next_obs.size(0), predicted_next_obs.size(1), self.n_agents, self.input_dim)

        return predicted_next_obs

    def get_embedding(self, obs_state):
        obs_state = obs_state.reshape([-1, self.n_agents, self.input_dim + self.state_dim])
        actions = F.one_hot(torch.arange(self.args.n_actions), num_classes=self.args.n_actions).to(obs_state.device)
        obs_state_embedding = self.obs_state_encoder(obs_state)
        act_embedding = self.action_encoder(actions.float())
        obs_state_embedding = obs_state_embedding.unsqueeze(-2).repeat(1, 1, self.args.n_actions, 1)
        act_embedding = act_embedding[None, None, :, :].repeat(*obs_state_embedding.size()[:2], 1, 1)
        merge = torch.cat([obs_state_embedding, act_embedding], dim=-1)

        node_features = self.merge_encoder(merge)
        return node_features
