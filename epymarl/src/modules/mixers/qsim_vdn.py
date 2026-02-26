import torch as th
import torch.nn as nn


class VDNMixer(nn.Module):
    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, agent_qs, batch):
        if len(agent_qs.size()) == 3:
            q_tot = th.sum(agent_qs, dim=2, keepdim=True)
            return q_tot
        else:
            q_tot = th.sum(agent_qs, dim=2, keepdim=True)
            return q_tot.squeeze(2)