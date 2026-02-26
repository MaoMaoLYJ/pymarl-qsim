REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_ppo_agent import RNNPPOAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ppo"] = RNNPPOAgent