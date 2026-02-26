from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .actor_critic_pac_learner import PACActorCriticLearner
from .actor_critic_pac_dcg_learner import PACDCGLearner
from .maddpg_learner import MADDPGLearner
from .ppo_learner import PPOLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .qsim_qplex_learner import QsimQplexLearner
from .fmac_learner import FMACLearner
from .max_q_learner import MAXQLearner
from .qsim_qmix_learner import QsimQmixLearner


REGISTRY = {}
REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["pac_learner"] = PACActorCriticLearner
REGISTRY["pac_dcg_learner"] = PACDCGLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["qsim_qplex_learner"] = QsimQplexLearner
REGISTRY["fmac_learner"] = FMACLearner
REGISTRY["max_q_learner"] = MAXQLearner
REGISTRY["qsim_qmix_learner"] = QsimQmixLearner
