from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .qsim_qplex_learner import QsimQplexLearner
from .qsim_qmix_learner import QsimQmixLearner
from .fmac_learner import FMACLearner
from .res_learner import RESLearner
from .qsim_qmix_learner import QsimQmixLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["qsim_qplex_learner"] = QsimQplexLearner
REGISTRY["fmac_learner"] = FMACLearner
REGISTRY["res_learner"] = RESLearner
REGISTRY["qsim_qmix_learner"] = QsimQmixLearner