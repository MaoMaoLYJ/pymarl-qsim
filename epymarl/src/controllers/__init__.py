REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC
from .basic_central_controller import CentralBasicMAC
from .lica_controller import LICAMAC
from .ppo_controller import PPOMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["lica_mac"] = LICAMAC
REGISTRY["ppo_mac"] = PPOMAC