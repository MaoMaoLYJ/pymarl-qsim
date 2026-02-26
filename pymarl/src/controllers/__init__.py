REGISTRY = {}

from .basic_controller import BasicMAC
from .lica_controller import LICAMAC
from .ppo_controller import PPOMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["lica_mac"] = LICAMAC
REGISTRY["ppo_mac"] = PPOMAC