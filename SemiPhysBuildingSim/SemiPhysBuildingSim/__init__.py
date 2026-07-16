import os
from gym.envs.registration import register

print("Register SPBS environment!")

register(
        id='SemiPhysBuildingSim-v0',
        entry_point="SemiPhysBuildingSim.SemiPhysBuildingSim:SemiPhysBuildingSimulation",
    )



