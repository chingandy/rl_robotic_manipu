""" gym’s main purpose is to provide a large collection of environments that expose a common interface and are versioned to allow for comparisons. To list the environments available in your installation, just ask gym.envs.registry:
"""

from gym import envs
envids = [spec.id for spec in envs.registry.all()]
for envid in sorted(envids):
    print(envid)
