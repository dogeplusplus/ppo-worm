import pytest
import numpy as np
from collections import namedtuple

from environment.env_wrapper import WormGymWrapper

@pytest.fixture(scope="session")
def worm_env():
    return WormGymWrapper("worm/worm.x86_64", no_graphics=True)


def test_reset(worm_env):
    obs = worm_env.reset()
    assert obs.shape == (10, 64)


def test_step(worm_env):
    action = np.random.normal(size=(10, 9))
    observation, reward, done, info = worm_env.step(action)
    assert observation.shape == (10, 64)
    assert reward.shape == (10,)
    assert done == False


def test_decision_to_observation(worm_env):
    DecisionStep = namedtuple("DecisionStep", ["obs", "reward"])
    
    # Each observation is 64 array
    # Each reward is a float
    decision_steps = [DecisionStep(np.ones((1, 64)), 1.) for _ in range(10)]
    observations, rewards  = worm_env._decision_to_observation(decision_steps)
    
    # Check shapes match when converting
    assert observations.shape == (10, 64)
    assert rewards.shape == (10,)


