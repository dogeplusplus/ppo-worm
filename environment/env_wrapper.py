import gym
import numpy as np

from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


class WormGymWrapper(gym.Env):
    def __init__(self, env_file, time_scale=10., no_graphics=False):
        self.env = self._create_env(env_file, time_scale, no_graphics)
        # Reset to get behavior names
        self.env.reset()
        
        self.behavior_name = list(self.env.behavior_specs)[0]
        behavior_spec = self.env.behavior_specs[self.behavior_name]
        self.observation_space = behavior_spec.observation_specs[0]
        self.action_space = behavior_spec.action_spec

    def _create_env(self, env_file, time_scale, no_graphics):
        channel = EngineConfigurationChannel()
        env = UnityEnvironment(
            file_name=env_file, 
            no_graphics=no_graphics, 
            side_channels=[channel]
        )
        channel.set_configuration_parameters(
            time_scale=time_scale,
        )
        return env

    def reset(self):
        self.env.reset()
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        observation, _ = self._decision_to_observation(decision_steps)
        return observation
    
    def step(self, action):
        act = ActionTuple(action)
        self.env.set_actions(self.behavior_name, act)
        
        self.env.step()
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        observation, reward = self._decision_to_observation(decision_steps)
        done = len(decision_steps) == 0
        info = {}
        return observation, reward, done, info

    def _decision_to_observation(self, decision_steps):
        steps = len(decision_steps)
        observations = np.concatenate([decision_steps[i].obs for i in range(steps)], axis=0)
        rewards = np.stack([decision_steps[i].reward for i in range(steps)])

        return observations, rewards

    def render(self):
        self.env.render()
    
        
def main():
    env_file = "environments/worm/worm.x86_64"
    env = WormGymWrapper(env_file)
    
    while True:
        x = np.random.normal(0, 1, size=(10, 9))
        obs, rew, done, info = env.step(x)
    
if __name__ == "__main__":
    main()

