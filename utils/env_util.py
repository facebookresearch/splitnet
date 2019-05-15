import gym
import torch
from baselines.common.vec_env.vec_env import VecEnvWrapper

from utils import pytorch_util as pt_util


class LambdaObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, lambda_func):
        super(LambdaObservationWrapper, self).__init__(env)
        self.lambda_func = lambda_func

    def step(self, action):
        result = self.env.step(action)
        return self.observation(result)

    def observation(self, observation):
        return self.lambda_func(observation)


class HabitatVecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv):
        observation_space = venv.observation_spaces[0]
        action_space = venv.action_spaces[0]
        super(HabitatVecEnvWrapper, self).__init__(venv, observation_space, action_space)

    @staticmethod
    def package_data(data):
        return {key: [dd[key] for dd in data] for key in data[0]}

    def reset(self):
        reset_val = self.venv.reset()
        return self.package_data(reset_val)

    def step_async(self, actions):
        self.venv.async_step(actions)

    def step_wait(self):
        results = self.venv.wait_step()
        observations, rewards, dones, infos = zip(*results)
        return self.package_data(observations), list(rewards), list(dones), list(infos)

    @property
    def unwrapped(self):
        return self.venv


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def convert_obs(self, obs):
        if isinstance(obs, tuple):
            obs = (pt_util.from_numpy(val) for val in obs)
        elif isinstance(obs, list):
            obs = [pt_util.from_numpy(val) for val in obs]
        elif isinstance(obs, dict):
            new_obs = {}
            for key, val in obs.items():
                new_val = pt_util.from_numpy(val)
                if new_val is not None:
                    new_obs[key] = new_val
            obs = new_obs
        else:
            obs = torch.from_numpy(obs)
        return obs

    def reset(self):
        obs = self.venv.reset()
        obs = self.convert_obs(obs)
        return obs

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = self.convert_obs(obs)
        reward = torch.tensor(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info
