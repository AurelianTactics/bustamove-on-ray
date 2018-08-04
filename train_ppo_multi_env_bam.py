from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonic_on_ray
import ray
import gym
from ray.rllib.agents import ppo
from ray.tune.registry import register_env


env_name = "multienv"

class MultiEnv(gym.Env):
    def __init__(self, env_config):
        # pick actual env based on worker and env indexes
        #print("worker index is {}".format(env_config.worker_index))
        #print("testing vector_index {}".format(env_config.vector_index))
        #BustAMove.Challengeplay0
        challenge_level = env_config.worker_index % 5
        self.env = sonic_on_ray.make(game='BustAMove-Snes', state='BustAMove.Challengeplay{}'.format(challenge_level)) #BustAMove.1pplay.Level10
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    def reset(self):
        return self.env.reset()
    def step(self, action):
        return self.env.step(action)

register_env("multienv", lambda c: MultiEnv(c))

ray.init()

config = ppo.DEFAULT_CONFIG.copy()

config.update({
    #"env_config": env_config,
    #'timesteps_per_batch': 40000,
    'timesteps_per_batch': 32,
    #'min_steps_per_task': 100,
    #'num_workers': 32,
    'num_workers': 4,
    'gamma': 0.995,
    'lambda': 0.95,
    'clip_param': 0.1,
    'num_sgd_iter': 3,
    #'sgd_batchsize': 4096,
    #'sgd_batchsize': 128,
    'sgd_stepsize': 1e-4,
    'use_gae': True,
    'horizon': 4096,
    'entropy_coeff': 0.0,
    # 'devices': ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3', '/gpu:4', '/gpu:5',
    #             '/gpu:6', 'gpu:7'],
    #'devices': ['/gpu:0'],
    #'num_gpus_per_worker':1,
    'tf_session_args': {
        'gpu_options': {'allow_growth': True}
    }
})

alg = ppo.PPOAgent(config=config, env=env_name)

for i in range(10000000):
    result = alg.train()
    print('result = {}'.format(result))

    if i % 10000 == 0:
        checkpoint = alg.save()
        print('checkpoint saved at', checkpoint)
