from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from ray.tune.async_hyperband import AsyncHyperBandScheduler
from ray.tune.registry import register_env
from ray.tune import Trainable, TrainingResult, register_trainable, run_experiments, grid_search
import sonic_on_ray
import random

ahb = AsyncHyperBandScheduler(
        time_attr="timesteps_total",
        reward_attr="episode_reward_mean",
        grace_period=50000,
        max_t=500000)




env_name = 'sonic_env'
# Note that the hyperparameters have been tuned for sonic, which can be used
# run by replacing the below function with:
#
#     register_env(env_name, lambda config: sonic_on_ray.make(
#                                game='SonicTheHedgehog-Genesis',
#                                state='GreenHillZone.Act1'))
#
# However, to try Sonic, you have to obtain the ROM yourself (see then
# instructions at https://github.com/openai/retro/blob/master/README.md).
register_env(env_name,
             lambda config: sonic_on_ray.make(game='BustAMove-Snes',
                                              state='BustAMove.1pplay.Level10'))

ray.init()

run_experiments({
    'bustamove-ppo-hyperband': {
        'run': 'PPO',
        'env': 'sonic_env',
        'stop':{'timesteps_total': 10000000},
        #'stop':{'training_iteration': 100},
        'repeat':10,
        # 'trial_resources': {
        #     'gpu': 2,  # note, keep this in sync with 'devices' config value
        #     'cpu': lambda spec: spec.config.num_workers,  # one cpu per worker
        # },
        "trial_resources": {
                            #"cpu": 4,
                            'cpu': lambda spec: spec.config.num_workers,
                            #'extra_cpu': 2,
                            #'extra_gpu' : ,
                            "gpu": 1
        },
        'config': {
            'horizon': lambda spec: random.randint(1024,8192),#1024, #grid_search([256,512,1024,2048]),
            # # grid search over learning rate
            'sgd_stepsize': lambda spec: random.choice([2e-4, 3e-5, 1e-5, 3e-6]),#grid_search([5e-4, 1e-4, 5e-5, 1e-5]),
            'timesteps_per_batch': lambda spec: random.randint(16,256),#64,#grid_search([16,32,64,128]),#40000,
            # #'min_steps_per_task': 100,
            'num_workers': 2,
            'gamma': lambda spec: random.uniform(0.99, 0.999),#0.99, #grid_search([0.99,0.995,0.999]),
            'lambda': lambda spec: random.uniform(0.9, 1.0),#0.95, #grid_search([0.9, 0.95, 1.0]),
            'clip_param': lambda spec: random.uniform(0.01, 0.4),#0.2,#grid_search([0.1, 0.2, 0.3]),
            'num_sgd_iter': lambda spec: random.randint(3, 8),#3,#grid_search([3, 4, 5, 6]),
            'vf_loss_coeff':lambda spec: random.uniform(0.3, 1),#1,#grid_search([0.5,0.75,1]),
            'entropy_coeff':lambda spec: random.choice([0.0,0.003,0.01]),#grid_search([0.0,0.05,0.1]),
            #'kl_coeff': lambda spec: random.choice([0.0,0.2]), #I think 0.0 turns off kl
            'kl_target': lambda spec: random.uniform(0.003, 0.03), #grid_search([0.003,0.01,0.02,0.03]),
            #'sgd_batchsize': 4096,
            'use_gae': True,
            #'devices': ['/gpu:0', '/gpu:1'],
            'tf_session_args': {
                'gpu_options': {'allow_growth': True}
            },
        },
    } ,
},scheduler=ahb)
