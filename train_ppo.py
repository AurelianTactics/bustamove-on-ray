from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonic_on_ray
import ray
from ray.rllib.agents import ppo
from ray.tune.registry import register_env


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
# register_env(env_name,
#              lambda config: sonic_on_ray.make(game='Airstriker-Genesis',
#                                               state='Level1'))

# register_env(env_name,
#              lambda config: sonic_on_ray.make(game='SonicTheHedgehog-Genesis',
#                                               state='GreenHillZone.Act1'))

register_env(env_name,
             lambda config: sonic_on_ray.make(game='BustAMove-Snes',
                                              state='BustAMove.1pplay.Level10'))

ray.init()

config = ppo.DEFAULT_CONFIG.copy()

config.update({
    #'timesteps_per_batch': 40000,
    'timesteps_per_batch': 128,
    #'min_steps_per_task': 100,
    #'num_workers': 32,
    'num_workers': 1,
    'gamma': 0.99,
    'lambda': 0.95,
    'clip_param': 0.1,
    'num_sgd_iter': 2,
    #'sgd_batchsize': 4096,
    'sgd_batchsize': 128,
    'sgd_stepsize': 5e-5,
    'use_gae': True,
    'horizon': 512,
    # 'devices': ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3', '/gpu:4', '/gpu:5',
    #             '/gpu:6', 'gpu:7'],
    #'devices': ['/gpu:0'],
    'num_gpus_per_worker':1,
    'tf_session_args': {
        'gpu_options': {'allow_growth': True}
    }
})

alg = ppo.PPOAgent(config=config, env=env_name)

for i in range(1000):
    result = alg.train()
    print('result = {}'.format(result))

    if i % 10 == 0:
        checkpoint = alg.save()
        print('checkpoint saved at', checkpoint)
