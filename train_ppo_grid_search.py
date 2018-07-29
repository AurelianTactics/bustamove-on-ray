from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from ray.tune import grid_search, run_experiments
from ray.tune.registry import register_env
import sonic_on_ray


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

register_env(env_name,
             lambda config: sonic_on_ray.make(game='BustAMove-Snes',
                                              state='BustAMove.1pplay.Level10'))

ray.init()

run_experiments({
    'sonic-ppo': {
        'run': 'PPO',
        'env': 'sonic_env',
        # 'trial_resources': {
        #     'gpu': 2,  # note, keep this in sync with 'devices' config value
        #     'cpu': lambda spec: spec.config.num_workers,  # one cpu per worker
        # },
        'config': {
            # grid search over learning rate
            'sgd_stepsize': grid_search([1e-4, 5e-5, 1e-5, 5e-6]),

            # fixed params for everything else
            'timesteps_per_batch': 128,#40000,
            #'min_steps_per_task': 100,
            'num_workers': 2,
            'gamma': 0.99,
            'lambda': grid_search([0.93, 0.95, 0.97, 0.99]),
            'clip_param': grid_search([0.1, 0.2]),
            'num_sgd_iter': grid_search([2, 3]),
            #'sgd_batchsize': 4096,
            'use_gae': True,
            'horizon': 512,#4000,
            #'devices': ['/gpu:0', '/gpu:1'],
            'tf_session_args': {
                'gpu_options': {'allow_growth': True}
            },
        },
    },
})
