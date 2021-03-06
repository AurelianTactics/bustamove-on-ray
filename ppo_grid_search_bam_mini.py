from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from ray.tune import grid_search, run_experiments
from ray.tune.registry import register_env
import sonic_on_ray


env_name = 'bustamove_env'
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
# env_name2 = 'bustamove_env2'
# register_env(env_name2,
#              lambda config: sonic_on_ray.make(game='BustAMove-Snes',
#                                               state='BustAMove.1pplay.Level1'))

ray.init()

run_experiments({
    'bustamove-ppo': {
        'run': 'PPO',
        'env':'bustamove_env',
        'stop':{'training_iteration': 50},
        #'env': grid_search(['bustamove_env','bustamove_env2']),
        # 'trial_resources': {
        #     'gpu': 2,  # note, keep this in sync with 'devices' config value
        #     'cpu': lambda spec: spec.config.num_workers,  # one cpu per worker
        # },
        "trial_resources": {
                            'cpu': lambda spec: spec.config.num_workers,
                            'extra_cpu': 1,
                            "gpu": 1
        },
        'config': {
            'horizon':grid_search([1024,4096]),#grid_search([256,512,1024,2048]),
            # grid search over learning rate
            'sgd_stepsize': 2e-4,#grid_search([1e-3, 1e-4, 1e-5]),#grid_search([5e-4, 1e-4, 5e-5, 1e-5]),
            'timesteps_per_batch': 32,#grid_search([32,64,128]),#grid_search([16,32,64,128]),#40000,
            #'min_steps_per_task': 100,
            'num_workers': 2,
            'gamma': grid_search([0.99,0.999]),#grid_search([0.99,0.995,0.999]),
            #'lambda': grid_search([0.9, 1.0]),#grid_search([0.9, 0.95, 1.0]),
            'clip_param': grid_search([0.1, 0.2, 0.3]),
            'num_sgd_iter': 3,#grid_search([3,5]),#grid_search([3, 4, 5, 6]),
            #'vf_loss_coeff':grid_search([0.5,1.0]),#grid_search([0.5,0.75,1]),
            'entropy_coeff':0.0,#grid_search([0.0,0.1]),#grid_search([0.0,0.05,0.1]),
            'kl_coeff':0.0,#grid_search([0.0,0.2]), #I think 0.0 turns off kl
            #'kl_target':grid_search([0.01,0.02]),# grid_search([0.003,0.01,0.02,0.03]),
            #'sgd_batchsize': 4096,#only for gpu
            'use_gae': True,
            #'devices': ['/gpu:0', '/gpu:1'],
            'tf_session_args': {
                'gpu_options': {'allow_growth': True}
            },
        },
    },
})
