from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from ray.tune import run_experiments
from ray.tune.registry import register_env
from ray.tune.pbt import PopulationBasedTraining
import sonic_on_ray
import random




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

# Postprocess the perturbed config to ensure it's still valid
def explore(config):
    #me: not doing GPU training so don't think I need this first part
    # ensure we collect enough timesteps to do sgd
    # if config["timesteps_per_batch"] < config["sgd_batchsize"] * 2:
    #     config["timesteps_per_batch"] = config["sgd_batchsize"] * 2
    # ensure we run at least one sgd iter
    if config["num_sgd_iter"] < 1:
        config["num_sgd_iter"] = 1
    return config


pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        reward_attr="episode_reward_mean",
        perturbation_interval=10,
        resample_probability=0.01,#0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations={
            "lambda": lambda: random.uniform(0.9, 1.0),
            "gamma": lambda: random.uniform(0.99, 0.999),
            "horizon": lambda: random.randint(256,2048),
            "clip_param": lambda: random.uniform(0.01, 0.4),
            "sgd_stepsize": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "num_sgd_iter": lambda: random.randint(2, 10),
            #"sgd_batchsize": lambda: random.randint(128, 16384),
            "timesteps_per_batch": lambda: random.randint(16,256),
            "vf_loss_coeff": lambda: random.uniform(0.3, 1),
            "entropy_coeff": lambda: random.uniform(0.0, 0.2),
            "kl_coeff": [0.0,0.2,1.0],
            "kl_target": lambda: random.uniform(0.003, 0.03)
        }
    #,custom_explore_fn=explore
)

ray.init()

run_experiments({
    'bustamove-ppo-pbt': {
        'run': 'PPO',
        'env': 'sonic_env',
        'stop':{'timesteps_total': 4000000},
        #'stop':{'training_iteration': 100},
        'repeat':20,
        # 'trial_resources': {
        #     'gpu': 2,  # note, keep this in sync with 'devices' config value
        #     'cpu': lambda spec: spec.config.num_workers,  # one cpu per worker
        # },
        "trial_resources": {
                            'cpu': lambda spec: spec.config.num_workers,
                            #'extra_cpu': 1,
                            #'extra_gpu' : ,
                            'gpu': 1
        },
        'config': {
            'horizon': 1024,#1024, #grid_search([256,512,1024,2048]),
            # grid search over learning rate
            'sgd_stepsize': 1e-4,#grid_search([5e-4, 1e-4, 5e-5, 1e-5]),
            'timesteps_per_batch': 32,#64,#grid_search([16,32,64,128]),#40000,
            #'min_steps_per_task': 100,
            'num_workers': 2,
            'gamma': 0.99, #grid_search([0.99,0.995,0.999]),
            'lambda': 0.95, #grid_search([0.9, 0.95, 1.0]),
            'clip_param': 0.2,#grid_search([0.1, 0.2, 0.3]),
            'num_sgd_iter': 2,#grid_search([3, 4, 5, 6]),
            'vf_loss_coeff':1,#grid_search([0.5,0.75,1]),
            'entropy_coeff':0.0,#grid_search([0.0,0.05,0.1]),
            #'kl_coeff':0.2,#grid_search([0.0,0.2]), #I think 0.0 turns off kl
            'kl_target':0.01,#grid_search([0.003,0.01,0.02,0.03]),
            #'sgd_batchsize': 4096,
            'use_gae': True,
            #'devices': ['/gpu:0', '/gpu:1'],
            'tf_session_args': {
                'gpu_options': {'allow_growth': True}
            },
        },
    },
},scheduler=pbt)
