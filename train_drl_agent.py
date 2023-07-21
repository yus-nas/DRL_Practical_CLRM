import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from reservoir_env import ReservoirEnv
from network_model_attention_cnn_reg import GTrXLNet as MyModel
from sim_opt_setup import Sim_opt_setup

import os

# parameters
num_cpus = 160 #int(sys.argv[1])
num_opt_ctrl_step = 7
num_sim_iter = 1
num_training_iter = 1000
memory_len = 7

sim_input = Sim_opt_setup()
sim_input["reg_ctrl"] = True
sim_input["epl_mode"] = "irr" # "negative_npv" #"negative_npv" #  # None (end of lease) or negative_npv or irr

if num_training_iter < 1000:
    sim_input["epl_start_iter"] = 50
else:
    sim_input["epl_start_iter"] = 100

cur_env_config = {"sim_input": sim_input}

ray.init(ignore_reinit_error=True, log_to_driver=False, address=os.environ["ip_head"], include_dashboard=False)#, memory=500 * 1024 * 1024)
ModelCatalog.register_custom_model("my_model", MyModel)

nstep = num_opt_ctrl_step*num_sim_iter
tune.run(
    "PPO",
    stop={ "training_iteration": num_training_iter,},
    config={
        "env": ReservoirEnv,
        "model": {
            "custom_model": "my_model",
            "max_seq_len": memory_len,
            "custom_model_config": {
                "num_transformer_units": 2, #base 2
                "attention_dim": 128, #64
                "num_heads": 2, #base 2
                "memory_inference": memory_len, #num_opt_ctrl_step, 
                "memory_training": memory_len, #num_opt_ctrl_step,  
                "head_dim": 64, #64
                "position_wise_mlp_dim": 64,  #base 64
            },
        },
        
        "num_envs_per_worker": 2, # 4
        "remote_worker_envs": True,
        
        "num_workers": num_cpus,
        "num_cpus_for_driver": 8, #8,
        "num_gpus": 0,
        "train_batch_size": num_cpus * nstep,  # Total number of steps per iterations
       
        "batch_mode": "complete_episodes",
        #"rollout_fragment_length": nstep,
        
        "sgd_minibatch_size": 128, 
        
        "gamma": 0.9997,

        # "lr": 5e-5,
        # "entropy_coeff": 1e-4, #1e-3, 
        "lr_schedule": [[0, 1e-4], [num_cpus * nstep * num_training_iter, 1e-5]], 
		"entropy_coeff_schedule": [[0, 1e-3], [num_cpus * nstep * num_training_iter, 1e-5]],
        "vf_loss_coeff": 1,
        "num_sgd_iter": 10,
        
        
        "env_config": cur_env_config,
    },
    sync_config=tune.SyncConfig(syncer=None),  # Disable syncing
    
   # checkpoint_at_end=True,
    checkpoint_freq = 5,
    local_dir="./logs", 
    #restore="/scratch/users/nyusuf/logs/PPO/PPO_ReservoirEnv_a1429_00000_0_2021-09-18_16-53-32/checkpoint_000300/checkpoint-300"
)

ray.shutdown()
