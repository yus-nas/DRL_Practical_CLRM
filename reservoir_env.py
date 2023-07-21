import gym
from gym import spaces
import numpy as np
from gym.utils import seeding

from reservoir_sim import Simulator
import gc
import psutil
import copy

class ReservoirEnv(gym.Env):
    def __init__(self, env_config):
               
        self.sim_input = copy.deepcopy(env_config["sim_input"])

        # simulator
        self.res_sim = Simulator(self.sim_input)
        
        # initialize history collector
        self.hist = self.History()
        
        # action and observation space 
        self._setup_spaces()
        
        # training realizations
        worker_ind = np.max([env_config.worker_index - 1, 0])
        num_cluster = np.max(self.sim_input["cluster_labels"]) + 1
        self.cluster_index = worker_ind % num_cluster  # zero-based
        self.realz_train = np.argwhere(self.sim_input["cluster_labels"] == self.cluster_index) + 1
        mask = np.in1d(self.realz_train, self.sim_input["models_to_exclude"], invert=True)
        self.realz_train = self.realz_train[mask]
        
        # track sim iterations
        self.sim_iter = 0 # reset at len(self.realz_train)
        self.total_num_sim = 0
        
        # random number generator
        self.rng = np.random.default_rng(env_config.worker_index)
          
    def _setup_spaces(self):
        
        self.num_obs_data = (3 * self.res_sim.num_prod + 2 * self.res_sim.num_inj + 1) * self.sim_input["num_run_per_step"]
        self.num_wells = self.res_sim.num_prod + self.res_sim.num_inj
        self.num_sim_obs = self.num_obs_data
        if self.sim_input["epl_mode"] == "irr":
            self.num_obs_data += (2*self.sim_input["num_run_per_step"]) 
        
        if self.sim_input["reg_ctrl"]:
            self.action_space = spaces.Box(-1.0, +1.0, shape=[self.num_wells], dtype=np.float32)
            self.num_reg_obs_data = self.num_obs_data + (self.num_wells + 1) * self.sim_input["num_run_per_step"]
            self.observation_space = spaces.Box(-10, 10, shape=(self.num_reg_obs_data,))
        else:
            self.action_space = spaces.Box(0.0, +1.0, shape=[self.num_wells], dtype=np.float32)   
            self.observation_space = spaces.Box(-10, 10, shape=(self.num_obs_data,))
  
    def reset(self, realz=None, reg_limit=None, irr_min=None):   
        
        # helps with memory issues
        #self.auto_garbage_collect()
        
        # select realization  
        if realz == None:
            if self.sim_iter == self.realz_train.shape[0]:
                self.sim_iter = 0
                self.rng.shuffle(self.realz_train)
            
            self.sim_input["realz"] = int(self.realz_train[self.sim_iter])
            self.sim_iter += 1
     
        else:
            self.sim_input["realz"] = realz
        
        # regularize well controls
        if self.sim_input["reg_ctrl"]:
            if reg_limit == None:
                self.reg_limit = self.rng.choice(self.sim_input["ctrl_regs"])
            else:
                self.reg_limit = reg_limit
        
        # irr
        if self.sim_input["epl_mode"] == "irr":
            if irr_min == None:
                self.irr_min = self.rng.choice(self.sim_input["irr_min_list"])
            else:
                self.irr_min = irr_min
          
        # reset
        self.res_sim.reset_vars(self.sim_input)
        self.hist.reset()
        self.cum_reward = 0
        self.cont_step = -1
        self.total_num_sim += 1
        self.prev_irr = 0
        
        # run historical period (assumes length of hist = length of ctrl step) #TODO: make general
        observation, _, _, _ = self.step(self.sim_input["hist_ctrl_scaled"])

        return observation
    
    def step(self, action):
        assert self.action_space.contains(action)
        
        # advance to next control step
        self.cont_step += 1

        if not self.sim_input["reg_ctrl"] or self.cont_step == 0: #history/no regularization
            reg_action = action
        else:
            if self.cont_step == 1: #first opt control step for reg
                reg_action = (action + 1) / 2   # smoothen the control by mapping from [-1, 1] to [0,1]
            else:
                reg_action = self.hist.actions[-1] * (1 + action * self.reg_limit)
                reg_action = reg_action.clip(0, 1)

        
        # add action to history
        self.hist.actions.append(reg_action)
        
        # change well controls
        controls = reg_action * (self.sim_input["upper_bound"] - self.sim_input["lower_bound"]) + self.sim_input["lower_bound"]
        self.res_sim.set_well_control(controls)
       
        # run simulation
        end_of_lease = self.res_sim.run_single_ctrl_step()
        
        # calculate reward and irr
        reward, irr = self.res_sim.calculate_npv_and_irr()
        self.cum_reward += reward
        
        # check if end of project
        pre_check = (self.total_num_sim > self.sim_input["epl_start_iter"] and self.cont_step > 2)
        epl_terminate_neg_npv = (self.sim_input["epl_mode"] in ["negative_npv", "irr"] and pre_check and reward <= 0)
        epl_terminate_irr = (self.sim_input["epl_mode"] == "irr" and pre_check and self.prev_irr > irr and \
                             (irr - self.sim_input["irr_eps"]) < self.irr_min)                                  
        self.prev_irr = irr
        if end_of_lease or epl_terminate_neg_npv or epl_terminate_irr:
            done = True
        else:
            done = False
         
        # get observation
        observation = self.get_observation()  
             
        return observation, reward, done, {} 
    
    class History():
        def __init__(self):
            self.reset()

        def reset(self):
            self.scaled_state = []
            self.unscaled_state = []
            self.reward_dollar = []
            self.actions = []
            self.done = []
            
    def experiment(self, actions):
        # advance from "current state" according to actions
        self.hist.reset()
        
        unscaled_obs = self.get_unscaled_state()
        self.hist.unscaled_state.append(unscaled_obs)
        obs = self.get_observation()
        self.hist.scaled_state.append(obs)

        for i_action in actions:
            obs, reward, done, _ = self.step(i_action)
            self.hist.scaled_state.append(obs)
            self.hist.reward_dollar.append(reward)
            self.hist.done.append(done)
            
            unscaled_obs = self.get_unscaled_state()
            self.hist.unscaled_state.append(unscaled_obs)
    
    def get_unscaled_state(self):
        _ , unscaled_state = self.res_sim.get_observation()
        
        return unscaled_state
                        
    def get_observation(self): 
        obs, _ = self.res_sim.get_observation()
        
        # add time
        norm_time = self.cont_step / (self.sim_input["num_cont_step"] - 1)
        time_obs = np.expand_dims(np.array([norm_time]*self.sim_input["num_run_per_step"]), axis=1)
        obs = np.hstack([obs.reshape(self.sim_input["num_run_per_step"], -1), time_obs]).flatten()
        
        # regularization
        if self.sim_input["reg_ctrl"]:
            reg_prev_act = np.concatenate((self.hist.actions[-1], [self.reg_limit/self.sim_input["ctrl_regs"][-1]]))
            reg_prev_act = np.array([reg_prev_act]*self.sim_input["num_run_per_step"])
            obs = np.hstack([obs.reshape(self.sim_input["num_run_per_step"], -1), reg_prev_act]).flatten()
        
        # irr
        if self.sim_input["epl_mode"] == "irr":
            low_irr, high_irr = self.sim_input["irr_min_list"][0], self.sim_input["irr_min_list"][-1]
            norm_irr_min = (self.irr_min - low_irr) / (high_irr - low_irr)
            irr_obs = np.concatenate(([self.prev_irr], [norm_irr_min]))
            irr_obs = np.array([irr_obs]*self.sim_input["num_run_per_step"])
            obs = np.hstack([obs.reshape(self.sim_input["num_run_per_step"], -1), irr_obs]).flatten()
        
        return obs
    
    
    def auto_garbage_collect(self, pct=55.0):
        """
        auto_garbage_collection - Call the garbage collection if memory used is greater than 65% of total available memory.
                                  This is called to deal with an issue in Ray not freeing up used memory.

            pct - Default value of 65%.  Amount of memory in use that triggers the garbage collection call.
        """
        if psutil.virtual_memory().percent >= pct:
            gc.collect()     
