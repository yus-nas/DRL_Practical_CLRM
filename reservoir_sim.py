from darts.models.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
from darts.models.physics.dead_oil_python import DeadOil
#from darts.models.physics.dead_oil import DeadOil
from darts.engines import value_vector, redirect_darts_output

import pandas as pd
import numpy as np
import os
import contextlib
import copy

#redirect_darts_output('m.log')

class Simulator(DartsModel):
    def __init__(self, sim_input):
        # call base class constructor
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f): # disable print
            super().__init__()
        
        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        # dimension
        self.nx = sim_input["nx"]
        self.ny = sim_input["ny"]
        self.nz = sim_input["nz"]
        
        # scaler
        self.scaler = sim_input["scaler"]
        
        # create sim and reservoir objects
        self.reset_vars(sim_input)
        
    def reset_vars(self, res_param):
        
        
        # create copy of input data
        sim_input = copy.deepcopy(res_param)
        
        # solver parameters
        self.params.first_ts = sim_input["first_ts"]
        self.params.mult_ts = sim_input["mult_ts"]
        self.params.max_ts = sim_input["max_ts"]
        self.params.tolerance_newton = sim_input["tolerance_newton"]
        self.params.tolerance_linear = sim_input["tolerance_linear"]

        # timing
        self.runtime = sim_input["runtime"]
        self.total_time = sim_input["total_time"]
        self.num_run_per_step = sim_input["num_run_per_step"]
        self.len_cont_step = sim_input["len_cont_step"]

        # well parameters
        self.num_prod  = 0
        self.num_inj   = 0
        self.prod_bhp = sim_input["prod_bhp"]
        self.inj_bhp = sim_input["inj_bhp"]
        self.well_radius = sim_input["well_radius"]
        self.skin = sim_input["skin"]
        self.wc_lim = sim_input["water_cut_limit"]
        self.max_liq_prod_rate = sim_input["max_liq_prod_rate"]
        self.max_water_inj_rate = sim_input["max_water_inj_rate"]
        
        # economic parameters
        self.oil_price = sim_input["oil_price"]
        self.wat_prod_cost = sim_input["wat_prod_cost"]
        self.wat_inj_cost = sim_input["wat_inj_cost"]
        self.capex = sim_input["capex"]
        self.opex = sim_input["opex"]
        self.discount_rate = sim_input["discount_rate"]
        self.npv_scale = sim_input["npv_scale"]
        self.cash_flow_list = []
        self.irr_list = []
        
        # noise
        self.noise = sim_input["noise"]
        self.std_rate_min = sim_input["std_rate_min"]
        self.std_rate_max = sim_input["std_rate_max"]
        self.std_rate = sim_input["std_rate"]
        self.std_pres = sim_input["std_pres"]
        
        # initial P and Zw
        self.init_p = sim_input["Pi"]
        self.init_sw = sim_input["Swi"]  

        # reservoir construction
        kx = np.loadtxt(sim_input["realz_path"]+"{}.in".format(sim_input["realz"]), skiprows=1, comments='/')
        if self.nz > 1: #3d
            kz = sim_input["kv_kh"] * kx
        else:
            kz = 100
        self.reservoir = None
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f): # disable print
            self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=self.ny, nz=self.nz, \
                  dx=sim_input["dx"], dy=sim_input["dy"], dz=sim_input["dz"], permx=kx, permy=kx,\
                  permz=kz, poro=sim_input["poro"], depth=sim_input["depth"], actnum=sim_input["actnum"])
        
        # setup physics
        self.physics = None
        file_name = sim_input["physics_file"]
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f): # disable print
            self.physics = DeadOil(timer=self.timer, physics_filename=file_name, n_points=300, \
                                   min_p=100, max_p=600, min_z=1e-8)
                   
        # add wells
        self.nw_added = len(sim_input["well_comp"])
        for comp_dat in sim_input["well_comp"]:
            self.add_well(*comp_dat)
       
        # equilibrium initialization
        if self.nz > 1: #3d
            self.initialize_equil_pres_comp(sim_input)
    
        # initialize DARTS
        self.init()
        self.timer.node["initialization"].stop()

        
    def initialize_equil_pres_comp(self, sim_input):
        '''
        Initializes pressure based on equilibrium condition and calculates initial composition.
        '''         
        dh_grav = 9.80665e-5 * (sim_input["depth"] - sim_input["datum"])
        
        P_initial = self.init_p * np.ones_like(sim_input["depth"])       
        error = 1e-2
        while True:
            P_old = P_initial       
            oil_dens = np.array([self.physics.property_data.do_oil_dens_ev.evaluate(value_vector([p, 0])) for p in P_initial])
            wat_dens = np.array([self.physics.property_data.do_wat_dens_ev.evaluate(value_vector([p, 0])) for p in P_initial])
            av_dens = self.init_sw * wat_dens + (1 - self.init_sw) * oil_dens
            P_initial = self.init_p + av_dens * dh_grav
            if np.linalg.norm(P_initial - P_old) < error:
                break
        
        # water saturation to composition
        oil_dens = np.array([self.physics.property_data.do_oil_dens_ev.evaluate(value_vector([p, 0])) for p in P_initial])
        wat_dens = np.array([self.physics.property_data.do_wat_dens_ev.evaluate(value_vector([p, 0])) for p in P_initial])
        self.init_zw = (wat_dens * self.init_sw)/(wat_dens * self.init_sw + oil_dens * (1 - self.init_sw)) 
        self.equil_p = P_initial    
        
    def set_initial_conditions(self):
        if self.nz > 1: #3d
            pres = np.append(self.equil_p, np.array([350]*(2*self.nw_added))) # add shadow blocks
            zw = np.append(self.init_zw, np.array([0.2]*(2*self.nw_added)))
            self.physics.set_nonuniform_initial_conditions(self.reservoir.mesh, nonuniform_pressure=pres, nonuniform_composition=[zw])     
        else:
            state = value_vector([self.init_p, 0]) # the sat value isnt important
            oil_dens = self.physics.property_data.do_oil_dens_ev.evaluate(state)
            wat_dens = self.physics.property_data.do_wat_dens_ev.evaluate(state)   
            Zw_init = (wat_dens * self.init_sw)/(wat_dens * self.init_sw + oil_dens * (1 - self.init_sw)) 
            self.physics.set_uniform_initial_conditions(self.reservoir.mesh, uniform_pressure=self.init_p,\
                                                    uniform_composition=[Zw_init])
        
    def set_boundary_conditions(self):   
        self.shut_water_cut_status = {}
        for well in self.reservoir.wells: 
            if well.name[0] == 'I':  
                well.control = self.physics.new_bhp_water_inj(self.inj_bhp)
                well.constraint = self.physics.new_rate_water_inj(self.max_water_inj_rate)
            else:     
                well.control = self.physics.new_bhp_prod(self.prod_bhp)
                well.constraint = self.physics.new_rate_liq_prod(self.max_liq_prod_rate)
                self.shut_water_cut_status[well.name] = False

    def set_well_control(self, cont):  
        assert len(cont) == len(self.reservoir.wells)
        
        pres = self.physics.engine.X[0::2]
        p_allowance = 0.2
        for well, ctrl in zip(self.reservoir.wells, cont):
            well_blck = well.perforations[0][1]
            if well.name[0] == 'I':                 
                if ctrl < pres[well_blck]+p_allowance:
                    well.control = self.physics.new_rate_water_inj(0)
                else:
                    well.control = self.physics.new_bhp_water_inj(ctrl)  
                    well.constraint = self.physics.new_rate_water_inj(self.max_water_inj_rate)
            else:
                if ctrl > pres[well_blck]-p_allowance or self.shut_water_cut_status[well.name]:
                    well.control = self.physics.new_rate_liq_prod(0)
                else:
                    well.control = self.physics.new_bhp_prod(ctrl)
                    well.constraint = self.physics.new_rate_liq_prod(self.max_liq_prod_rate) # needs to be repeated
      
    
    def check_inverse_flow_and_wc_lim(self):

        # extract data 
        td = pd.DataFrame.from_dict(self.physics.engine.time_data.copy())
        time_data = td.iloc[-1]
        
        for well in self.reservoir.wells:  
            WWR = time_data['{} : water rate (m3/day)'.format(well.name)]
            WOR = time_data['{} : oil rate (m3/day)'.format(well.name)] 
            
            if well.name[0] == 'I' and WWR < 0:     
                well.control = self.physics.new_rate_water_inj(0)
            elif well.name[0] == 'P':
                WC = WWR/(WWR+WOR+1e-6)
                if WOR > 0 or WC > self.wc_lim:                
                    well.control = self.physics.new_rate_liq_prod(0)
                    if WC > self.wc_lim:
                        self.shut_water_cut_status[well.name] = True
     
    def add_well(self, well_name, well_type, loc_x, loc_y, loc_z1, loc_z2):               
        rad = self.well_radius
        
        # drill well
        if well_type == 'INJ': # INJECTOR
            self.num_inj += 1
            self.reservoir.add_well(well_name, wellbore_diameter=rad*2)
            skin = 0
        elif well_type == 'PROD': # PRODUCER
            self.num_prod += 1
            self.reservoir.add_well(well_name, wellbore_diameter=rad*2)
            skin = self.skin
        else:
            raise ValueError('Wrong well type for well {}'.format(well_name))
        
        # perfs   
        for loc_z in range(loc_z1, loc_z2+1):
            self.reservoir.add_perforation(self.reservoir.wells[-1], loc_x, loc_y, loc_z,\
                                           well_radius=rad, skin=skin, multi_segment=False)
            
    def run_single_ctrl_step(self):    
        
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f): # disable print

            # advance simulation 
            for run in range(self.num_run_per_step):
    #                 if run == 0:
    #                     self.run_python(restart_dt=self.params.first_ts)
    #                 else:
                self.run_python()
                
        return self.physics.engine.t >= self.total_time
    
    
    def calculate_npv_and_irr(self):
        
        # beginning of current production stage
        time_from = self.physics.engine.t - self.len_cont_step
        
        # extract data for current production stage
        td = pd.DataFrame.from_dict(self.physics.engine.time_data.copy())
        time_data = td.truncate(before=td.loc[td.time > time_from].index[0])
        
        # time
        time = time_data['time'].values
        
        # time steps
        time_steps = time - np.concatenate(([time_from], time[:-1]))
        
        # production and injection data
        disc_cum_oil_prod = disc_cum_water_prod = disc_cum_water_inj = 0
        cum_oil_prod = cum_water_prod = cum_water_inj = 0
        for well in self.reservoir.wells:         
            
            WOR = 6.28981 * np.abs(time_data['{} : oil rate (m3/day)'.format(well.name)].values)
            WWR = 6.28981 * np.abs(time_data['{} : water rate (m3/day)'.format(well.name)].values)
            
            # temporal production/injection volume
            WO = time_steps * WOR
            WW = time_steps * WWR

            # discounted production/injection 
            discounted_time = np.float_power(1 + self.discount_rate, -time/365.0)
            well_disc_cum_oil = np.sum(WO * discounted_time)
            well_disc_cum_wat = np.sum(WW * discounted_time)
            
            # aggregate production/injection
            if well.name[0] == 'P':               
                disc_cum_oil_prod += well_disc_cum_oil
                disc_cum_water_prod += well_disc_cum_wat
                cum_oil_prod += np.sum(WO)
                cum_water_prod += np.sum(WW)
            else:
                disc_cum_water_inj += well_disc_cum_wat
                cum_water_inj += np.sum(WW)
                      
        # economic parameters
        po, cwp, cwi, opex = self.oil_price, self.wat_prod_cost, self.wat_inj_cost, self.opex
                
        # calculate npv
        opex_disc = np.float_power(1 + self.discount_rate, -time_from/365.0)
        npv = disc_cum_oil_prod * po - disc_cum_water_prod * cwp - disc_cum_water_inj * cwi - opex_disc * opex * self.len_cont_step
        
        # calculate cashflow (nondiscounted)
        cashflow = cum_oil_prod * po - cum_water_prod * cwp - cum_water_inj * cwi - opex * self.len_cont_step
        self.cash_flow_list.append(cashflow)
        
        # calculate IRR
        cont_step_time = np.arange(1, len(self.cash_flow_list)+1) * self.len_cont_step
        cash_flow_array = np.array(self.cash_flow_list)
        
        if any(cash_flow_array < 0): # negative cashflow
            mask = cash_flow_array < 0
            neg_cash_flow = cash_flow_array[mask] * np.float_power(1 + self.discount_rate, -cont_step_time[mask]/365.0)
        else:
            neg_cash_flow = 0
        present_investment_val = self.capex + np.abs(np.sum(neg_cash_flow))
        
        if any(cash_flow_array > 0): # positive cashflow
            mask = cash_flow_array > 0
            pos_cash_flow = cash_flow_array[mask] * \
                        np.float_power(1 + self.discount_rate, (cont_step_time[-1] - cont_step_time[mask])/365.0)
        else:
            pos_cash_flow = 0
        future_return_val = np.sum(pos_cash_flow)
        
        irr = np.float_power(future_return_val/present_investment_val, 365/cont_step_time[-1]) - 1
        self.irr_list.append(irr)
        irr = np.max([irr, 0])
        
   
        return npv/self.npv_scale, irr

    def get_observation(self): 
        
        # beginning of current production stage
        time_from = self.physics.engine.t - self.runtime * self.num_run_per_step
        time_of_interest = np.arange(time_from, self.physics.engine.t, self.runtime) + self.runtime
        
        # extract data for current production stage
        td = pd.DataFrame.from_dict(self.physics.engine.time_data.copy())
        time_data = td.truncate(before=td.loc[td.time > time_from].index[0])
        
        # mask for times of interest
        mask = time_data.isin({'time':time_of_interest})['time'].values
        
        obs_data = np.array([]).reshape(0, self.num_run_per_step)
        for well in self.reservoir.wells:                     
            WOR = np.abs(time_data['{} : oil rate (m3/day)'.format(well.name)].values)
            WWR = np.abs(time_data['{} : water rate (m3/day)'.format(well.name)].values)
            BHP = np.abs(time_data['{} : BHP (bar)'.format(well.name)].values)
            
            if self.noise:
                WOR = self.add_noise(WOR, "rate")
                WWR = self.add_noise(WWR, "rate")
                BHP = self.add_noise(BHP, "pressure")
            
            # observation
            obs_data = np.vstack((obs_data, BHP[mask]))
            if well.name[0] == 'I':
                obs_data = np.vstack((obs_data, WWR[mask]))     
            else:
                obs_data = np.vstack((obs_data, WOR[mask]))
                WC = WWR/(WWR+WOR+1e-6)
                obs_data = np.vstack((obs_data, WC[mask]))
                    
        unscaled = obs_data.T.flatten()    
        scaled =  self.scale_channel(unscaled, self.scaler[:,0], self.scaler[:,1])
        
        return scaled, unscaled
                
    def scale_channel(self, in_ch, min_ch, max_ch,  new_range=[0,1]):
            
        return (new_range[1] - new_range[0]) * (in_ch - min_ch)/(max_ch - min_ch) + new_range[0]
     
    def add_noise(self, qoi, qty_type):
        
        rand_vec = np.random.normal(0, 1, size=qoi.shape)
        if qty_type == "rate":
            qoi_noise = np.abs(qoi + (qoi*self.std_rate).clip(self.std_rate_min, self.std_rate_max) * rand_vec)
        elif qty_type == "pressure":
            qoi_noise = np.abs(qoi + self.std_pres * rand_vec)
        else:
            raise ValueError('Wrong quantity type')
                  
        return qoi_noise
       
