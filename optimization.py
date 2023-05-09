from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
import nmodl
import similaritymeasures
from nmodl import dsl
from neuron import h,gui
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import deap
import multiprocessing
import random
from scipy.signal import argrelmax, argrelmin

class Section():
    def __init__(self):
        self.soma = h.Section(name="soma", cell=self)
        self.soma.nseg = 1
        self.soma.diam = 4
        self.soma.L = 10
        self.soma.Ra = 120
        self.soma.cm = 1 
        self.soma.insert("na18a")
        self.soma.gbar_na18a = 0.1
        self.soma.ena = 60

def initialize():
    h.finitialize()
    h.run()

def voltage_clamp_activation(voltage):
    cell = Section()
    h.dt = 0.025
    h.celsius = 32
    h.tstop = 210
    h.v_init = -70 
    ina = h.Vector()
    ina.record(cell.soma(0.5)._ref_ina_na18a)
    v = h.Vector()
    v.record(cell.soma(0.5)._ref_v)
    time = h.Vector()
    time.record(h._ref_t)

    clamp = h.SEClamp(cell.soma(0.5))
    clamp.rs /= 100
    clamp.dur1 = 100
    clamp.amp1 = -70
    clamp.dur2 = 100
    clamp.amp2 = voltage
    clamp.dur3 = 10
    clamp.amp3 = -70

    initialize()
    return list(ina), v, time

def voltage_clamp_resurgent(voltage):
    cell = Section()
    h.dt = 0.025
    h.celsius = 32
    h.tstop = 250
    h.v_init = -70 
    ina = h.Vector()
    ina.record(cell.soma(0.5)._ref_ina_na18a)
    v = h.Vector()
    v.record(cell.soma(0.5)._ref_v)
    time = h.Vector()
    time.record(h._ref_t)

    clamp = h.SEClamp(cell.soma(0.5))
    clamp.rs /= 100
    clamp.dur1 = 10
    clamp.amp1 = -100
    clamp.dur2 = 20
    clamp.amp2 = 30
    clamp.dur3 = 200
    clamp.amp3 = voltage

    initialize()
    return list(ina), v, list(time)

def max_value(ina):
    if abs(min(ina)) > abs(max(ina)):
        return min(ina)
    else:
        return max(ina)

def extract_slice(to_slice, start, stop):
    a = to_slice[:stop*40]
    middle_part = a[start*40:]
    return middle_part

def find_distance(max_values,optimal):
    distance = []
    for x in max_values:
        if x < optimal:
            distance.append(optimal-x)
        else:
            distance.append(x-optimal)
    return np.array(distance)

sample_current = pd.read_csv('res_trace.csv', sep = ";") #/mnt/c/Users/Computer/Downloads/
sample_current.replace(',', '.', regex=True, inplace=True)
x = sample_current['time'].to_numpy()
x = x.astype(float)
x_0 = [n-min(x) for n in x]
x_norm = [n/max(x_0) for n in x_0]
voltages_current = np.array([n*217 for n in x_norm]) + 30

y = sample_current["I"].to_numpy()
y = y.astype(float)
y = [-y/max(y)]
sample_current = np.vstack([voltages_current,y])
fig_curr = go.Figure(go.Scatter(x=sample_current[0],y=sample_current[1]))

sample_activation_iv = pd.read_csv('activation_digitizer.csv', sep = ";") #/mnt/c/Users/Computer/Downloads/
sample_activation_iv.replace(',', '.', regex=True, inplace=True)
x = sample_activation_iv['Mv'].to_numpy()
voltages_activation = x.astype(float)
y = sample_activation_iv['Current'].to_numpy()
y = y.astype(float)
sample_activation_iv = np.vstack([voltages_activation,y]) 
#go.Figure(go.Scatter(x=x, y=y)).show()

sample = pd.read_csv('persistent_digitizer.csv', sep = ";")  #/mnt/c/Users/Computer/Downloads/
sample.replace(',', '.', regex=True, inplace=True)
x = sample['Voltage'].to_numpy()
voltages_persistent = x.astype(float)
y = sample['I/Imax'].to_numpy()
y = y.astype(float)
sample_persistent_data = np.vstack([voltages_persistent,y])
#go.Figure(go.Scatter(x=sample_persistent_data[0], y=sample_persistent_data[1])).show()

sample_resurgent_iv = pd.read_csv('resurgent_digitizer.csv', sep = ";") #/mnt/c/Users/Computer/Downloads/
sample_resurgent_iv.replace(',', '.', regex=True, inplace=True)
x = sample_resurgent_iv['Voltage'].to_numpy()
voltages_resurgent = x.astype(float)
y = sample_resurgent_iv["I/It(%)"].to_numpy()
y = y.astype(float)
sample_resurgent_iv = np.vstack([voltages_resurgent,y]) 
#go.Figure(go.Scatter(x = x, y = y)).show()



def activation_protocol(par,val):

    parval = np.vstack((par,val))
    parval = np.transpose(parval)
    for x in parval:
        attr = x[0]+'_na18a'
        setattr(h, attr, x[1].astype(float))

    max_i_list = []
    voltages = voltages_activation
    
    for x in voltages:
        ina, v, time = voltage_clamp_activation(x)
        ina_only_stim = extract_slice(ina, 100, 140)
        max_i_list.append(max_value(ina_only_stim))
    
    y = []
    for i in max_i_list:
        try:
            y.append(i/min(max_i_list))
        except ZeroDivisionError:
            y.append(0)

    activation_iv = np.vstack([voltages,y])
    #go.Figure(go.Scatter(x = activation_iv[0], y = activation_iv[1])).show()
    
    return activation_iv

def persistent_protocol(par,val):

    parval = np.vstack((par,val))
    parval = np.transpose(parval)
    for x in parval:
        attr = x[0]+'_na18a'
        setattr(h, attr, x[1].astype(float))

    max_i_list = []
    max_i_persistent = []
    voltages = voltages_persistent
    
    for x in voltages:
        ina, v, time = voltage_clamp_activation(x)
        ina_only_stim = extract_slice(ina, 100, 200)
        ina_persistent = extract_slice(ina, 180, 200)
        max_i_list.append(max_value(ina_only_stim))
        max_i_persistent.append(max_value(ina_persistent))
    
    y = []
    for i in max_i_persistent:
        try:
            y.append((i/min(max_i_list))*100)
        except ZeroDivisionError:
            y.append(0)

    persistent_sim = np.vstack([voltages,y]) 
    #go.Figure(go.Scatter(x=persistent_sim[0], y=persistent_sim[1])).show()
    
    return persistent_sim

def resurgent_protocol(par,val):
    
    parval = np.vstack((par,val))
    parval = np.transpose(parval)
    for x in parval:
        attr = x[0]+'_na18a'
        setattr(h, attr, x[1].astype(float))

    max_i_list = []
    max_i_transient = []
    voltages = voltages_resurgent
    
    for x in voltages:
        ina, v, time = voltage_clamp_resurgent(x)
        ina_transient = extract_slice(ina, 10,30)
        max_i_transient.append(max_value(ina_transient))
        ina_only_stim = extract_slice(ina, 30, 230)
        max_i_list.append(max_value(ina_only_stim))
        if x == -20.0:
            y = extract_slice(ina,30,233)
            y = [-n/min(y) for n in y]
            #go.Figure(go.Scatter(x=time, y=y)).update_layout(title="before").show()
            ids = [int(idx*40) for idx in sample_current[0]]
            #timestamps = [time[idx] for idx in ids]
            curr_value = [ina[idx] for idx in ids]
            y = [-n/min(curr_value) for n in curr_value]      
            current_20 = np.vstack([list(extract_slice(time,20,233)),y])
            #go.Figure(go.Scatter(x=timestamps, y=y)).update_layout(title="after").show()
            
    y = []
    for i in max_i_list:
        try:
            y.append(i/min(max_i_transient)*100)
        except ZeroDivisionError:
            y.append(0)
    
    resurgent_iv_sim = np.vstack([voltages,y])
    #go.Figure(go.Scatter(x=resurgent_iv_sim[0], y = resurgent_iv_sim[1])).show()
    
    return resurgent_iv_sim#, current_20

def get_params():
    mech_type = h.MechanismType(0)
    mech_type.select('na18a')
    code = mech_type.code()
    driver = dsl.NmodlDriver()
    modast = driver.parse_string(code)
    lookup_visitor = dsl.visitor.AstLookupVisitor()
    param_block = lookup_visitor.lookup(modast, dsl.ast.AstNodeType.PARAM_ASSIGN)

    params = {}
    for param in param_block:
        name = param.name.value.value
        value = None
        if param.value is not None:
            value = param.value.value
        params[name] = value
    
    not_to_optimize = ['v','ena','celsius','gbar','I1O1b1','I1O1k1','I1O1v1','I2I1b1','I2I1k1','I2I1v1','I1I2b2','I1I2v2','I1I2k2']
    for par in not_to_optimize:
        del params[par]

    return params

params = get_params()
parameters = np.array(list(params.keys()))

lower_bounds = []
upper_bounds = []

for key in params:
    a = float(params[key])
    if 'b' in key:
        lower_bounds.append(round(a/20,5))
        upper_bounds.append(a*10)
    else:
        lower_bounds.append(a-30)
        upper_bounds.append(a+30)

def distance_from_optimal(coordinates,optimal_current,optimal_voltage,currents=False):
    if currents == True:
        time_to_max = []
        time_to_half = []
        for x_y_pair in coordinates:
            # prendi solo la parte interessante da quando inizi a stimolare
            #slice_x = extract_slice(x_y_pair[0],12,100)
            slice_y = extract_slice(x_y_pair[1],12,100)
            #slice_for_max = np.vstack((slice_x,slice_y))
            #go.Figure(go.Scatter(x = slice_for_max[0], y = slice_for_max[1])).show()
            # se ha un massimo, allora attacca
            #print("value", slice_y[argrelmin(slice_y)], "time", slice_x[argrelmin(slice_y)])
            if np.size(argrelmin(slice_y)) == 1:
                max_value_idx = np.argmin(slice_y)
                append_to_max = x_y_pair[0][np.where(x_y_pair[1] == slice_y[max_value_idx])].tolist()
                time_to_max.append(append_to_max[0])
                # dato che ha un massimo controlla anche il dimezzamento
                idx = np.where(x_y_pair[1] == slice_y[max_value_idx])
                idx = idx[0][0] // 40 
                y_for_half = extract_slice(x_y_pair[1], idx,210)
                #x_for_half = extract_slice(x_y_pair[0], idx,210)
                #go.Figure(go.Scatter(x = x_for_half, y = y_for_half)).update_layout(title="half").show()
                half_value = slice_y[max_value_idx]/2
                half_value_idx = np.argmin(np.absolute(y_for_half - half_value))
                append_to_half = x_y_pair[0][np.where(x_y_pair[1] == y_for_half[half_value_idx])].tolist()
                time_to_half.append(append_to_half[0])
            # se non ha un massimo allora dai come risultato più di 70, cio+ il val massimo (25-100)
            else:
                time_to_max.append(np.random.uniform(100,120))
                # in più di sicuro non dimezza, quindi dai un valore anche qui maggiore
                time_to_half.append(np.random.uniform(210,230))
            
        max_ = find_distance(time_to_max,optimal_current)
        half = find_distance(time_to_half,optimal_voltage)
        return max_, half
            
    else: 
        max_coordinates = []
        for x_y_pair in coordinates:
            index = np.argmax(x_y_pair[1])
            max_coordinates.append([x_y_pair[0][index], x_y_pair[1][index]])
        max_coordinates = np.array(max_coordinates).T
        voltages = find_distance(max_coordinates[0],optimal_voltage)
        currents = find_distance(max_coordinates[1],optimal_current)

        return voltages, currents

def curve_similarity(curve1,curve2):
    a = np.absolute(curve1[1] - curve2[1])
    s = np.sum(a)
    return s

fitness_obj = []
fitness_con = []

class Myprob(Problem):
    
    def _evaluate(self, designs, out, *args, **kwargs):
        results_act = []
        results_per = []
        results_res = []
        currents = []
        for design in designs:
            results_act.append(activation_protocol(parameters,design))
            results_per.append(persistent_protocol(parameters,design))
            sim = resurgent_protocol(parameters,design)
            results_res.append(sim)
            #currents.append(curr)

        sim_act = np.array([similaritymeasures.area_between_two_curves(sample_activation_iv, results_act[n]) for n in range(len(results_act))])
        sim_per = np.array([similaritymeasures.area_between_two_curves(sample_persistent_data, results_per[n]) for n in range(len(results_per))])
        sim_res = np.array([similaritymeasures.area_between_two_curves(sample_resurgent_iv, results_res[n]) for n in range(len(results_res))])

#        sim_act = np.array([curve_similarity(sample_activation_iv, results_act[n]) for n in range(len(results_act))])
#        sim_per = np.array([curve_similarity(sample_persistent_data, results_per[n]) for n in range(len(results_per))])
#        sim_res = np.array([curve_similarity(sample_resurgent_iv, results_res[n]) for n in range(len(results_res))])
        #sim_cur = np.array([curve_similarity(sample_current, currents[n]) for n in range(len(results_res))])

	#first is distance from voltage, second from current
        g1,g3 = distance_from_optimal(results_per, 8.90,-10)
        g2,g4 = distance_from_optimal(results_res, 5.84,-20)
        g5,g6 = distance_from_optimal(results_res, 2.67, 10)
        g7,g8 = distance_from_optimal(results_res, 4.44, -5)

        # 55, 120 perchè 25, 90 + i 30 del protocollo prima che parta lo stimolo
        #g5,g6 = distance_from_optimal(currents, 55,120,currents=True)

        fitness_obj.append(sim_act)
        fitness_obj.append(sim_per)
        fitness_obj.append(sim_res)

        fitness_con.append(g1)
        fitness_con.append(g2)
        fitness_con.append(g3)
        fitness_con.append(g4)
        out["G"] = np.column_stack([g1,g2,g3,g4,g5,g6,g7,g8])
        out["F"] = np.column_stack([sim_act, sim_per, sim_res])

problem = Myprob(n_var=48, n_obj=3,n_ieq_constr=8, xl=lower_bounds, xu=upper_bounds)
ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
algorithm = NSGA3(pop_size=500,ref_dirs=ref_dirs)
terminator = ('n_gen', 30)
res = minimize(problem = problem,
               algorithm = algorithm,
               termination = terminator,
               save_history=True)

np.savetxt("500_30_3obj_8const_1_sim.txt",res.pop.get("X"))
np.savetxt("500_30_fitobj_1_sim.txt",fitness_obj)
np.savetxt("500_30_fitcon_1_sim.txt",fitness_con)


class Myprob(Problem):
    
    def _evaluate(self, designs, out, *args, **kwargs):
        results = []
        for design in designs:
            results.append(protocol(parameters,design))

        sim = np.array([curve_similarity(sample_iv_rel, results[n]) for n in range(len(results))])

        g1,g3 = distance_from_optimal(results, 10,-10)
        
        out["G"] = np.column_stack([g1,g2])
        out["F"] = np.column_stack([sim])

problem = Myprob(n_var=48, xl=lower_bounds, xu=upper_bounds)
ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
algorithm = NSGA3(pop_size=800,ref_dirs=ref_dirs)
terminator = ('n_gen', 50)
res = minimize(problem = problem,
               algorithm = algorithm,
               termination = terminator,
               save_history=True)