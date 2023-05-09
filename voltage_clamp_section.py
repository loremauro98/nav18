import nmodl
import similaritymeasures
from nmodl import dsl
from neuron import h,gui
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from scipy.signal import argrelmax, argrelmin


class Section():
    def __init__(self):
        self.soma = h.Section("name=soma")
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
    clamp.dur1 = 100
    clamp.amp1 = -70
    clamp.dur2 = 100
    clamp.amp2 = voltage
    clamp.dur3 = 50
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
    return list(ina), v, time

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

def resurgent_protocol(par,val,currents=False,iv=False):
    
    parval = np.vstack((par,val))
    parval = np.transpose(parval)
    for x in parval:
        attr = x[0]+'_na18a'
        setattr(h, attr, x[1].astype(float))

    max_i_list = []
    max_i_transient = []
    voltages = np.linspace(-45,15,7)#voltages_resurgent#
    #i_traces = go.Figure()
    i_transient = go.Figure()
    i_traces=go.Figure()
    for x in voltages:
        ina, v, time = voltage_clamp_resurgent(x)
        ina_transient = extract_slice(ina, 10,30)
        i_transient.add_trace(go.Scatter(x=time,y=ina_transient))
        max_i_transient.append(max_value(ina_transient))
        ina_only_stim = extract_slice(ina, 30, 200)
        max_i_list.append(max_value(ina_only_stim))
        #i_traces.append(ina)
        i_traces.add_trace(go.Scatter(x=time, y=ina,name=f"{x}"))
        if x == -20.09575595:
            y = extract_slice(ina,20,233)
            y = [-n/min(y) for n in y]
            #go.Figure(go.Scatter(x=list(time), y=y)).show()
            current_20 = np.vstack([list(extract_slice(list(time),20,233)),y])
    
    #i_transient.show()
    y = []
    for i in max_i_list:
        try:
            y.append(i/min(max_i_transient)*100)
        except ZeroDivisionError:
            y.append(0)
    
    resurgent_iv_sim = np.vstack([voltages,y])
    
    if iv == True:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=resurgent_iv_sim[0], y = resurgent_iv_sim[1],name="Modified",line=dict(color="blue"),mode="lines"))
        fig.add_trace(go.Scatter(x=sample_resurgent_iv[0][2:], y = sample_resurgent_iv[1][2:],name="Real data",line=dict(color="black"),mode="lines"))
        fig.update_layout(title={"text" : "INaR","x":0.5,'xanchor': 'center','yanchor': 'top'},plot_bgcolor="white",showlegend=False)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black',ticks="outside", tickwidth=2, tickcolor='black', ticklen=10,
                title="% of the INaT", title_font=dict(color="black"))
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black',ticks="outside", tickwidth=2, tickcolor='black', ticklen=10,
                title="Membrane voltage [mV]", title_font=dict(color="black"))
        #go.Figure(go.Scatter(x=resurgent_iv_sim[0], y = resurgent_iv_sim[1])).show()
        fig.show()
    if currents == True:
        i_traces.update_layout(autosize=False, width=500, height=500, plot_bgcolor="white",showlegend=False)
        i_traces.update_xaxes(visible=False)
        i_traces['data'][0]['line']['color']="black"
        i_traces['data'][0]['line']['color']="black"
        i_traces['data'][0]['line']['color']="black"
        i_traces['data'][0]['line']['color']="#00ff00"
        i_traces['data'][0]['line']['color']="#00ff00"
        i_traces['data'][0]['line']['color']="#00ff00"
        i_traces['data'][0]['line']['color']="#00ff00"
        i_traces.update_yaxes(visible=False)
        i_traces.show()
    
    return i_traces#resurgent_iv_sim, current_20

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

X = np.genfromtxt("sim_results/good_curves_formulapy1.txt")
good = [1,18,19,50,77,140]
len(X)
X1 = np.genfromtxt("sim_results/good_curves_formulapy2.txt")
good1 = [7,9,13,29,49,59,61,63,65,67,68,69,73,75,76,77,79,80,81,82,
         96,97,101,105,106,110,117,118,126,131,132,134,139,144,168,
         177,178,215,217,218,222,223,227,228,230,232,233,234,235,237,
         239,240,241,242,244,246,247,248,250,253,255,257,259,260,261,
         262,263,267,268,270,271,273,276,280,283,284,285,288,289,290,
         297,296,300,301,302,303,305,318,324,330,336,341,342,344,346,
         348,349,351,352,354,355,358,361,362,363,364,366,369,370,376,
         377,382,385,386,387,388,392,393,395,398,399,400,404,405,408,409,410,413,416]

notgood1 = [14,19,20,24,27,31,37,38,39,42,44,45,49,50,51,53,54,55,56,57,59,60,66,69,71,
            72,73,74,75,78,79,80,82,84,88,90,91,92,94,96,97,98,99,101,102,108,112,116,114,117,118]

realgood1 = [n for n in good1 if not good1.index(n) in notgood1]
goddiss = [X1[realgood1[4]],X1[realgood1[66]],X1[realgood1[69]]]


#resurgent_protocol(parameters,X1[0],currents=True)

possibili = np.genfromtxt("sim_results/parametri_possibili.txt")
veryposs = [2,9,11,21,22,24,26,27,34,35,36]
davvero = [possibili[x] for x in veryposs] + goddiss

resurgent_protocol(parameters,davvero[0],currents=True)