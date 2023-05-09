from dbbs_models import quick_test, quick_plot, Nav18PurkinjeCell_ma, PurkinjeCell, Ball
from patch import p
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import neuron
from neuron import h

def record_section(self):
    self.Vm = self.record()
    return self.Vm

def quick_test(*models, duration=300, temperature=32, v_init=-65):
    from patch import p

    p.time
    p.celsius = temperature
    p.v_init = v_init
    for model in models:
        record_section(model)
    p.finitialize(v_init)
    p.continuerun(duration)

    return list(p.time), list(model.Vm)

def quick_plot(*args, **kwargs):
    time, model_results = quick_test(*args, **kwargs)

    go.Figure(
            go.Scatter(x=time, y=model_results)
    ).show()

protocol = []

def single_clamp(voltage):
    s = p.Section()
    s.L = 20
    s.diam = 20
    s.insert("na18a")    
    clamp = s.vclamp(delay=100, duration=10, after=10, voltage=voltage)
    clamp.rs /= 100
    ina = p.record(s(0.5).na18a._ref_ina)
    vm = p.record(s(0.5)._ref_v)
    quick_test(s, duration=120, temperature=32)
    protocol.append(list(vm))
    return list(ina) 

def find_max_i(list_of_currents):
    max_negative = min(list_of_currents)
    max_positive = max(list_of_currents)
    if abs(max_negative) > abs(max_positive):
        return max_negative
    else:
        return max_positive 

def voltage_clamp_protocol(voltage_list):
    ina_list_every_voltage = [single_clamp(voltage) for voltage in voltage_list]
    max_i_list = [find_max_i(ina) for ina in ina_list_every_voltage]
    normalized_i_list = [i/min(max_i_list) for i in max_i_list]

    return voltage_list, normalized_i_list, max_i_list

voltage_list, normalized_i_list, max_i_list = voltage_clamp_protocol(np.linspace(-100,110,43))

fig = go.Figure()
fig.add_trace(go.Scatter(x = voltage_list, y = normalized_i_list))
fig.update_layout(xaxis_title="Voltage (mV)", yaxis_title="I/Imax", autosize=False, width=600, height=500)
fig.update_xaxes(dtick = 10)
fig['layout']['yaxis']['autorange'] = "reversed"
#fig.show()

fig1 = go.Figure()
for x in protocol:
    fig1.add_trace(go.Scatter(x = list(p.time), y = x))
fig1.show()

def single_clamp(cell,delay,duration,hold,voltage,temperature):
    clamp = cell.vclamp(delay=delay, duration=duration, after=10, voltage=voltage, holding=hold)
    i_nav18 = p.record(cell(0.5).na18a._ref_ina)
    vm = p.record(cell(0.5)._ref_v)
    quick_test(cell, duration=duration+delay+10, temperature = temperature)
    return i_nav18, vm

def find_max_current(list_of_currents):
    max_negative = min(list_of_currents)
    max_positive = max(list_of_currents)
    if abs(max_negative) > abs(max_positive):
        return max_negative
    else:
        return max_positive 

def I_V_relationship(delay,duration,hold,voltage,temperature):
    stacked = []
    vms = []
    for holding_potential in hold:
        s = p.Section()
        s.L = 20
        s.diam = 20
        s.insert("na18a")
        i_nav18, vm = single_clamp(s,delay,duration,holding_potential,voltage,temperature)
        stacked.append(list(i_nav18))
        vms.append(list(vm))

    true_hold = [v[1] for v in vms]
    i_max = [find_max_current(i) for i in stacked]
    time_ = list(p.time)
    normalized_i = [i/min(i_max) for i in i_max]

    return normalized_i, true_hold, stacked, vms

normalized_i, true_hold, stacked, vms = I_V_relationship(20,0,range(-70,100,5),0,temperature = 32)
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x = true_hold, y = normalized_i))
fig4.update_layout(xaxis_title="Voltage (mV)", yaxis_title="I/Imax", autosize=False,
    width=700, height=500)
fig4.update_xaxes(dtick = 10)
fig4['layout']['yaxis']['autorange'] = "reversed"
fig4.show()

go.Figure(go.Scatter(x = list(p.time), y = vms[1])).show()









