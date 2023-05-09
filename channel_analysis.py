from dbbs_models import quick_test, quick_plot, Nav18PurkinjeCell_ma, PurkinjeCell, Ball
from patch import p
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import neuron
from neuron import h

#10ms: len(current_trace)=400, 20:800 -> 1ms:40 

#qual'è il canale migliore per i_v rel? pare ma, mima anche meglio quello del topo
#la forma del pda non cambia, nè la sua frequenza
#aumenta la durata: 4 contro 2,49

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

def single_current_activation(voltage,hold,cell,delay,duration):

    Section.clamp = Section().soma[0].vclamp(delay=delay, duration=duration, after=0, voltage=voltage, holding=hold)
    Section.clamp.rs /= 100

    clamp = cell.soma[0].vclamp(delay=delay, duration=duration, after=0, voltage=voltage, holding=hold)
    i_nav18 = p.record(cell.soma[0](0.5).glia__local__Nav1_8__balbi._ref_ina)
    i_tot = p.record(clamp._ref_i)
    vm = p.record(cell.soma[0](0.5)._ref_v)
    quick_test(cell, duration=duration+delay)
    return i_nav18, vm

def single_current_inactivation(voltage,hold,cell,delay,duration, temperature):
    if hasattr(cell,"clamp"):
        cell.clamp.amp1 = hold
    else:
        cell.clamp = cell.soma[0].vclamp(delay=delay, duration=duration, after=0, voltage=voltage, holding=hold)
        cell.clamp.rs /= 100

    clamp = cell.soma[0].vclamp(delay=delay, duration=duration, after=0, voltage=voltage, holding=hold)
    i_nav18 = p.record(cell.soma[0](0.5).glia__local__Nav1_8__balbi._ref_ina)
    i_tot = p.record(clamp._ref_i)
    vm = p.record(cell.soma[0](0.5)._ref_v)
    quick_test(cell, duration=duration+delay, temperature = temperature)
    return i_nav18, vm

def find_max_current(list_of_currents):
    max_negative = min(list_of_currents)
    max_positive = max(list_of_currents)
    if abs(max_negative) > abs(max_positive):
        return max_negative
    else:
        return max_positive 

def activation(voltages,hold,cell,delay,duration):
    stacked = []
    vms = []
    for voltage in voltages:
        i_nav18, vm = single_current_activation(voltage,hold,cell,delay,duration)
        stacked.append(list(i_nav18))
        vms.append(list(vm))

    true_vms = [v[delay*40+2] for v in vms]
    n = 0
    gNa = []
    i_max_list = []

    for i_nav18 in stacked:
        i_max_list.append(min(i_nav18[(delay*40):]))
        gNa.append(i_max_list[n] / (true_vms[n]-60))
        n += 1

    normalized_g = [g/max(gNa) for g in gNa]
    transient_i_max = min(i_max_list)

    return normalized_g, true_vms, transient_i_max, i_max_list

def inactivation(voltages,hold,cell,delay,duration, temperature):
    stacked = []
    vms = []
    for voltage in hold:
        i_nav18, vm = single_current_inactivation(voltages,voltage,cell,delay,duration, temperature)
        stacked.append(list(i_nav18))
        vms.append(list(vm))

    true_hold = [v[delay*40] for v in vms]
    i_max = [min(i[delay*40:]) for i in stacked]
    time_ = list(p.time)
    normalized_i = [i/min(i_max) for i in i_max]
    i_max_inac = min(i_max)

    return normalized_i, true_hold, stacked, i_max_inac

def persistent_current(voltages,hold,cell,delay,duration,transient_i_max):
    stacked = []
    vms = []
    for voltage in voltages:
        i_nav18, vm = single_current_activation(voltage,hold,cell,delay,duration)
        stacked.append(list(i_nav18))
        vms.append(list(vm))

    time_for_persistent = (delay+(duration-7))*40
    true_vms = [v[time_for_persistent] for v in vms]
    n = 0
    i_max = []

    for i_nav18 in stacked:
        i_max.append(min(i_nav18[(time_for_persistent):]))

    normalized_i = [(-i/transient_i_max)*100 for i in i_max]

    return normalized_i, true_vms

def I_V_relationship(voltages,hold,cell,delay,duration, temperature):
    stacked = []
    vms = []
    for voltage in hold:
        i_nav18, vm = single_current_inactivation(voltages,voltage,cell,delay,duration, temperature)
        stacked.append(list(i_nav18))
        vms.append(list(vm))

    true_hold = [v[1] for v in vms]
    i_max = [find_max_current(i) for i in stacked]
    time_ = list(p.time)
    normalized_i = [i/min(i_max) for i in i_max]

    return normalized_i, true_hold, stacked


'''
normalized_i, true_hold, stacked = I_V_relationship(0,range(-70,100,10),cell,20,0, temperature = 32)
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x = true_hold, y = normalized_i))
fig4.update_layout(xaxis_title="Voltage (mV)", yaxis_title="I/Imax", autosize=False,
    width=500, height=500)
fig4.update_xaxes(dtick = 10)
fig4['layout']['yaxis']['autorange'] = "reversed"
#fig4.show()

normalized_g, voltages, transient_i_max = activation(range(-90,60,5),-70,Ball(),5,5)
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x = voltages, y = normalized_g))
fig1.update_layout(xaxis_title="Voltage (mV)", yaxis_title="G/Gmax", autosize=False,
    width=500, height=500)
fig1.update_xaxes(dtick = 10)
fig1.show()

normalized_i, true_hold, stacked, i_max_inac = inactivation(0,range(-70,70,5),Ball(),20,5)
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x = true_hold, y = normalized_i))
fig2.update_layout(xaxis_title="Voltage (mV)", yaxis_title="I/Imax", autosize=False,
    width=500, height=500)
fig2.update_xaxes(dtick = 10)
fig2.show()

normalized_i_persistent, true_vms = persistent_current(range(-70,50,5),-70,Ball(),0,100,transient_i_max)
to_reverse = np.array(normalized_i_persistent)
reverse_i_persistent = -to_reverse  
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x = true_vms, y = reverse_i_persistent))
fig3.update_layout(xaxis_title="Voltage (mV)", yaxis_title="Persistent Current (%)", autosize=False,
    width=500, height=500)
fig3['layout']['yaxis']['autorange'] = "reversed"
fig3.update_xaxes(dtick = 10)

fig3.show()
'''

normalized_g, voltages, transient_i_max = activation(range(-90,60,5),-70,Section(),5,5)
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x = voltages, y = normalized_g))
fig1.update_layout(xaxis_title="Voltage (mV)", yaxis_title="G/Gmax", autosize=False,
    width=500, height=500)
fig1.update_xaxes(dtick = 10)
fig1.show()


normalized_i_persistent, true_vms = persistent_current(range(-70,50,5),-70,Section(),0,100,transient_i_max)
to_reverse = np.array(normalized_i_persistent)
reverse_i_persistent = -to_reverse  
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x = true_vms, y = reverse_i_persistent))
fig3.update_layout(xaxis_title="Voltage (mV)", yaxis_title="Persistent Current (%)", autosize=False,
    width=500, height=500)
fig3['layout']['yaxis']['autorange'] = "reversed"
fig3.update_xaxes(dtick = 10)

fig3.show()

