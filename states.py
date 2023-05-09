from neuron import h, gui
from dbbs_models import quick_test, quick_plot, Nav18PurkinjeCell_ma
from patch import p
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import numpy as np
import pandas as pd

cell = PurkinjeCell()

C1 = p.record(cell.soma[0](0.5).glia__local__Nav1_8__balbi._ref_C1)
C2 = p.record(cell.soma[0](0.5).glia__local__Nav1_8__balbi._ref_C2)
O1 = p.record(cell.soma[0](0.5).glia__local__Nav1_8__balbi._ref_O1)
O2 = p.record(cell.soma[0](0.5).glia__local__Nav1_8__balbi._ref_O2)
I1 = p.record(cell.soma[0](0.5).glia__local__Nav1_8__balbi._ref_I1)
I2 = p.record(cell.soma[0](0.5).glia__local__Nav1_8__balbi._ref_I2)
OB = p.record(cell.soma[0](0.5).glia__local__Nav1_8__balbi._ref_OB)
vm = p.record(cell.soma[0](0.5)._ref_v)
time = p.time

quick_test(cell,duration=50)
# quick_plot(PurkinjeCell(), duration=1000)

stacked = np.vstack((np.array(time), np.array(vm), np.array(C1), np.array(C2), 
np.array(O1), np.array(O2), np.array(I1), np.array(I2), np.array(OB)))

df = pd.DataFrame(stacked).T
# ap = df[750*40:850*40]

np.savetxt("test.txt",df)

#cell = Nav18PurkinjeCell_ma()

data = np.genfromtxt("test.txt", delimiter = ",")
# take ap from 790 to 820
df = pd.DataFrame(data).T
ap = df[790*40:820*40]

ap = ap.set_axis(["time","mv","C1","C2","O1","O2","I1","I2","ina"], axis=1, inplace=False)

fig = make_subplots(rows=2, cols=2, subplot_titles=(
    "Action potential","Closed","Open","Inactive"))
#ap
fig.add_trace(go.Scatter(x = ap["time"], y = ap["mv"], name = "mv"), row=1, col=1)
#closed
fig.add_trace(go.Scatter(x = ap["time"], y = ap["C1"], name = "C1"), row=1, col=2)
fig.add_trace(go.Scatter(x = ap["time"], y = ap["C2"], name = "C2"), row=1,col=2)
#open
fig.add_trace(go.Scatter(x = ap["time"], y = ap["O1"], name = "O1"), row=2,col=1)
fig.add_trace(go.Scatter(x = ap["time"], y = ap["O2"], name = "O2"), row=2,col=1)
#inactive
fig.add_trace(go.Scatter(x = ap["time"], y = ap["I1"], name = "I1"), row=2,col=2)
fig.add_trace(go.Scatter(x = ap["time"], y = ap["I2"], name = "I2"), row=2,col=2)

fig.show()

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x = ap["time"], y = ap["I1"], name = "I1"))
fig1.add_trace(go.Scatter(x = ap["time"], y = ap["C1"], name = "C1"))
fig1.add_trace(go.Scatter(x = ap["time"], y = ap["C2"], name = "C2"))
fig1.add_trace(go.Scatter(x = ap["time"], y = ap["O1"], name = "O1"))
#fig1.show()



























'''
C1 = p.record(cell.soma[0](0.5).glia__local__Nav1_8__balbi._ref_C1)
C2 = p.record(cell.soma[0](0.5).glia__local__Nav1_8__balbi._ref_C2)
O1 = p.record(cell.soma[0](0.5).glia__local__Nav1_8__balbi._ref_O1)
O2 = p.record(cell.soma[0](0.5).glia__local__Nav1_8__balbi._ref_O2)
I1 = p.record(cell.soma[0](0.5).glia__local__Nav1_8__balbi._ref_I1)
I2 = p.record(cell.soma[0](0.5).glia__local__Nav1_8__balbi._ref_I2)
time = h.Vector()
time.record(h._ref_t)
mv = p.record(cell.soma[0](0.5)._ref_v)
ina = p.record(cell.soma[0](0.5).glia__local__Nav1_8__balbi._ref_ina)

quick_test(cell,duration=1000,temperature= 32)
_save = np.vstack((np.array(time),np.array(mv),np.array(C1),np.array(C2),np.array(O1),np.array(O2),np.array(I1),np.array(I2),np.array(ina)))

np.savetxt('test.txt', _save, delimiter = ",")
'''




















