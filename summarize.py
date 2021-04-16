import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit, fit_report
from scipy.integrate import odeint
from scipy import stats
import pandas as pd
import datetime
import os
import json
import csv
import random

import uncertainties.unumpy as unp
import uncertainties as unc

def getStartDate(date,incidence):

    if incidence > 1:
        return date.year, date.month, date.day-5
    else:
        return date.year, date.month, date.day

def ODEfun(y,t,params):

    """
    Your system of differential equations
    """
    Im = y[0]
    Is = y[1]
    Ic = y[2]
    R = y[3]
    

    try:

        r = params['r']
        p = params['p']
        N = params['N']
        R0 = params['R0']
        k = params['k']
        m = params['m']
        #n = paras['n'].value
        ps = 0.1
    except:
        print('error')

    S = (1-R/N)
    M = np.exp(-m*t)
    # the model equations
    f0 = (1-ps)*R0*(k+(1-k)*M)*S*(Im + p*Is) - Im
    f1 = ps*R0*(k+(1-k)*M)*S*(Im + p*Is) - Is
    f2 = r*(1-ps)*R0*(k+(1-k)*M)*S*(Im + p*Is) + ps*R0*(k+(1-k)*M)*S*(Im + p*Is) #r*Im/10 + ps*R0*(k+(1-k)*M)*S*(Im + p*Is)#
    f3 = R0*(k+(1-k)*M)*S*(Im + p*Is)

    return [f0, f1, f2, f3]

def getReff(t,R,params):

    r = params['r']
    p = params['p']
    N = params['N']
    R0 = params['R0']
    k = params['k']
    m = params['m']

    S = (1-R/N)
    M = np.exp(-m*t)

    return R0*(k+(1-k)*M)*S

def get_parameter(file_name,param_name):

    with open(file_name) as file:
        param_data = file.read()

    ind = param_data.find('[[Variables]]')
    #print(ind)
    ind1 = param_data.find(param_name+':',ind)
    #print(ind1)
    ind2 = param_data.find('(', ind1)
    #print(ind2)
    value_s = param_data[ind1+len(param_name+':'):ind2]
    value_s = value_s.replace(' ','')
    
    return unc.ufloat_fromstr(value_s)

D = 10
predict = 10
G = 0

param_list = ['x10','x20','x30','x40','R0','p','r','N','k','m']

province = 'NewBrunswick'
files = os.listdir('Sensitivity/'+province)

data = pd.read_csv(province+'Data.csv',parse_dates=True)
data['date'] = pd.to_datetime(data['date'], dayfirst=True)
y, m, d = getStartDate(data['date'][0],data['numtotal'][0])
start_date1 = datetime.datetime(y,m,d)
today = str(datetime.datetime.today().date())
start_date = datetime.datetime(2020,3,15)
if province in ['Ontario','BritishColumbia']:
    start_date = datetime.datetime(2020,3,15)
if province in ['Canada']:
    start_date = datetime.datetime(2020,3,15)
    
ind = data.index[data['date']==start_date]
ind = ind[0]
days_since = (data['date'] - start_date1).dt.days
days_since = days_since/D+G/D
data['numtotal'] = data['numtotal']
x = days_since.values
y = data['numtotal'].values
z = data['numtoday'].values

t_measured = x[:]
x2_measured = y[:]
z_measured = z[:]
x = x[ind:len(x)-predict]
y = y[ind:len(y)-predict]
z = z[ind:len(z)-predict]

t = np.linspace(x[0], 36.5+x[0],366)
mild = np.zeros((len(files),len(t)))
severe = np.zeros((len(files),len(t)))
known = np.zeros((len(files),len(t)))
total = np.zeros((len(files),len(t)))
newcases = np.zeros((len(files),len(t)))
Rt = np.zeros((len(files),len(t)))
sim_num = 0

params_calc = {}

for p in param_list:
    params_calc[p] = []

for file in files:

    params={}
    file_bad = False
    for p in param_list:
        try:
            params[p] = get_parameter('Sensitivity/'+province+"/"+file,p).nominal_value
            params_calc[p].append(get_parameter('Sensitivity/'+province+"/"+file,p).nominal_value)
        except:
            file_bad = True
            break
    if file_bad:
        mild = np.delete(mild, -1, axis=0)
        severe = np.delete(severe, -1, axis=0)
        known = np.delete(known, -1, axis=0)
        total = np.delete(total, -1, axis=0)
        newcases = np.delete(newcases, -1, axis=0)
        Rt = np.delete(Rt, -1, axis=0)
        continue

    y_calc = odeint(ODEfun, [params['x10'],params['x20'],params['x30'],params['x10']+params['x20']+params['x40']], t, args=(params,),rtol=1e-11,atol=1e-11)

    dx = np.zeros((len(t),4))
    px_i = 0
    for px_t in t:
        Rt[sim_num,px_i] = getReff(px_t,y_calc[px_i,3],params)
        dxt = ODEfun(y_calc[px_i,:],px_t,params)
        dx[px_i,:] = dxt
        dx[px_i,:] /= D
        px_i += 1
    
    mild[sim_num,:] = y_calc[:,0]
    severe[sim_num,:] = y_calc[:,1]
    known[sim_num,:] = y_calc[:,2]
    total[sim_num,:] = y_calc[:,3]
    newcases[sim_num,:] = dx[:,2]

    sim_num += 1

l1 = len(mild[0,:])
l2 = len(mild[:,0])
print('************')
print(sim_num)
print('************')
print(l1)
print('************')
print(l2)

real = []
for i in range(1,l2+1):
    real.extend((i*np.ones((1,l1))).tolist())

flatreal = [item for sublist in real for item in sublist]

column_names = ['realization','mild','severe','known','C','DailyPresentations']
df = pd.DataFrame(columns = column_names)

df['realization'] = flatreal
print(len(df['realization']))
print(np.product(mild.shape))
df['mild'] = np.reshape(mild,np.product(mild.shape))
df['severe'] = np.reshape(severe,np.product(severe.shape))
df['known'] = np.reshape(known,np.product(known.shape))
df['C'] = np.reshape(total,np.product(total.shape))
df['DailyPresentations'] = np.reshape(newcases,np.product(newcases.shape))

df.to_csv('out'+province+'.csv', index=False)