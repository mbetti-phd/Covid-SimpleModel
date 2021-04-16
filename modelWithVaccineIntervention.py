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

def ODEfunVax(y,t,params):

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
        ps = 0.1
    except:
        print('error')
    if t > 9.7:
        Vx = 1-0.75/(1+np.exp(-0.433*(t-24.73)))
    else:
        Vx=1
    #if t>10 and t < 12:
    #    m=0
    #elif t>=12:
    #    m=2*m
    #    k=0.9*k
    S = (1-R/(Vx*N))
    M = k+(1-k)*np.exp(-m*t)
    #if t > 23.5:
    #    M = (1+(R0-1)*(t-23.5)/(20+(t-23.5)))
    # the model equations
    f0 = (1-ps)*R0*M*S*(Im + p*Is) - Im
    f1 = ps*R0*M*S*(Im + p*Is) - Is
    f2 = r*(1-ps)*R0*M*S*(Im + p*Is) + ps*R0*M*S*(Im + p*Is) #r*Im/10 + ps*R0*(k+(1-k)*M)*S*(Im + p*Is)#
    f3 = R0*M*S*(Im + p*Is)

    return [f0, f1, f2, f3]

def getReff(t,R,params):

    r = params['r']
    p = params['p']
    N = params['N']
    R0 = params['R0']
    k = params['k']
    m = params['m']
    #n = params['n']

    S = (1-R/N)
    M = np.exp(-m*t)

    return R0*(k+(1-k)*M)*S

def getReff_vax(t,R,params):

    r = params['r']
    p = params['p']
    N = params['N']
    R0 = params['R0']
    k = params['k']
    m = params['m']
    #n = params['n']

    if t > 9.7:
        Vx = 1-0.75/(1+np.exp(-0.433*(t-24.73)))
    else:
        Vx=1

    S = (1-R/(Vx*N))
    M = k+(1-k)*np.exp(-m*t)
    if t > 23.5:
        M = (1+(R0-1)*(t-23.5)/(20+(t-23.5)))
    #if t > 14.5:
    #    M = (1+(R0-1)*(t-14.5)/(12+(t-14.5)))
    #if t > 20.4:
    #    M = (1+(R0-1)*(t-20.4)/(0+(t-20.4)))
    # the model equations
    return R0*M*S

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
predict = 0
G = 0

param_list = ['x10','x20','x30','x40','R0','p','r','N','k','m']

province = 'Alberta'
files = os.listdir('Sensitivity12/'+province+'/2')

data = pd.read_csv(province+'Data2.csv',parse_dates=True)
data['date'] = pd.to_datetime(data['date'], dayfirst=True)
y, m, d = getStartDate(data['date'][0],data['numtotal'][0])
start_date1 = datetime.datetime(y,m,d)
today = str(datetime.datetime.today().date())
start_date = datetime.datetime(2020,9,8)
if province in ['Ontario','BritishColumbia']:
    start_date = datetime.datetime(2020,9,8)
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

t = np.linspace(x[0], 47.9,479-x[0])
mild = np.zeros((len(files),len(t)))
severe = np.zeros((len(files),len(t)))
known = np.zeros((len(files),len(t)))
total = np.zeros((len(files),len(t)))
newcases = np.zeros((len(files),len(t)))
mild_vax = np.zeros((len(files),len(t)))
severe_vax = np.zeros((len(files),len(t)))
known_vax = np.zeros((len(files),len(t)))
total_vax = np.zeros((len(files),len(t)))
newcases_vax = np.zeros((len(files),len(t)))
Rt = np.zeros((len(files),len(t)))
Rt_vax = np.zeros((len(files),len(t)))
sim_num = 0

params_calc = {}

for p in param_list:
    params_calc[p] = []

for file in files:
    params={}
    file_bad = False
    for p in param_list:
        try:
            params[p] = get_parameter('Sensitivity12/'+province+"/2/"+file,p).nominal_value
            params_calc[p].append(get_parameter('Sensitivity12/'+province+"/2/"+file,p).nominal_value)
        except:
            file_bad = True
            break
    if file_bad:
        mild = np.delete(mild, -1, axis=0)
        severe = np.delete(severe, -1, axis=0)
        known = np.delete(known, -1, axis=0)
        total = np.delete(total, -1, axis=0)
        newcases = np.delete(newcases, -1, axis=0)

        mild_vax = np.delete(mild_vax, -1, axis=0)
        severe_vax = np.delete(severe_vax, -1, axis=0)
        known_vax = np.delete(known_vax, -1, axis=0)
        total_vax = np.delete(total_vax, -1, axis=0)
        newcases_vax = np.delete(newcases_vax, -1, axis=0)
        Rt = np.delete(Rt, -1, axis=0)
        Rt_vax = np.delete(Rt_vax, -1, axis=0)
        continue

    y_calc = odeint(ODEfun, [params['x10'],params['x20'],params['x30'],params['x10']+params['x20']+params['x40']], t, args=(params,),rtol=1e-11,atol=1e-11)
    y_calc_vax = odeint(ODEfunVax, [params['x10'],params['x20'],params['x30'],params['x10']+params['x20']+params['x40']], t, args=(params,),rtol=1e-11,atol=1e-11)


    dx = np.zeros((len(t),4))
    dx_vax = np.zeros((len(t),4))
    px_i = 0
    for px_t in t:
        Rt[sim_num,px_i] = getReff(px_t,y_calc[px_i,3],params)
        Rt_vax[sim_num,px_i] = max([getReff_vax(px_t,y_calc_vax[px_i,3],params),0])
        dxt = ODEfun(y_calc[px_i,:],px_t,params)
        dxt_vax = ODEfunVax(y_calc_vax[px_i,:],px_t,params)
        dx[px_i,:] = dxt
        dx[px_i,:] /= D
        dx_vax[px_i,:] = dxt_vax
        dx_vax[px_i,:] /= D
        px_i += 1
    
    mild[sim_num,:] = y_calc[:,0]
    severe[sim_num,:] = y_calc[:,1]
    known[sim_num,:] = y_calc[:,2]
    total[sim_num,:] = y_calc[:,3]
    newcases[sim_num,:] = dx[:,2]

    mild_vax[sim_num,:] = y_calc_vax[:,0]
    severe_vax[sim_num,:] = y_calc_vax[:,1]
    known_vax[sim_num,:] = y_calc_vax[:,2]
    total_vax[sim_num,:] = y_calc_vax[:,3]
    newcases_vax[sim_num,:] = dx_vax[:,2]

    sim_num += 1

param_mean = {}
param_sem = {}
param_std = {}

for p in param_list:
    param_mean[p] = np.mean(params_calc[p])
    param_sem[p] = stats.sem(params_calc[p])
    param_std[p] = np.std(params_calc[p])

with open("Sensitivity12/"+province+"/4/Params.csv", "w") as file:
        filew = csv.writer(file,delimiter = ',')
        for p in param_list:
            filew.writerow([p,param_mean[p],param_sem[p],param_std[p]])

mean_mild = np.mean(mild,axis=0)
mean_severe = np.mean(severe,axis=0)
mean_known = np.mean(known,axis=0)
mean_total = np.mean(total,axis=0)
mean_new = np.mean(newcases,axis=0)
Rt_mean = np.mean(Rt,axis=0)

std_mild = np.std(mild,axis=0)
std_severe = np.std(severe,axis=0)
std_known = np.std(known,axis=0)
std_total = np.std(total,axis=0)
std_new = np.std(newcases,axis=0)
Rt_std = np.std(Rt,axis=0)

mean_mild_vax = np.mean(mild_vax,axis=0)
mean_severe_vax = np.mean(severe_vax,axis=0)
mean_known_vax = np.mean(known_vax,axis=0)
mean_total_vax = np.mean(total_vax,axis=0)
mean_new_vax = np.mean(newcases_vax,axis=0)

Rt_mean_vax = np.mean(Rt_vax,axis=0)

px_date = []
px_date1 = []
for pp in t:
    px_date.append(start_date1 + datetime.timedelta(days=10*pp))
for pp in t:
    px_date1.append(start_date + datetime.timedelta(days=10*pp))
Peak_k = px_date[np.argmax(mean_new[:50])]
Peak_m = px_date[np.argmax(mean_mild[:50])]
Peak_s = px_date[np.argmax(mean_severe[:50])]
Peak_t = px_date[np.argmax(mean_mild[:50]+mean_severe[:50])]
Total_All = mean_total[-1]
Total_Alls = 1.96*std_total[-1]
Total_Known = mean_known[-1]
Total_Knowns = 1.96*std_known[-1]
Peak_Mag = max(mean_mild[:50]+mean_severe[:50])
Peak_Mags = 1.96*max(std_mild[:50]+std_severe[:50])
Peak_Mild = max(mean_mild[:50])
Peak_Milds = 1.96*max(std_mild[:50])
Peak_Sev = max(mean_severe[:50])
Peak_Sevs = 1.96*max(std_severe[:50])

with open("Sensitivity12/"+province+"/4/Things.csv", "w") as file:
        filew = csv.writer(file,delimiter = ',')
        filew.writerow(['Peak k',Peak_k])
        filew.writerow(['Peak m',Peak_m])
        filew.writerow(['Peak s',Peak_s])
        filew.writerow(['Peak t',Peak_t])
        filew.writerow(['Total All',Total_All,Total_Alls])
        filew.writerow(['Total Known',Total_Known,Total_Knowns])
        filew.writerow(['Peak Mag',Peak_Mag,Peak_Mags])
        filew.writerow(['Peak Mag Mild',Peak_Mild,Peak_Milds])
        filew.writerow(['Peak Mag Sev',Peak_Sev,Peak_Sevs])

months = mdates.MonthLocator()
days = mdates.DayLocator(interval=7)
fig=plt.figure()
ax = fig.add_subplot(111)
ax.scatter(data['date'][ind:len(t_measured)-predict], data['numtotal'][ind:len(t_measured)-predict], marker='o', color='xkcd:navy blue', label='measured data', s=20)
ax.scatter(data['date'][len(t_measured)-predict:], data['numtotal'][len(x2_measured)-predict:], marker='o', color='xkcd:red', label='measured data', s=20)
ax.plot(px_date,mean_mild,label = 'mean: Mild',color = 'tab:red',linewidth=2)
ax.plot(px_date,mean_mild_vax,label = 'mean: Mild Vax',color = 'xkcd:burgundy',linewidth=2)
ax.fill_between(px_date,mean_mild-1.96*std_mild,mean_mild+1.96*std_mild,color='tab:red',alpha=0.25)
ax.plot(px_date,mean_severe,label = 'mean: Severe',color = 'tab:blue',linewidth=2)
ax.plot(px_date,mean_severe_vax,label = 'mean: Severe Vax',color = 'xkcd:navy',linewidth=2)
ax.fill_between(px_date,mean_severe-1.96*std_severe,mean_severe+1.96*std_severe,color='tab:blue',alpha=0.25)
ax.plot(px_date,mean_known,label = 'mean: Known',color = 'xkcd:mint green',linewidth=2)
ax.plot(px_date,mean_known_vax,label = 'mean: Known Vax',color = 'tab:green',linewidth=2)
ax.fill_between(px_date,mean_known-1.96*std_known,mean_known+1.96*std_known,color='xkcd:mint green',alpha=0.25)
ax.plot(px_date,mean_total,label = 'mean: Total',color = 'tab:purple',linewidth=2)
ax.plot(px_date,mean_total_vax,label = 'mean: Total Vax',color = 'xkcd:plum',linewidth=2)
plt.fill_between(px_date,mean_total-1.96*std_total,mean_total+1.96*std_total,color='tab:purple',alpha=0.25)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
ax.set_ylim((0,max([max(mean_total+1.96*std_known),max(mean_total_vax)])*1.1))
ax.set_xlabel('Date')
ax.set_ylabel('Cases')
ax.set_title(province)
ax.xaxis.set_major_locator(months)
ax.xaxis.set_minor_locator(days)
fig.savefig("Sensitivity12/"+province+'/4/'+province+'FIGURE-1.png',dpi=300,bbox_inches='tight')
plt.fill_between(px_date,mean_known-1.96*std_known,mean_known+1.96*std_known,color='xkcd:mint green',alpha=0.25)
ax.fill_between(px_date,mean_total-1.96*std_total,mean_total+1.96*std_total,color='tab:purple',alpha=0.25)
ax.set_yscale('log')
ax.set_ylim((1,max([max(mean_total+1.96*std_known),max(mean_total_vax)])*1.1))
fig.savefig("Sensitivity12/"+province+'/4/'+province+'FIGURE-log-1.png',dpi=300,bbox_inches='tight')

plt.clf()
ax = fig.add_subplot(111)
ax.scatter(data['date'][:len(t_measured)-predict], data['numtoday'][:len(t_measured)-predict],color='xkcd:navy blue',s = 20)
ax.scatter(data['date'][len(t_measured)-predict:], data['numtoday'][len(t_measured)-predict:],color='tab:red', s = 20)
ax.plot(px_date,mean_new,linewidth=2)
ax.plot(px_date,mean_new_vax,linewidth=2,color='tab:pink')
ax.set_ylim((0,max(mean_new)*1.2))
ax.xaxis.set_major_locator(months)
ax.xaxis.set_minor_locator(days)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
ax.set_xlabel('Date')
ax.set_ylabel('New Cases')
ax.set_title(province)
fig.savefig("Sensitivity12/"+province+'/4/'+province+'FIGUREnewcases-1-NOBARS.png',dpi=300,bbox_inches='tight')
ax.set_ylim((0,max([max(mean_new+1.96*std_new),max(mean_new_vax)])*1.2))
#ax.set_ylim((0,7500))
ax.fill_between(px_date,mean_new-1.96*std_new,mean_new+1.96*std_new,color='tab:purple',alpha=0.25)
fig.savefig("Sensitivity12/"+province+'/4/'+province+'FIGUREnewcases-1.png',dpi=300,bbox_inches='tight')

plt.clf()
ax = fig.add_subplot(111)
ax.plot(px_date,Rt_mean,linewidth=2)
ax.plot(px_date,Rt_mean_vax,linewidth=2,color="tab:pink")
ax.fill_between(px_date,Rt_mean-1.96*Rt_std,Rt_mean+1.96*Rt_std,color='tab:purple',alpha=0.25)
ax.plot(px_date,np.ones((len(px_date),1)))
ax.set_xlabel('Date')
ax.set_ylabel('$R_{eff}$')
ax.set_title(province)
ax.xaxis.set_major_locator(months)
ax.xaxis.set_minor_locator(days)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
ax.set_ylim(-0.1,2)
fig.savefig("Sensitivity12/"+province+'/4/'+province+'FIGURERt-1.png',dpi=300,bbox_inches='tight',transparent=True)

