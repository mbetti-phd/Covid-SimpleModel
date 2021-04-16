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

def main(province):

    D = 10
    predict = 0
    G = 0

    today = datetime.date.today()
    today = datetime.datetime(2021,3,2)
    today_str = today.strftime("%b-%d-%Y")

    param_list = ['x10','x20','x30','x40','R0','p','r','N','k','m']

    files = os.listdir('Sensitivity-C/'+province.replace(' ','')+'/'+today_str+"/")

    data = pd.read_csv(province+'Data.csv',parse_dates=True)
    data['date'] = pd.to_datetime(data['date'], dayfirst=True)
    y, m, d = getStartDate(data['date'][0],data['numtotal'][0])
    start_date1 = datetime.datetime(2020,9,8)
    today = str(datetime.datetime.today().date())
    start_date = datetime.datetime(2020,9,8)
    if province in ['Ontario','BritishColumbia']:
        start_date = datetime.datetime(2020,9,8)
    if province in ['Canada']:
        start_date = datetime.datetime(2020,9,8)
        
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

    t = np.linspace(x[0], 36.5,366-x[0])
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
                params[p] = get_parameter('Sensitivity-C/'+province.replace(' ','')+'/'+today_str+"/"+file,p).nominal_value
                params_calc[p].append(get_parameter('Sensitivity-C/'+province.replace(' ','')+'/'+today_str+"/"+file,p).nominal_value)
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

    param_mean = {}
    param_sem = {}
    param_std = {}

    for p in param_list:
        param_mean[p] = np.mean(params_calc[p])
        param_sem[p] = stats.sem(params_calc[p])
        param_std[p] = np.std(params_calc[p])

    with open('Sensitivity-C/'+province.replace(' ','')+'/'+today_str+"/"+"Params.csv", "w") as file:
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

    with open('Sensitivity-C/'+province.replace(' ','')+'/'+today_str+"/"+"Things.csv", "w") as file:
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
    ax.fill_between(px_date,mean_mild-1.96*std_mild,mean_mild+1.96*std_mild,color='tab:red',alpha=0.25)
    ax.plot(px_date,mean_severe,label = 'mean: Severe',color = 'tab:blue',linewidth=2)
    ax.fill_between(px_date,mean_severe-1.96*std_severe,mean_severe+1.96*std_severe,color='tab:blue',alpha=0.25)
    ax.plot(px_date,mean_known,label = 'mean: Known',color = 'tab:green',linewidth=2)
    ax.fill_between(px_date,mean_known-1.96*std_known,mean_known+1.96*std_known,color='tab:green',alpha=0.25)
    ax.plot(px_date,mean_total,label = 'mean: Total',color = 'tab:purple',linewidth=2)
    #plt.fill_between(px_date,mean_total-1.96*std_total,mean_total+1.96*std_total,color='tab:purple',alpha=0.25)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    ax.set_ylim((0,max(mean_total+1.96*std_known)*1.1))
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Cases')
    ax.set_title(province)
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_minor_locator(days)
    fig.savefig('Sensitivity-C/'+province.replace(' ','')+'/'+today_str+"/"+'FIGURE-1.png',dpi=300,bbox_inches='tight')
    #plt.fill_between(px_date,mean_known-1.96*std_known,mean_known+1.96*std_known,color='tab:green',alpha=0.25)
    ax.fill_between(px_date,mean_total-1.96*std_total,mean_total+1.96*std_total,color='tab:purple',alpha=0.25)
    ax.set_yscale('log')
    ax.set_ylim((1,max(mean_total+1.96*std_total)*1.1))
    fig.savefig('Sensitivity-C/'+province.replace(' ','')+'/'+today_str+"/"+'FIGURE-log-1.png',dpi=300,bbox_inches='tight')

    plt.clf()
    ax = fig.add_subplot(111)
    ax.scatter(data['date'][:len(t_measured)-predict], data['numtoday'][:len(t_measured)-predict],color='xkcd:navy blue',s = 20)
    ax.scatter(data['date'][len(t_measured)-predict:], data['numtoday'][len(t_measured)-predict:],color='tab:red', s = 20)
    ax.plot(px_date,mean_new,linewidth=2)
    ax.set_ylim((0,max(mean_new)*1.2))
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_minor_locator(days)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    ax.set_xlabel('Date')
    ax.set_ylabel('New Cases')
    ax.set_title(province)
    fig.savefig('Sensitivity-C/'+province.replace(' ','')+'/'+today_str+"/"+'FIGUREnewcases-1-NOBARS.png',dpi=300,bbox_inches='tight')
    ax.set_ylim((0,max(mean_new+1.96*std_new)*1.2))
    ax.fill_between(px_date,mean_new-1.96*std_new,mean_new+1.96*std_new,color='tab:purple',alpha=0.25)
    fig.savefig('Sensitivity-C/'+province.replace(' ','')+'/'+today_str+"/"+'FIGUREnewcases-1.png',dpi=300,bbox_inches='tight')

    plt.clf()
    ax = fig.add_subplot(111)
    ax.plot(px_date,Rt_mean,linewidth=2)
    ax.fill_between(px_date,Rt_mean-1.96*Rt_std,Rt_mean+1.96*Rt_std,color='tab:purple',alpha=0.25)
    ax.plot(px_date,np.ones((len(px_date),1)))
    ax.set_xlabel('Date')
    ax.set_ylabel('$R_{eff}$')
    ax.set_title(province)
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_minor_locator(days)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    fig.savefig('Sensitivity-C/'+province.replace(' ','')+'/'+today_str+"/"+'FIGURERt-1.png',dpi=300,bbox_inches='tight')

    #d_save = {'date':px_date,'cumulative_cases_mean':mean_known,'lower_confidence':mean_known-1.96*std_known,'upper_confidence':mean_known+1.96*std_known}
    cols = ["Sim"+str(i) for i in range(0, len(newcases[:,0]))]
    df = pd.DataFrame(data=np.transpose(newcases),columns=cols)
    df.to_csv('Sensitivity-C/'+province.replace(' ','')+'/'+today_str+"/"+province+'forecast.csv')

main('BritishColumbia')