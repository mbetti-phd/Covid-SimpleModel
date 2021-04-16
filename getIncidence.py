import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt    
from scipy import stats
import pandas as pd
import csv
import datetime

def buildIncidence(file,province):

    case_data = pd.read_csv(file)

    case_data = case_data.loc[case_data['prname']==province]

    case_data['date'] = pd.to_datetime(case_data['date'], dayfirst=True)

    case_data = case_data.sort_values(by=['date'])

    total_cases_per_day = case_data[['date','numtotal','numtoday','percentactive']]

    province = province.replace(' ','')

    total_cases_per_day.to_csv(province+'Data.csv',index=False,header=True)


def main():
    provinces = ['Canada','Newfoundland and Labrador','Prince Edward Island', 'Nova Scotia', 'New Brunswick', 'Quebec','Ontario','Manitoba','Saskatchewan','Alberta','British Columbia','Yukon','Northwest Territories','Nunavut']

    for pr in provinces:
        buildIncidence('https://health-infobase.canada.ca/src/data/covidLive/covid19-download.csv',pr)
