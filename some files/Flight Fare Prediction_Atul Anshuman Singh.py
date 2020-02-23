"""
@author: u346442
"""
# import packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from xgboost import XGBRegressor
import itertools as it
from xgboost import plot_importance
os.chdir('C:/Users/u346442/Desktop/Data Science')

# read datasets
train_fares = pd.read_csv('training/train_fares.csv')
train_schedules = pd.read_csv('training/train_schedules.csv')
service_index = pd.read_csv('training/service_index.csv')

test_fares = pd.read_csv('test/test_fares_data.csv')
test_schedules = pd.read_csv('test/test_schedules.csv')

# target
target = train_fares['total_fare']

# check for nulls 
# : None of files has any missing values - Hurray! :P
train_fares.isna().sum()
train_schedules.isna().sum()
service_index.isna().sum()

test_fares.isna().sum()
test_schedules.isna().sum()

# Let's do some EDA:
# Problem statement: to predict daliy fare for carriers L1, L2, L3

# 1. Train: 2,160,016 data points
#    Test: 29,825 data point
# 2. There are 8 unique airports in the data sets, these are common
#    in both train and test. 

# Data has 6 unique carriers, but we need to predict only for L1, L2 and L3
# We'll drop other carriers from our analysis
train_fares.groupby('carrier')['carrier'].count()
test_fares.groupby('carrier')['carrier'].count()

# Build vanila model with available features
# Create Basic Features - convert categorical features to one-hot

# But first, drop unnecessary columns and extra carriers
train_ = train_fares.drop(['Unnamed: 0', 'total_fare'] , axis = 1)
train_2 = train_.iloc[np.where((train_['carrier'] =='L1')
                           | (train_['carrier'] =='L2')
                           | (train_['carrier'] =='L3')
                          )]
target_ = target.iloc[np.where((train_['carrier'] =='L1')
                           | (train_['carrier'] =='L2')
                           | (train_['carrier'] =='L3')
                          )]

test_ = test_fares.drop('Unnamed: 0', axis = 1)
test_2 = test_.iloc[np.where((test_['carrier'] =='L1')
                           | (test_['carrier'] =='L2')
                           | (test_['carrier'] =='L3')
                          )]

del test_, train_

len(train_2)  #1493689
len(test_2)   #20404

1493689+ 20404
# concat the data toghether
full_data = pd.concat([train_2, test_2], axis = 0)
del test_2, train_2

# Convert origin and destnation --> Market and convert to one-hot encoding
full_data['dir_mkt'] = full_data['origin'].str.cat(full_data['destination'])

full_data['dir_mkt_cat'] = pd.Categorical(full_data['dir_mkt'])
dfDummies = pd.get_dummies(full_data['dir_mkt_cat'], prefix = 'dir_mkt_cat')
full_data = pd.concat([full_data, dfDummies], axis=1)

# carrier to one-hot encoding
full_data['car_cat'] = pd.Categorical(full_data['carrier'])
dfDummies = pd.get_dummies(full_data['car_cat'], prefix = 'car_cat')
full_data = pd.concat([full_data, dfDummies], axis=1)

# flt num to one-hot encoding
# Too-many sparse features - ***don't include***
full_data['flt_num_cat'] = pd.Categorical(full_data['flt_num'])
dfDummies = pd.get_dummies(full_data['flt_num_cat'], prefix = 'flt_num_cat')
full_data = pd.concat([full_data, dfDummies], axis=1)

# extract month, year, day of the month, week of the month, day of the week from observation date
# and departure date - at the same time conver each into cyclical feature - if applicable

full_data['deprt_dt_day'] = pd.to_datetime(full_data['flt_departure_dt']).dt.day
full_data['deprt_dt_day_sin'] = np.sin((full_data['deprt_dt_day']-1)*(2.*np.pi/31))
full_data['deprt_dt_day_cos'] = np.cos((full_data['deprt_dt_day']-1)*(2.*np.pi/31))

full_data['deprt_dt_weekofmonth'] = (full_data['deprt_dt_day'] - 1)//7 + 1
full_data['deprt_dt_weekofmonth_sin'] = np.sin((full_data['deprt_dt_weekofmonth']-1)*(2.*np.pi/5))
full_data['deprt_dt_weekofmonth_cos'] = np.cos((full_data['deprt_dt_weekofmonth']-1)*(2.*np.pi/5))

full_data['deprt_dt_month'] = pd.to_datetime(full_data['flt_departure_dt']).dt.month
full_data['deprt_dt_month_sin'] = np.sin((full_data['deprt_dt_month']-1)*(2.*np.pi/12))
full_data['deprt_dt_month_cos'] = np.cos((full_data['deprt_dt_month']-1)*(2.*np.pi/12))

full_data['deprt_dt_week'] = pd.to_datetime(full_data['flt_departure_dt']).dt.week
full_data['deprt_dt_week_sin'] = np.sin((full_data['deprt_dt_week']-1)*(2.*np.pi/52))
full_data['deprt_dt_week_cos'] = np.cos((full_data['deprt_dt_week']-1)*(2.*np.pi/52))

full_data['deprt_dt_weekday'] = pd.to_datetime(full_data['flt_departure_dt']).dt.weekday
full_data['deprt_dt_weekday_sin'] = np.sin((full_data['deprt_dt_weekday'])*(2.*np.pi/7))
full_data['deprt_dt_weekday_cos'] = np.cos((full_data['deprt_dt_weekday'])*(2.*np.pi/7))



full_data['obser_dt_day'] = pd.to_datetime(full_data['observation_date']).dt.day
full_data['obser_dt_day_sin'] = np.sin((full_data['obser_dt_day']-1)*(2.*np.pi/31))
full_data['obser_dt_day_cos'] = np.cos((full_data['obser_dt_day']-1)*(2.*np.pi/31))

full_data['obser_dt_weekofmonth'] = (full_data['obser_dt_day'] - 1)//7 + 1
full_data['obser_dt_weekofmonth_sin'] = np.sin((full_data['obser_dt_weekofmonth']-1)*(2.*np.pi/5))
full_data['obser_dt_weekofmonth_cos'] = np.cos((full_data['obser_dt_weekofmonth']-1)*(2.*np.pi/5))

full_data['obser_dt_month'] = pd.to_datetime(full_data['observation_date']).dt.month
full_data['obser_dt_month_sin'] = np.sin((full_data['obser_dt_month']-1)*(2.*np.pi/12))
full_data['obser_dt_month_cos'] = np.cos((full_data['obser_dt_month']-1)*(2.*np.pi/12))

full_data['obser_dt_week'] = pd.to_datetime(full_data['observation_date']).dt.week
full_data['obser_dt_week_sin'] = np.sin((full_data['obser_dt_week']-1)*(2.*np.pi/52))
full_data['obser_dt_week_cos'] = np.cos((full_data['obser_dt_week']-1)*(2.*np.pi/52))

full_data['obser_dt_weekday'] = pd.to_datetime(full_data['observation_date']).dt.weekday
full_data['obser_dt_weekday_sin'] = np.sin((full_data['obser_dt_weekday'])*(2.*np.pi/7))
full_data['obser_dt_weekday_cos'] = np.cos((full_data['obser_dt_weekday'])*(2.*np.pi/7))


###########################################################################
# Advanced features
###########################################################################

# DFD

full_data['DFD'] = pd.to_datetime(full_data['flt_departure_dt']) - pd.to_datetime(full_data['observation_date'])
full_data['DFD'] = full_data['DFD'].apply(lambda x: x.days if hasattr(x,'days') else np.nan)
full_data['DFD'] = np.abs(full_data['DFD'])

##########################################################################
# Flight hours and mins

# The data contains NaN after merge, we'll fill the nan's with corresponding
# from schedules table at 'carrier', 'flt_num', 'origin', 'destination' level
# final fall back: 'carrier', 'origin', 'destination' level
##########################################################################

train_schedules['flt_hours'] = (pd.to_datetime(train_schedules['flt_arrival_gmt']) - pd.to_datetime(train_schedules['flt_departure_gmt'])).apply(lambda x: x.total_seconds()//3600)
train_schedules['flt_mins'] = (pd.to_datetime(train_schedules['flt_arrival_gmt']) - pd.to_datetime(train_schedules['flt_departure_gmt'])).apply(lambda x: x.total_seconds()//60)

test_schedules['flt_hours'] = (pd.to_datetime(test_schedules['flt_arrival_gmt']) - pd.to_datetime(test_schedules['flt_departure_gmt'])).apply(lambda x: x.total_seconds()//3600)
test_schedules['flt_mins'] = (pd.to_datetime(test_schedules['flt_arrival_gmt']) - pd.to_datetime(test_schedules['flt_departure_gmt'])).apply(lambda x: x.total_seconds()//60)

# Prepare to merge with the main data set
train_schedules['flt_departure_dt'] = pd.to_datetime(train_schedules['flt_departure_dt'])
test_schedules['flt_departure_dt'] = pd.to_datetime(test_schedules['flt_departure_dt'])

schedules = pd.concat([train_schedules, test_schedules], axis = 0)

full_data['flt_departure_dt'] = pd.to_datetime(full_data['flt_departure_dt'])

# Merge to datasets
full_data = pd.merge(full_data,  
                     schedules,
                     how = 'left', 
                     left_on = ['carrier', 'flt_num', 'origin','destination', 'flt_departure_dt'], 
                     right_on = ['carrier', 'flt_num', 'origin','destination', 'flt_departure_dt'] ).drop(['Unnamed: 0','flt_departure_local_time','flt_arrival_local_time','flt_departure_gmt','flt_arrival_gmt'], axis = 1)


SkedRolledUp = schedules.groupby(['carrier', 'flt_num', 'origin', 'destination']).agg({'flt_hours': 'mean', 'flt_mins':'mean'}).reset_index()
SkedRolledUp2 = schedules.groupby(['carrier', 'origin', 'destination']).agg({'flt_hours': 'mean', 'flt_mins':'mean'}).reset_index()

full_data = pd.merge(full_data,  
                      SkedRolledUp,
                      how = 'left',
                      left_on = ['carrier', 'flt_num', 'origin','destination'], 
                      right_on = ['carrier', 'flt_num', 'origin','destination'])

full_data = pd.merge(full_data,  
                      SkedRolledUp2,
                      how = 'left',
                      left_on = ['carrier', 'origin','destination'], 
                      right_on = ['carrier', 'origin','destination'])


def func(a, b, c):
    
    if np.isnan(a):   
        if np.isnan(b):
            return c
        else:
            return b
    else:
        return a

full_data['flt_hours_x'] = full_data.apply(lambda x: func(x['flt_hours_x'],x['flt_hours_y'],x['flt_hours']), axis = 1)
full_data['flt_mins_x'] = full_data.apply(lambda x: func(x['flt_mins_x'],x['flt_mins_y'],x['flt_mins']), axis = 1)

full_data.drop(['flt_hours_y', 'flt_hours', 'flt_mins_y', 'flt_mins'], axis = 1, inplace= True)
full_data.rename(columns = {'flt_hours_x':'flt_hours','flt_mins_x':'flt_mins'}, inplace = True)

# Scale feature
full_data['flt_mins']  = (full_data['flt_mins'] - np.mean(full_data['flt_mins']))/(np.std(full_data['flt_mins']))

del SkedRolledUp, SkedRolledUp2

##############################################################################
# Monthly number of flights  -- doesn't give good result, do not include

# for every flight number get the count at origin, destination, carrier level
##############################################################################

schedules.columns
schedules['month'] = pd.to_datetime(schedules['flt_departure_dt']).dt.month
schedules['year'] = pd.to_datetime(schedules['flt_departure_dt']).dt.year

number_of_flight = schedules.groupby(['flt_num','carrier', 'origin', 'destination', 'year','month']).agg({'flt_num':'count'}).rename(columns = {'flt_num':'flt_num_count'}).reset_index()
number_of_flight2 = schedules.groupby(['flt_num','origin', 'destination', 'year','month']).agg({'flt_num':'count'}).rename(columns = {'flt_num':'flt_num_count2'}).reset_index()

full_data['deprt_dt_year'] = pd.to_datetime(full_data['flt_departure_dt']).dt.year

full_data = pd.merge(full_data,  
                      number_of_flight,
                      how = 'left',
                      left_on = ['flt_num','carrier', 'origin','destination','deprt_dt_year','deprt_dt_month'], 
                      right_on = ['flt_num','carrier', 'origin','destination', 'year', 'month'])

full_data = pd.merge(full_data,  
                      number_of_flight2,
                      how = 'left',
                      left_on = ['flt_num','origin','destination','deprt_dt_year','deprt_dt_month'], 
                      right_on = ['flt_num', 'origin','destination', 'year', 'month'])

full_data['flt_num_count_y'].fillna(0, inplace = True)


def func(a, b):
    
    if np.isnan(a):   
        return b
    else:
        return a

full_data['flt_num_count'] = full_data.apply(lambda x: func(x['flt_num_count'],x['flt_num_count2']), axis = 1)

full_data = full_data.drop(['destination_y','flt_num_count2'], axis = 1)
full_data.rename(columns = {'destination_x':'destination'}, inplace = True)

# Scale features
full_data['flt_num_count_y'] = (full_data['flt_num_count_y'] - np.mean(full_data['flt_num_count_y']))/np.std(full_data['flt_num_count_y'])

del number_of_flight, number_of_flight2

##############################################################################

# Monthly market demand and Monthly Share at the time of *observation month*

##############################################################################

full_data['obser_dt_year'] = pd.to_datetime(full_data['observation_date']).dt.year

# We do not have Nov, 2017 demand and share data, so we'll append Nov, 2018 as it is

nov_2018 = service_index[(service_index['yr']==2018) & (service_index['mo']==11)]

nov_2018['yr'] = 2017

service_index = pd.concat([service_index, nov_2018], axis = 0)

full_data = pd.merge(full_data,  
                     service_index,
                     how = 'left',
                     left_on = ['obser_dt_year','obser_dt_month','origin','destination', 'carrier'], 
                     right_on = ['yr', 'mo', 'origin','destination', 'carrier'])

full_data.drop(['Unnamed: 0','yr', 'mo'], axis = 1, inplace = True)

full_data['scaled_share'] = (full_data['scaled_share'] - np.mean(full_data['scaled_share']))/np.std(full_data['scaled_share'])
full_data['scaled_demand'] = (full_data['scaled_demand'] - np.mean(full_data['scaled_demand']))/np.std(full_data['scaled_demand'])

len(full_data)

########################################################################

# Prepare data to feed into the model

########################################################################


model_features = ['dir_mkt_cat_Airport17Airport4',
                'dir_mkt_cat_Airport20Airport4',
                'dir_mkt_cat_Airport26Airport30',
                'dir_mkt_cat_Airport30Airport26',
                'dir_mkt_cat_Airport30Airport31',
                'dir_mkt_cat_Airport30Airport60',
                'dir_mkt_cat_Airport31Airport30',
                'dir_mkt_cat_Airport43Airport4',
                'dir_mkt_cat_Airport4Airport17',
                'dir_mkt_cat_Airport4Airport20',
                'dir_mkt_cat_Airport4Airport43',
                'dir_mkt_cat_Airport60Airport30',
                'car_cat_L1',
                'car_cat_L2',
                'car_cat_L3',
                'deprt_dt_day_sin',
                'deprt_dt_day_cos',
                'deprt_dt_weekofmonth_sin',
                'deprt_dt_weekofmonth_cos',
                'deprt_dt_month_sin',
                'deprt_dt_month_cos',
                'deprt_dt_week_sin',
                'deprt_dt_week_cos',
                'deprt_dt_weekday_sin',
                'deprt_dt_weekday_cos',
                'obser_dt_year',
                'obser_dt_day_sin',
                'obser_dt_day_cos',
                'obser_dt_weekofmonth_sin',
                'obser_dt_weekofmonth_cos',
                'obser_dt_month_sin',
                'obser_dt_month_cos',
                'obser_dt_week_sin',
                'obser_dt_week_cos',
                'obser_dt_weekday_sin',
                'obser_dt_weekday_cos',
                'DFD',
                'flt_hours',
                'flt_mins',
                'scaled_share']

full_data_2_model = full_data[model_features]

train_to_model = full_data_2_model.iloc[0:1493689,:]
test_to_model = full_data_2_model.iloc[1493689:1493689+20404,:]

del full_data_2_model

###########################################################################
# Model building
###########################################################################

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from itertools import product

###########################################################################
########################################################################### 
# Test
X, X_Test, Y, Y_Test = train_test_split(train_to_model, target_, test_size=0.20, random_state = 1234567890)

# Validation
X, X_Val, Y, Y_Val = train_test_split(X, Y, test_size=0.10, random_state = 1234567890)

###########################################################################
###########################################################################

params = {  
            'n_estimators': [25, 50, 75, 100, 125],
            'min_child_weight':[4, 5 ,6], 
            'gamma':[0.3,0.6],  
            'subsample':[0.6, 0.7, 0.8],
            'colsample_bytree':[0.4, 0.6, 0.7, 0.9], 
            'max_depth': [10, 12, 15, 17, 18, 20]
            #'verbosity':2
         }

###########################################################################
# Some utility functions
###########################################################################
def param_generate(params):
    for i in product(*params.values()):
        yield dict(zip(params.keys(),i))

def add_score(param, score, param_list, score_list):
    param_list.append(param)
    score_list.append(score)

###########################################################################
# Grid search
###########################################################################
param_list = []
score_list = []
param0 = param_generate(params)
while True:
    score = 0

    param = next(param0, None)
    if param == None:
        break
    
    print("----------- New iteration ----------------","\n")
    
    model = XGBRegressor(**param, nthread=-1, verbosity = 2)
    print('Iteration Parmeters:',param,"\n")
    model.fit(X, Y, eval_metric=["rmse"], eval_set= [(X, Y),(X_Val, Y_Val)], verbose=True)
    score = model.evals_result()['validation_1']['rmse'][-1]
    
    add_score(param, score, param_list, score_list)

best_index = np.argmin(score_list)

best_param = param_list[best_index]
    
###########################################################################
# Train on train and validation
###########################################################################
params = {  
            'n_estimators': 50,
            'min_child_weight':10, 
            'gamma':0.5,  
            'subsample':0.5,
            'colsample_bytree': 0.8, 
            'max_depth':15,
            'verbosity':2
         }

model = XGBRegressor(**params,nthread=-1) 
model.fit(X, Y, eval_metric=["rmse"], eval_set = [(X,Y), (X_Val,Y_Val)], verbose=True, early_stopping_rounds=10)

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X, Y)
reg.score(X, Y)
reg_prd = reg.predict(X_Test)
np.sqrt(np.var(preds_test))

# Plot RMSE to check overfitting

val = model.evals_result()['validation_0']['rmse']
train_rmse = model.evals_result()['validation_1']['rmse']

plt.plot(range(0,len(val)),val)
plt.plot(range(0,len(val)),train_rmse)

# Check variable importance
plot_importance(model, max_num_features=10)
plt.show()

###########################################################################
###### Perfomance on unseen data
###########################################################################
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error
preds_test = model.predict(X_Test)
np.sqrt(mean_squared_error( Y_Test, reg_prd ))
np.sqrt(mean_squared_log_error( Y_Test, reg_prd ))
r2_score(Y_Test, reg_prd)


###########################################################################
# Train on entire train data with best parameters
###########################################################################
params = {  
            'n_estimators': 100,
            'min_child_weight':10, 
            'gamma':0.5,  
            'subsample':0.5,
            'colsample_bytree': 0.8, 
            'max_depth':15,
            'verbosity':2
         }

model = XGBRegressor(**params,nthread=-1) 
model.fit(train_to_model, target_, eval_metric=["rmse"], eval_set = [(train_to_model, target_)], verbose=True, early_stopping_rounds=10)

###########################################################################
###### Final performance in test
###########################################################################

final_pred = model.predict(test_to_model)

###########################################################################
###### Prepare submission file
###########################################################################

test_submission = full_data.iloc[1493689:1493689+20404,:][['origin',
                                                         'destination',
                                                         'carrier',
                                                         'flt_num',
                                                         'flt_departure_dt',
                                                         'observation_date',
                                                         'origin_city',
                                                         'destination_city']]

test_submission['fare_prediction'] = final_pred

test_submission['flt_departure_dt'] = pd.to_datetime(test_submission['flt_departure_dt'])
test_submission['observation_date'] = pd.to_datetime(test_submission['observation_date'])

test_fares['flt_departure_dt'] = pd.to_datetime(test_fares['flt_departure_dt'])
test_fares['observation_date'] = pd.to_datetime(test_fares['observation_date'])

final_test_submission = pd.merge(test_fares,
                                 test_submission,
                                 how = 'left',
                                 left_on = ['origin','destination', 'carrier', 'flt_num','flt_departure_dt','observation_date','origin_city','destination_city'], 
                                 right_on = ['origin','destination', 'carrier', 'flt_num','flt_departure_dt','observation_date','origin_city','destination_city']
                                 )


final_test_submission.to_csv('Flight_Prediction_Atul_Anshuman_Singh.csv', index = False)