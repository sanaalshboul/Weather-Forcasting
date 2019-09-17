# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 12:12:51 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.ensemble import AdaBoostRegressor

plt.style.use('ggplot')
weather_data=pd.read_csv('C:\\Users\\Administrator.Omnisec421_05\\.spyder-py3\\Weather_Project\\weather_data_day.csv')

#delete rows with any empty cells more than 7
weather_data=weather_data.dropna(axis=0, how='any', thresh=7)

#convert the TimeStamp to datetime
weather_data['TimeStamp']=pd.to_datetime(weather_data['TimeStamp'], errors='coerce')

#set column index
weather_data=weather_data.set_index(['TimeStamp'])

#split the dataset to train and test sets
predictors=['Wind_Speed','Wind_Direction','Solar_Radiation','Humidity','Rain_fall','Visibility','Pressure']
X=weather_data[predictors]
y=weather_data['Air_Temp']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
#******************************************************************************
#****************************plotting part*************************************
#plot relations between all columns (EDA)
#Air_Temp has linear relations with pressure/1, visibility/1, humidity/-1,solar radiation/1
#solar radiation has linear relations with pressure/2, visibility/1, humidity/-1, Air_Temp/1
#Humedity has linear relation with pressure/-1, visibility/-1, Air_Temp/-1,solar radiation/-1
pd.scatter_matrix(weather_data, c=y, figsize=[14,10], s=100, marker='D')


#plot the reltion between every feature and Air_Temp feature
weather_data.plot(x='Wind_Speed', y='Air_Temp', kind= 'scatter', s=70, c='m')
weather_data.plot(x='Wind_Direction', y='Air_Temp', kind= 'scatter', s=70, c='m')
weather_data.plot(x='Solar_Radiation', y='Air_Temp', kind= 'scatter', s=70, c='m')
weather_data.plot(x='Humidity', y='Air_Temp', kind= 'scatter', s=70, c='m')
weather_data.plot(x='Rain_fall', y='Air_Temp', kind= 'scatter', s=70, c='m')
weather_data.plot(x='Visibility', y='Air_Temp', kind= 'scatter', s=70, c='m')
weather_data.plot(x='Pressure', y='Air_Temp', kind= 'scatter', s=70, c='m')
weather_data.plot(x='Wind_Speed', y='Air_Temp', kind= 'scatter', s=70, c='m')


#Box plot for every feature to show the outliers
plt.rcParams['figure.figsize']=[6,6]
for i in predictors:
    weather_data.plot(y=i,kind='box',patch_artist=True, notch='True', color='m')    
    plt.ylabel(i+' percentage')
    plt.show()
    
    
#Box plot for all features
plt.rcParams['figure.figsize']=[10,10]
weather_data.plot.box(patch_artist=True, notch='True', color='m')


#plot the distribution of every feature 2 years
for i in predictors:
    weather_data[i].plot(color='m')
    plt.ylabel(i)
    plt.show()
    
#lasso for feature selection  
names = weather_data.drop('Air_Temp', axis=1).columns
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X, y).coef_
_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.xticks(range(len(names)), names, rotation=60)
_ = plt.ylabel('Coefficients')
plt.show()
#******************************************************************************
discribe=weather_data.describe().T
IQR = discribe['75%'] - discribe['25%']

# create an outliers column which is either 3 IQRs below the first quartile or
# 3 IQRs above the third quartile
discribe['outliers'] = (discribe['min']<(discribe['25%']-(3*IQR)))|(discribe['max'] > (discribe['75%']+3*IQR))

# just display the features containing extreme outliers
discribe.ix[discribe.outliers,]
#Wind_speed an Rain_fall have the most outliers
#******************************************************************************
#plot the histogram of Rain, visability, and wind speed since they are outliers
plt.rcParams['figure.figsize'] = [14, 8]
weather_data.Rain_fall.hist(color='m')
plt.title('Avg Rain fall')
plt.xlabel('Rain fall')
plt.show()

weather_data.Wind_Speed.hist(color='m')
plt.title('Wind Speed')
plt.xlabel('Wind Speed')
plt.show()
#*******************************************************************************************************  
#calculating the Pearson correlation coefficient (r) is a measurement of the amount of linear correlation between equal length arrays which outputs a value ranging -1 to 1
print(weather_data.corr()[['Air_Temp']].sort_values('Air_Temp'))

#calculate the correlation coefficient between any two features in the data  and plot them
f,ax=plt.subplots(figsize=(12,12))
sns.heatmap(weather_data.corr(), annot=True, linewidths=0.4, fmt='1f', ax=ax)
plt.show()

#******************************************************************************************************
#plotting the relation between learning rate and train and test scores using gradient boost regresso
neighbors = [0.0001,0.001,0.01,0.02,0.03,0.04,0.05,0.1,0.2,0.25,0.3,0.4,0.55,0.63]
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))


for i, k in enumerate(neighbors):
    print("i: ",i, "K: ", k)
    gb = GradientBoostingRegressor(n_estimators= 600, max_depth=1,max_features=1, min_samples_split=2,
              learning_rate= k, loss='ls')
    gb.fit(X_train, y_train)
    train_accuracy[i] = gb.score(X_train, y_train)
    test_accuracy[i] = gb.score(X_test, y_test)


plt.title('GradientBoost: Varying Number of learning rate')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Learning rate')
plt.ylabel('Accuracy')
plt.show()


gb = GradientBoostingRegressor(n_estimators= 600, max_depth=1,max_features=1, min_samples_split=2,
              learning_rate= .5, loss='ls')
gb.fit(X_train, y_train)

print('*****************Model1: GradientBoostingRegressor*******************')
print('Training score: ',gb.score(X_train, y_train))
print("Test score: ",gb.score(X_test, y_test))
print('********************************End**********************************')
#*******************************************************************************************************  
regressor=LinearRegression()
# fit the build the model by fitting the regressor to the training data
regressor.fit(X_train, y_train)
# make a prediction set using the test set
prediction=regressor.predict(X_test)

print('*****************Model2: Linear Regressor*******************')
print('Training score: ',regressor.score(X_train, y_train))
print("Test score: ", regressor.score(X_test, y_test))
print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_test, prediction))
print("The Median Absolute Error: %.2f degrees celsius" % median_absolute_error(y_test, prediction))
print('***************************End******************************')

#***************************************************************************************************
xgb=XGBRegressor(objective="reg:linear", random_state=42, alpha=1, max_depth=1).fit(X_train,y_train)
xgb_predict=xgb.predict(X_test)
train_predict=xgb.predict(X_train)

print('*****************Model3: XGBRegressor*******************')
print('Training score: ',xgb.score(X_train, y_train))
print("Test score: ", xgb.score(X_test, y_test))
print('**************************End***************************')

#***************************************************************************************************

ridge=Ridge(alpha=0.1, normalize=True)
ridge.fit(X_train, y_train)
ridge_predict=ridge.predict(X_test)

print('*****************Model4: Ridge Regressor*******************')
print('Training score: ',ridge.score(X_train, y_train))
print("Test score: ", ridge.score(X_test, y_test))
print('***************************End*****************************')
#accurcy=71
#***************************************************************************************************
lasso=Lasso(alpha=0.1, normalize=True)
lasso.fit(X_train, y_train)
lasso_predict=lasso.predict(X_test)

print('*****************Model5: Lasso Regressor*******************')
print('Training score: ',lasso.score(X_train, y_train))
print("Test score: ", lasso.score(X_test, y_test))
print('***************************End*****************************')
#accuracy=72
#***************************************************************************************************

mlp=MLPRegressor(solver='lbfgs',hidden_layer_sizes=1000 ).fit(X_train, y_train)
mlp_predict=mlp.predict(X_test)

print('*****************Model6: MLPRegressor*******************')
print('Training score: ',mlp.score(X_train, y_train))
print("Test score: ", mlp.score(X_test, y_test))
print('***************************End**************************')
#accuracy=65
#**************************************************************************************************
dtr=DecisionTreeRegressor(random_state=0, max_depth=4, max_features=None,max_leaf_nodes=500).fit(X_train,y_train)
dtr_predict=dtr.predict(X_test)

print('***************Model7: DecisionTreeRegressor*****************')
print('Training score: ',dtr.score(X_train, y_train))
print("Test score: ", dtr.score(X_test, y_test))
print('******************************End****************************')
#***************************************************************************************************
parameters={'base_estimator':[None], 'learning_rate':[.001,.01,.25,.5,.75,1.0,], 'loss':['linear','square', 'exponential'],
        'n_estimators':[10,20,30,50,100,400,600], 'random_state':[0,1,2,10,15,20]}
regr = AdaBoostRegressor()
regr_cv=GridSearchCV(regr, parameters,cv=5)
regr_cv.fit(X_train, y_train)  

print(regr_cv.best_params_)
print(regr_cv.best_score_)
print(regr.score(X_train, y_train))
print(regr.score(X_test, y_test))