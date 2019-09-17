# Weather-Forcasting
Weather temperature prediction  

Prediction weather temprature has been trained on weather data, that has been collected at Hussain Technichal University in Jordan. the dataset contains about 701006 observations in the period 25-01-2018 to 21-07-2019. Every record in the dataset represents weather information per minute. The dataset has nine columns; the first column shows the timestamp for every insight, while the reset columns are the weather information per minute. 


Our aim is to predict the weather temprature per day. So, I upsampled the dataset per day, then I divided the dataset to training and testing sets ( 80% training set and 20% testing set).

Using multiple regressors from sklearn library the dataset has been trained to achieve the best accuracy in the testing set as follows:

GradientBoostingRegressor
Training score:  0.8744019897809936
Test score:  0.7505109681311617

Linear Regressor
Training score:  0.7220992522262739
Test score:  0.7130470269807083

XGBRegressor
Training score:  0.7839359038431786
Test score:  0.7222213732625233

Ridge Regressor
Training score:  0.7181833876286197
Test score:  0.714372034437952

Lasso Regressor
Training score:  0.5997644408455675
Test score:  0.6048324150052011

MLPRegressor
Training score:  0.6607118446613041
Test score:  0.6454255654735612

DecisionTreeRegressor
Training score:  0.8407921895943988
Test score:  0.6428879845545679

Gradient Boosting regressor has achieved the best testing accuracy, which is 75%.




Requierments: 
Install the following libraries:
1- Pandas.
2- Numpy.
3- Matplotlib.
4- Sklearn.
5- Xgboost.
6- Seaborn.

Run temprature_predection.py 



