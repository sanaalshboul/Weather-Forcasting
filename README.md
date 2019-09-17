# Weather-Forcasting
Weather temperature prediction  

Prediction weather temprature has been trained on weather data, that has been collected at Hussain Technichal University in Jordan. the dataset contains about 701006 observations in the period 25-01-2018 to 21-07-2019. Every record in the dataset represents weather information per minute. The dataset has nine columns; the first column shows the timestamp for every insight, while the reset columns are the weather information per minute. 

TimeStamp	Wind_Speed	Wind_Direction	Solar_Radiation	Air_Temp	Humidity	Rain_fall	Visibility	Pressure
1/25/2018 13:04	0	0	0.03905404	5.814	96.312	0	0	995.9869
1/25/2018 13:05	2.9375	279.3135	0.03463924	5.82925	96.462	0.254	2705	995.6366
1/25/2018 13:06	3.8875	206.6951	0.03013906	5.78675	96.394	0	2085	995.4615
1/25/2018 13:07	3.8625	127.4307	0.02844074	5.7015	96.352	0	1544	995.4614
1/25/2018 13:08	3.9625	207.3859	0.02648783	5.65325	96.414	0.254	1630	995.2863
1/25/2018 13:09	4.75	182.193	0.0291194	5.64325	96.703	0.254	1843	995.4615


Our aim is to predict the weather temprature per day. So, I upsampled the dataset per day as follows:

TimeStamp	Wind_Speed	Wind_Direction	Solar_Radiation	Air_Temp	Humidity	Rain_fall	Visibility	Pressure
1/25/2018	3.890929857	186.0468321	0.019682798	5.274912337	98.67570732	0.013551829	6111.72561	995.0524944
1/26/2018	5.318654442	172.7533281	0.049187378	2.795219792	93.58752083	0.005997222	14692.54583	996.538384
1/27/2018	1.846597223	215.5391824	0.030725039	4.068115619	99.99783542	0.001234722	3298.190278	995.602468
1/28/2018	1.334939238	240.7198386	0.105149542	6.038954603	91.93746528	0	12423.22639	999.4369912
1/29/2018	0.931571184	218.5771519	0.074216997	5.643803606	95.0768875	0.000176389	9995.508333	997.8202369
1/30/2018	1.32886285	238.6529323	0.122782611	6.75970339	90.98296736	0.000176389	9269.991667	1000.362858

I divided the dataset to training and testing sets ( 80% training set and 20% testing set).

Using multiple regressors from sklearn library the dataset has been trained to achieve the best accuracy in the testing set as follows: 


