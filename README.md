## **Body Mass Index (BMI): Application of predictive and descriptive methods**

The data belong to a study made to estimate obesity levels in individuals from Mexico, Peru and Colombia, based on their eating habits and physical condition. The data set has 17 features and 2111 rows.

The data are available at the UCI Machine Learning Repository. Follow the link:  
https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+#

The purpose of this case study is:
1. Create a Linear Regression Model (MAPE = 14%) and BMI Program as an alternative to the formula that is used today: BMI = kg/m2, where kg is a person's weight in kilograms and m2 is their height in metres squared;
2. Detect the groups of people at greatest health risk due to the level of obesity and the low (or zero) frequency of physical activity;
3. Check which algorithm has the best performance in classifying Body Mass Index;

### **How?**

1. With the Linear Regression algorithm;
2. Through KMeans, Agglomerative Hierarchical and DBSCAN clustering algorithms;
3. Through Logistic Regression, Decision Tree and Random Forest classification algorithms;

### **Why?**

Contribute to the initial idea of the research article in the creation of intelligent computational tools to identify the level of obesity of an individual.

### **Warning!**

I'm currently reviewing the project, specifically on cleaning, exploring and preprocessing data to improve the models created and add algorithms for classification.
### _Regression_

In the first predictive method of the project the Linear Regression algorithm was applied to make the model for predicting the level of Body Mass Index according to Weight. 

The model created was: 
```
The model is: Body Mass Index = -1.8815 + 0.069223X
```
Recalling the dictionary used for Body Mass Index levels:
* Insufficient_Weight: 1;
* Normal_Weight: 2;
* Overweight_Level_I: 3;
* Overweight_Level_II: 4;
* Obesity_Type_I: 5;
* Obesity_Type_II: 6;
* Obesity_Type_III: 7;

The quality of the model was assessed using the "_R-squared_" and the "_p-value_":
```
                        OLS Regression Results                            
==============================================================================
Dep. Variable:        Body_Mass_Index   R-squared:                       0.834
Model:                            OLS   Adj. R-squared:                  0.834
Method:                 Least Squares   F-statistic:                 1.060e+04
Date:                Wed, 05 May 2021   Prob (F-statistic):               0.00
Time:                        21:38:15   Log-Likelihood:                -2546.1
No. Observations:                2111   AIC:                             5096.
Df Residuals:                    2109   BIC:                             5108.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const         -1.8815      0.061    -30.942      0.000      -2.001      -1.762
Weight         0.0692      0.001    102.979      0.000       0.068       0.071
==============================================================================
Omnibus:                       36.905   Durbin-Watson:                   0.952
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               31.861
Skew:                           0.240   Prob(JB):                     1.21e-07
Kurtosis:                       2.636   Cond. No.                         313.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```
Warning! R-squared represents how strong my model represents linear behavior. But the model can still be improved even with a high R-squared value.

- R-squared was 0.834. This says that variable "Body Mass Index" was explained a good part by the "Weight" variable; 
- Prob (F-Statistic) was zero which implies that, in general, the regression is significant;
- Prob (Omnibus) also was zero implies that the OLS assumption is not satisfied;
- Durbin-Watson a value between 1 and 2 would be preferable. Here, it was 0.952 which indicates that the results of the regression are not reliable on the interpretation side of this metric;
- Jarque-Bera (JB) is supposed to comply with the results of the Omnibus test. A large JB test value indicates that errors are not normally distributed;
- P> | t | very low, the Null Hypothesis was rejected. The Null Hypothesis means that there is no correlation between the predicted and predictive variables, that is, for a model to work, it must be false. Generally if the "_p-value_" is less than 0.05, there is a strong relationship between the variables.

The correlation between the variables Body Mass Index and Overweight History also had a considerable value, according to the _Heatmap_ chart. But as the Overweight History is a binary diagnosis, the most suitable to be used is Logistic Regression and not Linear Regression. Logistic Regression was used later in the classification part of the project.

I created another Linear Regression model for the Age and Body Mass Index variables.

The model created was: 
```
The model is: Body Mass Index = 1.9553 + 0.088708X
```
Below is the OLS Regression results:

```
                        OLS Regression Results                            
==============================================================================
Dep. Variable:        Body_Mass_Index   R-squared:                       0.081
Model:                            OLS   Adj. R-squared:                  0.080
Method:                 Least Squares   F-statistic:                     185.1
Date:                Thu, 06 May 2021   Prob (F-statistic):           1.77e-40
Time:                        11:07:21   Log-Likelihood:                -4353.5
No. Observations:                2111   AIC:                             8711.
Df Residuals:                    2109   BIC:                             8722.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          1.9553      0.164     11.933      0.000       1.634       2.277
Age            0.0887      0.007     13.607      0.000       0.076       0.101
==============================================================================
Omnibus:                      961.729   Durbin-Watson:                   0.249
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              114.537
Skew:                           0.102   Prob(JB):                     1.34e-25
Kurtosis:                       1.877   Cond. No.                         99.5
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```
R-squared was very low compared to the previous model and higher AIC, in this case 8711, implies a worse model.
##### _Performance Metrics_

After applying the Linear Regression algorithm for Machine Learning across the whole data set, the error that the model generated was calculated using the metrics: Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE). The following results were obtained:
```
MAE_test: 0.38
MAPE_test: 0.14
```
###### _Cross-Validation_

The result of the _Cross-Validation_ metric was also satisfactory:
```
0.94 accuracy with a standard deviation of 0.01
```
### _Clustering_

At this step, the descriptive method was approached by clustering, to verify a more select group at risk, according to the level of Body Mass Index and Physical Activity performed per week.

1. _Centroids-based clustering: K-Means_

Before applying the algorithm, it is necessary to define a ‘K’, that is, a number of clusters.
##### _Elbow Method_

The _Elbow Method_ was used to find out the number of clusters that was used in KMeans.

![](/Charts/ElbowMethod.png)

According to chart I chose k = 7. It can be seen that the most worrying group in the sample is Cluster 4, which has the highest rates of obesity and does not practice physical activity. There are 256 people in the purple cluster, as shown below:

![](/Charts/kmeans.png)

##### _Performance Metrics_

Through the _silhouette_score_ metrics, there was a return of approximately 0.5 which can be considered an acceptable performance of KMeans.
```
The Silhouette_Score of K-means is: 0.50
```
2. _Connectivity clustering: Agglomerative Hierarchical_

To know the number of clusters that was used in this algorithm, I applied the _Dendrogram_ method.
##### _Dendrogram_

![](/Charts/Dendrogram.png)

The number of clusters used was 2.

According to the graph, the risk group in the sample is the blue cluster or zero cluster. This cluster represents 1342 people, as shown below:

![](/Charts/Hierarchical.png)
##### _Performance Metrics_

Through the _silhouette_score_ metrics, there was a return of approximately 0.5 which can be considered an acceptable performance of Agglomerative Hierarchical.
```
The Silhouette_Score of Hierarchical is: 0.49
```
3. _Density-Based Spatial Clustering of Applications with Noise: DBSCAN_

Various values of _eps_ and _min_samples_ were tested and the values that gave the best _Silhouette Score_ result were with _eps_= .5 and _min_samples_= 15. With these values, DBSCAN returned 26 clusters.

![](/Charts/DBSCAN.png)

According to the chart, the highest risk group is Cluster 22. In this cluster there are 187 people. DBSCAN returns a better result, compared to previous clustering algorithms. Because with a more select group it is possible to do more effective work.
##### _Performance Metrics_

The best value was obtained through the _silhouette_score_ metrics:
```
The Silhouette_Score of DBSCAN is: 1.0
```
### _Classification_

In the second predictive method of the project the _Logistic Regression_, _Decision Tree_ and _Random Forest_ algorithms were applied.
##### _Performance Metrics_

Below the _Accuracy_ results of the models:
```
Accuracy of Logistic Regression Classifier on test set: 0.70
Accuracy of Logistic Regression Classifier on train set: 0.71
```
```
Accuracy of Decision Tree Classifier on test set: 0.91
Accuracy of Decision Tree Classifier on train set: 1.00
```
```
Accuracy of Random Forest Classifier on test set: 0.93
Accuracy of Random Forest Classifier on train set: 1.00
```
As a complement to the _Accuracy_, the _Confusion Matrix_ was applied to confirm the performance of the models.
| ![](/Charts/LogisticRegressionConfusionMatrix.png) | ![](/Charts/DecisionTreeConfusionMatrix.png)| ![](/Charts/RandomForestConfusionMatrix.png)|
|:-:|:-:|:-:|

As shown in the charts, as in _Accuracy_, the model generated by _Random Forest_ had a better performance.






