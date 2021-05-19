# Obesity Estimate

## **Case Study**

The data belong to a study made to estimate obesity levels in individuals from Mexico, Peru and Colombia, based on their eating habits and physical condition. The data set has 17 features and 2111 rows.

The data are available at the UCI Machine Learning Repository. Follow the link:  
https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+#

The purpose of this case study was:
1. Make the Linear Regression model of Body Mass Index;
2. Detect the group with the highest risk of life related to level of Body Mass Index and frequency of physical activity;
3. Check which algorithm has the best performance in the classification of Body Mass Index;

### **How?**

1. With the Linear Regression algorithm;
2. Through KMeans, Agglomerative Hierarchical and DBSCAN clustering algorithms;
3. Through Logistic Regression, Decision Tree and Random Forest classification algorithms;

### **Why?**

Contribute to the initial idea of the research article in the creation of intelligent computational tools to identify the level of obesity of an individual, complementing with the detection of the group with the highest risk of life and providing the algorithm with a better performance for future classification analyzes.

## **Load Data**

There are ordinal and nominal categorical variables and numerical variables with different orders of magnitude.

As there are a lot of acronyms in the original column names, I renamed them for a better understanding.
```
Index(['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
       'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
       'CALC', 'MTRANS', 'NObeyesdad'],
      dtype='object')
```
 ## **Exploratory Data Analysis (EDA)**
### _Data Processing_

One of the most time-consuming steps when working with Machine Learning models is data processing. It is also essential to understand the conclusions that can be drawn from the data. 

The scatter plot chart shows how the variables Physical Activity, Weight and Body Mass Index are distributed.

![](/Charts/scatter_weight_faf.png)

For a better understanding of physical activity numbers, follow the dictionary:
```
Physical_Activity  = {'No': 1}, {'1 or 2 days': 2}, {'2 or 4 days': 3}, {'4 or 5 days': 4}
```

It can be seen that there is a variation in the frequency of physical activity from individuals who do not perform to individuals who perform 4 or 5 days a week, in the following groups of Body Mass Index:
- Insufficient Weight;
- Normal Weight;
- Overweight Level I;
- Overweight Level II;
- Obesity Type I;

The sample classified by “Obesity Type II” and “Obesity Type III” also have variation in physical activity but no more than 2 or 4 days a week.

There are data of type object and float. I treated the object type data with the _category_encoders_ method, which uses a dictionary to determine the order of the attributes. The _Label Encoder_ method could also be used, which is indicated for ordinal categorical variables, which is the case of the "Foods_between_Main_Meals", "Alcoholic_Drinks" and "Body_Mass_Index" columns. However, as this method assigns the order of values through the alphabetical order of the classes, it did not return an expected order. I could also use the _One Hot Encoding_ method or _get_dummies_ from the Pandas library but as my number of columns would increase, I chose not to use it.

```
      Gender        Age    Height      Weight  ...  Time_Spent_on_Technologies  Alcoholic_Drinks  Type_of_Transport_Used  Body_Mass_Index
0          2  21.000000  1.620000   64.000000  ...                    1.000000                 1                       3                2
1          2  21.000000  1.520000   56.000000  ...                    0.000000                 2                       3                2
2          1  23.000000  1.800000   77.000000  ...                    1.000000                 3                       3                2
3          1  27.000000  1.800000   87.000000  ...                    0.000000                 3                       2                3
4          1  22.000000  1.780000   89.800000  ...                    0.000000                 2                       3                4
...      ...        ...       ...         ...  ...                         ...               ...                     ...              ...
2106       2  20.976842  1.710730  131.408528  ...                    0.906247                 2                       3                7
2107       2  21.982942  1.748584  133.742943  ...                    0.599270                 2                       3                7
2108       2  22.524036  1.752206  133.689352  ...                    0.646288                 2                       3                7
2109       2  24.361936  1.739450  133.346641  ...                    0.586035                 2                       3                7
2110       2  23.664709  1.738836  133.472641  ...                    0.714137                 2                       3                7

[2111 rows x 17 columns]
```
For the other float type data, I rounded the values using the _round()_ function. But I kept the column "Height" with the float type.

Through the _info()_ function it was verified that there were no missing values and the data types.
```
#   Column                          Non-Null Count  Dtype
---  ------                          --------------  -----
 0   Gender                          2111 non-null   int32
 1   Age                             2111 non-null   int64
 2   Height                          2111 non-null   float64
 3   Weight                          2111 non-null   int64
 4   Overweight_History              2111 non-null   int32
 5   Hypercaloric_Foods_Consumption  2111 non-null   int32
 6   Vegetable_Consumption           2111 non-null   int64
 7   Main_Meals_Number               2111 non-null   int64
 8   Foods_between_Main_Meals        2111 non-null   int32
 9   Smoke                           2111 non-null   int32
 10  Drink_Water                     2111 non-null   int64
 11  Calorie_Monitoring              2111 non-null   int32
 12  Physical_Activity               2111 non-null   int64
 13  Time_Spent_on_Technologies      2111 non-null   int64
 14  Alcoholic_Drinks                2111 non-null   int32
 15  Type_of_Transport_Used          2111 non-null   int32
 16  Body_Mass_Index                 2111 non-null   int32
dtypes: float64(1), int32(9), int64(7)
```
### _Descriptive Analysis_

It's time to visualize the distribution of the data!

![](/Charts/distplot.png)

Below is the dictionary of the ordinal and nominal categorical variables for a better understanding of the data distribution.
```
Dictionary: 
Gender = {'Male': 1}, {'Female': 2}
Overweight_History = {'no': 1}, {'yes': 2}
Hypercaloric_Foods_Consumption = {'no': 1}, {'yes': 2}
Foods_between_Main_Meals = {'no': 1}, {'Sometimes': 2}, {'Frequently': 3}, {'Always': 4}
Smoke = {'no': 1}, {'yes': 2}
Calorie_Monitoring = {'no': 1}, {'yes': 2}
Alcoholic_Drinks = {'no': 1}, {'Sometimes': 2}, {'Frequently': 3}, {'Always': 4}
Type_of_Transport_Used = {'Bike': 1}, {'Walking': 2}, {'Public_Transportation': 3}, {'Motorbike': 4}, {'Automobile': 5}
Body_Mass_Index = {'Insufficient_Weight': 1}, {'Normal_Weight': 2}, {'Overweight_Level_I': 3}, {'Overweight_Level_II': 4}, {'Obesity_Type_I': 5}, {'Obesity_Type_II': 6}, {'Obesity_Type_III':7}
```
The _describe()_ function returns descriptive statistics values, as follows:

```
            Gender          Age       Height       Weight  ...  Time_Spent_on_Technologies  Alcoholic_Drinks  Type_of_Transport_Used  Body_Mass_Index
count  2111.000000  2111.000000  2111.000000  2111.000000  ...                 2111.000000       2111.000000             2111.000000      2111.000000
mean      1.494079    24.315964     1.701847    86.586452  ...                    0.664614          1.731407                3.405021         4.112269
std       0.500083     6.357078     0.100385    26.190136  ...                    0.674009          0.515498                0.864439         1.985062
min       1.000000    14.000000     1.400000    39.000000  ...                    0.000000          1.000000                1.000000         1.000000
25%       1.000000    20.000000     1.600000    65.500000  ...                    0.000000          1.000000                3.000000         2.000000
50%       1.000000    23.000000     1.700000    83.000000  ...                    1.000000          2.000000                3.000000         4.000000
75%       2.000000    26.000000     1.800000   107.000000  ...                    1.000000          2.000000                3.000000         6.000000
max       2.000000    61.000000     2.000000   173.000000  ...                    2.000000          4.000000                5.000000         7.000000

[8 rows x 17 columns]
```
Having a visual summary of the information makes it easier to identify patterns and trends than to look at lines of a spreadsheet. For that I used seaborn which is a Python data visualization library based on Matplotlib.

| ![](/Charts/Gender.png) | ![](/Charts/Overweight_History.png) | ![](/Charts/Hypercaloric_Foods.png)|
|:-:|:-:|:-:|
| ![](/Charts/Main_Meals_Number.png) | ![](/Charts/Physical_Activity.png) | ![](/Charts/Drink_Water.png)|
|![](/Charts/Alcoholic_Drinks.png)|![](/Charts/Time_Spent_on_Technologies.png)|![](/Charts/Type_of_Transport_Used.png)|

Through charts it is can see:

- The genders are balanced but it can be noted that the group classified as Obesity Type II (6), there is a predominance of the male gender (1) and the group classified as Obesity Type III (7), there is a predominance of the female gender (2);
- Most of the sample has a history of overweight in the family (2);
- Majority consume hypercaloric foods (2);
- Most eat three main meals;
- Most people with the highest level of obesity do not practice physical activity (0);
- The vast majority drink two liters of water a day;
- Most drink alcohol sometimes (2);
- The minority spends more than 5 hours a day using technologies (2);
- Most of the group use public transport (3);
### _Relationships Between Variables_

![](/Charts/heatmap.png)

Through the Heatmap it can see a strong correlation between the variables:
- Body Mass Index x Weight;
- Body Mass Index x Overweight History;
- Type of Transport Used x Age;

### _Outlier Detection_

One of the most important parts of Exploratory Data Analysis (EAD) is the detection of outliers because if outliers are not studied at this stage, they can affect statistical modeling and Machine Learning. In statistics, an outlier is an observation point that is distant from other observations. Outlier may be due only to the variability in the measurement or it may indicate experimental errors. Machine Learning algorithms are sensitive to the range and distribution of attribute values. An outlier can be detected by visualization techniques such as box plot or with mathematical functions such as: Z-score and IQR score. 

I used box plot for data visualization. And it can be seen that there is a high number of outliers in the "Age" column and outliers have also been detected in other columns.

![](/Charts/boxplot.png)

Analyzing the Age column with the function _value_counts()_, I saw that there are no such discrepant ages to be removed. I also analyzed the Weight column and did not see values that justified delete.
### _Regression_

In the first predictive method of the project the Linear Regression algorithm was applied to make the model for predicting the level of Body Mass Index according to Weight. It seems obvious to choose the Weight variable to create the Linear Regression model for the Body Mass Index, but I chose it due to the strong relationship, through the _Heatmap_ chart, and to analyze the OLS Regression Results.

The model created was: 
```
The model is: Body Mass Index = -1.8815 + 0.069223X
```
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






