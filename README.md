# Obesity Estimate

## **Case Study**

The data belongs to a study made to estimate obesity levels in individuals from Mexico, Peru and Colombia, based on their eating habits and physical condition. The data set contains 17 features and 2111 rows. The data analysis project is in progress.

The data were taken from the UCI Machine Learning Repository. Follow the link:  
https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+#

## **Load Data**

The Pandas library was used to load and manipulate the data. When loading the data, it can be seen that there are ordinal and nominal categorical variables and numerical variables with different orders of magnitude.
```
      Gender        Age    Height      Weight  ...       TUE        CALC                 MTRANS           NObeyesdad
0     Female  21.000000  1.620000   64.000000  ...  1.000000          no  Public_Transportation        Normal_Weight
1     Female  21.000000  1.520000   56.000000  ...  0.000000   Sometimes  Public_Transportation        Normal_Weight
2       Male  23.000000  1.800000   77.000000  ...  1.000000  Frequently  Public_Transportation        Normal_Weight
3       Male  27.000000  1.800000   87.000000  ...  0.000000  Frequently                Walking   Overweight_Level_I
4       Male  22.000000  1.780000   89.800000  ...  0.000000   Sometimes  Public_Transportation  Overweight_Level_II
...      ...        ...       ...         ...  ...       ...         ...                    ...                  ...
2106  Female  20.976842  1.710730  131.408528  ...  0.906247   Sometimes  Public_Transportation     Obesity_Type_III
2107  Female  21.982942  1.748584  133.742943  ...  0.599270   Sometimes  Public_Transportation     Obesity_Type_III
2108  Female  22.524036  1.752206  133.689352  ...  0.646288   Sometimes  Public_Transportation     Obesity_Type_III
2109  Female  24.361936  1.739450  133.346641  ...  0.586035   Sometimes  Public_Transportation     Obesity_Type_III
2110  Female  23.664709  1.738836  133.472641  ...  0.714137   Sometimes  Public_Transportation     Obesity_Type_III

[2111 rows x 17 columns]
```

## **Exploratory Data Analysis (EDA)**
### _Data Processing_

One of the most time-consuming steps when working with Machine Learning models is data processing. It is also essential to understand the conclusions that can be drawn from the data. 

Analyzing the _sccater plot_ it can be seen the variation in the frequency of physical activity per week by the groups classified as "Insufficient Weight", "Normal Weight", "Overweight Lvel I", "Overweight Level II" and "Obesity Type I" and the increase in weight as the classification increases. And the higher weight can be noted in the groups classified as "Obesity Type II" and "Obesity Type III", as well as not performing physical activity more than twice a week by these groups.

![](/Graphics/scatter_weight_faf.png)

Through the _info()_ function it is verified that there are not no missing values and the data types.

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2111 entries, 0 to 2110
Data columns (total 17 columns):
 #   Column                          Non-Null Count  Dtype
---  ------                          --------------  -----
 0   Gender                          2111 non-null   object
 1   Age                             2111 non-null   float64
 2   Height                          2111 non-null   float64
 3   Weight                          2111 non-null   float64
 4   family_history_with_overweight  2111 non-null   object
 5   FAVC                            2111 non-null   object
 6   FCVC                            2111 non-null   float64
 7   NCP                             2111 non-null   float64
 8   CAEC                            2111 non-null   object
 9   SMOKE                           2111 non-null   object
 10  CH2O                            2111 non-null   float64
 11  SCC                             2111 non-null   object
 12  FAF                             2111 non-null   float64
 13  TUE                             2111 non-null   float64
 14  CALC                            2111 non-null   object
 15  MTRANS                          2111 non-null   object
 16  NObeyesdad                      2111 non-null   object
dtypes: float64(8), object(9)
memory usage: 206.2+ KB
```

There are columns with object and floating data. First, treat the object type data with the _category_encoders_ method, which uses a dictionary to determine the order of the attributes. The _Label Encoder_ method could also be used, which is indicated for ordinal categorical variables, which is the case of the "CAEC", "CALC" and "NObeyesdad" columns. However, as this method assigns the order of values through the alphabetical order of the classes, it did not return an expected order. I could also use the _One Hot Encoding_ method or _get_dummies_ from the Pandas library but as my number of columns would increase, I chose not to use it.

```
      Gender        Age    Height      Weight  family_history_with_overweight  ...       FAF       TUE  CALC  MTRANS  NObeyesdad
0          2  21.000000  1.620000   64.000000                               2  ...  0.000000  1.000000     1       3
2
1          2  21.000000  1.520000   56.000000                               2  ...  3.000000  0.000000     2       3
2
2          1  23.000000  1.800000   77.000000                               2  ...  2.000000  1.000000     3       3
2
3          1  27.000000  1.800000   87.000000                               1  ...  2.000000  0.000000     3       2
3
4          1  22.000000  1.780000   89.800000                               1  ...  0.000000  0.000000     2       3
4
...      ...        ...       ...         ...                             ...  ...       ...       ...   ...     ...         ...
2106       2  20.976842  1.710730  131.408528                               2  ...  1.676269  0.906247     2       3
7
2107       2  21.982942  1.748584  133.742943                               2  ...  1.341390  0.599270     2       3
7
2108       2  22.524036  1.752206  133.689352                               2  ...  1.414209  0.646288     2       3
7
2109       2  24.361936  1.739450  133.346641                               2  ...  1.139107  0.586035     2       3
7
2110       2  23.664709  1.738836  133.472641                               2  ...  1.026452  0.714137     2       3
7

[2111 rows x 17 columns]  
```

For the other floating type columns, I chose to round the values using the _round()_ function. But I kept the column "Height" with the floating type.

```
   Gender  Age  Height  Weight  family_history_with_overweight  FAVC  FCVC  ...  CH2O  SCC  FAF  TUE  CALC  MTRANS  NObeyesdad
0          2   21     1.6      64                               2     1     2  ...     2    1    0    1     1       3
2
1          2   21     1.5      56                               2     1     3  ...     3    2    3    0     2       3
2
2          1   23     1.8      77                               2     1     2  ...     2    1    2    1     3       3
2
3          1   27     1.8      87                               1     1     3  ...     2    1    2    0     3       2
3
4          1   22     1.8      90                               1     1     2  ...     2    1    0    0     2       3
4
...      ...  ...     ...     ...                             ...   ...   ...  ...   ...  ...  ...  ...   ...     ...         ...
2106       2   21     1.7     131                               2     2     3  ...     2    1    2    1     2       3
7
2107       2   22     1.7     134                               2     2     3  ...     2    1    1    1     2       3
7
2108       2   23     1.8     134                               2     2     3  ...     2    1    1    1     2       3
7
2109       2   24     1.7     133                               2     2     3  ...     3    1    1    1     2       3
7
2110       2   24     1.7     133                               2     2     3  ...     3    1    1    1     2       3
7

[2111 rows x 17 columns]
```

### _Descriptive Analysis_

Time to briefly describe the content of the data!

![](/Graphics/distplot.png)

1. **Gender**: the genders are balanced;
2. **Age**: there is a predominance in the age group below 30;
3. **Height**: heights are concentrated in the range of 1.60 to 1.80 meters;
4. **Weight**: a greater number of people weighing 80 kg;
5. **Family** history with overweight: a predominance of cases with a family history of overweight;
6. **FAVC**: predominance of sample with frequent consumption of hypercaloric foods;
7. **FCVC**: most of the sample answered that they consume vegetables sometimes and always;
8. **NCP**: predominance of three main meals;
9. **CAEC**: most of the sample sometimes consumes food between main meals;
10. **SMOKE**: predominance of nonsmokers;
11.	**CH2O**: most of the sample consumes between 1 and 2 liters of water per day;
12. **SCC**: there is no calorie monitoring by most of the sample;
13.	**FAF**: most of the sample does physical activity 1 to 2 times a week, followed by the second most who do not perform physical activity;
14. **TUE**: most of the sample spends up to two hours a day using technologies, followed by the second portion that spends three to five hours a day;
15. **CALC**: most of the sample sometimes drinks alcohol;
16.	**MTRANS**: most of the sample uses public transport;
17. **NObeyesdad** (body mass index): most of the sample was with some level of obesity;

The _describe()_ function returns descriptive statistics values, as follows:

```
            Gender          Age       Height       Weight  ...          TUE         CALC       MTRANS   NObeyesdad
count  2111.000000  2111.000000  2111.000000  2111.000000  ...  2111.000000  2111.000000  2111.000000  2111.000000
mean      1.494079    24.315964     1.701847    86.586452  ...     0.664614     1.731407     3.405021     4.112269
std       0.500083     6.357078     0.100385    26.190136  ...     0.674009     0.515498     0.864439     1.985062
min       1.000000    14.000000     1.400000    39.000000  ...     0.000000     1.000000     1.000000     1.000000
25%       1.000000    20.000000     1.600000    65.500000  ...     0.000000     1.000000     3.000000     2.000000
50%       1.000000    23.000000     1.700000    83.000000  ...     1.000000     2.000000     3.000000     4.000000
75%       2.000000    26.000000     1.800000   107.000000  ...     1.000000     2.000000     3.000000     6.000000
max       2.000000    61.000000     2.000000   173.000000  ...     2.000000     4.000000     5.000000     7.000000

[8 rows x 17 columns]
```

#### _Central Trend Measures_

1. Mean: indicates where the values are concentrated. Focusing on the "Age", "Height" and "Weight" columns, it can be noted that the sample average is 24 years old, 1.70 m and 86 kg.

2. Median and Quantil: the median is the value that separates the top half from the bottom half of a data distribution, or the value at the center of the distribution. The median is a concept less susceptible to large outliers than the average. The 50% quantile is a median by default. Focusing on the "Age", "Height" and "Weight" columns, it can be seen that the median values are 23 years old, 1.70 m and 83 kg.

3. Mode: is the most repeated value in the data. Through the _mode()_ function you can see that it is 21 yeras old, 1,70 m and 80 kg.
```
   Gender  Age  Height  Weight  family_history_with_overweight  FAVC  FCVC  NCP  ...  SMOKE  CH2O  SCC  FAF  TUE  CALC  MTRANS  NObeyesdad
0       1   21     1.7      80                               2     2     2    3  ...      1     2    1    1    0     2       3           5 

[1 rows x 17 columns]
```

#### _Dispersion Measures_

1. Amplitude: difference between the highest and lowest value of the data set. There is a greater variation in the "Weight" column.
- Age: 47;
- Height: 0.6;
- Weight: 134;

2. Variance: expressed as the data of a set are far from its expected value. The variance of the "Weight" column is too high.
```
Gender                              0.250083
Age                                40.412442
Height                              0.010077
Weight                            685.923210
family_history_with_overweight      0.149187
FAVC                                0.102638
FCVC                                0.340945
NCP                                 0.655582
CAEC                                0.219533
SMOKE                               0.020418
CH2O                                0.474192
SCC                                 0.043429
FAF                                 0.801852
TUE                                 0.454288
CALC                                0.265738
MTRANS                              0.747254
NObeyesdad                          3.940470
dtype: float64
```

3. Standard deviation: indicates how far the data is away from the mean. Again indicating greater attention to the "Age" and "Weight" columns.
- Age: 6.3;
- Height: 0.1;
- Weight: 26.2;

### _Relationships Between Variables_

Through the seaborn heatmap the relationships between the variables can be analyzed. It shows us a strong relationship between the variables "Weight" and "NObeyesdad" and a considerable relationship between "Age" and "MTRANS".

![](/Graphics/heatmap.png)

### _Outlier Detection_

One of the most important parts of Exploratory Data Analysis (EAD) is the detection of outliers because if outliers are not studied at this stage, they can affect statistical modeling and Machine Learning. In statistics, an outlier is an observation point that is distant from other observations. Outlier may be due only to the variability in the measurement or it may indicate experimental errors. Machine Learning algorithms are sensitive to the range and distribution of attribute values. An outlier can be detected by visualization techniques such as box plot or with mathematical functions such as: Z-score and IQR score. 

I used box plot for data visualization. And it can be seen that there is a high number of outliers in the "Age" column.

![](/Graphics/boxplot.png)

### _IQR_

Before starting Machine Learning, I treated the outliers of the "Weight" and "Age" columns using the technique based on the interquartile range (IQR) that to identify the upper and lower limits of the data and removing values above and below the limits. 

Now no more outliers are now seen in the box plot chart:
| ![](/Graphics/boxplot_age.png) | ![](/Graphics/boxplot_weight.png)|
|:-:|:-:|

#### _Linear Regression_

The model created was: 
```
The model is: NObeyesdad = -1.9888 + 0.070014X
```

There is a positive correlation between the predictive variable "Weight" and the predicted variable "NObeyesdad":
![](/Graphics/LinearRegressionModel.png)

The quality of the model was assessed using the "R²" and the "p-value":
```
 OLS Regression Results
==============================================================================
Dep. Variable:             NObeyesdad   R-squared:                       0.847
Model:                            OLS   Adj. R-squared:                  0.847
Method:                 Least Squares   F-statistic:                 1.076e+04
Date:                Mon, 08 Feb 2021   Prob (F-statistic):               0.00
Time:                        13:11:35   Log-Likelihood:                -2324.9
No. Observations:                1950   AIC:                             4654.
Df Residuals:                    1948   BIC:                             4665.
Df Model:                           1
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const         -1.9888      0.061    -32.505      0.000      -2.109      -1.869
Weight         0.0700      0.001    103.709      0.000       0.069       0.071
==============================================================================
Omnibus:                       38.839   Durbin-Watson:                   0.973
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               34.987
Skew:                           0.276   Prob(JB):                     2.53e-08
Kurtosis:                       2.645   Cond. No.                         307.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```
R² is 0.847, this means that approximately 80% of the behavior of the variable "NObeyesdad" is explained by the variable "Weight".

With P> | t | very low, the Null Hypothesis is rejected. The Null Hypothesis means that there is no correlation between the predicted and predictive variables, that is, for a model to work, it must be false. In general, if the "p-value" is less than 0.05, there is a strong relationship between the variables.

##### _Performance Metrics_

The error generated by the model was calculated using Mean Absolute Arror (MAE) and Mean Absolute Percentage Error (MAPE) returning the following results:
```
MAE_test: 0.3720322818993708
MAPE_test: 0.1309782290989399
```
###### _Cross-Validation_

The result of the _Cross-Validation_ metric was also satisfactory:
```
0.95 accuracy with a standard deviation of 0.01
```
#### _Clustering_

1. _Centroids-based clustering: K-Means_

In this step of the project, the groups formed by frequency of physical activity ("FAF") and level of body mass index ("NObeyesdad") columns of the data set were analysed. 

##### _Elbow Method_

The _Elbow Method_ was used to find out the number of clusters that was used in KMeans.

![](/Graphics/ElbowMethod.png)

According to the graph I could choose the value of k between 6 and 8. I chose k = 7.

![](/Graphics/kmeans.png)

It can be seen that the most worrying group in the sample is Cluster 4, which has the highest rates of obesity and does not practice physical activity. There are 256 people in the purple cluster, as shown below:
```
      Gender        Age    Height      Weight  ...        CALC                 MTRANS        NObeyesdad  Cluster
68      Male  30.000000  1.760000  112.000000  ...  Frequently             Automobile   Obesity_Type_II        4
197     Male  41.000000  1.750000  118.000000  ...   Sometimes                   Bike   Obesity_Type_II        4
202   Female  26.000000  1.560000  102.000000  ...   Sometimes  Public_Transportation  Obesity_Type_III        4
210     Male  20.000000  1.800000  114.000000  ...          no  Public_Transportation   Obesity_Type_II        4
229     Male  32.000000  1.750000  120.000000  ...          no             Automobile   Obesity_Type_II        4
...      ...        ...       ...         ...  ...         ...                    ...               ...      ...
2098  Female  25.992348  1.606474  104.954291  ...   Sometimes  Public_Transportation  Obesity_Type_III        4
2099  Female  25.974446  1.628855  108.090006  ...   Sometimes  Public_Transportation  Obesity_Type_III        4
2100  Female  25.777565  1.628205  107.378702  ...   Sometimes  Public_Transportation  Obesity_Type_III        4
2101  Female  25.722004  1.628470  107.218949  ...   Sometimes  Public_Transportation  Obesity_Type_III        4
2102  Female  25.765628  1.627839  108.107360  ...   Sometimes  Public_Transportation  Obesity_Type_III        4

[256 rows x 18 columns]
```
##### _Performance Metrics_

Through the _silhouette_score_ metrics, there was a return of approximately 0.5 which can be considered an acceptable performance of KMeans.
```
The Silhouette_Score of K-means is: 0.4979439571648936
```
2. _Connectivity clustering: Agglomerative Hierarchical_

To know the number of clusters that was used in this algorithm, I applied the _Dendrogram_ method.

##### _Dendrogram_

![](/Graphics/Dendrogram_2.png)

The number of clusters used was 2.

![](/Graphics/Hierarchical_2.png)

According to the graph, the risk group in the sample is the blue cluster or zero cluster. This cluster represents 1342 people, as shown below:
```
      Gender        Age    Height      Weight  ...        CALC                 MTRANS           NObeyesdad  Cluster
3       Male  27.000000  1.800000   87.000000  ...  Frequently                Walking   Overweight_Level_I        0
4       Male  22.000000  1.780000   89.800000  ...   Sometimes  Public_Transportation  Overweight_Level_II        0
10      Male  26.000000  1.850000  105.000000  ...   Sometimes  Public_Transportation       Obesity_Type_I        0
11    Female  21.000000  1.720000   80.000000  ...   Sometimes  Public_Transportation  Overweight_Level_II        0
13      Male  41.000000  1.800000   99.000000  ...  Frequently             Automobile       Obesity_Type_I        0
...      ...        ...       ...         ...  ...         ...                    ...                  ...      ...
2106  Female  20.976842  1.710730  131.408528  ...   Sometimes  Public_Transportation     Obesity_Type_III        0
2107  Female  21.982942  1.748584  133.742943  ...   Sometimes  Public_Transportation     Obesity_Type_III        0
2108  Female  22.524036  1.752206  133.689352  ...   Sometimes  Public_Transportation     Obesity_Type_III        0
2109  Female  24.361936  1.739450  133.346641  ...   Sometimes  Public_Transportation     Obesity_Type_III        0
2110  Female  23.664709  1.738836  133.472641  ...   Sometimes  Public_Transportation     Obesity_Type_III        0

[1342 rows x 18 columns]
```
##### _Performance Metrics_

Through the _silhouette_score_ metrics, there was a return of approximately 0.5 which can be considered an acceptable performance of Agglomerative Hierarchical.
```
The Silhouette_Score of Hierarchical is: 0.4907216842031665
```





