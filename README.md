# Obesity Estimate

## **Case Study**

The data belongs to a study made to estimate obesity levels in individuals from Mexico, Peru and Colombia, based on their eating habits and physical condition. The data set contains 17 features and 2111 rows. The data analysis project is in progress.

The data were taken from the UCI Machine Learning Repository. Follow the link:  
https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+#

## **Load Data**

When loading data with the Pandas library, it is noted that there are categorical variables and numeric variables. In addition, there are variables with different orders of magnitude.

## **Exploratory Data Analysis (EDA)**
### _Data Processing_

One of the most time-consuming steps when working with Machine Learning models is data processing. It is also essential to understand the conclusions that can be drawn from the data. 

Analyzing the _sccater plot_ I can see the variation in the frequency of physical activity per week by the groups classified as "Insufficient Weight", "Normal Weight", "Overweight Lvel I", "Overweight Level II" and "Obesity Type I" and the increase in weight as the classification increases. And the higher weight can be noted in the groups classified as "Obesity Type II" and "Obesity Type III", as well as not performing physical activity more than twice a week by these groups.

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
      Gender        Age    Height      Weight  family_history_with_overweight  FAVC  FCVC  NCP  CAEC  SMOKE      CH2O  SCC       FAF       TUE  CALC  MTRANS  NObeyesdad
0          1  21.000000  1.620000   64.000000                               1     0   2.0  3.0     1      0  2.000000    0  0.000000  1.000000     0       2           1
1          1  21.000000  1.520000   56.000000                               1     0   3.0  3.0     1      1  3.000000    1  3.000000  0.000000     1       2           1
2          0  23.000000  1.800000   77.000000                               1     0   2.0  3.0     1      0  2.000000    0  2.000000  1.000000     2       2           1
3          0  27.000000  1.800000   87.000000                               0     0   3.0  3.0     1      0  2.000000    0  2.000000  0.000000     2       1           2
...      ...        ...       ...         ...                             ...   ...   ...  ...   ...    ...       ...  ...       ...       ...   ...     ...         ...
2106       1  20.976842  1.710730  131.408528                               1     1   3.0  3.0     1      0  1.728139    0  1.676269  0.906247     1       2           6
2107       1  21.982942  1.748584  133.742943                               1     1   3.0  3.0     1      0  2.005130    0  1.341390  0.599270     1       2           6
2108       1  22.524036  1.752206  133.689352                               1     1   3.0  3.0     1      0  2.054193    0  1.414209  0.646288     1       2           6
2109       1  24.361936  1.739450  133.346641                               1     1   3.0  3.0     1      0  2.852339    0  1.139107  0.586035     1       2           6
2110       1  23.664709  1.738836  133.472641                               1     1   3.0  3.0     1      0  2.863513    0  1.026452  0.714137     1       2           6

[2111 rows x 17 columns]
```

For the other floating type columns, I chose to round the values using the _round()_ function. But I kept the column "Height" with the floating type.

```
      Gender  Age  Height  Weight  family_history_with_overweight  FAVC  FCVC  NCP  CAEC  SMOKE  CH2O  SCC  FAF  TUE  CALC  MTRANS  NObeyesdad
0          1   21     1.6      64                               1     0     2    3     1      0     2    0    0    1     0       2           1
1          1   21     1.5      56                               1     0     3    3     1      1     3    1    3    0     1       2           1
2          0   23     1.8      77                               1     0     2    3     1      0     2    0    2    1     2       2           1
3          0   27     1.8      87                               0     0     3    3     1      0     2    0    2    0     2       1           2
4          0   22     1.8      90                               0     0     2    1     1      0     2    0    0    0     1       2           3
...      ...  ...     ...     ...                             ...   ...   ...  ...   ...    ...   ...  ...  ...  ...   ...     ...         ...
2106       1   21     1.7     131                               1     1     3    3     1      0     2    0    2    1     1       2           6
2107       1   22     1.7     134                               1     1     3    3     1      0     2    0    1    1     1       2           6
2108       1   23     1.8     134                               1     1     3    3     1      0     2    0    1    1     1       2           6
2109       1   24     1.7     133                               1     1     3    3     1      0     3    0    1    1     1       2           6
2110       1   24     1.7     133                               1     1     3    3     1      0     3    0    1    1     1       2           6

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
            Gender          Age       Height       Weight  family_history_with_overweight         FAVC  ...          SCC          FAF          TUE         CALC       MTRANS   NObeyesdad
count  2111.000000  2111.000000  2111.000000  2111.000000                     2111.000000  2111.000000  ...  2111.000000  2111.000000  2111.000000  2111.000000  2111.000000  2111.000000
mean      0.494079    24.315964     1.701847    86.586452                        0.817622     0.883941  ...     0.045476     1.006632     0.664614     0.731407     2.405021     3.112269
std       0.500083     6.357078     0.100385    26.190136                        0.386247     0.320371  ...     0.208395     0.895462     0.674009     0.515498     0.864439     1.985062
min       0.000000    14.000000     1.400000    39.000000                        0.000000     0.000000  ...     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
25%       0.000000    20.000000     1.600000    65.500000                        1.000000     1.000000  ...     0.000000     0.000000     0.000000     0.000000     2.000000     1.000000
50%       0.000000    23.000000     1.700000    83.000000                        1.000000     1.000000  ...     0.000000     1.000000     1.000000     1.000000     2.000000     3.000000
75%       1.000000    26.000000     1.800000   107.000000                        1.000000     1.000000  ...     0.000000     2.000000     1.000000     1.000000     2.000000     5.000000
max       1.000000    61.000000     2.000000   173.000000                        1.000000     1.000000  ...     1.000000     3.000000     2.000000     3.000000     4.000000     6.000000

[8 rows x 17 columns]
```

#### _Central Trend Measures_

1. Mean: indicates where the values are concentrated. Focusing on the "Age", "Height" and "Weight" columns, it can be noted that the sample average is 24 years old, 1.70 m and 86 kg.

2. Median and Quantil: the median is the value that separates the top half from the bottom half of a data distribution, or the value at the center of the distribution. The median is a concept less susceptible to large outliers than the average. The 50% quantile is a median by default. Focusing on the "Age", "Height" and "Weight" columns, it can be seen that the median values are 23 years old, 1.70 m and 83 kg.

3. Mode: is the most repeated value in the data. Through the _mode()_ function you can see that it is 21 yeras old, 1,70 m and 80 kg.
```
   Gender  Age  Height  Weight  family_history_with_overweight  FAVC  FCVC  NCP  CAEC  SMOKE  CH2O  SCC  FAF  TUE  CALC  MTRANS  NObeyesdad
0       0   21     1.7      80                               1     1     2    3     1      0     2    0    1    0     1       2           4
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

One of the most important parts of Exploratory Data Analysis (EAD) is the detection of outliers because if outliers are not studied at this stage, they can affect statistical modeling and Machine Learning. In statistics, an outlier is an observation point that is distant from other observations. Outlier may be due only to the variability in the measurement or it may indicate experimental errors. Machine learning algorithms are sensitive to the range and distribution of attribute values. An outlier can be detected by visualization techniques such as box plot or with mathematical functions such as: Z-score and IQR score. 

I used box plot for data visualization. And it can be seen that there is a high number of outliers in the "Age" column.

![](/Graphics/boxplot.png)

### _Scaling of Data_

To enter the Machine Learning, data needs to be prepared. And what remains to start modeling the data is to do is to place the data in the same order of magnitude. For that, I used the _QuantileTransformer()_ method because it also deals with outliers. This method transforms the values in such a way that the distribution tends to approximate a normal distribution.

```
 0            1            2            3            4            5            6   ...           10           11           12           13           14           15           16
count  2111.000000  2111.000000  2111.000000  2111.000000  2111.000000  2111.000000  2111.000000  ...  2111.000000  2111.000000  2111.000000  2111.000000  2111.000000  2111.000000  2111.000000
mean      0.494079     0.499981     0.500021     0.499999     0.817622     0.883941     0.610154  ...     0.503395     0.045476     0.443339     0.404981     0.454255     0.523298     0.503387
std       0.500083     0.288009     0.277159     0.288820     0.386247     0.320371     0.373462  ...     0.344413     0.208395     0.348496     0.380610     0.305795     0.260697     0.316702
min       0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000  ...     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
25%       0.000000     0.259259     0.192192     0.247247     1.000000     1.000000     0.288288  ...     0.492492     0.000000     0.000000     0.000000     0.000000     0.403904     0.196697
50%       0.000000     0.535536     0.491491     0.501502     1.000000     1.000000     0.288288  ...     0.492492     0.000000     0.524525     0.667668     0.634635     0.403904     0.470470
75%       1.000000     0.721221     0.798799     0.747748     1.000000     1.000000     1.000000  ...     0.492492     0.000000     0.826326     0.667668     0.634635     0.403904     0.776276
max       1.000000     1.000000     1.000000     1.000000     1.000000     1.000000     1.000000  ...     1.000000     1.000000     1.000000     1.000000     1.000000     1.000000     1.000000

[8 rows x 17 columns]
```
### _Model Training_

#### _Clustering_

1. Centroids-based clustering: K-Means.

When applying the _Elbow Method_, the data does not form such a sharp elbow in the graph. But there is a significant change in angle from 5 clusters. I defined the number of clusters 8.

![](/Graphics/NumberOfClusters_KMeans.png)

The good thing about K-means is that it is a fast method because it does not perform many calculations. However, it is difficult to identify and classify groups. As it starts with a random choice of cluster centers, the results can be inconsistent.
Analyzing the columns "NObeyesdad"(body mass index) and "FAF"(frequency of physical activity) cannot find a pattern.

| ![](/Graphics/relplot_NObeyesdad.png) | ![](/Graphics/relplot_FAF.png) |
|:-:|:-:|

To check how good our cluster is, I used the Silhouette coefficient. The result was 0.17. As it is close to zero, I can say that the sample is very close to the neighboring clusters.

2. Density-based clustering: DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

DBSCAN frames the points according to the parameters: _eps_ and _min_samples_. In practice, it joins the points within a certain distance with a minimum number of samples to close the cluster.

Depending on the value you use for these parameters, the number of clusters can change, as shown in the graphs below. The first graph was used _eps_ = 1 and _min_samples_ = 20. The second graph was used _eps_ = .95 and _min_samples_ = 50. And the third graph was used _eps_ = .95 and _min_samples_ = 20.

| ![](/Graphics/DBSCAN_eps1_min_sample20.png) | ![](/Graphics/DBSCAN_eps.95_min_sample50.png) | ![](/Graphics/DBSCAN_eps.95_min_sample20.png) |
|:-:|:-:|:-:|

In the first and third graphs, there is no trend in cluster behavior. In the first, it presented only one cluster and in the third it presented several clusters. In the second graph I can see a behavioral trend of the blue cluster, where it was better suited to the group that performs physical activities 4 to 5 days a week.

#### _IQR_

As can be seen in the Seaborn Heatmap graph, there is a strong correlation between the variables "Nobeyesdad" and "Weight". I applied the Linear Regression model to these variables. 

But in the box plot graph I can see that there is an outlier in the "Weight" variable. The Linear Regression algorithm is very sensitive to outliers. So before applying the model, I treated the outlier of the "Weight" variable. I used the IQR (Inter-Quartile Range) technique to identify the upper and lower limits of the data and removing values above and below the limits. 

No more outliers are now seen in the box plot chart:
![](/Graphics/boxplot_weight.png) 

#### _Linear Regression_

The model created was: 
```
The model is: NObeyesdad = -2.8961 + 0.069409X
```

There is a positive correlation between the predictive variable "Weight" and the predicted variable "NObeyesdad":
![](/Graphics/LinearRegression.png)

The quality of the model was assessed using the "R²" and the "p-value":
```
OLS Regression Results
==============================================================================
Dep. Variable:             NObeyesdad   R-squared:                       0.835
Model:                            OLS   Adj. R-squared:                  0.835
Method:                 Least Squares   F-statistic:                 1.068e+04
Date:                Thu, 04 Feb 2021   Prob (F-statistic):               0.00
Time:                        16:45:57   Log-Likelihood:                -2538.0
No. Observations:                2110   AIC:                             5080.
Df Residuals:                    2108   BIC:                             5091.
Df Model:                           1
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const         -2.8961      0.061    -47.690      0.000      -3.015      -2.777
Weight         0.0694      0.001    103.325      0.000       0.068       0.071
==============================================================================
Omnibus:                       49.734   Durbin-Watson:                   0.952
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               40.137
Skew:                           0.260   Prob(JB):                     1.92e-09
Kurtosis:                       2.568   Cond. No.                         313.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```
R² is 0.835, this means that approximately 80% of the behavior of the variable "NObeyesdad" is explained by the variable "Weight".

With P> | t | very low, probably because some data very close to zero was covered by rounding. So the null hypothesis is rejected. The null hypothesis means that there is no correlation between the predicted and predictive variables, that is, for a model to perform, it must be false. In general, if the "p-value" is less than 0.05, there is a strong relationship between the variables. 

It is also important to note the information given by "F-Statistics". This number shows that there is a very high variance in the data set.












