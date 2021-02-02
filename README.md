# Obesity Estimate

## **Case Study**

The data belongs to a study made to estimate obesity levels in individuals from Mexico, Peru and Colombia, based on their eating habits and physical condition. The data set contains 17 attributes and 2111 features. The data analysis project is in progress.

The data were taken from the UCI Machine Learning Repository. Follow the link:  
https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+#

## **Load Data**

Load data with the Pandas library. There are categories variables and numeric variables. Also there are variables with different orders of magnitude.

## **Exploratory Data Analysis (EDA)**

### _Data Processing_

One of the most time-consuming steps when working with Machine Learning models is data processing. It is also essential to understand the conclusions that can be drawn from the data. Through the _info()_ function it is verified that there are not no missing values and the data types.

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

There are columns with object and floating data. First, treat the object type data with the _category_encoders_ method, which uses a dictionary to determine the order of the attributes. The _Label Encoder_ method could also be used, which is indicated for ordinal categorical variables, which is the case of the "CAEC", "CALC" and "NObeyesdad" columns. However, as this method assigns the order of values through the alphabetical order of the classes, it did not return an expected order.

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

For the other floating type columns, I chose to round the values using the _round()_ function. I could also use the _One Hot Encoding_ method or _get_dummies_ from the Pandas library but as my number of columns would increase, I chose not to use it.

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






