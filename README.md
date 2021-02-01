# Obesity Estimate

## **_Case Study_**

The data belongs to a study made to estimate obesity levels in individuals from Mexico, Peru and Colombia, based on their eating habits and physical condition. The data set contains 17 attributes and 2111 features. My data analysis project is in progress.

The data were taken from the UCI Machine Learning Repository. Follow the link:  
https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+#

## **Load Data**

I load the data with the library Pandas. There are categories variables and numeric variables. Also there are variables with different orders of magnitude.

## **Exploratory Data Analysis (EDA)**

### _Data Processing_

One of the most time-consuming steps when working with Machine Learning models is data processing. It is also essential to make sense of any conclusions I can draw from the data. Through the _info()_ function I see that there are not no missing values and the data types.

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

There are columns with object and floating data. First I treated the object type data with the _category_encoders_ method that uses a dictionary to determine the order of the attributes. You could use the Label Encoder method which is indicated for ordinal categorical variables, which is the case of the "CAEC", "CALC" and "NObeyesdad" columns. However, as this method assigns the order of values through the alphabetical order of the classes, it did not return an expected order.

For the other floating type columns, I chose to round the values using the _round()_ function. I could also use the One Hot Encoding method or get_dummies from the Pandas library but as my number of columns would increase, I chose not to use it.

### _Descriptive Analysis_

Time to briefly describe the content of the data!

1. Gender: the genders are balanced;
2. Age: there is a predominance in the age group below 30;
3. Height: heights are concentrated in the range of 1.60 to 1.80 meters;
4. Weight: a greater number of people weighing 80 kg;
5. Family history with overweight: a predominance of cases with a family history of overweight;
6. FAVC: predominance of sample with frequent consumption of hypercaloric foods;
7. FCVC: most of the sample answered that they consume vegetables sometimes and always;
8. NCP: predominance of three main meals;
9. CAEC: most of the sample sometimes consumes food between main meals;
10. SMOKE: predominance of nonsmokers;
11.	CH2O: most of the sample consumes between 1 and 2 liters of water per day;
12. SCC: there is no calorie monitoring by most of the sample;
13.	FAF: most of the sample does physical activity 1 to 2 times a week, followed by the second most who do not perform physical activity;
14. TUE: most of the sample spends up to two hours a day using technologies, followed by the second portion that spends three to five hours a day;
15. CALC: most of the sample sometimes drinks alcohol;
16.	MTRANS: most of the sample uses public transport;
17. NObeyesdad (body mass index): most of the sample was with some level of obesity;

### _Relationships Between Variables_

Through the seaborn heatmap I can see the relationships between the variables. It shows us a strong relationship between the variables "Weight" and "NObeyesdad" and a considerable relationship between "Age" and "MTRANS".

### _Outlier Detection_

One of the most important parts of Exploratory Data Analysis (EAD) is the detection of outliers because if outliers are not studied at this stage, they can affect statistical modeling and Machine Learning. In statistics, an outlier is an observation point that is distant from other observations. Outlier may be due only to the variability in the measurement or it may indicate experimental errors. Machine learning algorithms are sensitive to the range and distribution of attribute values. An outlier can be detected by visualization techniques such as box plot or with mathematical functions such as: Z-score and IQR score. 

I used box plot for data visualization. And it can be seen that there is a high number of outliers in the "Age" column.






