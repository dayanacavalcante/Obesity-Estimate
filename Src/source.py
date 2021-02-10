# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

# Load Data

data = pd.read_csv('C:\\Users\\RenanSardinha\\Documents\\Data Science\\EstimatedObesityLevels\\Data\\ObesityDataSet_raw_and_data_sinthetic.csv')
print(data)

# EDA
## Data Processing

print(data.info())

"""
Plot Weight vs FAF 
Insufficient_Weight - 'red'; Normal_Weight - 'blue'; Overweight_Level_I - 'green'; 
Overweight_Level_II - 'yellow'; Obesity_Type_I - 'purple'; Obesity_Type_II - 'grey'; Obesity_Type_III - 'orange' 
"""

labels = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
colors = ['red', 'blue', 'green', 'yellow', 'purple', 'grey', 'orange']

filter = lambda type: data['NObeyesdad'] == type

plt.figure(figsize=(20,16))    
 
for n in range(len(labels)):
    plt.scatter(data['Weight'][filter(labels[n])], data['FAF'][filter(labels[n])], color = colors[n], label = labels[n])

plt.xlabel('Weight')
plt.ylabel('FAF')
plt.legend()

# Applying Ordinal Encoder 
encoder = ce.OrdinalEncoder(mapping = [{'col':'Gender','mapping':{'Male':1,'Female':2}},
                            {'col':'family_history_with_overweight','mapping':{'no':1,'yes':2}},
                            {'col':'FAVC','mapping':{'no':1,'yes':2}},
                            {'col':'CAEC','mapping':{'no':1,'Sometimes':2,'Frequently':3,'Always':4}},
                            {'col':'SMOKE','mapping':{'no':1,'yes':2}},
                            {'col':'SCC','mapping':{'no':1,'yes':2}},
                            {'col':'CALC','mapping':{'no':1,'Sometimes':2,'Frequently':3,'Always':4}},
                            {'col':'MTRANS','mapping':{'Bike':1,'Walking':2,'Public_Transportation':3,'Motorbike':4,'Automobile':5}},
                            {'col':'NObeyesdad','mapping':{'Insufficient_Weight':1,'Normal_Weight':2,'Overweight_Level_I':3,'Overweight_Level_II':4,'Obesity_Type_I':5,'Obesity_Type_II':6,'Obesity_Type_III':7}}])


encoder.fit(data)
encoder.transform(data)
data_encoder = encoder.fit_transform(data)
print(data_encoder)

data_encoder['Age'] = data_encoder['Age'].apply(lambda age : round(age)) 
data_encoder['Height'] = data_encoder['Height'].apply(lambda height : round(height, 1)) 
data_encoder['Weight'] = data_encoder['Weight'].apply(lambda weight : round(weight)) 
data_encoder['FCVC'] = data_encoder['FCVC'].apply(lambda fcvc : round(fcvc)) 
data_encoder['NCP'] = data_encoder['NCP'].apply(lambda ncp : round(ncp)) 
data_encoder['CH2O'] = data_encoder['CH2O'].apply(lambda ch2o : round(ch2o)) 
data_encoder['FAF'] = data_encoder['FAF'].apply(lambda faf : round(faf)) 
data_encoder['TUE'] = data_encoder['TUE'].apply(lambda tue : round(tue))

print(data_encoder)
print(data_encoder['NObeyesdad'])

## Descriptive Analysis

plt.figure(figsize=(20,16)) 

for i,col in enumerate(list(data_encoder.columns.values)):
    plt.subplot(5,4,i+1)
    sns.distplot(data_encoder[col], color='b', kde= 0, label='data_encoder')
    plt.grid()
    plt.tight_layout()

print(data_encoder.describe())
print(data_encoder.mode())
print(data_encoder.var())

## Relationships Between Variables

plt.figure(figsize=(15,15))
sns.heatmap(data_encoder.corr(), color='b', annot=True)

## Outliers Detection

plt.figure(figsize=(10,15))

for i, col in enumerate(list(data_encoder.columns.values)):
    plt.subplot(5,4,i+1)
    data_encoder.boxplot(col)
    plt.grid()
    plt.tight_layout()

# IQR
# Weight
Q1_w, Q3_w = np.percentile(data_encoder.Weight, [25,75])
print(Q1_w,Q3_w)

IQR_w = Q3_w - Q1_w
print(IQR_w)

# Lower Range
low_range_w = Q1_w - (1.5*IQR_w)
print(low_range_w)
print('The Wheight low range is: {}'.format(low_range_w))

# Upper Range
upper_range_w = Q3_w + (1.5*IQR_w)
print(upper_range_w)
print('The Wheight upper range is: {}'.format(upper_range_w))

# Age
Q1_a, Q3_a = np.percentile(data_encoder.Age, [25,75])
print(Q1_a,Q3_a)

IQR_a = Q3_a - Q1_a
print(IQR_a)

# Lower Range
low_range_a = Q1_a - (1.5*IQR_a)
print('The Age low range is: {}'.format(low_range_a))

# Upper Range
upper_range_a = Q3_a + (1.5*IQR_a)
print(upper_range_a)
print('The Age upper range is: {}'.format(upper_range_a))

data_iqr = data_encoder.copy()

data_iqr.drop(data_iqr[(data_iqr.Weight > 169.25) | (data_iqr.Weight < 3.25)].index, inplace = True)
data_iqr.drop(data_iqr[(data_iqr.Age > 35.0) | (data_iqr.Age < 11.0)].index, inplace = True)

print(data_iqr)

plt.figure(figsize=(8,6))
sns.boxplot(y = "Weight", data = data_iqr)
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(y = "Age", data = data_iqr)
plt.show()

# Creating the Linear Regression Model with "Weight" and "NObeyesdad" columns.

X = data_iqr['Weight'].values.reshape(-1,1)
y = data_iqr['NObeyesdad'].values.reshape(-1,1)

reg = LinearRegression()
reg.fit(X,y)

print("The model is: NObeyesdad = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))

pred = reg.predict(X)

plt.figure(figsize = (16,8))
plt.scatter(
    data_iqr['Weight'],
    data_iqr['NObeyesdad'],
    c = 'blue')

plt.plot(
    data_iqr['Weight'],
    pred,
    c = 'red',
    linewidth = 3,
    linestyle = ':')

plt.xlabel('Weight')
plt.ylabel('NObeyesdad')
plt.show()

Xols = data_iqr['Weight']
yols = data_iqr['NObeyesdad']
X2 = sm.add_constant(Xols)
est = sm.OLS(yols,X2)
est2 = est.fit()
print(est2.summary())

# Training the model with the data set

Xlr = data_iqr.drop(['NObeyesdad'], axis = 1)
ylr = data_iqr['NObeyesdad']

# Separating training and test data

x_train, x_test, y_train, y_test = train_test_split(Xlr,ylr,test_size = .3, random_state=1)

print(x_train)
print(y_train)

# Model Training

lr = LinearRegression()
lr.fit(x_train,y_train)

pred_train = lr.predict(x_train)
pred_test = lr.predict(x_test)

# Performance Metrics

mae_test = mean_absolute_error(y_test, pred_test)
mape_test = mean_absolute_percentage_error(y_test, pred_test)
print('MAE_test: {}'.format(mae_test))
print('MAPE_test: {}'.format(mape_test))

# Cross Validation

result_cv = cross_val_score(lr, x_test, y_test, cv = 10)
print('Cross Validation: {}'.format(result_cv))
print("%0.2f accuracy with a standard deviation of %0.2f" % (result_cv.mean(), result_cv.std()))

# Clustering: K-means

Xc = data_encoder.iloc[:, [12,16]].values

print(Xc)

# Elbow Method

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(Xc)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.xlabel('Numbers of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

k = 7
kmeans = KMeans(n_clusters = 7, random_state = 0)
pred_k = kmeans.fit_predict(Xc)
print(pred_k)

sns.scatterplot(data = data_encoder, x = 'NObeyesdad', y = 'FAF', hue = pred_k, palette = "deep", s = 100)
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.title("KMeans")
plt.show()

data_kmeans = data.copy()
data_kmeans['Cluster'] = pred_k
print(data_kmeans.head())

boolArray = data_kmeans['Cluster'] == 4
print(data_kmeans[boolArray])

# Performance Metrics

kmeans_metrics = silhouette_score(Xc, kmeans.labels_, metric = 'euclidean')
print('The Silhouette_Score of K-means is: {}'.format(kmeans_metrics))


# Clustering: Agglomerative Hierarchical

Xc = data_encoder.iloc[:, [12,16]].values

hc = sch.dendrogram(sch.linkage(Xc, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Sample')
plt.ylabel('Euclidean Distances')
plt.show()

hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward' )
pred_h = hc.fit_predict(Xc)

sns.scatterplot(data = data_encoder, x = 'NObeyesdad', y = 'FAF', hue = pred_h, palette = "deep", s = 100)
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.title("Hierarchical")
plt.show()

data_hc = data.copy()
data_hc['Cluster'] = pred_h
print(data_hc.head())

boolArrayhc = data_hc['Cluster'] == 0
print(data_hc[boolArrayhc])

# Performance Metrics

hc_metrics = silhouette_score(Xc, hc.labels_, metric = 'euclidean')
print('The Silhouette_Score of Hierarchical is: {}'.format(hc_metrics))

"""
# Clustering: DBSCAN

Xc = data_encoder.iloc[:, [12,16]].values
scaler = StandardScaler()
Xc = scaler.fit_transform(Xc)

dbscan = DBSCAN(eps = .5, min_samples = 15)
dbscan.fit(Xc)
pred_d = dbscan.labels_
print(pred_d)

colors = plt.cm.rainbow(np.linspace(0, 1, len(Xc)))
for i in range(len(Xc)):
    plt.plot(Xc[i][0], Xc[i][1], colors[pred_d[i]], markersize = 15)
    plt.xlabel('NObeyesdad')
    plt.ylabel('FAF')
    plt.title('DBSCAN')
    plt.legend()
    plt.show()


plt.scatter(Xc[pred_d == 0, 0], Xc[pred_d == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(Xc[pred_d == 1, 0], Xc[pred_d == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(Xc[pred_d == 2, 0], Xc[pred_d == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(Xc[pred_d == 3, 0], Xc[pred_d == 3, 1], s = 100, c = 'yellow', label = 'Cluster 4')
plt.xlabel('NObeyesdad')
plt.ylabel('FAF')
plt.title('DBSCAN')
plt.legend()
plt.show()

# Performance Metrics

dbscan_metrics = silhouette_score(Xc, dbscan.labels_, metric = 'euclidean')
print('The Silhouette_Score of DBSCAN is: {}'.format(dbscan_metrics))
"""