# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

# Load Data

data = pd.read_csv('C:\\Users\\RenanSardinha\\Documents\\Data Science\\EstimatedObesityLevels\\Data\\ObesityDataSet_raw_and_data_sinthetic.csv')
print(data)

# EDA
## Data Processing

print(data.info())

encoder = ce.OrdinalEncoder(mapping = [{'col':'Gender','mapping':{'Male':0,'Female':1}},
                            {'col':'family_history_with_overweight','mapping':{'no':0,'yes':1}},
                            {'col':'FAVC','mapping':{'no':0,'yes':1}},
                            {'col':'CAEC','mapping':{'no':0,'Sometimes':1,'Frequently':2,'Always':3}},
                            {'col':'SMOKE','mapping':{'no':0,'yes':1}},
                            {'col':'SCC','mapping':{'no':0,'yes':1}},
                            {'col':'CALC','mapping':{'no':0,'Sometimes':1,'Frequently':2,'Always':3}},
                            {'col':'MTRANS','mapping':{'Bike':0,'Walking':1,'Public_Transportation':2,'Motorbike':3,'Automobile':4}},
                            {'col':'NObeyesdad','mapping':{'Insufficient_Weight':0,'Normal_Weight':1,'Overweight_Level_I':2,'Overweight_Level_II':3,'Obesity_Type_I':4,'Obesity_Type_II':5,'Obesity_Type_III':6}}])


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

# Scaling of Data

X_quantile = data_encoder.copy()
X_quantile = QuantileTransformer().fit_transform(X_quantile)
print(pd.DataFrame(X_quantile).describe())

# Clustering
# Centroids-based clustering: K-Means Clustering

# Elbow Method

SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_clusters = cluster, init = 'k-means++')
    kmeans.fit(X_quantile)
    SSE.append(kmeans.inertia_)

frame = pd.DataFrame({'Cluster': range(1,20), 'SSE': SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')

# K = 8
kmeans = KMeans(n_clusters = 8, init = 'k-means++')
kmeans.fit(X_quantile)
pred = kmeans.predict(X_quantile)

final_data = pd.DataFrame(data_encoder)
final_data['Cluster'] = pred
print(final_data['Cluster'].value_counts())

sns.relplot(data = final_data, x = 'Cluster', y = 'NObeyesdad', hue = pred, legend = "full", palette = "pastel", s = 200)

sns.relplot(data = final_data, x = 'Cluster', y = 'FAF', hue = pred, legend = "full", palette = "pastel", s = 200)

# Performance Metrics

metrics.silhouette_score(X_quantile, kmeans.labels_,metric='euclidean')

# DBSCAN

X = data_encoder.iloc[:,[12,16]].values
print(X)

clustering_model = DBSCAN(eps=.95, min_samples = 50)
clustering_model.fit(X)
pred_labels = clustering_model.labels_

print(pred_labels)

plt.scatter(X[:,0], X[:,1], c = pred_labels, cmap = 'Paired')
plt.xlabel('FAF')
plt.ylabel('NObeyesdad')
plt.title('DBSCAN')

# IQR

Q1, Q3 = np.percentile(data_encoder.Weight, [25,75])
print(Q1,Q3)

IQR = Q3 - Q1
print(IQR)

# Lower Range
low_range = Q1 - (1.5*IQR)
print(low_range)

# Upper Range
upper_range = Q3 + (1.5*IQR)
print(upper_range)

data_iqr = data_encoder.copy()

data_iqr.drop(data_iqr[(data_iqr.Weight > 169.25) | (data_iqr.Weight < 3.25)].index, inplace = True)

ax = sns.boxplot(y = "Weight", data = data_iqr)

# Creating the model
# LinearRegression
X = data_iqr['Weight'].values.reshape(-1,1)
y = data_iqr['NObeyesdad'].values.reshape(-1,1)

lr = LinearRegression()
lr.fit(X,y)

print("The model is: NObeyesdad = {:.5} + {:.5}X".format(lr.intercept_[0], lr.coef_[0][0]))

pred = lr.predict(X)

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

X = data_iqr['Weight']
y = data_iqr['NObeyesdad']
X2 = sm.add_constant(X)
est = sm.OLS(y,X2)
est2 = est.fit()
print(est2.summary())