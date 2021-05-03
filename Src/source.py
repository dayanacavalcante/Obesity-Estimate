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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Load Data

data = pd.read_csv('C:\\Users\\RenanSardinha\\Documents\\Data Science\\EstimatedObesityLevels\\Data\\ObesityDataSet_raw_and_data_sinthetic.csv')
print(data)

# EDA
## Data Processing

"""
Plot Weight vs FAF 
Insufficient_Weight - 'red'; Normal_Weight - 'blue'; Overweight_Level_I - 'green'; 
Overweight_Level_II - 'yellow'; Obesity_Type_I - 'purple'; Obesity_Type_II - 'grey'; Obesity_Type_III - 'orange' 
"""

labels = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
colors = ['red', 'blue', 'green', 'yellow', 'purple', 'grey', 'orange']

filter = lambda type: data['NObeyesdad'] == type

plt.figure(figsize=(18,7)) 
 
for n in range(len(labels)):
    plt.scatter(data['Weight'][filter(labels[n])], data['FAF'][filter(labels[n])], color = colors[n], label = labels[n])

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Weight', fontsize=16)
plt.ylabel('FAF', fontsize=16)
plt.title('Body Mass Index ("NObeyesdad") plot with "Weight" and Frequency of Physical Activity ("FAF")', fontsize=19)
plt.legend()

print(data.info())

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
    plt.subplots_adjust(left=0.125,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.2, 
                        hspace=0.35)

print(data_encoder.describe())
print(data_encoder.mode())
print(data_encoder.var())

## Relationships Between Variables

plt.figure(figsize=(15,15))
sns.heatmap(data_encoder.corr(), color='b', annot=True)
plt.title('Correlation Heatmap',fontsize=16)

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
plt.yticks(fontsize=14)
plt.ylabel('Weight', fontsize=16)
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(y = "Age", data = data_iqr)
plt.yticks(fontsize=14)
plt.ylabel('Age', fontsize=16)
plt.show()

# Regression
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

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Weight', fontsize=16)
plt.ylabel('NObeyesdad', fontsize=16)
plt.title('Linear Regression Model', fontsize=19)
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

X_train, X_test, y_train, y_test = train_test_split(Xlr,ylr,test_size = .3, random_state=1)

print(X_train)
print(y_train)

# Model Training

lr = LinearRegression()
lr.fit(X_train,y_train)

pred_train = lr.predict(X_train)
pred_test = lr.predict(X_test)

# Performance Metrics

mae_test = mean_absolute_error(y_test, pred_test)
mape_test = mean_absolute_percentage_error(y_test, pred_test)
print('MAE_test: {}'.format(mae_test))
print('MAPE_test: {}'.format(mape_test))

# Cross Validation

result_cv = cross_val_score(lr, X_test, y_test, cv = 10)
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
plt.xlabel('Numbers of Clusters', fontsize=14)
plt.ylabel('WCSS', fontsize=14)
plt.title('Elbow Method', fontsize=16)
plt.show()

k = 7
kmeans = KMeans(n_clusters = 7, random_state = 0)
pred_k = kmeans.fit_predict(Xc)
print(pred_k)

sns.scatterplot(data = data_encoder, x = 'NObeyesdad', y = 'FAF', hue = pred_k, palette = "deep", s = 100)
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.title("KMeans", fontsize=16)
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
plt.title('Dendrogram', fontsize=16)
plt.xlabel('Sample', fontsize=14)
plt.ylabel('Euclidean Distances', fontsize=14)
plt.show()

hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward' )
pred_h = hc.fit_predict(Xc)

sns.scatterplot(data = data_encoder, x = 'NObeyesdad', y = 'FAF', hue = pred_h, palette = "deep", s = 100)
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.title("Hierarchical", fontsize=16)
plt.show()

data_hc = data.copy()
data_hc['Cluster'] = pred_h
print(data_hc.head())

boolArrayhc = data_hc['Cluster'] == 0
print(data_hc[boolArrayhc])

# Performance Metrics

hc_metrics = silhouette_score(Xc, hc.labels_, metric = 'euclidean')
print('The Silhouette_Score of Hierarchical is: {}'.format(hc_metrics))

# Clustering: DBSCAN

Xc = data_encoder.iloc[:, [12,16]].values

dbscan = DBSCAN(eps = .5, min_samples = 15)
dbscan.fit(Xc)
pred_d = dbscan.labels_
print(pred_d)

sns.relplot(x = 'NObeyesdad', y = 'FAF', hue = pred_d, data = data_encoder, palette = ["black", "dimgray", "bisque", "darkorange", "burlywood", 
"forestgreen", "limegreen", "darkgreen", "slategrey", "lightsteelblue", "cornflowerblue", "rosybrown", "lightcoral", "indianred", "darkgoldenrod",
"goldenrod", "cornsilk", "lightseagreen", "mediumturquoise", "darkslategray", "mediumpurple", "rebeccapurple", "blueviolet", "palevioletred", "crimson",
"pink"], s = 100)
plt.title("DBSCAN", fontsize=16)
plt.show()

data_d = data.copy()
data_d['Cluster'] = pred_d
print(data_d.head())

boolArraydb = data_d['Cluster'] == 22
print(data_d[boolArraydb])

# Performance Metrics

dbscan_metrics = silhouette_score(Xc, dbscan.labels_, metric = 'euclidean')
print('The Silhouette_Score of DBSCAN is: {}'.format(dbscan_metrics))

# Classification
# Training the model with the data set

Xlor = data_iqr.drop(['NObeyesdad'], axis = 1)
ylor = data_iqr['NObeyesdad']

# Separating training and test data

X_train, X_test, y_train, y_test = train_test_split(Xlor,ylor,test_size = .3,random_state=1)

# Logistic Regression Classifier

lor = LogisticRegression()
lor.fit(X_train,y_train)

pred_train_lor = lor.predict(X_train)
pred_test_lor = lor.predict(X_test)

print('Accuracy of Logistic Regression Classifier on test set: {:.2f}'.format(lor.score(X_test,y_test)))
print('Accuracy of Logistic Regression Classifier on train set: {:.2f}'.format(lor.score(X_train,y_train)))

# Confusion Matrix

cmlr = confusion_matrix(y_test, pred_test_lor)
sns.heatmap(cmlr,annot=True,fmt='g',cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predict")
plt.ylabel("Real")
plt.show()

# Decision Tree Classifier

tree = DecisionTreeClassifier()
tree.fit(X_train,y_train)

pred_train_tree = tree.predict(X_train)
pred_test_tree = tree.predict(X_test)

print('Accuracy of Decision Tree Classifier on test set: {:.2f}'.format(tree.score(X_test,y_test)))
print('Accuracy of Decision Tree Classifier on train set: {:.2f}'.format(tree.score(X_train,y_train)))

# Confusion Matrix

cmdt = confusion_matrix(y_test, pred_test_tree)
sns.heatmap(cmdt,annot=True,fmt='g',cmap="Blues")
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predict")
plt.ylabel("Real")
plt.show()

# Random Forest Classifier

rf = RandomForestClassifier()
rf.fit(X_train,y_train)

pred_train_rf = rf.predict(X_train)
pred_test_rf = rf.predict(X_test)

print('Accuracy of Random Forest Classifier on test set: {:.2f}'.format(rf.score(X_test,y_test)))
print('Accuracy of Random Forest Classifier on train set: {:.2f}'.format(rf.score(X_train,y_train)))

# Confusion Matrix

cmrf = confusion_matrix(y_test, pred_test_rf)
sns.heatmap(cmrf,annot=True,fmt='g',cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predict")
plt.ylabel("Real")
plt.show()