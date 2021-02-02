# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce

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

