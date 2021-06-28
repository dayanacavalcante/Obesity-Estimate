## **Application of predictive and descriptive methods in the case study to estimate obesity levels**

The data belongs to a study made to estimate obesity levels in individuals from Mexico, Peru and Colombia, based on their eating habits and physical condition. The data set has 17 features and 2111 rows.

The purpose of this case study was:

1. Created a Regression model (r2_score = 0.95) to check which variables based on eating habits and physical condition most influence the prediction of obesity levels;
2. Detected a group of people with possible high health risk due to the highest level of obesity and low (or none) frequency of physical activity with clustering algorithms: K-means (silhouette_score = 0.49), Hierarchical (silhouette_score = 0.47), and DBSCAN (silhouette_score = 1.0);
3. Checked which algorithm had the best performance for obesity levels classifier: Decision Tree (Accuracy = 93.2%), Random Forest (Accuracy = 93.1%), Logistic Regression (Accuracy = 84.8%), Support Vector Classifier (Accuracy = 83.1%), K-nearest neighbors (Accuracy = 77%), and Naive Bayes (Accuracy = 49.1%);

### **How?**

1. Applying the Ordinary Least Squares (OLS) method and Linear Regression algorithm;
2. Applying K-means, Agglomerative Hierarchical, and DBSCAN clustering algorithms;
3. Applying Decision Tree, Random Forest, Logistic Regression, Support Vector Classifier, K-nearest neighbors, and Naive Bayes algorithms;

### **Why?**

Contribute to the initial idea of the research article in the creation of intelligent computational tools to identify the level of obesity of an individual.

### **The data**

The data are available at the UCI Machine Learning Repository. Follow the link:  
https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+#

If you would like any clarification, feel free to send an email.