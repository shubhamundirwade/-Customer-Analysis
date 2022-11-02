## Importing libraries
import numpy as np
import pandas as pd
import scipy

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.preprocessing import  StandardScaler
from scipy.cluster.hierarchy import  dendrogram, linkage
from sklearn.cluster import  KMeans
from sklearn.decomposition import PCA

## Importing dataset
df_segmentation = pd.read_csv("segmentation data.csv", index_col=0)
df_segmentation

## Exploring Dataset
df_segmentation.head()
df_segmentation.describe()
  
## Getting Correlation
df_segmentation.corr()

## Plotting heatmap

plt.figure(figsize = (12,9) )
s = sns.heatmap(df_segmentation.corr(),
                annot = True,
                cmap = 'rainbow',#'RdBu',#'viridis','autumn',
                vmax = 1,
                vmin = -1)
s.set_yticklabels(s.get_yticklabels(), rotation = 0, fontsize = 12)
s.set_xticklabels(s.get_xticklabels(), rotation = 90, fontsize = 12)
plt.title("Correlation Heatmap")

## Visualization of Raw Data
plt.figure(figsize = (12,9))
plt.scatter(df_segmentation.iloc[:,2], df_segmentation.iloc[:,4])
plt.xlabel("Age")
plt.ylabel("Income")
plt.title("Visualization of Raw Data")

## Hierarchical Clusering

hier_clust = linkage(segmentation_std, method = 'ward')

## Dendogram Plot
plt.figure(figsize=(12,9))
plt.title("Hierarchical Clusering Dendogram") 
plt.xlabel("Observation")
plt.ylabel("Distance")
dendrogram(hier_clust,
            show_leaf_counts = False,
            no_labels = True,
            color_threshold = 0
            )
plt.show()

# Short Hirarchial Clustering

plt.figure(figsize=(12,9))
plt.title("Hierarchical Clusering Dendogram")
plt.xlabel("Observation")
plt.ylabel("Distance")
dendrogram(hier_clust,
            truncate_mode = 'level',
            p = 5,
            show_leaf_counts = True,
            no_labels = True
            )
plt.show()

## K-Means Clustering
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state=42)
    kmeans.fit(segmentation_std)
    wcss.append(kmeans.inertia_)
    
#Ploting for number of clusters 
plt.figure(figsize=(10,8))
plt.plot(range(1,11), wcss, marker = 'o', linestyle = '--')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("K-Means Clustering")
plt.show()

kmeans = KMeans(n_clusters=4, init= 'k-means++', random_state=42)
kmeans.fit(segmentation_std)

## Results
 df_segm_kmeans = df_segmentation.copy()
 df_segm_kmeans["Segment K-means"] = kmeans.labels_

df_segm_analysis = df_segm_kmeans.groupby(["Segment K-means"]).mean()
df_segm_analysis

df_segm_analysis["N Obs"] = df_segm_kmeans[["Segment K-means", "Sex"]].groupby(["Segment K-means"]).count()
df_segm_analysis["Prop Obs"] = df_segm_analysis["N Obs"] / df_segm_analysis["N Obs"].sum()

#renaming the column names
df_segm_analysis.rename({0:"well-off",
                         1:"fewer-opportunities",
                         2: 'standard',
                         3: "career focused"})

#lets plot raw data 
df_segm_kmeans["Labels"] = df_segm_kmeans["Segment K-means"].map({0:"well-off",
                                                                  1:"fewer-opportunities",
                                                                  2: 'standard',
                                                                  3: "career focused"})

x_axis = df_segm_kmeans["Age"]
y_axis = df_segm_kmeans["Income"]
plt.figure(figsize=(10,8) )
sns.scatterplot(x_axis, y_axis, hue=df_segm_kmeans["Labels"], palette=['g','r','c','m'])
plt.title("Segmentation K-means")
plt.show()

#PCA
pca = PCA()
pca.fit(segmentation_std)

pca.explained_variance_ratio_

#to select features
plt.figure(figsize=(12,9))
plt.plot(range(1,8), pca.explained_variance_ratio_.cumsum(), marker = 'o', linestyle = '--')
plt.title("Explained Variance by Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.show()

# selecting features
pca = PCCA(n_components = 3) # selecting first 3 components as features

pca.fit(segmentation.std)

## PCA Results
pca.components_

# Creating dataframe for the PCA component
df_pca_comp = pd.DataFrame(data = pca.components_,
                           columns = df_segmentation.columns.values,
                           index = ['Component 1', 'Component 2','Component 3'])

df_pca_comp

# generating heatmap for the dataframe4
sns.heatmap(df_pca_comp,
            vmin = -1,
            vmax = 1,
            cmap = 'RdBu',
            annot = True)
plt.yticks([0,1,2],
           ['Component 1', 'Component 2', 'Component 3'],
           rotation = 45,
           fontsize = 9)

#transofmation to 3 array 
pca.transform(segmentation_std)

# Scores
scores_pca = pca.transform(segmentation_std)

## K-means clustering with PCA component
wcss = []
for i in range(1,11):
    kmeans_pca = KMeans(n_clusters = i, init = 'k-means++', random_state=42)
    kmeans_pca.fit(scores_pca)  #changing the fit values to scores_pca
    wcss.append(kmeans_pca.inertia_)
    
 # Plotting 
plt.figure(figsize=(10,8))
plt.plot(range(1,11), wcss, marker = 'o', linestyle = '--')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("K-Means with PCA Clustering")
plt.show()

kmeans_pca = KMeans(n_clusters=4, init='k-means++',random_state=42)
kmeans_pca.fit(scores_pca)

# K-means clustering with PCA results
df_segm_pca_kmeans = pd.concat([df_segmentation.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
df_segm_pca_kmeans.columns.values[-3:] = ['Component 1','Component 2','Component 3']
df_segm_pca_kmeans["Segment K-means PCA"] = kmeans_pca.labels_
df_segm_pca_kmeans

df_segm_pca_kmeans_freq = df_segm_pca_kmeans.groupby(['Segment K-means PCA']).mean()
df_segm_pca_kmeans_freq


df_segm_pca_kmeans_freq["N Obs"] = df_segm_pca_kmeans[["Segment K-means PCA", "Sex"]].groupby(["Segment K-means PCA"]).count()
df_segm_pca_kmeans_freq["Prop Obs"] = df_segm_pca_kmeans_freq["N Obs"] / df_segm_pca_kmeans_freq["N Obs"].sum()

# Renaming the segments
df_segm_pca_kmeans_freq = df_segm_pca_kmeans_freq.rename({0:"standard",
                                                          1:"career focused",
                                                          2: 'fewer oportunities',
                                                          3: "well-off"})
df_segm_pca_kmeans_freq

df_segm_pca_kmeans["Legend"] = df_segm_pca_kmeans["Segment K-means PCA"].map({0:"standard",
                                                                              1:"career focused",
                                                                              2: 'fewer oportunities',
                                                                              3: "well-off"})

x_axis = df_segm_pca_kmeans["Component 2"]
y_axis = df_segm_pca_kmeans["Component 1"]
plt.figure(figsize=(10,8))
sns.scatterplot(x_axis, y_axis, hue=df_segm_pca_kmeans["Legend"], palette=["g",'r','c','m'])
plt.title("Clusters by PCA Components")
plt.show()

## Saving pickle file
pickle.dump(scalar, open('scalar.pickle', 'wb'))
pickle.dump(pca, open('pca.pickle', 'wb'))
pickle.dump(kmeans_pca, open('kmeans_pca.pickle', 'wb'))
