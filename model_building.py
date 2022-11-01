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
