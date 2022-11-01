## Importing libraries
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.preprocessing import  StandardScaler
from scipy.cluster.hierarchy import  dendrogram, linkage

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
