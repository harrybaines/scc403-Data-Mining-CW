#####
# SCC.403 Data Mining Coursework
# Clustering Code
#
# Author: Harry Baines
#####

# Import clustering algorithms
from clustering.kmeans import Kmeans
from clustering.hierarchical import Hierarchical

# Import utility and ML functions
from classification.evaluator import ClassificationEvaluator
from preprocessing.pca import PCA
from preprocessing.encoding import LabelEncoder
from preprocessing.encoding import OneHotEncoder
from preprocessing.sampler import Sampler
import preprocessing.fileopts as fo
import preprocessing.scalers as scalers
import preprocessing.ml as ml

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import pandas as pd # for visualising correlations
import seaborn as sns
import math
import random
import time

# Using seaborn's style and custom appearance
plt.style.use('seaborn')
mp.rcParams.update({
    'font.family': 'STIXGeneral',
    'mathtext.fontset': 'stix',
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

########################################################
#################### Preprocessing  ####################
########################################################

# Read in pulsar dataset
pulsar = fo.open_csv("./data/Pulsar/HTRU_2.csv")
pulsar_columns = fo.open_cols('./data/Pulsar/pulsar_cols.txt')

# Normalize pulsar data
pulsar = scalers.normalize(pulsar.astype(float))

###
## Pulsar Exploratory Data Analysis
###

# Create pandas dataframe to facilitate EDA
# df = pd.DataFrame(scalers.normalize(pulsar))
#
# # Visualise pair plot
# g = sns.PairGrid(df, vars=list(range(9)), hue=8, palette='RdBu_r')
# g.map(plt.scatter, alpha=0.8)
# g.add_legend()
#
# # Obtain correlation matrix and plot on heatmap
# corr = df.corr()
# ax = sns.heatmap(corr, annot = True, fmt='.2f')
# ax.set_ylim(9.0, 0)
#
# # Save to file
# fig = ax.get_figure()
# fig.savefig('corr.pdf', dpi=400, bbox_inches='tight')

# Pulsar original VS transformed plots
# plot(
#     data=pulsar,
#     x_col_ind=0,
#     y_col_ind=3,
#     cols=pulsar_columns,
#     filename="./plots/scaling_results/standardized_pulsar_1.pdf",
#     scale_type="stand"
# )
# plot(
#     data=pulsar,
#     x_col_ind=5,
#     y_col_ind=4,
#     cols=pulsar_columns,
#     filename="./plots/scaling_results/standardized_pulsar_2.pdf",
#     scale_type="stand"
# )

##
# Pulsar PCA
##

# Instantiate PCA class and fit to pulsar data
pulsar_pca = PCA(n_components=8)
pulsar_pca.fit(pulsar[:, :-1])

# Plot principal components with most of the variance
legend = [(0, "Not Pulsar Star"), (1, "Pulsar Star")]
pulsar_pca.plot(
    labels=pulsar[:, -1],
    legend=legend,
    filename='./plots/PCAResults/PulsarPCA.pdf'
)

# Plot explained variance
pulsar_pca.scree_plot(filename='./plots/PCAResults/PulsarScree.pdf')

# Visualise and analyse correlation between features
pulsar_pca.get_corr_coef()

# PCA with Libraries (to verify result of PCA code from scratch)
# from sklearn.decomposition import PCA
#
# # Fit PCA
# pca = PCA(n_components=5)
# pca.fit(scalers.standardize(pulsar))
#
# # Transform and display data
# transformed = pca.transform(scalers.standardize(pulsar))
# plt.plot(transformed[:, 0], transformed[:, 1], '.', markersize = 5)
# plt.show()

########################################################
################ Clustering Algorithms  ################
########################################################

###
## 2.1 K-means Clustering
###

# Pulsar K-means - we know we need 2 clusters for k-means (either pulsar or not pulsar)
start = time.time()

# Split full PCA pulsar data into training and testing sets
pulsar_pca_data = pulsar_pca.pca_by_hand_data
train_X, train_Y, test = ml.train_test_split(
    data=pulsar_pca_data,
    labels=pulsar[:, -1],
    train_size=0.8
)

# Use kmeans on PCA training set
km = Kmeans(k=2)
km.fit(train_X)

# Plot PCA training results after k-means
legend = [(0, "Not Pulsar Star"), (1, "Pulsar Star")]
km.plot(
    xlabel="Principal Component 1",
    ylabel="Principal Component 2",
    title="K-means clustering on 2 component PCA",
    legend=legend
)

# Predict labels for test data items
test_items = pulsar_pca_data[test]
actual_labels = pulsar[test, -1]

cluster_nums, metrics = km.predict(
    test_data=test_items,
    actual_labels=actual_labels
)

print(f'Kmeans Pulsar Metrics: {metrics}')

########################################################################

###
## 2.2 Hierarchical Clustering (from scratch)
###

# Create new hierarchical clustering instance
hc = Hierarchical(data=pulsar_pca_data)

# Fit to data and plot dendrogram results
hc.fit(
    clusters_to_vis=10,
    filename="./plots/clustering_results/pulsar_dendrogram.pdf"
)

########################################################################

###
## Whole Process : Pulsar Preprocessing, SMOTE, PCA, k-means clustering and prediction
###

# 1. SMOTE sampling

# Create new Sampler instance to use SMOTE on abalone data
sampler = Sampler()

label_to_oversample = 1
pulsar_labels = pulsar[:, -1]

# Obtain number of 1's and 0's first
num_ones = ml.get_num_labels(pulsar_labels, 1)
num_zeros = ml.get_num_labels(pulsar_labels, 0)

# No. to oversample = num 0's - num 1's
# no_to_oversample = num_zeros - num_ones
no_to_oversample = 0

print("1. Running SMOTE on pulsar data...")
pulsar_data_smote, pulsar_labels_smote = sampler.smote(
    data=pulsar[:, :-1],
    labels=pulsar_labels,
    label=label_to_oversample,
    k_neighbours=2,
    iterations=no_to_oversample
)

label_counts = ml.get_label_counts(pulsar_labels_smote)
print(f"[After SMOTE oversampling with {no_to_oversample} label {label_to_oversample} items]\n{label_counts}")

# 2. PCA
pca = PCA(n_components=2)

print("2. Running PCA on pulsar data after SMOTE...")
pca.fit(pulsar_data_smote)

# 3. Train-test split
print("3. Splitting PCA pulsar data into train and test sets...")
pca_pulsar_data = pca.pca_by_hand_data

train_X, train_Y, test = ml.train_test_split(
    data=pca_pulsar_data,
    labels=pulsar_labels_smote,
    train_size=0.8
)

# 4. K-means clustering
kmeans = Kmeans(k=2)

print("4. Running k-means clustering on pulsar PCA training set...")
kmeans.fit(train_X)

# 5. Predict on test set
test_data = pca_pulsar_data[test]
actual_labels = pulsar_labels_smote[test]

print("5. Running predictions on k-means clustering result using test set...")
pred_labels, metrics = kmeans.predict(
    test_data=test_data,
    actual_labels=actual_labels
)

# 6. Evaluating metrics
print("6. Obtaining metrics for classified test items on k-means...")
evaluator = ClassificationEvaluator(
    pred_labels=pred_labels,
    actual_labels=actual_labels
)

metrics = evaluator.evaluate()
print(metrics)

########################################################################

###
## EXTRA: k-means visualise multiple different numbers of clusters in 1 plot
###

# max_no_k = 7
# print(f"Generating plot of k-means clustering for clusters 2-{max_no_k}...")
#
# # Multiple kmeans in subplots
# km.plot_k(
#     k=max_no_k,
#     xlabel="Principal Component 1",
#     ylabel="Principal Component 2"
# )
#
# print(f'Completed in {(time.time() - start)}s')

# Store class labels in separate variable
# pulsar_nolbl = pulsar[:, :-1]
# pulsar_labels = pulsar[:, -1]
#
# # Get training and testing split
# train_X, train_Y, test = ml.train_test_split(pulsar, pulsar_labels)
#
# test_X = pulsar[test, :-1]
# test_Y = pulsar[test, -1]
#
# # K-means for train and evaluate on test
# km = Kmeans(k=2)
# km.fit(train_X[:, 0:2])
# km.plot()

# Y_preds = km.predict(test_X)

# Plots of kmeans on chosen features
# reduced_data = pulsar[:, 0:2].astype(float)

# # Perform K-means clustering
# start = time.time()
# k = 2
#
# print(f"Running k-means clustering with {k} clusters...")
#
# # Perform K-means clustering
# kmeans = Kmeans(k=k, threshold=0.001)
# # kmeans.fit(reduced_data)
# # kmeans.plot()
#
# print(f'Completed in {(time.time() - start)}s')

# start = time.time()
# max_no_k = 7
# #
# print(f"Generating plot of k-means clustering for clusters 2-{max_no_k}...")
#
# # K-means - multiple kmeans in subplots
# kmeans = Kmeans(k=max_no_k, threshold=0.00001)
# kmeans.fit(reduced_data)
# kmeans.plot_k(max_no_k)
#
# print(f'Completed in {(time.time() - start)}s')
