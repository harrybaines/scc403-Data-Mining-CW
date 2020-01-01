#####
# SCC.403 Data Mining Coursework
# Abalone Classification Code
#
# Author: Harry Baines
#####

# Import classification algorithms
from classification.neural_network.rbfnn import RBFNN
from classification.neural_network.shallownn import ShallowNeuralNetwork
from classification.neural_network.lr import LogisticRegression
from classification.knn import KNN
from classification.decision_tree import DecisionTree

# Import utility and ML functions
from classification.evaluator import ClassificationEvaluator
from preprocessing.pca import PCA
from preprocessing.encoding import LabelEncoder
from preprocessing.encoding import OneHotEncoder
import preprocessing.fileopts as fo
import preprocessing.scalers as scalers
import preprocessing.ml as ml

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import seaborn as sns
import math
import random
import time
import csv

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

# Read in Abalone dataset
abalone = fo.open_csv("./data/Abalone/abalone19.csv")
abalone_columns = fo.open_cols('./data/Abalone/abalone_cols.txt')

########################################################################

###
## Abalone preprocessing
###

# Set labels to 1 for positive and 0 for negative
abalone = np.char.strip(abalone)
abalone[abalone == 'positive'] = 1
abalone[abalone == 'negative'] = 0
abalone_labels = abalone[:, -1].astype(int)

# Label encoding from scratch
label_encoder = LabelEncoder()

# Encode sex using label encoding
abalone = label_encoder.fit_transform(
    data=abalone,
    col_idx=0
)

# Normalize abalone columns
abalone = scalers.normalize(abalone)[:, :-1]

########################################################################

###
## Abalone EDA
###

# # Extract numerical columns and normalize them
# abalone_columns_sub = abalone_columns[1:8]
#
# plt.bar(['Not Age Class 19', 'Age Class 19'], [ml.get_num_labels(abalone_labels, 0), ml.get_num_labels(abalone_labels, 1)], color=['green', '#eb3734'])
# plt.title('Counts of positive and negative classes')
# plt.show()

# Abalone original VS transformed plots
# abalone_sub = abalone[:, 1:8].astype(float)
# norm_abalone = scalers.normalize(abalone_sub)
# plot(
#     data=abalone_sub,
#     x_col_ind=0,
#     y_col_ind=4,
#     cols=abalone_columns_sub,
#     filename="normalized_abalone.pdf",
#     scale_type="stand"
# )
#
# plot(
#     data=abalone_sub,
#     x_col_ind=3,
#     y_col_ind=6,
#     cols=abalone_columns_sub,
#     filename="standardized_abalone.pdf",
#     scale_type="norm"
# )

########################################################################

###
## Abalone PCA
###

# Instantiate PCA class and fit to abalone data
# pca = PCA(n_components=8)
# pca.fit(abalone)
#
# # Obtain centralized data
# centralized_data = pca.centralized
#
# # Plot principal components with most of the variance
# legend = [(0, "Not Age Class 19"), (1, "Age Class 19")]
# pca.plot(
#     # data=centralized_data,
#     labels=abalone[:, -1],
#     legend=legend,
#     filename='./plots/PCAResults/AbalonePCA.pdf'
# )
#
# # Plot explained variance
# pca.scree_plot(filename='./plots/PCAResults/AbaloneScree.pdf')
#
# # Visualise and analyse correlation between features
# pca.get_corr_coef()

########################################################################

###
## Sampling: dealing with class imbalance for abalone data
###
from preprocessing.sampler import Sampler

# Obtain minority class from labels
minority_class = ml.get_minority_class(abalone_labels)

label_counts = ml.get_label_counts(abalone_labels)
print(f"[Before sampling]\n{label_counts}")

sampler = Sampler()

## Oversampling
no_to_oversample = 4000
label_to_oversample = 1
os_abalone_data, os_abalone_labels = sampler.over_sample(abalone, abalone_labels, k=no_to_oversample, l=label_to_oversample)

label_counts = ml.get_label_counts(os_abalone_labels)
print(f"[After oversampling {no_to_oversample} label {label_to_oversample} items]\n{label_counts}")

# Automatic detection and oversampling of minority class
os_abalone_data, os_abalone_labels = sampler.over_sample(abalone, abalone_labels, auto=True)

label_counts = ml.get_label_counts(os_abalone_labels)
print(f"[After automatic oversampling]\n{label_counts}")

## Undersampling
no_to_undersample = ml.get_num_labels(abalone_labels, 0) - ml.get_num_labels(abalone_labels, 1)
label_to_undersample = 0
us_abalone_data, us_abalone_labels = sampler.under_sample(abalone, abalone_labels, k=no_to_undersample, l=label_to_undersample)

label_counts = ml.get_label_counts(us_abalone_labels)
print(f"[After undersampling {no_to_undersample} label {label_to_undersample} items]\n{label_counts}")

# Automatic detection and oversampling of minority class
us_abalone_data, us_abalone_labels = sampler.under_sample(abalone, abalone_labels, auto=True)

label_counts = ml.get_label_counts(us_abalone_labels)
print(f"[After automatic undersampling]\n{label_counts}")

# SMOTE - create new Sampler instance to use SMOTE on abalone data
sampler = Sampler()

no_to_oversample = ml.get_num_labels(abalone_labels, 0) - ml.get_num_labels(abalone_labels, 1)
label_to_oversample = 1

smote_abalone_data, smote_abalone_labels = sampler.smote(
    data=abalone,
    labels=abalone_labels,
    label=label_to_oversample,
    k_neighbours=2,
    iterations=no_to_oversample
)

label_counts = ml.get_label_counts(smote_abalone_labels)
print(f"[After SMOTE oversampling with {no_to_oversample} label {label_to_oversample} items]\n{label_counts}")

########################################################################
###################### Classification Algorithms #######################
########################################################################

# Run each classifier 'iterations' times
iterations = 1

roc_curve_data = []

###
## 3.1 Radial Basis Function Neural Network (RBF)
###

metrics = []

# Run training and testing multiple times and compute average
for i in range(iterations):

    # Create new RBF neural network for abalone
    model = RBFNN(
        n_prototypes=4,
        n_classes=2
    )

    # Get random training and testing sets
    train_X, train_Y, test = ml.train_test_split(smote_abalone_data, smote_abalone_labels)

    # Training
    model.train(train_X, train_Y)

    # Testing
    print(f"[RBFNN Abalone] Computing metrics for test set (iteration {i})")
    metrics_res, probs = model.test(smote_abalone_data, smote_abalone_labels, test, verbose=False)
    metrics.append(metrics_res)

# Obtain averages for each classification metric
evaluator = ClassificationEvaluator()
evaluator.compute_avg_metrics(
    metrics=metrics,
    iterations=iterations,
    test_str=f"[RBFNN Abalone] Test set results after {iterations} iterations:"
)

roc_curve_data.append((smote_abalone_labels[test], probs))

########################################################################

###
## 3.2 Shallow Neural Network
###

metrics = []

# Run training and testing multiple times and compute average
for i in range(iterations):

    # Create new shallow neural network for abalone
    nn = ShallowNeuralNetwork(
        learning_rate=0.001,
        n_epochs=100,
        n_hidden_units=1
    )

    # Get train and test sets
    train_X, train_Y, test = ml.train_test_split(
        data=smote_abalone_data,
        labels=smote_abalone_labels,
        train_size=0.7
    )

    # Train the model
    model = nn.train(train_X, train_Y)

    # Test the model
    print(f"[Shallow NN Abalone] Computing metrics for test set (iteration {i})")
    metrics_res, probs = nn.test(model, test, smote_abalone_data, smote_abalone_labels, verbose=False)
    metrics.append(metrics_res)

# Obtain averages for each classification metric
ClassificationEvaluator().compute_avg_metrics(
    metrics=metrics,
    iterations=iterations,
    test_str=f"[Shallow NN Abalone] Test set results after {iterations} iterations:"
)

roc_curve_data.append((smote_abalone_labels[test], probs))

########################################################################

##
# 3.2.2 Accuracy against parameters plot
##
# def test_nn(data, labels, train_size, learning_rate, n_epochs, n_hidden_units):
#     # Create new neural network
#     nn = NeuralNetwork(
#         learning_rate=learning_rate,
#         n_epochs=n_epochs,
#         n_hidden_units=n_hidden_units
#     )
#
#     # Get train and test sets
#     train_X, train_Y, test = ml.train_test_split(
#         data=data,
#         labels=labels,
#         train_size=0.7
#     )
#
#     start = time.time()
#
#     # Train the model
#     model = nn.train(train_X, train_Y)
#
#     # Test the model
#     test_results = nn.test(model, test, labels, summary=True)
#     return test_results['accuracy']
#
#
# # normalize numeric columns
# norm_data = scalers.normalize(data[:, 3:])
# data = np.column_stack((data[:, :3], norm_data))
#
# start = time.time()
#
# # Plot how no. of hidden units affects accuracy
# iter = 3
# inc = 4
#
# incs = []
# accs = []
#
# for i in range(iter):
#     cur_inc = 1 + (i * inc)
#
#     acc = test_nn(
#         data=data,
#         labels=labels,
#         train_size=0.7,
#         learning_rate=0.01,
#         n_epochs=2000,
#         n_hidden_units=cur_inc
#     )
#
#     incs.append(cur_inc)
#     accs.append(acc)
#
# print(f'Time elapsed: {round(time.time() - start, 2)}s')
#
# # Plot results
# plt.plot(incs, accs)
# plt.xlabel('Hidden Units')
# plt.ylabel('Accuracy')
# plt.ylim(0, 100)
# plt.show()

########################################################################

###
## 3.3 Logistic Regression Classifier (1-layer Neural Network) (built entirely from scratch)
###

metrics = []

# Run training and testing multiple times and compute average
for i in range(iterations):

    # Create new logistic regression unit for abalone
    model = LogisticRegression(
        epochs=2000,
        alpha=0.01
    )

    # Get train and test sets
    train_X, train_Y, test = ml.train_test_split(
        data=smote_abalone_data,
        labels=np.array([smote_abalone_labels]).T, # for SMOTE do: np.array([smote_abalone_labels]).T, normal labels array otherwise
        train_size=0.7
    )

    # Fit the model
    model.fit(train_X, train_Y)

    # Test the model (predict labels for test data)
    print(f"[Logistic Regression Abalone] Computing metrics for test set (iteration {i})")
    metrics_res, probs = model.predict(
        data=smote_abalone_data[test],
        labels=np.array([smote_abalone_labels]).T, # for SMOTE do: np.array([smote_abalone_labels]).T, normal labels array otherwise
        test=test,
        summary=False
    )

    metrics.append(metrics_res)

# Obtain averages for each classification metric
ClassificationEvaluator().compute_avg_metrics(
    metrics=metrics,
    iterations=iterations,
    test_str=f"[Logistic Regression Abalone] Test set results after {iterations} iterations:"
)

roc_curve_data.append((smote_abalone_labels[test], probs))

########################################################################

###
## 3.4 Decision Tree
## (OOP implementation from scratch with guidance from https://towardsdatascience.com/decision-tree-from-scratch-in-python-46e99dfea775)
###

metrics = []

# Run training and testing multiple times and compute average
for i in range(iterations):

    # Create train and test sets
    train_X, train_Y, test = ml.train_test_split(
        data=smote_abalone_data,
        labels=smote_abalone_labels,
        train_size=0.7
    )

    # Fit decision tree
    tree = DecisionTree(max_depth=1)
    tree.fit(train_X, train_Y)

    # Obtain metrics from classifier
    print(f"[Decision Tree Abalone] Computing metrics for test set (iteration {i})")
    metrics_res, probs = tree.test(smote_abalone_data[test], smote_abalone_labels[test])

    metrics.append(metrics_res)

# Obtain averages for each classification metric
ClassificationEvaluator().compute_avg_metrics(
    metrics=metrics,
    iterations=iterations,
    test_str=f"[Decision Tree Neighbor Abalone] Test set results after {iterations} iterations:"
)

roc_curve_data.append((smote_abalone_labels[test], probs))

########################################################################

# Plot ROC curve
true_labels = [roc_data[0] for roc_data in roc_curve_data]
probs = [roc_data[1] for roc_data in roc_curve_data]

alg_strs = ['RBFNN', 'Shallow NN', 'Logistic Regression', 'Decision Tree']

evaluator.plot_roc_curve(
    true_labels=true_labels,
    probs=probs,
    alg_strs=alg_strs,
    filename="./plots/classification_results/ROC_abalone.pdf"
)

########################################################################

##
# Evaluating max_depth parameter of decision tree
##

# iterations = 5
#
# def test_depths(data, labels, max_val=10):
#     avg_metrics = []
#
#     evaluator = ClassificationEvaluator()
#
#     for depth in range(1, max_val+1):
#         metrics = []
#
#         for i in range(iterations):
#             print(f"[Decision Tree Abalone] Training and testing on depth {depth}")
#
#             # Get random training and testing sets
#             train_X, train_Y, test = ml.train_test_split(
#                 data=data.astype(int),
#                 labels=labels.astype(int),
#                 train_size=0.7
#             )
#
#             dtree = DecisionTree(max_depth=depth)
#             dtree.fit(train_X, train_Y)
#
#             metric_result = dtree.test(data[test], labels[test])
#             metrics.append(metric_result)
#
#         avg_metrics_result = evaluator.compute_avg_metrics(metrics=metrics, iterations=iterations)
#         avg_metrics.append(avg_metrics_result)
#
#     return avg_metrics
#
# plt.figure(figsize = (6,4))
#

# max_val = 8
# depths = range(1, max_val+1)
#
# # Test: test depths for abalone data with imputed values
# metrics_test = test_depths(
#     data=abalone,
#     labels=abalone_labels,
#     max_val=max_val
# )


# Plot saved metrics (accuracy, precision, recall, F1) from saved file
# with open('abalone_imputed_decisiontree_depths.csv', 'r') as file:
#     metrics = list(csv.reader(file))
#     for row in metrics:
#         row = [float(i) for i in row]
#         plt.plot(depths, row)

# accuracies = [str(round(metric['avg_accuracy'], 2)) for metric in metrics_test]
# precisions = [str(round(metric['avg_precision'], 2)) for metric in metrics_test]
# recalls = [str(round(metric['avg_recall'], 2)) for metric in metrics_test]
# f1s = [str(round(metric['avg_f1'], 2)) for metric in metrics_test]
#
# plt.plot(depths, accuracies)
# plt.plot(depths, precisions)
# plt.plot(depths, recalls)
# plt.plot(depths, f1s)

# with open('abalone_imputed_decisiontree_depths.csv', 'a') as file:
#     file.write(','.join(accuracies)+'\n')
#     file.write(','.join(precisions)+'\n')
#     file.write(','.join(recalls)+'\n')
#     file.write(','.join(f1s)+'\n')

# plt.legend(['Accuracy', 'Precision', 'Recall', 'F1'], loc='upper left')
# plt.xlabel('Maximum Depth')
# plt.ylabel('Evaluation Metric (%)')
# plt.title('Decision Tree Metrics against Maximum Depths')
# plt.savefig('./plots/evaluation/dectree_maxdepth.pdf', dpi=300)
# plt.show()

########################################################################

###
## 3.5 k-nearest neighbours
## (OOP implementation from scratch with guidance from https://www.edureka.co/blog/k-nearest-neighbors-algorithm/)
###

# Create new kNN model
# knn = KNN()
#
# # Create train and test sets
# train_X, train_Y, test = ml.train_test_split(
#     data=abalone,
#     labels=abalone_labels,
#     train_size=0.7
# )
#
# # Obtain testing data items and corresponding class labels
# test_labels = abalone_labels[test]
# test = abalone[test]

# Fit model to training data with testing item
# preds = knn.fit(
#     train_data=train_X,
#     train_labels=train_Y,
#     test_data=test,
#     test_labels=test_labels,
#     k=1
# )
#
# # Calculate accuracy of predictions based on true labels
# acc = knn.calc_accuracy(
#     test_labels=test_labels,
#     preds=preds
# )

########################################################################

###
## 3.6 Linear Discriminant Analysis (LDA)
###

# from classification.lda import LDA
#
# # Initialise new LDA instance
# lda = LDA()
#
# # Train classifier
# lda.train(abalone, abalone_labels)
#
# # Plot results
# lda.plot()
