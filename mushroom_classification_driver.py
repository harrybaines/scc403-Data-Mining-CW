#####
# SCC.403 Data Mining Coursework
# Mushroom Classification Code
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

# Read in and Mushroom dataset
mushroom = fo.open_csv("./data/Mushroom/agaricus-lepiota.csv")
mushroom_columns = fo.open_cols('./data/Mushroom/mushroom_cols.txt')

########################################################################

###
## Mushroom Exploratory Data Analysis
###

# # Get total number of edible mushrooms
# num_edible = get_row_count(mushroom, 0, 'e')
# print(f"{num_edible} edible mushrooms")
#
# # Get total number of poisonous mushrooms
# num_poisonous = get_row_count(mushroom, 0, 'p')
# print(f"{num_poisonous} poisonous mushrooms")

# fo.head(mushroom)

# plt.bar(['Edible', 'Poisonous'], [num_edible, num_poisonous], color=['green', '#eb3734'])
# plt.title('Count of Poisonous and Edible Mushrooms')
# plt.show()

# counts = []
# for i in range(len(mushroom_columns)):
#     count = np.sum(mushroom[:, i] == '?')
#     count > 0 and print(f'Feature {i}: {count} missing value(s)')
#
# missing_values = ml.get_num_missing_values(mushroom, '?')
# print(f'Total missing values in mushroom: {missing_values}')

########################################################################

###
## Mushroom preprocessing
###

# Deal with missing values in columns
mushroom = ml.impute_mode(mushroom, missing_char='?')
# missing = ml.get_num_missing_values(mushroom, missing_char='?')

# Label encoding from scratch
label_encoder = LabelEncoder()

# One-hot encoding from scratch
one_hot_encoder = OneHotEncoder()

# (Used for labelling the one-hot encoded columns for clarity)
unique_col_vals = list(np.unique(mushroom[:, col]) for col in range(1, mushroom.shape[1]))

# Label encode all categorical values before one-hot encoding
mushroom = label_encoder.fit_transform(
    data=mushroom,
    col_idx=-1
)

mushroom_labels = mushroom[:, 0]

# One-hot encoding for categorical features in mushroom
mushroom, mushroom_enc_columns = one_hot_encoder.fit_transform(
    data=mushroom[:, 1:],
    columns=mushroom_columns[1:],
    unique_col_vals=unique_col_vals
)

# Append class to front (didn't need one-hot encoding)
mushroom = np.concatenate((mushroom_labels.reshape((len(mushroom), 1)), mushroom), axis=1)

mushroom_enc_columns = np.insert(mushroom_enc_columns, 0, 'class', axis=0)

# Remove uninformative features with mean of 0 or 1
new_mushroom = mushroom.copy()
inds = []

for col_idx in range(mushroom.shape[1]):
    col_mean = np.mean(mushroom[:, col_idx])
    if col_mean == 0 or col_mean == 1:
        inds.append(col_idx)

mushroom = np.delete(mushroom, inds, axis=1)

# Store copy for later testing
mushroom_copy = mushroom.copy()
mushroom_labels_copy = mushroom[:, 0]

# Remove class label column
mushroom = mushroom[:, 1:]
fo.head(mushroom)

########################################################################

###
## Mushroom PCA
###

# Instantiate PCA class and fit to mushroom data
pca = PCA(n_components=10)
pca.fit(mushroom)

# Obtain centralized data
centralized_data = pca.centralized

# Plot principal components with most of the variance
legend = [(0, "Edible"), (1, "Poisonous")]
pca.plot(
    labels=mushroom_labels,
    legend=legend,
    filename='./plots/PCAResults/MushroomPCA.pdf'
)

# Plot explained variance
pca.scree_plot(filename='./plots/PCAResults/MushroomScree.pdf')

# Visualise and analyse correlation between features
pca.get_corr_coef()


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

    # Create new RBF neural network for mushroom
    model = RBFNN(
        n_prototypes=4,
        n_classes=2
    )

    # Get random training and testing sets
    train_X, train_Y, test = ml.train_test_split(mushroom, mushroom_labels)

    # Training
    model.train(train_X, train_Y)

    # Testing
    print(f"[RBFNN Mushroom] Computing metrics for test set (iteration {i})")
    metrics_res, probs = model.test(mushroom, mushroom_labels, test)
    metrics.append(metrics_res)

# Obtain averages for each classification metric
evaluator = ClassificationEvaluator()
evaluator.compute_avg_metrics(
    metrics=metrics,
    iterations=iterations,
    test_str=f"[RBFNN Mushroom] Test set results after {iterations} iterations:"
)

roc_curve_data.append((mushroom_labels[test], probs))

########################################################################

###
## 3.2 Shallow Neural Network
###

metrics = []

# Run training and testing multiple times and compute average
for i in range(iterations):

    # Create new shallow neural network for mushroom
    nn = ShallowNeuralNetwork(
        learning_rate=0.001,
        n_epochs=100,
        n_hidden_units=1
    )

    # Get train and test sets
    train_X, train_Y, test = ml.train_test_split(
        data=mushroom,
        labels=mushroom_labels,
        train_size=0.7
    )

    # Train the model
    model = nn.train(train_X, train_Y)

    # Test the model
    print(f"[Shallow NN Mushroom] Computing metrics for test set (iteration {i})")
    metrics_res, probs = nn.test(model, test, mushroom, mushroom_labels, verbose=False)
    metrics.append(metrics_res)

# Obtain averages for each classification metric
ClassificationEvaluator().compute_avg_metrics(
    metrics=metrics,
    iterations=iterations,
    test_str=f"[Shallow NN Mushroom] Test set results after {iterations} iterations:"
)

roc_curve_data.append((mushroom_labels[test], probs))

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

    # Create new logistic regression unit for mushroom
    model = LogisticRegression(
        epochs=2000,
        alpha=0.01
    )

    # Get train and test sets
    mushroom_labels_t = np.array([mushroom_labels]).T
    train_X, train_Y, test = ml.train_test_split(
        data=mushroom,
        labels=mushroom_labels_t,
        train_size=0.7
    )

    # Fit the model
    model.fit(train_X, train_Y)

    # Test the model (predict labels for test data)
    print(f"[Logistic Regression Mushroom] Computing metrics for test set (iteration {i})")
    metrics_res, probs = model.predict(
        data=mushroom[test],
        labels=mushroom_labels_t,
        test=test,
        summary=False
    )

    # metrics_res = nn.test(model, test, mushroom, mushroom_labels, verbose=False)
    metrics.append(metrics_res)

# Obtain averages for each classification metric
ClassificationEvaluator().compute_avg_metrics(
    metrics=metrics,
    iterations=iterations,
    test_str=f"[Logistic Regression Mushroom] Test set results after {iterations} iterations:"
)

roc_curve_data.append((mushroom_labels[test], probs))

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
        data=mushroom.astype(int),
        labels=mushroom_labels.astype(int),
        train_size=0.7
    )

    # Fit decision tree
    tree = DecisionTree(max_depth=1)
    tree.fit(train_X, train_Y)

    # Obtain metrics from classifier
    print(f"[Decision Tree Mushroom] Computing metrics for test set (iteration {i})")
    metrics_res, probs = tree.test(mushroom[test], mushroom_labels[test])

    metrics.append(metrics_res)

# Obtain averages for each classification metric
ClassificationEvaluator().compute_avg_metrics(
    metrics=metrics,
    iterations=iterations,
    test_str=f"[Decision Tree Neighbor Mushroom] Test set results after {iterations} iterations:"
)

roc_curve_data.append((mushroom_labels[test], probs))

########################################################################

# Plot ROC curve
true_labels = [roc_data[0] for roc_data in roc_curve_data]
probs = [roc_data[1] for roc_data in roc_curve_data]

alg_strs = ['RBFNN', 'Shallow NN', 'Logistic Regression', 'Decision Tree']

evaluator.plot_roc_curve(
    true_labels=true_labels,
    probs=probs,
    alg_strs=alg_strs,
    filename="./plots/classification_results/ROC_mushroom.pdf"
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
#             print(f"[Decision Tree Mushroom] Training and testing on depth {depth}")
#
#             # Get random training and testing sets
#             train_X, train_Y, test = ml.train_test_split(
#                 data=data.astype(int),
#                 labels=labels.astype(int),
#                 train_size=0.8
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
# # Test: test depths for mushroom data with imputed values
# metrics_test = test_depths(
#     data=mushroom,
#     labels=mushroom_labels,
#     max_val=max_val
# )


# Plot saved metrics (accuracy, precision, recall, F1) from saved file
# with open('mushroom_imputed_decisiontree_depths.csv', 'r') as file:
#     metrics = list(csv.reader(file))
#     for row in metrics:
#         row = [float(i) for i in row]
#         plt.plot(depths, row)

# accuracies = [str(round(metric['avg_accuracy'], 2)) for metric in metrics_test]
# precisions = [str(round(metric['avg_precision'], 2)) for metric in metrics_test]
# recalls = [str(round(metric['avg_recall'], 2)) for metric in metrics_test]
# f1s = [str(round(metric['avg_f1'], 2)) for metric in metrics_test]

# with open('mushroom_imputed_decisiontree_depths.csv', 'a') as file:
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
## 3.5 k-Nearest Neighbors
## (OOP implementation from scratch with guidance from https://www.edureka.co/blog/k-nearest-neighbors-algorithm/)
###

# metrics = []
#
# # Run training and testing multiple times and compute average
# for i in range(iterations):
#     print(i)
#
#     # Create new kNN model
#     knn = KNN()
#
#     # Create train and test sets
#     train_X, train_Y, test = ml.train_test_split(
#         data=mushroom,
#         labels=mushroom_labels,
#         train_size=0.7
#     )
#
#     # Fit model to training data with testing item
#     preds = knn.fit(
#         train_data=train_X,
#         train_labels=train_Y,
#         test_data=mushroom[test],
#         test_labels=mushroom_labels[test],
#         k=1
#     )
#
#     # Evaluate trained model
#     acc = knn.test(
#         test_labels=test_labels,
#         preds=preds
#     )
#
#     # Test the model (predict labels for test data)
#     print(f"[k-Nearest Neighbor Mushroom] Computing metrics for test set (iteration {i})")
#     metrics_res = model.predict(
#         data=mushroom[test],
#         labels=mushroom_labels_t,
#         test=test,
#         summary=False
#     )
#
#     metrics.append(metrics_res)
#
# # Obtain averages for each classification metric
# ClassificationEvaluator().compute_avg_metrics(
#     metrics=metrics,
#     iterations=iterations,
#     test_str=f"[k-Nearest Neighbor Mushroom] Test set results after {iterations} iterations:"
# )
