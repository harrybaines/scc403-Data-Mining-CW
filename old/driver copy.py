# Import dependencies
import numpy as np # matrices
import matplotlib.pyplot as plt # plotting
import matplotlib as mp # plotting
import pandas as pd # visualising correlations
import seaborn as sns # plotting
import math
import random
import time

from scipy import linalg # linear algebra

import preprocessing.scalers as scalers
import preprocessing.ml as ml
import preprocessing.fileopts as fo

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

###
## Reading data and column name files
###

# Read in pulsar data
pulsar_filepath = "./Data/Pulsar/HTRU_2.csv"
pulsar = fo.open_csv(pulsar_filepath).astype(float)

# Open pulsar column names
pulsar_cols_filepath = './Data/Pulsar/pulsar_cols.txt'
pulsar_columns = fo.open_cols(pulsar_cols_filepath)

# Read in abalone data
abalone_filepath = "./Data/Abalone/abalone19.csv"
abalone = fo.open_csv(abalone_filepath)

# Open abalone column names
abalone_cols_filepath = './Data/Abalone/abalone_cols.txt'
abalone_columns = fo.open_cols(abalone_cols_filepath)

# Read in mushroom data
mushroom_filepath = "./Data/Mushroom/agaricus-lepiota.csv"
mushroom = fo.open_csv(mushroom_filepath)

# Open mushroom column names
mushroom_cols_filepath = './Data/Mushroom/mushroom_cols.txt'
mushroom_columns = fo.open_cols(mushroom_cols_filepath)


###
## Abalone preprocessing
###

# Set labels to 1 for positive and 0 for negative
abalone = np.char.strip(abalone)
abalone[abalone == 'positive'] = 1
abalone[abalone == 'negative'] = 0
abalone_labels = abalone[:, -1].astype(int)

from preprocessing.encoding import LabelEncoder
from preprocessing.encoding import OneHotEncoder

# Label encoding from scratch
label_encoder = LabelEncoder()

# Encode sex using label encoding
abalone = label_encoder.fit_transform(
    data=abalone,
    col_idx=0
)


###
## Mushroom preprocessing
###

import pandas as pd

# One-hot encoding from scratch
one_hot_encoder = OneHotEncoder()

# (Used for labelling the one-hot encoded columns for clarity)
unique_col_vals = list(np.unique(mushroom[:, col]) for col in range(1, mushroom.shape[1]))

# Label encode categorical values before one-hot encoding
mushroom = label_encoder.fit_transform(
    data=mushroom,
    col_idx=-1
)

mushroom_classes = mushroom[:, 0]

# One-hot encoding for categorical features in mushroom
mushroom, mushroom_enc_columns = one_hot_encoder.fit_transform(
    data=mushroom[:, 1:],
    columns=mushroom_columns[1:],
    unique_col_vals=unique_col_vals
)

# Append class to front (didn't need one-hot encoding)
mushroom = np.concatenate((mushroom_classes.reshape((len(mushroom), 1)), mushroom), axis=1)
mushroom_enc_columns = np.insert(mushroom_enc_columns, 0, 'class', axis=0)


###
## Missing Data
###
def impute_mode(data, missing_char='?'):
    pass

def remove_missing_rows(data, missing_char='?'):
    pass

def remove_column_with_missing(data, missing_char='?'):
    pass


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
#     filename="standardized_pulsar_1.pdf",
#     scale_type="stand"
# )
# plot(
#     data=pulsar,
#     x_col_ind=5,
#     y_col_ind=4,
#     cols=pulsar_columns,
#     filename="standardized_pulsar_2.pdf",
#     scale_type="stand"
# )


###
## Abalone EDA
###
# # Extract numerical columns and normalize them
# abalone_columns_sub = abalone_columns[1:8]
#
# plt.bar(['Not Age Class 19', 'Age Class 19'], [get_num_labels(abalone_labels, 0), get_num_labels(abalone_labels, 1)], color=['green', '#eb3734'])
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


###
## Mushroom Exploratory Data Analysis
###

# Get total number of edible mushrooms
num_edible = get_row_count(mushroom, 0, 'e')
print(f"{num_edible} edible mushrooms")

# Get total number of poisonous mushrooms
num_poisonous = get_row_count(mushroom, 0, 'p')
print(f"{num_poisonous} poisonous mushrooms")

# fo.head(mushroom)

# plt.bar(['Edible', 'Poisonous'], [num_edible, num_poisonous], color=['green', '#eb3734'])
# plt.title('Count of Poisonous and Edible Mushrooms')
# plt.show()

counts = []
for i in range(len(mushroom_columns)):
    count = np.sum(mushroom[:, i] == '?')
    count > 0 and print(f'Feature {i}: {count} missing value(s)')

missing_values = get_num_missing_values(mushroom, '?')
print(f'Total missing values in mushroom: {missing_values}')


###
## Principal Component Analysis (PCA)
###

##
# Pulsar PCA
##
from preprocessing.pca import PCA

# Instantiate PCA class and fit to pulsar data
pulsar_pca = PCA(n_components=6)
pulsar_pca.fit(pulsar)

# Plot principal components with most of the variance
legend = [(0, "Not Pulsar Star"), (1, "Pulsar Star")]
pulsar_pca.plot(
    labels=pulsar[:, -1],
    legend=legend,
    color_labels=False,
    filename='PulsarPCA.pdf'
)

# Plot explained variance
pulsar_pca.scree_plot(filename='PulsarScree.pdf')

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

##
# Mushroom PCA
##

# Remove uninformative features with mean of 0 or 1
new_mushroom = mushroom.copy()
inds = []

for col_idx in range(mushroom.shape[1]):
    col_mean = np.mean(mushroom[:, col_idx])
    if col_mean == 0 or col_mean == 1:
        inds.append(col_idx)

mushroom = np.delete(mushroom, inds, axis=1)

# Instantiate PCA class and fit to mushroom data
# pca = PCA(n_components=10)
# pca.fit(mushroom)
#
# # Obtain centralized data
# centralized_data = pca.centralized
#
# # Plot principal components with most of the variance
# legend = [(0, "Edible"), (1, "Poisonous")]
# pca.plot(
#     data=centralized_data,
#     labels=mushroom[:, 0],
#     legend=legend,
#     filename='MushroomPCA.pdf'
# )
#
# # Plot explained variance
# pca.scree_plot(filename='MushroomScree.pdf')

# Visualise and analyse correlation between features
# pca.get_corr_coef()

##
# Abalone PCA
##

# Instantiate PCA class and fit to mushroom data
pca = PCA(n_components=8)
pca.fit(abalone)

# Obtain centralized data
centralized_data = pca.centralized

# Plot principal components with most of the variance
legend = [(0, "Not Age Class 19"), (1, "Age Class 19")]
pca.plot(
    # data=centralized_data,
    labels=abalone[:, -1],
    legend=legend,
    filename='AbalonePCA.pdf'
)

# Plot explained variance
pca.scree_plot(filename='AbaloneScree.pdf')

# Visualise and analyse correlation between features
pca.get_corr_coef()


###
## Sampling: dealing with class imbalance for abalone data
###
from preprocessing.sampler import Sampler

# Obtain minority class from labels
minority_class = get_minority_class(abalone_labels)

label_counts = get_label_counts(abalone_labels)
print(f"[Before sampling]\n{label_counts}")

sampler = Sampler()

## Oversampling
no_to_oversample = 4000
label_to_oversample = 1
os_abalone_data, os_abalone_labels = sampler.over_sample(abalone, abalone_labels, k=no_to_oversample, l=label_to_oversample)

label_counts = get_label_counts(os_abalone_labels)
print(f"[After oversampling {no_to_oversample} label {label_to_oversample} items]\n{label_counts}")

# Automatic detection and oversampling of minority class
os_abalone_data, os_abalone_labels = sampler.over_sample(abalone, abalone_labels, auto=True)

label_counts = get_label_counts(os_abalone_labels)
print(f"[After automatic oversampling]\n{label_counts}")

## Undersampling
no_to_undersample = 2500
label_to_undersample = 0
us_abalone_data, us_abalone_labels = sampler.under_sample(abalone, abalone_labels, k=no_to_undersample, l=label_to_undersample)

label_counts = get_label_counts(us_abalone_labels)
print(f"[After undersampling {no_to_undersample} label {label_to_undersample} items]\n{label_counts}")

# Automatic detection and oversampling of minority class
us_abalone_data, us_abalone_labels = sampler.under_sample(abalone, abalone_labels, auto=True)

label_counts = get_label_counts(us_abalone_labels)
print(f"[After automatic undersampling]\n{label_counts}")

## SMOTE
# Create new Sampler instance to use SMOTE on abalone data
sampler = Sampler()

no_to_oversample = 10
label_to_oversample = 1

smote_abalone_data, smote_abalone_labels = sampler.smote(
    data=abalone,
    labels=abalone_labels,
    label=label_to_oversample,
    k_neighbours=2,
    iterations=no_to_oversample
)

label_counts = get_label_counts(smote_abalone_labels)
print(f"[After SMOTE oversampling with {no_to_oversample} label {label_to_oversample} items]\n{label_counts}")

###
## Outliers
###

# Find outliers for pulsar data
pulsar_sub = pulsar[:, 0:8]
# find_outliers(pulsar_sub)
