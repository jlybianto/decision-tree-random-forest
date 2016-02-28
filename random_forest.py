# ----------------
# IMPORT PACKAGES
# ----------------

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as skm

# ----------------
# OBTAIN DATA
# ----------------

# Data Source: https://archive.ics.uci.edu/ml/machine-learning-databases/00240/

# ----------------
# PROFILE DATA
# ----------------

# Determine number of observations or data points in the training data set.
subjects = pd.read_csv("uci-har-dataset/train/subject_train.txt", header=None, delim_whitespace=True, index_col=False)
observations = len(subjects)
participants = len(subjects.stack().value_counts())
subjects.columns = ["Subject"]
print("Number of Observations: " + str(observations))
print("Number of Participants: " + str(participants))

# Determine the number of features in the data set.
features = pd.read_csv("uci-har-dataset/features.txt", header=None, delim_whitespace=True, index_col=False)
num_features = len(features)
print("Number of Features: " + str(num_features))
print("")

# Data munging of the predictor and target variables.
x = pd.read_csv("uci-har-dataset/train/X_train.txt", header=None, delim_whitespace=True, index_col=False)
y = pd.read_csv("uci-har-dataset/train/y_train.txt", header=None, delim_whitespace=True, index_col=False)