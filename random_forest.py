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

# Data munging of the predictor and target variables starting with the column names.
x = pd.read_csv("uci-har-dataset/train/X_train.txt", header=None, delim_whitespace=True, index_col=False)
y = pd.read_csv("uci-har-dataset/train/y_train.txt", header=None, delim_whitespace=True, index_col=False)

col = [i.replace("()-", "") for i in features[1]] # Remove inclusion of "()-" in column names
col = [i.replace(",", "") for i in col] # Remove inclusion of "," in column names
col = [i.replace("()", "") for i in col] # Remove inclusion of "()" in column names
col = [i.replace("Body", "") for i in col] # Drop "Body" and "Mag" from column names
col = [i.replace("Mag", "") for i in col]
col = [i.replace("mean", "Mean") for i in col] # Rename "Mean" and "Standard Deviation"
col = [i.replace("std", "STD") for i in col]

x.columns = col
y.columns = ["Activity"]
# 1 = Walking, 2 = Walking Upstairs, 3 = Walking Downstairs, 4 = Sitting, 5 = Standing, 6 = Laying

data = pd.merge(y, x, left_index=True, right_index=True)
data = pd.merge(data, subjects, left_index=True, right_index=True)
data["Activity"] = pd.Categorical(data["Activity"]).labels

# ----------------
# MODEL DATA
# ----------------

# Partitioning of aggregate data into training, testing and validation data sets
train = data.query("Subject >= 27")
test = data.query("Subject <= 6")
valid = data.query("(Subject >= 21) & (Subject < 27)")

train_target = train["Activity"]
train_data = train.ix[:, 1:-2]
rfc = RandomForestClassifier(n_estimators=500, oob_score=True)
rfc.fit(train_data, train_target)

rfc.oob_score_

importance = rfc.feature_importances_
rank = np.argsort(importance)[::-1]