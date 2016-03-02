# ----------------
# IMPORT PACKAGES
# ----------------

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as skm
import numpy as np
import matplotlib.pyplot as plt

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

# Fit random forest model with training data.
n = raw_input("Insert number of estimators to be used (10-500): ")
train_target = train["Activity"]
train_data = train.ix[:, 1:-2]
rfc = RandomForestClassifier(n_estimators=int(n), oob_score=True)
rfc.fit(train_data, train_target)
print("")

# Calculate Out-Of-Bag (OOB) score
print("Out-Of-Bag (OOB) Score: %f" % rfc.oob_score_)
print("")

# Determine the important features
rank = rfc.feature_importances_
index = np.argsort(rank)[::-1]
print("Top 10 Important Features:")
for i in range(10):
	print("%d. Feature #%d: %s (%f)" % (i + 1, index[i], x.columns[index[i]], rank[index[i]]))
print("")

# Define validation and test set to make predictions
valid_target = valid["Activity"]
valid_data = valid.ix[:, 1:-2]
valid_pred = rfc.predict(valid_data)

test_target = test["Activity"]
test_data = test.ix[:, 1:-2]
test_pred = rfc.predict(test_data)

# Calculation of scores
print("Mean Accuracy score for validation data set = %f" %(rfc.score(valid_data, valid_target)))
print("Mean Accuracy score for test data set = %f" %(rfc.score(test_data, test_target)))

print("Precision = %f" %(skm.precision_score(test_target, test_pred)))
print("Recall = %f" %(skm.recall_score(test_target, test_pred)))
print("F1 Score = %f" %(skm.f1_score(test_target, test_pred)))

# ----------------
# VISUALIZE DATA
# ----------------

# Visualization through a confusion matrix
graph = skm.confusion_matrix(test_target, test_pred)
plt.matshow(graph)
plt.title('Confusion Matrix for Test Data')
plt.colorbar()
plt.show()