# ================================================================================================================================
# Author: Ryan C Hood
#
# Description: This is the python file responsible for training a Random Forest Classifier using the popular sklearn library.
#
# ================================================================================================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ******* NEW USERS MAY NEED TO CHANGE THIS VALUE ********
data_directory = "C:\\Users\\User\\Dropbox\\CS 6375 - Machine Learning\\homework\\CS6375 HW1\\hw1_data\\all_data\\"

print("Please Wait.  This should take about 30 seconds.")

# Unlike the decision tree code, I just decide to print the accuracies for all of the trees at the same time.
# So let's start out by getting all of the training and testing data.
c300d100_train = pd.read_csv(data_directory + "train_c300_d100.csv", header=None)
c300d1000_train = pd.read_csv(data_directory + "train_c300_d1000.csv", header=None)
c300d5000_train = pd.read_csv(data_directory + "train_c300_d5000.csv", header=None)
c500d100_train = pd.read_csv(data_directory + "train_c500_d100.csv", header=None)
c500d1000_train = pd.read_csv(data_directory + "train_c500_d1000.csv", header=None)
c500d5000_train = pd.read_csv(data_directory + "train_c500_d5000.csv", header=None)
c1000d100_train = pd.read_csv(data_directory + "train_c1000_d100.csv", header=None)
c1000d1000_train = pd.read_csv(data_directory + "train_c1000_d1000.csv", header=None)
c1000d5000_train = pd.read_csv(data_directory + "train_c1000_d5000.csv", header=None)
c1500d100_train = pd.read_csv(data_directory + "train_c1500_d100.csv", header=None)
c1500d1000_train = pd.read_csv(data_directory + "train_c1500_d1000.csv", header=None)
c1500d5000_train = pd.read_csv(data_directory + "train_c1500_d5000.csv", header=None)
c1800d100_train = pd.read_csv(data_directory + "train_c1800_d100.csv", header=None)
c1800d1000_train = pd.read_csv(data_directory + "train_c1800_d1000.csv", header=None)
c1800d5000_train = pd.read_csv(data_directory + "train_c1800_d5000.csv", header=None)

c300d100_test = pd.read_csv(data_directory + "test_c300_d100.csv", header=None)
c300d1000_test = pd.read_csv(data_directory + "test_c300_d1000.csv", header=None)
c300d5000_test = pd.read_csv(data_directory + "test_c300_d5000.csv", header=None)
c500d100_test = pd.read_csv(data_directory + "test_c500_d100.csv", header=None)
c500d1000_test = pd.read_csv(data_directory + "test_c500_d1000.csv", header=None)
c500d5000_test = pd.read_csv(data_directory + "test_c500_d5000.csv", header=None)
c1000d100_test = pd.read_csv(data_directory + "test_c1000_d100.csv", header=None)
c1000d1000_test = pd.read_csv(data_directory + "test_c1000_d1000.csv", header=None)
c1000d5000_test = pd.read_csv(data_directory + "test_c1000_d5000.csv", header=None)
c1500d100_test = pd.read_csv(data_directory + "test_c1500_d100.csv", header=None)
c1500d1000_test = pd.read_csv(data_directory + "test_c1500_d1000.csv", header=None)
c1500d5000_test = pd.read_csv(data_directory + "test_c1500_d5000.csv", header=None)
c1800d100_test = pd.read_csv(data_directory + "test_c1800_d100.csv", header=None)
c1800d1000_test = pd.read_csv(data_directory + "test_c1800_d1000.csv", header=None)
c1800d5000_test = pd.read_csv(data_directory + "test_c1800_d5000.csv", header=None)

# Now lets split our attributes into features and classes for both training and testing data.
X_c300d100_train = c300d100_train.iloc[:,0:500].values
X_c300d1000_train = c300d1000_train.iloc[:,0:500].values
X_c300d5000_train = c300d5000_train.iloc[:,0:500].values
X_c500d100_train = c500d100_train.iloc[:,0:500].values
X_c500d1000_train = c500d1000_train.iloc[:,0:500].values
X_c500d5000_train = c500d5000_train.iloc[:,0:500].values
X_c1000d100_train = c1000d100_train.iloc[:,0:500].values
X_c1000d1000_train = c1000d1000_train.iloc[:,0:500].values
X_c1000d5000_train = c1000d5000_train.iloc[:,0:500].values
X_c1500d100_train = c1500d100_train.iloc[:,0:500].values
X_c1500d1000_train = c1500d1000_train.iloc[:,0:500].values
X_c1500d5000_train = c1500d5000_train.iloc[:,0:500].values
X_c1800d100_train = c1800d100_train.iloc[:,0:500].values
X_c1800d1000_train = c1800d1000_train.iloc[:,0:500].values
X_c1800d5000_train = c1800d5000_train.iloc[:,0:500].values

y_c300d100_train = c300d100_train.iloc[:,500].values
y_c300d1000_train = c300d1000_train.iloc[:,500].values
y_c300d5000_train = c300d5000_train.iloc[:,500].values
y_c500d100_train = c500d100_train.iloc[:,500].values
y_c500d1000_train = c500d1000_train.iloc[:,500].values
y_c500d5000_train = c500d5000_train.iloc[:,500].values
y_c1000d100_train = c1000d100_train.iloc[:,500].values
y_c1000d1000_train = c1000d1000_train.iloc[:,500].values
y_c1000d5000_train = c1000d5000_train.iloc[:,500].values
y_c1500d100_train = c1500d100_train.iloc[:,500].values
y_c1500d1000_train = c1500d1000_train.iloc[:,500].values
y_c1500d5000_train = c1500d5000_train.iloc[:,500].values
y_c1800d100_train = c1800d100_train.iloc[:,500].values
y_c1800d1000_train = c1800d1000_train.iloc[:,500].values
y_c1800d5000_train = c1800d5000_train.iloc[:,500].values


X_c300d100_test = c300d100_test.iloc[:,0:500].values
X_c300d1000_test = c300d1000_test.iloc[:,0:500].values
X_c300d5000_test = c300d5000_test.iloc[:,0:500].values
X_c500d100_test = c500d100_test.iloc[:,0:500].values
X_c500d1000_test = c500d1000_test.iloc[:,0:500].values
X_c500d5000_test = c500d5000_test.iloc[:,0:500].values
X_c1000d100_test = c1000d100_test.iloc[:,0:500].values
X_c1000d1000_test = c1000d1000_test.iloc[:,0:500].values
X_c1000d5000_test = c1000d5000_test.iloc[:,0:500].values
X_c1500d100_test = c1500d100_test.iloc[:,0:500].values
X_c1500d1000_test = c1500d1000_test.iloc[:,0:500].values
X_c1500d5000_test = c1500d5000_test.iloc[:,0:500].values
X_c1800d100_test = c1800d100_test.iloc[:,0:500].values
X_c1800d1000_test = c1800d1000_test.iloc[:,0:500].values
X_c1800d5000_test = c1800d5000_test.iloc[:,0:500].values


y_c300d100_test = c300d100_test.iloc[:,500].values
y_c300d1000_test = c300d1000_test.iloc[:,500].values
y_c300d5000_test = c300d5000_test.iloc[:,500].values
y_c500d100_test = c500d100_test.iloc[:,500].values
y_c500d1000_test = c500d1000_test.iloc[:,500].values
y_c500d5000_test = c500d5000_test.iloc[:,500].values
y_c1000d100_test = c1000d100_test.iloc[:,500].values
y_c1000d1000_test = c1000d1000_test.iloc[:,500].values
y_c1000d5000_test = c1000d5000_test.iloc[:,500].values
y_c1500d100_test = c1500d100_test.iloc[:,500].values
y_c1500d1000_test = c1500d1000_test.iloc[:,500].values
y_c1500d5000_test = c1500d5000_test.iloc[:,500].values
y_c1800d100_test = c1800d100_test.iloc[:,500].values
y_c1800d1000_test = c1800d1000_test.iloc[:,500].values
y_c1800d5000_test = c1800d5000_test.iloc[:,500].values


# Now we contruct the RandomForestClassifier for each training data-set.
classifier_c300d100 = RandomForestClassifier()
classifier_c300d100.fit(X_c300d100_train, y_c300d100_train)
y_pred_c300d100 = classifier_c300d100.predict(X_c300d100_test)

classifier_c300d1000 = RandomForestClassifier()
classifier_c300d1000.fit(X_c300d1000_train, y_c300d1000_train)
y_pred_c300d1000 = classifier_c300d1000.predict(X_c300d1000_test)

classifier_c300d5000 = RandomForestClassifier()
classifier_c300d5000.fit(X_c300d5000_train, y_c300d5000_train)
y_pred_c300d5000 = classifier_c300d5000.predict(X_c300d5000_test)

classifier_c500d100 = RandomForestClassifier()
classifier_c500d100.fit(X_c500d100_train, y_c500d100_train)
y_pred_c500d100 = classifier_c500d100.predict(X_c500d100_test)

classifier_c500d1000 = RandomForestClassifier()
classifier_c500d1000.fit(X_c500d1000_train, y_c500d1000_train)
y_pred_c500d1000 = classifier_c500d1000.predict(X_c500d1000_test)

classifier_c500d5000 = RandomForestClassifier()
classifier_c500d5000.fit(X_c500d5000_train, y_c500d5000_train)
y_pred_c500d5000 = classifier_c500d5000.predict(X_c500d5000_test)

classifier_c1000d100 = RandomForestClassifier()
classifier_c1000d100.fit(X_c1000d100_train, y_c1000d100_train)
y_pred_c1000d100 = classifier_c1000d100.predict(X_c1000d100_test)

classifier_c1000d1000 = RandomForestClassifier()
classifier_c1000d1000.fit(X_c1000d1000_train, y_c1000d1000_train)
y_pred_c1000d1000 = classifier_c1000d1000.predict(X_c1000d1000_test)

classifier_c1000d5000 = RandomForestClassifier()
classifier_c1000d5000.fit(X_c1000d5000_train, y_c1000d5000_train)
y_pred_c1000d5000 = classifier_c1000d5000.predict(X_c1000d5000_test)

classifier_c1500d100 = RandomForestClassifier()
classifier_c1500d100.fit(X_c1500d100_train, y_c1500d100_train)
y_pred_c1500d100 = classifier_c1500d100.predict(X_c1500d100_test)

classifier_c1500d1000 = RandomForestClassifier()
classifier_c1500d1000.fit(X_c1500d1000_train, y_c1500d1000_train)
y_pred_c1500d1000 = classifier_c1500d1000.predict(X_c1500d1000_test)

classifier_c1500d5000 = RandomForestClassifier()
classifier_c1500d5000.fit(X_c1500d5000_train, y_c1500d5000_train)
y_pred_c1500d5000 = classifier_c1500d5000.predict(X_c1500d5000_test)

classifier_c1800d100 = RandomForestClassifier()
classifier_c1800d100.fit(X_c1800d100_train, y_c1800d100_train)
y_pred_c1800d100 = classifier_c1800d100.predict(X_c1800d100_test)

classifier_c1800d1000 = RandomForestClassifier()
classifier_c1800d1000.fit(X_c1800d1000_train, y_c1800d1000_train)
y_pred_c1800d1000 = classifier_c1800d1000.predict(X_c1800d1000_test)

classifier_c1800d5000 = RandomForestClassifier()
classifier_c1800d5000.fit(X_c1800d5000_train, y_c1800d5000_train)
y_pred_c1800d5000 = classifier_c1800d5000.predict(X_c1800d5000_test)


# We get the accuracies by comparing the predicitons just found to the actual values.
c300d100_accuracy = accuracy_score(y_c300d100_test, y_pred_c300d100)
c300d1000_accuracy = accuracy_score(y_c300d1000_test, y_pred_c300d1000)
c300d5000_accuracy = accuracy_score(y_c300d5000_test, y_pred_c300d5000)
c500d100_accuracy = accuracy_score(y_c500d100_test, y_pred_c500d100)
c500d1000_accuracy = accuracy_score(y_c500d1000_test, y_pred_c500d1000)
c500d5000_accuracy = accuracy_score(y_c500d5000_test, y_pred_c500d5000)
c1000d100_accuracy = accuracy_score(y_c1000d100_test, y_pred_c1000d100)
c1000d1000_accuracy = accuracy_score(y_c1000d1000_test, y_pred_c1000d1000)
c1000d5000_accuracy = accuracy_score(y_c1000d5000_test, y_pred_c1000d5000)
c1500d100_accuracy = accuracy_score(y_c1500d100_test, y_pred_c1500d100)
c1500d1000_accuracy = accuracy_score(y_c1500d1000_test, y_pred_c1500d1000)
c1500d5000_accuracy = accuracy_score(y_c1500d5000_test, y_pred_c1500d5000)
c1800d100_accuracy = accuracy_score(y_c1800d100_test, y_pred_c1800d100)
c1800d1000_accuracy = accuracy_score(y_c1800d1000_test, y_pred_c1800d1000)
c1800d5000_accuracy = accuracy_score(y_c1800d5000_test, y_pred_c1800d5000)




# Put the accuracies in a list.
accuracies = [c300d100_accuracy,
c300d1000_accuracy,
c300d5000_accuracy,
c500d100_accuracy,
c500d1000_accuracy,
c500d5000_accuracy,
c1000d100_accuracy,
c1000d1000_accuracy,
c1000d5000_accuracy,
c1500d100_accuracy,
c1500d1000_accuracy,
c1500d5000_accuracy,
c1800d100_accuracy,
c1800d1000_accuracy,
c1800d5000_accuracy]

# Print the list.
print(accuracies)


print("c300d100 accuracy: " + str(accuracies[0]) + "\n"
        + "c300d1000 accuracy: " + str(accuracies[1]) + "\n"
        + "c300d5000 accuracy: " + str(accuracies[2]) + "\n"
        + "c500d100 accuracy: " + str(accuracies[3]) + "\n"
        + "c500d1000 accuracy: " + str(accuracies[4]) + "\n"
        + "c500d5000 accuracy: " + str(accuracies[5]) + "\n"
        + "c1000d100 accuracy: " + str(accuracies[6]) + "\n"
        + "c1000d1000 accuracy: " + str(accuracies[7]) + "\n"
        + "c1000d5000 accuracy: " + str(accuracies[8]) + "\n"
        + "c1500d100 accuracy: " + str(accuracies[9]) + "\n"
        + "c1500d1000 accuracy: " + str(accuracies[10]) + "\n"
        + "c1500d5000 accuracy: " + str(accuracies[11]) + "\n"
        + "c1800d100 accuracy: " + str(accuracies[12]) + "\n"
        + "c1800d1000 accuracy: " + str(accuracies[13]) + "\n"
        + "c1800d5000 accuracy: " + str(accuracies[14]) + "\n")
