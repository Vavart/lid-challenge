# File: lid_challenge.py
# Author: Maxime Sciare
# Date: June 11, 2023
# Description: This script is used to train a model for the LID challenge

# Importing the libraries
import numpy as np

# Sklearn libraries
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Personal libraries
from lid_functions import *

"""
Load the audio files and their labels
"""

x_data, y_data = loadData()

"""
Split the dataset into training and validation sets
"""

x_train, x_test, y_train, y_test = shuffleAndSplit(x_data, y_data, test_size=0.3)


"""
Classifier (train a classifier on the training set and evaluate it on the validation set)
"""

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf, accuracy = trainAndEvaluate(clf, x_train, y_train, x_test, y_test)

"""
Evaluation (save the predictions on the evaluation set in a csv file)
"""

print("Save this try ? (y/n)")
answer = input()

if answer == "y" :
    savePredictOnEvaluation(clf, accuracy)
else : 
    print("No predictions saved")
    exit()