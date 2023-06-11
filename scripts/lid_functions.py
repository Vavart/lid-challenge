# File: lid_functions.py
# Author: Maxime Sciare
# Date: June 11, 2023
# Description: This script contains the functions used in the LID challenge

"""
Libraries
"""

# Importing the libraries for feature extraction and data processing
import librosa
import numpy as np
import csv
from datetime import datetime
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Personal libraries
import env

"""
Global variables
"""
# Create a dictionary for labels    
lang_dic={'EN':0,'FR':1,'AR':2,'JP':3}
dic_lang={0:"EN",1:"FR",2:"AR",3:"JP"}


"""
Feature extraction and data processing functions
"""

# Function to extract the features from the audio files
def featureExtractor(audio_file_dir):

    # Load the audio files
    x,freq = librosa.load(audio_file_dir,sr=16000)
    
    x_normalized = normalize_audio(x)
    
    # Extract 20 MFCCs
    # mfcc=librosa.feature.mfcc(y=x,sr=freq,n_mfcc=20)
    mfcc=librosa.feature.mfcc(y=x_normalized,sr=freq,n_mfcc=20)
    
    # Calculate statistics for each MFCC
    mean_mfccs=np.mean(mfcc,axis=1)
    var_mfccs=np.var(mfcc,axis=1)
    std_mfccs=np.std(mfcc,axis=1)
    min_mfccs=np.min(mfcc,axis=1)
    max_mfccs=np.max(mfcc,axis=1)
    
    # Return the list of features 
    return list(mean_mfccs)+list(var_mfccs)+list(std_mfccs)+list(max_mfccs)+list(min_mfccs)

# Get x_data and y_data
def loadData() :

    print("Loading data...")

    # Read file info file to get the list of audio files and their labels
    file_list=[]
    label_list=[]
    with open(env.DATASET_DIR + "Info.txt", 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # The first column contains the file name
            file_list.append(row[0])
            # The last column contains the lable (language)
            label_list.append(row[-1]) 
            
            

    lang_dic={'EN':0,'FR':1,'AR':2,'JP':3}

    # Create a list of extracted feature (MFCC) for files
    x_data=[]

    # Extract the features for each audio file
    for audio_file in file_list:
        
        # Extract the features
        file_feature = featureExtractor(env.DATASET_DIR + audio_file)
        
        # Add extracted feature to dataset 
        x_data.append(file_feature)

    # Create a list of labels for files
    y_data=[]
    for lang_label in label_list:
        # Convert the label to a value in {0,1,2,3} as the class label
        y_data.append(lang_dic[lang_label])
        
    # Return the dataset
    print("Data loaded")
    return x_data, y_data


# Function to scale the features
def scale_features(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features


# Normalize the features
def normalize_audio(audio):
    max_value = np.max(np.abs(audio))
    normalized_audio = audio / max_value
    return normalized_audio



"""
Shuffle and split the dataset functions
"""

def shuffleAndSplit(x_data, y_data, test_size=0.3) :
    
    # Shuffle data before splitting
    temp_list = list(zip(x_data, y_data))
    random.shuffle(temp_list)
    shuffled_x_data, shuffled_y_data = zip(*temp_list)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(shuffled_x_data, shuffled_y_data, test_size=test_size, shuffle=True)
    
    # Return the splitted dataset
    return x_train, x_test, y_train, y_test


"""
Classifier training and testing functions
"""

# Function to train the classifier
def trainAndEvaluate(clf, x_train, y_train, x_test, y_test) :
        
        print("Training the classifier...")
        
        # Train the classifier
        clf.fit(x_train, y_train)
        
        # Test the classifier
        accuracy = clf.score(x_test, y_test)
        
        print(f"Classifier trained with accuracy = {accuracy}")
    
        # Return the trained classifier and the accuracy
        return clf, accuracy


"""
Evaluation set submission (generate the csv file)
"""

def savePredictOnEvaluation(clf, accuracy) :
    
    print("Saving predictions on the evaluation set")
    
    # Prepare the csv file
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename = env.SUBMISSION_DIR + "(" + f"{accuracy:.2f}" + ")" + date_time + ".csv"
    with open(filename,'w') as file:
        file.write("File_name,Lang\n")
    
    # Read file info file to get the list of audio files and their labels
    file_list=[]
    with open(env.EVALUATION_DIR + "Info.txt", 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # The first column contains the file name
            file_list.append(row[0])

    for test_sample in file_list:
        test_sample_feature = featureExtractor(env.EVALUATION_DIR + test_sample)
        predicted = dic_lang[clf.predict([test_sample_feature])[0]]

        # Save the predicted output in filename.csv
        with open(filename,'a+') as file:
            file.write(f"{test_sample},{predicted}\n")
    
    print(f"Predictions saved in {filename}.csv")