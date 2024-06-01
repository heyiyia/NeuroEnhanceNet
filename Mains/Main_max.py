'''
Maximum predicted score as an individual predicted score.
'''

import numpy as np
import os
import random
import math
import cv2
import tensorflow as tf
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import pandas as pd
from Final.Models.Model import Conformer_Model

# Directory containing the dataset
data_directory = 'D:\\Document_TongyueHe\\Datasets_TongyueHe\\mPower_walking_dataset\\zaoqi\\deviceMotion_rest_all1'
txt_files = [f for f in os.listdir(data_directory) if f.endswith('.txt')]

def process_file(file_path):
    """
    Process a single file and extract the relevant data and label.
    """
    df = pd.read_csv(file_path, sep=',', header=None)
    for col in df.columns[8:14]:
        if df[col].dtype == 'object':
            df[col] = df[col].str.rstrip(';').astype(float)
    last_three_columns = df.iloc[:, 8:14].to_numpy().T
    num_rows = len(df)
    sub_data = np.zeros((6, 2500))
    if num_rows >= 2500:
        sub_data[:, :2500] = last_three_columns[:, 0:2500]
    else:
        sub_data[:, 0:num_rows] = last_three_columns
    label = 1 if df.iloc[2, 2] else 0
    return sub_data, label

def norm_axis(a, b, c):
    """
    Normalize the axis vector.
    """
    newa = a / (math.sqrt(float(a * a + b * b + c * c)))
    newb = b / (math.sqrt(float(a * a + b * b + c * c)))
    newc = c / (math.sqrt(float(a * a + b * b + c * c)))
    return [newa, newb, newc]

def rotation_matrix(axis, theta):
    """
    Generate a rotation matrix for rotating the data.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def rotateC(image, theta, a, b, c):
    """
    Rotate the given data.
    """
    axis = norm_axis(a, b, c)
    imagenew = np.dot(rotation_matrix(axis, theta), image)
    return imagenew

def scaleImage(image, scale):
    """
    Scale the given data.
    """
    [x, y] = image.shape
    y1 = int(y * scale)
    x1 = 3
    image = cv2.resize(image, (y1, x1))
    new = np.zeros((x, y))
    if y1 > y:
        start = 0
        end = start + y
        new = image[:, start:end]
    else:
        new_start = 0
        new_end = new_start + y1
        new[:, new_start:new_end] = image
    return new

# Define data augmentation
def augment_data(X_train, max_scale=1.1, min_scale=0.9):
    for i in range(len(X_train)):
        sub_data = X_train[i]
        if sub_data.shape[0] == 6:
            # First augmentation method for 6-dimensional data
            for j in range(6):
                rrr = random.random()
                rrr_scale = rrr * (max_scale - min_scale) + min_scale
                sub_data[j, :] = sub_data[j, :] * rrr_scale
        elif sub_data.shape[0] == 3:
            # Second augmentation method for 3-dimensional data
            rrr = random.random()
            rrr_scale = rrr * (max_scale - min_scale) + min_scale
            theta = random.random() * math.pi * 2
            a = random.random()
            b = random.random()
            c = random.random()
            sub_data = scaleImage(sub_data, rrr_scale)
            sub_data = rotateC(sub_data, theta, a, b, c)
            for j in range(3):
                rrr = random.random()
                rrr_scale = rrr * (max_scale - min_scale) + min_scale
                sub_data[j, :] = sub_data[j, :] * rrr_scale
        X_train[i] = sub_data
    return X_train

# Load individual data files and their corresponding labels
individuals = {}
for file in txt_files:
    with open(os.path.join(data_directory, file), 'r') as f:
        lines = f.readlines()
        if len(lines) > 0:
            data_parts = lines[0].strip().split(',')
            individual_id = data_parts[1]
            individuals.setdefault(individual_id, []).append(file)

individual_ids = list(individuals.keys())
labels = []
for ind_id in individual_ids:
    file_path = os.path.join(data_directory, individuals[ind_id][0])
    with open(file_path, 'r') as f:
        lines = f.readlines()
        if len(lines) > 0:
            label = int(lines[0].strip().split(',')[2] == 'True')
            labels.append(label)
labels = np.array(labels)

def main():
    num_splits = 5  # Number of splits for cross-validation
    num_epochs = 25  # Number of epochs for training
    batch_size = 64  # Batch size for training
    learning_rate = 0.001  # Learning rate for the optimizer

    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)

    fold_aucs = []  # Store AUC values for each fold
    fold_sensitivities = []  # Store sensitivity values for each fold
    fold_fnrs = []  # Store false negative rate values for each fold

    for fold, (train_index, test_index) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        # Separate training and testing individuals based on the current fold
        train_individuals = [individual_ids[i] for i in train_index]
        test_individuals = [individual_ids[i] for i in test_index]

        # Get the corresponding files for training and testing
        train_files = [file for ind_id in train_individuals for file in individuals[ind_id]]
        test_files = [file for ind_id in test_individuals for file in individuals[ind_id]]

        train_data = []
        train_labels = []
        test_data = []
        test_labels = []

        # Process training files
        for file in train_files:
            file_path = os.path.join(data_directory, file)
            data, label = process_file(file_path)
            train_data.append(data)
            train_labels.append(label)

        # Process testing files
        for file in test_files:
            file_path1 = os.path.join(data_directory, file)
            data1, label1 = process_file(file_path1)
            test_data.append(data1)
            test_labels.append(label1)

        X_train = np.array(train_data)
        y_train = np.array(train_labels)
        X_test = np.array(test_data)
        y_test = np.array(test_labels)

        # Data augmentation for training data
        X_train = augment_data(X_train)

        # Reshape data for the model
        X_train = X_train.transpose(0, 2, 1)
        X_test = X_test.transpose(0, 2, 1)

        # Apply SMOTE to balance the training data
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train.reshape(-1, 6 * 2500), y_train)
        X_train_resampled = X_train_resampled.reshape(-1, 2500, 6)

        # Define the model input shape
        input_shape = (2500, 6)
        data_input = tf.keras.Input(shape=input_shape)

        # Create the model using Conformer_Model
        Conformer_output = Conformer_Model(data_input)
        model = tf.keras.Model(inputs=data_input, outputs=Conformer_output)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, name='Adam')
        metrics = ['binary_accuracy']
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)
        model.summary()

        # Train the model
        model.fit(X_train_resampled, y_train_resampled, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))

        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {test_accuracy}")

        individual_predictions = {}
        true_labels = []
        highest_predictions = []

        # Predict for each test file and store the highest prediction for each individual
        for i, file in enumerate(test_files):
            file_path = os.path.join(data_directory, file)
            data, _ = process_file(file_path)
            data = np.array(data)
            data = data.T
            data = np.expand_dims(data, axis=0)

            prediction = model.predict(data)[0][0]

            with open(os.path.join(data_directory, file), 'r') as f:
                lines = f.readlines()
                if len(lines) > 0:
                    data_parts = lines[0].strip().split(',')
                    individual_id = data_parts[1]
                    individual_label = 1 if data_parts[2].strip().lower() == 'true' else 0

            if individual_id in individual_predictions:
                if prediction > individual_predictions[individual_id]:
                    individual_predictions[individual_id] = prediction
            else:
                individual_predictions[individual_id] = prediction

            true_labels.append(individual_label)
            highest_predictions.append(individual_predictions[individual_id])

        # Calculate AUC for the current fold
        model_auc = roc_auc_score(true_labels, highest_predictions)
        print(f"Fold {fold + 1} AUC: {model_auc}")
        fold_aucs.append(model_auc)

        # Calculate sensitivity and false negative rate for the current fold
        tn, fp, fn, tp = confusion_matrix(true_labels, [1 if p > 0.5 else 0 for p in highest_predictions]).ravel()
        sensitivity = tp / (tp + fn)
        fnr = fn / (tp + fn)
        print(f"Fold {fold + 1} Sensitivity: {sensitivity}")
        print(f"Fold {fold + 1} False Negative Rate: {fnr}")
        fold_sensitivities.append(sensitivity)
        fold_fnrs.append(fnr)

    # Calculate mean and confidence intervals for AUC
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    conf_interval_auc = stats.t.interval(0.95, len(fold_aucs) - 1, loc=mean_auc, scale=stats.sem(fold_aucs))
    print(f"Mean AUC across 5 folds: {mean_auc}")
    print(f"Standard Deviation of AUC across 5 folds: {std_auc}")
    print(f"95% Confidence Interval for AUC: {conf_interval_auc}")

    # Calculate mean and confidence intervals for sensitivity
    mean_sensitivity = np.mean(fold_sensitivities)
    std_sensitivity = np.std(fold_sensitivities)
    conf_interval_sensitivity = stats.t.interval(0.95, len(fold_sensitivities) - 1,
                                                 loc=mean_sensitivity,
                                                 scale=stats.sem(fold_sensitivities))
    print(f"Mean Sensitivity across 5 folds: {mean_sensitivity}")
    print(f"Standard Deviation of Sensitivity across 5 folds: {std_sensitivity}")
    print(f"95% Confidence Interval for Sensitivity: {conf_interval_sensitivity}")

    # Calculate mean and confidence intervals for false negative rate
    mean_fnr = np.mean(fold_fnrs)
    std_fnr = np.std(fold_fnrs)
    conf_interval_fnr = stats.t.interval(0.95, len(fold_fnrs) - 1, loc=mean_fnr,
                                         scale=stats.sem(fold_fnrs))
    print(f"Mean False Negative Rate across 5 folds: {mean_fnr}")
    print(f"Standard Deviation of False Negative Rate across 5 folds: {std_fnr}")
    print(f"95% Confidence Interval for False Negative Rate: {conf_interval_fnr}")

if __name__ == '__main__':
    main()
