#!/usr/bin/env python3

import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tqdm import tqdm

# Treat Warnings as errors
import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

# MacOS Fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Start total file time
file_time = time.time()

# Load Data
print("Loading data...", end="\r")
start_time = time.time()
directory = "obj/"
X_test = np.load(directory + "X_test.npy")
y_test = np.load(directory + "y_test.npy")
end_time = np.round(time.time() - start_time, 2)
print(f"Loading data...DONE! [{end_time} seconds]")

# Load Best Weights
model = load_model("weights/model.h5")

# Make prediction on samples
y_preds = model.predict(X_test)

# Round percents to binary labeling
y_preds = np.where(y_preds > 0.5, 1, 0)

num_categories = len(y_preds[0])
num_samples = len(y_preds)
print(f'Number of Categories: {num_categories}')
print(f'Number of Samples: {num_samples}')

# Precision & Recall
# Pulled from https://stackoverflow.com/questions/9004172/precision-recall-for-multiclass-multilabel-classification
precisions = []
recalls = []
mccs = []
for i in tqdm(range(0, num_categories), desc='Getting Metrics'):
    class_preds = y_preds[:, i]
    class_test = y_test[:, i]
    cm = confusion_matrix(class_test, class_preds)

    if len(cm) != 1:
        tp = cm[0][0]
        fp = cm[0][1]
        tn = cm[1][1]
        fn = cm[1][0]

        # Precision
        if tp == 0 and fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        precisions.append(precision)

        # Recall
        if tp == 0 and fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
        recalls.append(recall)

        # Matthew's Correlation Coefficient
        try:
            if ((tp*tn) - (fp*fn)) != 0 or (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) != 0:
                mcc = ((tp*tn) - (fp*fn)) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
            else:
                mcc = 0
            mccs.append(mcc)
        except RuntimeWarning:
            print(((tp*tn) - (fp*fn)))
            print((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    else:
        precisions.append(1)
        recalls.append(1)
        mccs.append(1)

avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)
avg_mcc = np.mean(mccs)

# Loss & Accuracy
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)

# Print Metrics
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')
print(f'Average Precisions across all classes: {avg_precision}')
print(f'Average Recalls across all classes: {avg_recall}')
print(f'Average MCC across all classes: {avg_mcc}')

end_time = np.round(time.time() - file_time, 2)
print(f"Total Training Time: {end_time} seconds")
