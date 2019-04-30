#!/usr/bin/env python3

import os
import time

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

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

# Results
print()
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing - Acc: {:.4f}, Loss: {:.4f}".format(accuracy, loss))

end_time = np.round(time.time() - file_time, 2)
print(f"Total Training Time: {end_time} seconds")
