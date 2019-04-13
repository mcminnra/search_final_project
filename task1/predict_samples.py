#!/usr/bin/env python3

import time

import numpy as np
import pandas as pd
from keras.models import load_model

# Load data
print("Loading data...", end="\r")
start_time = time.time()
directory = "obj/"
ids_test = np.load(directory + "ids_test.npy")
X_test = np.load(directory + "X_test.npy")
y_test = np.load(directory + "y_test.npy")
categories = np.load(directory + "categories.npy")
business = pd.read_json("../data/business.json", lines=True)
end_time = np.round(time.time() - start_time, 2)
print(f"Loading data...DONE! [{end_time} seconds]")

# Load model
model = load_model("weights/model.h5")

# Get 5 samples
samples = 5
sample_idx = np.random.randint(len(ids_test), size=samples)
ids_sample = ids_test[sample_idx]
X_samples = X_test[sample_idx, :]
y_samples = y_test[sample_idx]

# Make prediction on samples
y_preds = model.predict(X_samples)

# Print Output
print()
for bus_id, y_pred, y_sample in zip(ids_sample, y_preds, y_samples):
    name = business["name"][business["business_id"] == bus_id].iloc[0]

    cat_pred = sorted(
        [
            (category, percent)
            for category, percent in zip(categories, y_pred)
            if percent >= 0.5
        ],
        key=lambda x: x[1],
    )
    cat_real = sorted(
        [
            (category, percent)
            for category, percent in zip(categories, y_sample)
            if percent >= 0.5
        ],
        key=lambda x: x[1],
    )
    print(name)
    
    print(f'Predicted Categories: {[x[0] for x in cat_pred]}')
    print(f'Real Categories: {[x[0] for x in cat_real]}\n')

