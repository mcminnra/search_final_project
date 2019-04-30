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
user = pd.read_json("../data/user.json", lines=True)
business = pd.read_json("../data/business.json", lines=True)
end_time = np.round(time.time() - start_time, 2)
print(f"Loading data...DONE! [{end_time} seconds]")

# Load model
model = load_model("weights/model.h5")

# Get user and business index
business["business_index"] = business.index
user["user_index"] = user.index

# Get random user
user_sample_index = user["user_index"][
    user["user_id"] == np.random.choice(user["user_id"], size=1)[0]
].iloc[0]

# Print user
user_sample = user[user['user_index'] == user_sample_index]
print(f"\n{user_sample['user_id'].iloc[0]}:: {user_sample['name'].iloc[0]}")

# Make index arrays for model
X = business
X["user_index"] = user_sample_index
X_user_sample = X["user_index"].values
X_business_sample = X["business_index"].values

# Predict Stars
X["ratings_bar"] = model.predict([X_user_sample, X_business_sample])

# Get top 25 reviews and print
X = X.sort_values(by=["ratings_bar"], ascending=False).iloc[:10]

for i, (index, row) in enumerate(X.iterrows()):
    print(
        f"{i+1}:: {row['name']} - {np.round(row['ratings_bar'], 2)}"  # noqa
    )
