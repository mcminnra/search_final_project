#!/usr/bin/env python3

import os
import time

import numpy as np
import pandas as pd

# Start total file time
file_time = time.time()

# Read Data
print("Loading data...", end="\r")
start_time = time.time()
review = pd.read_json("../data/review_1000000.json", lines=True)
user = pd.read_json("../data/user.json", lines=True)
business = pd.read_json("../data/business.json", lines=True)
params = {}
end_time = np.round(time.time() - start_time, 2)
print(f"Loading data...DONE! [{end_time} seconds]")
print()

print("Merging data and creating indices...", end="\r")
start_time = time.time()
# Drop extra columns in review
review = review[['user_id', 'business_id', 'stars']]

# Get User Indices and merge into reviews
params['num_user'] = user.shape[0]
user['user_index'] = user.index
user = user[['user_id', 'user_index']]
review = review.merge(user, on="user_id", how="left")

# Get Business indices and merge into reviews
params['num_business'] = business.shape[0]
business['business_index'] = business.index
business = business[['business_id', 'business_index']]
review = review.merge(business, on="business_id", how="left")

# Drop ids and convert to features and target DataFrames
X_user = review[['user_index']]
X_business = review[['business_index']]
y = review['stars']
end_time = np.round(time.time() - start_time, 2)
print(f"Merging data and creating indices...DONE! [{end_time} seconds]")
print()

print("Saving data...", end="\r")
start_time = time.time()
directory = "obj/"
if not os.path.exists(directory):
    os.makedirs(directory)
np.save(directory + "X_user.npy", X_user)
np.save(directory + "X_business.npy", X_business)
np.save(directory + "y.npy", y)
np.save(directory + "params.npy", params)
end_time = np.round(time.time() - start_time, 2)
print(f"Saving data...DONE! [{end_time} seconds]")

end_time = np.round(time.time() - file_time, 2)
print(f"Total Processing Time: {end_time} seconds")
