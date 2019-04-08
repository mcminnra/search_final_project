#!/usr/bin/env python3

import os
import time

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

# MacOS Fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Load Data
print("Loading data...", end="\r")
start_time = time.time()
directory = "obj/"
X = np.load(directory + "X.npy")
y = np.load(directory + "y.npy")
real_X = np.load(directory + "real_X.npy")
categories = np.load(directory + "categories.npy")
real = np.load(directory + "real.npy")
params = np.load(directory + "params.npy")
business = pd.read_json("../data/business.json", lines=True)
end_time = np.round(time.time() - start_time, 2)
print(f"Loading data...DONE! [{end_time} seconds]")

# Train Validation Split
print("Creating Train/Validation Set...", end="\r")
start_time = time.time()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)
end_time = np.round(time.time() - start_time, 2)
print(f"Creating Train/Validation Set...DONE! [{end_time} seconds]")

# Model
# Epoch 00006: early stopping
# Training Accuracy: 0.9978
# Testing Accuracy:  0.9973
# just 100 dense layer

# Params
vocab_size = params.item().get('vocab_size')
maxlen = params.item().get('maxlen')
embedding_dim = 50
output_dim = y_train.shape[1]  # Number of labels (1300)
pool_size = 4

model = Sequential()
model.add(
    layers.Embedding(
        input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen
    )
)
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPool1D())
#model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(output_dim, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# Training Callbacks
if not os.path.exists("weights/"):
    os.makedirs("weights/")
checkpoint = ModelCheckpoint(
    "weights/model.h5",
    monitor="val_acc",
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode="auto",
    period=1,
)
early = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=2, verbose=1, mode="auto"
)

# Train Model
history = model.fit(
    X_train,
    y_train,
    epochs=1000,
    batch_size=25,
    verbose=True,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, early],
)

# Results
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_val, y_val, verbose=False)
print("Testing Accuracy:  {:.4f}\n".format(accuracy))

# Get Predictions on real_X (reviews that go to a business with no categories)
print("--Real X--")
pred_real = model.predict(real_X)
for bus_id, row in zip(real[:, 0], pred_real):
    ind = np.argpartition(row, -3)[-3:]
    sorted_ind = ind[np.argsort(row[ind])][::-1]
    top_3 = zip(categories[sorted_ind], row[sorted_ind])

    name = business["name"][business["business_id"] == bus_id].iloc[0]
    city = business["city"][business["business_id"] == bus_id].iloc[0]
    state = business["state"][business["business_id"] == bus_id].iloc[0]
    print(f"{name} - {city}, {state}")
    for cat, cat_pred in top_3:
        print(f"{cat}: {np.round(cat_pred, 4)}")
    print()
