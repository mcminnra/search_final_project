#!/usr/bin/env python3

import os
import time

import numpy as np
from sklearn.model_selection import train_test_split
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model

# MacOS Fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Start total file time
file_time = time.time()

# Load Data
print("Loading data...", end="\r")
start_time = time.time()
directory = "obj/"
ids = np.load(directory + "ids.npy")
X = np.load(directory + "X.npy")
y = np.load(directory + "y.npy")
params = np.load(directory + "params.npy")
end_time = np.round(time.time() - start_time, 2)
print(f"Loading data...DONE! [{end_time} seconds]")

# Train Validation Split
print("Creating Train/Validation/Test Set...", end="\r")
start_time = time.time()
idx = np.arange(len(X))
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, idx, test_size=0.20, random_state=42
)
ids_test = ids[idx_test]
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.20, random_state=42
)
end_time = np.round(time.time() - start_time, 2)
print(f"Creating Train/Validation/Test Set...DONE! [{end_time} seconds]")

# Params
vocab_size = params.item().get("vocab_size")
maxlen = params.item().get("maxlen")
embedding_dim = 50
output_dim = y_train.shape[1]  # Number of labels (1300)

# Model
model = Sequential()
model.add(
    layers.Embedding(
        input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen
    )
)
model.add(layers.Conv1D(128, 5, activation="relu"))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(1000, activation="relu"))
model.add(layers.Dense(output_dim, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# Training Callbacks
if not os.path.exists("weights/"):
    os.makedirs("weights/")
checkpoint = ModelCheckpoint(
    "weights/model.h5",
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode="auto",
    period=1,
)
early = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=15, verbose=1, mode="auto"
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

# Load Best Weights
model = load_model("weights/model.h5")

# Results
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training - Acc: {:.4f}, Loss: {:.4f}".format(accuracy, loss))
loss, accuracy = model.evaluate(X_val, y_val, verbose=False)
print("Validation - Acc: {:.4f}, Loss: {:.4f}".format(accuracy, loss))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing - Acc: {:.4f}, Loss: {:.4f}".format(accuracy, loss))

# Save Test Data
print("Saving test data...", end="\r")
start_time = time.time()
directory = "obj/"
if not os.path.exists(directory):
    os.makedirs(directory)
np.save(directory + "ids_test.npy", ids_test)
np.save(directory + "X_test.npy", X_test)
np.save(directory + "y_test.npy", y_test)
end_time = np.round(time.time() - start_time, 2)
print(f"Saving test data...DONE! [{end_time} seconds]")

end_time = np.round(time.time() - file_time, 2)
print(f"Total Training Time: {end_time} seconds")
