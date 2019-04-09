#!/usr/bin/env python3

import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential

# MacOS Fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# f1 metric
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


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
# Epoch 00008: early stopping
# Training Accuracy: 0.9987
# Testing Accuracy:  0.9981
# Best val_loss: .0066
# model = Sequential()
# model.add(
#     layers.Embedding(
#         input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen
#     )
# )
# model.add(layers.Conv1D(128, 5, activation='relu'))
# model.add(layers.GlobalMaxPool1D())
# model.add(layers.Dense(1000, activation="relu"))
# model.add(layers.Dense(output_dim, activation="sigmoid"))

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
model.add(layers.Conv1D(128, 7, activation="relu"))
model.add(layers.Conv1D(128, 5, activation="relu"))
model.add(layers.Conv1D(128, 3, activation="relu"))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(1000, activation="relu"))
model.add(layers.Dense(output_dim, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy", f1])
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
    monitor="val_loss", min_delta=0, patience=5, verbose=1, mode="auto"
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
loss, accuracy, f1 = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy, f1 = model.evaluate(X_val, y_val, verbose=False)
print("Testing Accuracy:  {:.4f}\n".format(accuracy))

# Get Predictions on real_X (reviews that go to a business with no categories)
print("--Real X--")
pred_real = model.predict(real_X)
for bus_id, row in zip(real[:, 0], pred_real):
    ind = np.argpartition(row, -10)[-10:]
    sorted_ind = ind[np.argsort(row[ind])][::-1]
    top_10 = [
        (cat, cat_pred)
        for cat, cat_pred in zip(categories[sorted_ind], row[sorted_ind])
        if cat_pred >= 0.50
    ]

    name = business["name"][business["business_id"] == bus_id].iloc[0]
    city = business["city"][business["business_id"] == bus_id].iloc[0]
    state = business["state"][business["business_id"] == bus_id].iloc[0]
    print(f"{name} - {city}, {state}")
    if top_10:
        for cat, cat_pred in top_10:
            print(f"{cat}: {np.round(cat_pred, 4)}")
        print()
    else:
        print('?\n')
