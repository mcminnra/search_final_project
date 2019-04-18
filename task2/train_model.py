#!/usr/bin/env python3

import os
import time

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model, load_model

# MacOS Fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Start total file time
file_time = time.time()

# Load Data
print("Loading data...", end="\r")
start_time = time.time()
directory = "obj/"
X_user = np.load(directory + "X_user.npy")
X_business = np.load(directory + "X_business.npy")
y = np.load(directory + "y.npy")
params = np.load(directory + "params.npy")
end_time = np.round(time.time() - start_time, 2)
print(f"Loading data...DONE! [{end_time} seconds]")

# Train Validation Split
# print("Creating Train/Validation/Test Set...", end="\r")
# start_time = time.time()
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.20, random_state=42
# )
# X_train, X_val, y_train, y_val = train_test_split(
#     X_train, y_train, test_size=0.20, random_state=42
# )
# end_time = np.round(time.time() - start_time, 2)
# print(f"Creating Train/Validation/Test Set...DONE! [{end_time} seconds]")

# Model
user_index_input = layers.Input(shape=[1], name="user")
business_index_input = layers.Input(shape=[1], name="business")

embedding_size = 50
user_embedding = layers.Embedding(
    output_dim=embedding_size,
    input_dim=params.item().get("num_user"),
    input_length=1,
    name="user_embedding",
)(user_index_input)
business_embedding = layers.Embedding(
    output_dim=embedding_size,
    input_dim=params.item().get("num_business"),
    input_length=1,
    name="business_embedding",
)(business_index_input)

user_vecs = layers.Reshape([embedding_size])(user_embedding)
business_vecs = layers.Reshape([embedding_size])(business_embedding)
input_vecs = layers.Concatenate()([user_vecs, business_vecs])

l1 = layers.Dense(512, activation="relu")(input_vecs)
l1do = layers.Dropout(.2)(l1)
l2 = layers.Dense(256, activation="relu")(l1do)
l2do = layers.Dropout(.2)(l2)
l3 = layers.Dense(128, activation="relu")(l2do)
out = layers.Dense(1)(l3)

model = Model(inputs=[user_index_input, business_index_input], outputs=out)

model.compile(loss="mse", optimizer="adam")
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
    monitor="val_loss", min_delta=0, patience=10, verbose=1, mode="auto"
)

# Train Model
history = model.fit(
    [X_user, X_business],
    y,
    epochs=1000,
    batch_size=25,
    verbose=True,
    validation_split=0.2,
    callbacks=[checkpoint, early],
)

# Load Best Weights
model = load_model("weights/model.h5")

# Results
loss = model.evaluate([X_user, X_business], y, verbose=False)
print("Train Loss: {:.4f}".format(loss))

end_time = np.round(time.time() - file_time, 2)
print(f"Total Training Time: {end_time} seconds")
