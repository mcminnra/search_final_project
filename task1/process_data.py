#!/usr/bin/env python3

import os
import string
import time

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer # noqa
from tqdm import tqdm

# Start total file time
file_time = time.time()

# Read Data
print("Loading data...", end="\r")
start_time = time.time()
business = pd.read_json("../data/business.json", lines=True)
# review = pd.read_json("../data/review_1000.json", lines=True)
# review = pd.read_json("../data/review_200000.json", lines=True)
review = pd.read_json("../data/review_1000000.json", lines=True)
end_time = np.round(time.time() - start_time, 2)
print(f"Loading data...DONE! [{end_time} seconds]")
print()

# Get Categories from business.json
cat_counts = {}
for index, row in tqdm(business.iterrows(), desc="Getting Categories"):
    if row["categories"] is None:
        continue

    cat_row = [x.strip() for x in row["categories"].split(",")]
    for cat in cat_row:
        # Get Category Counts
        if cat in cat_counts:
            cat_counts[cat] += 1
        else:
            cat_counts[cat] = 1

# Get Sorted Categories
categories = sorted([k for k, v in cat_counts.items()])

# Categories Statistics
top_25_categories = sorted(
    [(k, v) for k, v in cat_counts.items()], key=lambda x: x[1], reverse=True
)[:25]

print(f"Number of Categories: {len(categories)}")
print("--Top 25 Categories--")
for k, v in top_25_categories:
    print(f"{k}: {v}")
print()

# Merge business into reviews
merge_df = business[["business_id", "name", "categories"]]
review = review.merge(merge_df, on="business_id", how="left")
review = review[["business_id", "name", "text", "categories"]]
del business

# Add business name to review
review["text"] = review["name"] + " " + review["text"]
review = review.drop(["name"], axis=1)

# Convert Reviews Categories to OneHotEncoding
for index, row in tqdm(
    review.iterrows(), desc="Converting Categories to OneHotEncoding"
):
    if row["categories"] is None:
        continue

    cat_row = [x.strip() for x in row["categories"].split(",")]

    # Get indices of categories for row
    indices = []
    for cat in cat_row:
        indices.append(categories.index(cat))

    # Build OneHot Vector and flip indices to 1
    onehot = np.zeros(len(categories))
    for i in indices:
        onehot[i] = 1

    review.at[index, "categories"] = onehot

print("Cleaning and Vectorizing Text...", end="\r")
start_time = time.time()

# Remove Punctuation
review["text"] = (
    review["text"]
    .str.replace("[{}]".format(string.punctuation), "")
    .str.lower()
    .str.split()
)

# Remove Stop Words
stop = stopwords.words("english")
review["text"] = review["text"].apply(
    lambda x: [item for item in x if item not in stop]
)

# SnowballStemmer (Best)
stemmer = SnowballStemmer("english")
review["text"] = review["text"].apply(lambda x: [stemmer.stem(item) for item in x])

# Lemmatization
# lemmatizer = WordNetLemmatizer()
# review["text"] = review["text"].apply(
#     lambda x: [lemmatizer.lemmatize(item) for item in x]
# )

# Drop reviews that have no categories
real = review[review["categories"].isna()]
review = review[review["categories"].notnull()]
ids = review['business_id']

# Convert to numpy matrix
review_text = np.array(review["text"].values)
real_text = np.array(real["text"].values)

# Get words and line length
words = []
maxlen = 0
for line in review_text:
    if len(line) > maxlen:
        maxlen = len(line)
    for word in line:
        if word not in words:
            words.append(word)

# Tokenizer - creates sequence of word ids
# Num_words is the most X common words are kept
tokenizer = Tokenizer(num_words=len(words))
tokenizer.fit_on_texts(review_text)

X = tokenizer.texts_to_sequences(review_text)
real_X = tokenizer.texts_to_sequences(real_text)
y = np.array(review["categories"].tolist())

# Pad X
X = pad_sequences(X, padding="post", maxlen=maxlen)
real_X = pad_sequences(real_X, padding="post", maxlen=maxlen)

# Set some params for training
params = {}
params["maxlen"] = maxlen
params["vocab_size"] = len(tokenizer.word_index) + 1

end_time = np.round(time.time() - start_time, 2)
print(f"Cleaning and Vectorizing Text...DONE! [{end_time} seconds]")
print(f"Number of Reviews: {len(review) + len(real)}")
print(f"Number of Training Reviews: {len(review)}")
print(
    f"Number of Reviews to Businesses without Categories (Real World Data): {len(real)}"
)
print(f"X Shape: {X.shape}")
print(f"y Shape: {y.shape}")
print()

# Save Some Stuff
print("Saving data...", end="\r")
start_time = time.time()
directory = "obj/"
if not os.path.exists(directory):
    os.makedirs(directory)
np.save(directory + "ids.npy", ids)
np.save(directory + "X.npy", X)
np.save(directory + "y.npy", y)
np.save(directory + "real_X.npy", real_X)
np.save(directory + "real.npy", real)
np.save(directory + "categories.npy", categories)
np.save(directory + "params.npy", params)
end_time = np.round(time.time() - start_time, 2)
print(f"Saving data...DONE! [{end_time} seconds]")

end_time = np.round(time.time() - file_time, 2)
print(f"Total Processing Time: {end_time} seconds")
