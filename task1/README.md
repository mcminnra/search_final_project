# Task 1

## How to run
First, Process data. This is merges business and reviews plus does various NLP tasks to prep the reviews.
```bash
make data
```

Then, train the model
```bash
make train
```

There is a script that makes predictions on a subsample of the test set to see how well it performs.
```bash
make samples
```

## Results

Task 1 is trained on all the businesses in the dataset, and 1,000,000 reviews. 

### Process Data Steps

1. Merge business name into reviews
2. Convert categories to one-hot encoding (1300 categories total)
3. Remove Puncuation
4. Remove Stop Words
5. Snowball Stemming (NLTK)
6. Create sequence of word ids for embedding training
7. Pad sequences, so they are all the same length

```
Total Processing Time: 13210.18 seconds (3.67 hours)
```

### Training and CNN Architecture Overview (model.summary())
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 821, 50)           22015100
_________________________________________________________________
conv1d (Conv1D)              (None, 817, 128)          32128
_________________________________________________________________
global_max_pooling1d (Global (None, 128)               0
_________________________________________________________________
dense (Dense)                (None, 1000)              129000
_________________________________________________________________
dense_1 (Dense)              (None, 1300)              1301300
=================================================================
Total params: 23,477,528
Trainable params: 23,477,528
Non-trainable params: 0
```

```
28.4G RAM
11.7G VRAM on 1 Titan X GPU
```

```
Total Training Time: 8709.53 seconds (2.42 hours)
```

```
Training - Acc: 0.9988, Loss: 0.0036
Validation - Acc: 0.9986, Loss: 0.0049
Testing - Acc: 0.9986, Loss: 0.0049
```

### Some Predictions on the Test Set
```
Name: Beef 'N Bottle
Predicted Categories: ['Restaurants']
Real Categories: ['Restaurants', 'Seafood', 'Steakhouses']
Percent Overlap: 0.3333

Name: Ah-So Sushi & Steak
Predicted Categories: ['Sushi Bars', 'Steakhouses', 'Japanese', 'Restaurants']
Real Categories: ['Japanese', 'Restaurants']
Percent Overlap: 1.0

Name: Chang's Hong Kong Cuisine
Predicted Categories: ['Cantonese', 'Diners', 'Seafood', 'Dim Sum', 'Chinese', 'Restaurants']
Real Categories: ['Cantonese', 'Chinese', 'Dim Sum', 'Diners', 'Restaurants', 'Seafood']
Percent Overlap: 1.0

Name: Sushi Wa
Predicted Categories: ['Japanese', 'Sushi Bars', 'Restaurants']
Real Categories: ['Japanese', 'Restaurants', 'Sushi Bars']
Percent Overlap: 1.0

Name: Rancho Pinot
Predicted Categories: ['Mexican', 'Restaurants']
Real Categories: ['American (New)', 'Bars', 'Italian', 'Nightlife', 'Restaurants', 'Wine Bars']
Percent Overlap: 0.1667
```
