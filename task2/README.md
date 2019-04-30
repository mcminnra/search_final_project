# Task 2

## How to run
First, Process data. This is merges users and business plus does various conversion tasks to prep the data
```bash
make data
```

Then, train the model
```bash
make train
```

There is a script that makes recommendations on a sample user of the test set to see how well it performs.
```bash
make samples
```

## Results

Task 2 is trained on all the businesses and users in the dataset.


### Training and CNN Architecture Overview (model.summary())
```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
user (InputLayer)               (None, 1)            0
__________________________________________________________________________________________________
business (InputLayer)           (None, 1)            0
__________________________________________________________________________________________________
user_embedding (Embedding)      (None, 1, 50)        81856900    user[0][0]
__________________________________________________________________________________________________
business_embedding (Embedding)  (None, 1, 50)        9630450     business[0][0]
__________________________________________________________________________________________________
reshape (Reshape)               (None, 50)           0           user_embedding[0][0]
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 50)           0           business_embedding[0][0]
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 100)          0           reshape[0][0]
                                                                 reshape_1[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 512)          51712       concatenate[0][0]
__________________________________________________________________________________________________
dropout (Dropout)               (None, 512)          0           dense[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 256)          131328      dropout[0][0]
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 256)          0           dense_1[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 128)          32896       dropout_1[0][0]
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 1)            129         dense_2[0][0]
==================================================================================================
Total params: 91,703,415
Trainable params: 91,703,415
Non-trainable params: 0
```
