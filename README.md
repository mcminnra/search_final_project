# Search Final Project

edit bashrc for which gpu to use

## Task 1
Multi-Label Text Classification via Word Embeddings + CNN Binary-Crossentropy

Tried Methods:
- CNN -> LSTM (Long-Term Dependancies aren't important in short text reviews)
- TFIDF -> MLP (Dimensionality becomes a huge problem in any reasonable number of training data)

Done:
- [X] Each Review apply categories + business name
- [X] Train on reviews to learn multi-label categories
- [X] then from real world reviews about a business with no categories, you can suggest some. 
- [X] Maybe look into Embedding then CNN/LSTM
- [X] add title to text
- [X] Maybe switch to lemmization?
- [X] N-Gram features? (Takes an unrealistically long time to processs)
- [X] Skip-Gram features (Would assume the same about these as N-Grams)
- [X] Million training data?

Todo:
- is there a better metric than accuracy?
- Make Preds on val (or at least some subset)
- Maybe add tips?
- look at better neural network structure
- Maybe look into batches for training data
- glove word embedding?


1000000 Reviews:
28.4G of 251G RAM
11.7G VRAM Titan X GPU

approach: CNN+SnowballStemming
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

Total Processing Time: 13210.18 seconds (about 3 hours and 45 min)
Total Training Time: 8709.53 seconds
Training - Acc: 0.9988, Loss: 0.0036
Validation - Acc: 0.9986, Loss: 0.0049
Testing - Acc: 0.9986, Loss: 0.0049

Benchmarks (Normalization)
Training on single Titan X GPU with 200,000 sampled reviews
- CNN+SnowballStemming
Total Processing Time: 1263.51 seconds
Total Training Time: 813.04 seconds
Training - Acc: 0.9984, Loss: 0.0048
Validation - Acc: 0.9980, Loss: 0.0070
Testing - Acc: 0.9980, Loss: 0.0069
- CNN+WordNetLemmatization
Total Processing Time: 1506.33 seconds
Total Training Time: 874.52 seconds
Training - Acc: 0.9984, Loss: 0.0048
Validation - Acc: 0.9979, Loss: 0.0071
Testing - Acc: 0.9980, Loss: 0.0070
- CNN+NOWordNormalization
Total Processing Time: 1654.51 seconds
Total Training Time: 891.55 seconds
Training - Acc: 0.9984, Loss: 0.0047
Validation - Acc: 0.9980, Loss: 0.0071
Testing - Acc: 0.9980, Loss: 0.0071

Benchmarks (Pretrained embeddings)
- CNN+NoPretrained
- CNN+GloveEmbeddings

Cites:
- https://www.aclweb.org/anthology/D14-1181

## Task 2
Deep Matrix Factorization
https://pdfs.semanticscholar.org/35e7/4c47cf4b3a1db7c9bfe89966d1c7c0efadd0.pdf?_ga=2.157815098.1290675275.1554405980-1148675878.1554405980

