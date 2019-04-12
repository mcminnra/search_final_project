# Search Final Project

edit bashrc for which gpu to use

## Task 1
Multi-Label Text Classification via Word Embeddings + CNN Binary-Crossentropy

Done:
- [X] Each Review apply categories + business name
- [X] Train on reviews to learn multi-label categories
- [X] then from real world reviews about a business with no categories, you can suggest some. 
- [X] Maybe look into Embedding then CNN/LSTM
- [X] add title to text
- [X] Maybe switch to lemmization?
- [X] N-Gram features? (Takes an unrealistically long time to processs)
- [X] Skip-Gram features (Would assume the same about these as N-Grams)

Todo:
- is there a better metric than accuracy?
- Make Preds on val (or at least some subset)
- Maybe add tips?
- look at better neural network structure
- Maybe look into batches for training data
- glove word embedding?
- Million training data?

1000000 Reviews:
approach: CNN+SnowballStemming
Total Processing Time: 13210.18 seconds (about 3 hours and 45 min)

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

