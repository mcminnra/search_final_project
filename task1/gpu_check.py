#!/usr/bin/env python3

from tensorflow.keras import backend as K
print(K.tensorflow_backend._get_available_gpus())
