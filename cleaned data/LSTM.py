import glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

path = "."
documents = []

for filename in glob.iglob(path+"/fairytale/*.txt"):
    #iterate through each file in the folder, opening it
    with open(filename, encoding="utf8") as f:
        lines = f.read()
        documents.append(lines)
        f.close()

for filename in glob.iglob(path+"/updated_cleaned_mystery/*.txt"):
    #iterate through each file in the folder, opening it
    with open(filename, encoding="utf8") as f:
        lines = f.read()
        documents.append(lines)
        f.close()

vectorizer = CountVectorizer(max_features = 1000, min_df=5,max_df=0.7)
X = vectorizer.fit_transform(documents).toarray()
m = vectorizer.get_feature_names()

y = np.zeros(len(documents))
for i in range(15, len(documents)): 
        y[i] = 1

#Splitting into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
y_train = y_train.reshape((len(y_train),1))
y_test = y_test.reshape((len(y_test),1))

print(X_train.shape)
print(np.shape(y_train))
print(X_test.shape)
print(np.shape(y_test))

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 1000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100

from keras.layers import Layer
from keras.legacy import interfaces
from keras import backend as K
from keras.layers import LSTM

class InputSpec(object):
    """Specifies the ndim, dtype and shape of every input to a layer.
    Every layer should expose (if appropriate) an `input_spec` attribute:
    a list of instances of InputSpec (one per input tensor).
    A None entry in a shape is compatible with any dimension,
    a None shape is compatible with any shape.
    # Arguments
        dtype: Expected datatype of the input.
        shape: Shape tuple, expected shape of the input
            (may include None for unchecked axes).
        ndim: Integer, expected rank of the input.
        max_ndim: Integer, maximum rank of the input.
        min_ndim: Integer, minimum rank of the input.
        axes: Dictionary mapping integer axes to
            a specific dimension value.
    """

    def __init__(self, dtype=None,
                 shape=None,
                 ndim=None,
                 max_ndim=None,
                 min_ndim=None,
                 axes=None):
        self.dtype = dtype
        self.shape = shape
        if shape is not None:
            self.ndim = len(shape)
        else:
            self.ndim = ndim
        self.max_ndim = max_ndim
        self.min_ndim = min_ndim
        self.axes = axes or {}

    def __repr__(self):
        spec = [('dtype=' + str(self.dtype)) if self.dtype else '',
                ('shape=' + str(self.shape)) if self.shape else '',
                ('ndim=' + str(self.ndim)) if self.ndim else '',
                ('max_ndim=' + str(self.max_ndim)) if self.max_ndim else '',
                ('min_ndim=' + str(self.min_ndim)) if self.min_ndim else '',
                ('axes=' + str(self.axes)) if self.axes else '']
        return 'InputSpec(%s)' % ', '.join(x for x in spec if x)

class Dropout(Layer):
    """Applies Dropout to the input.
    Dropout consists in randomly setting
    a fraction `rate` of input units to 0 at each update during training time,
    which helps prevent overfitting.
    # Arguments
        rate: float between 0 and 1. Fraction of the input units to drop.
        noise_shape: 1D integer tensor representing the shape of the
            binary dropout mask that will be multiplied with the input.
            For instance, if your inputs have shape
            `(batch_size, timesteps, features)` and
            you want the dropout mask to be the same for all timesteps,
            you can use `noise_shape=(batch_size, 1, features)`.
        seed: A Python integer to use as random seed.
    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](
           http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
    """
    @interfaces.legacy_dropout_support
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)

    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape,
                                 seed=self.seed)
            return K.in_train_phase(dropped_inputs, inputs,
                                    training=training)
        return inputs

    def get_config(self):
        config = {'rate': self.rate,
                  'noise_shape': self.noise_shape,
                  'seed': self.seed}
        base_config = super(Dropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
    
    
class SpatialDropout1D(Dropout):
    """Spatial 1D version of Dropout.
    This version performs the same function as Dropout, however it drops
    entire 1D feature maps instead of individual elements. If adjacent frames
    within feature maps are strongly correlated (as is normally the case in
    early convolution layers) then regular dropout will not regularize the
    activations and will otherwise just result in an effective learning rate
    decrease. In this case, SpatialDropout1D will help promote independence
    between feature maps and should be used instead.
    # Arguments
        rate: float between 0 and 1. Fraction of the input units to drop.
    # Input shape
        3D tensor with shape:
        `(samples, timesteps, channels)`
    # Output shape
        Same as input
    # References
        - [Efficient Object Localization Using Convolutional Networks](
           https://arxiv.org/abs/1411.4280)
    """

    @interfaces.legacy_spatialdropout1d_support
    def __init__(self, rate, **kwargs):
        super(SpatialDropout1D, self).__init__(rate, **kwargs)
        self.input_spec = InputSpec(ndim=3)

    def _get_noise_shape(self, inputs):
        input_shape = K.shape(inputs)
        noise_shape = (input_shape[0], 1, input_shape[2])
        return noise_shape

    
from keras.models import Sequential
from keras.layers import Dense, Activation,Embedding
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.8, recurrent_dropout=0.8))
model.add(Dense(1, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 2
batch_size = 2

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
