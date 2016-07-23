from os.path import join

import numpy as np
from sklearn import cross_validation
import theano

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dropout, Dense, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint


class NN(object):
    def __init__(self, seed=1337, verbosity=1,
                 job_id=None, epochs=100,
                 batch_size=128):
        self.seed = seed
        self.verbosity = verbosity
        self.job_id = job_id
        self.epochs = epochs
        self.batch_size = batch_size

    def __str__(self):
        pass


class LSTMNet(NN):
    def __init__(self, X, y, vocab_size, embedding_size, maxlen,
                 lstm_size, dropout_rate, patience, savepath, weights=None,
                 optimizer='sgd', trainable=True, train_valid=0.8,
                 load=None, *args, **kwargs):

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.maxlen = maxlen
        self.lstm_size = lstm_size

        self.train_valid = train_valid

        super(LSTMNet, self).__init__(*args, **kwargs)
        self.model = Sequential()

        embeddings_layer = Embedding(vocab_size,
                                     embedding_size,
                                     input_length=maxlen)
        if weights is not None:
            embeddings_layer.set_weights([weights])
        embeddings_layer.trainable = trainable

        self.model.add(embeddings_layer)
        self.model.add(LSTM(lstm_size))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(1))
        self.model.add(Activation('linear'))

        # try using different optimizers and different optimizer configs
        self.model.compile(loss='mse', optimizer=optimizer)

        self.es = EarlyStopping(patience=patience, verbose=self.verbosity)
        self.mc = ModelCheckpoint(filepath=join(savepath, "lstm.pkl"),
                                  verbose=self.verbosity, save_best_only=True)
        self._splitDataset(X, y)
        self._train()
        self.docvecs = self.extract_vectors(X)

    def _splitDataset(self, X, y):
        (self.X_train,
         self.X_valid,
         self.y_train,
         self.y_valid) = cross_validation.train_test_split(
            X, y, test_size=self.train_valid, random_state=self.seed)

    def _train(self):

        self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size,
                       nb_epoch=self.epochs, validation_data=(self.X_valid,
                                                              self.y_valid),
                       verbose=self.verbosity, callbacks=[self.es, self.mc])

    def extract_vectors(self, X):

        # get lstm vectors
        fun = theano.function([self.model.layers[0].get_input()],
                              self.model.layers[1].get_output(train=False),
                              allow_input_downcast=True)

        ans = []
        for i in xrange(0, len(X), self.batch_size):
            ans.append(fun(X[i:i+self.batch_size]))
        ans = np.concatenate(ans)
        return ans
