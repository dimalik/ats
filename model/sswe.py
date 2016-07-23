"""SSWE trainer

Train score specific word embeddings as described in Alikaniotis,
Yannakoudakis and Rei (2016)
"""

import cPickle as pkl

import numpy as np

import theano
import theano.tensor as T

from sklearn.preprocessing import normalize


FLOAT = 'float32'


def htanh(net_input):
    """Calculate the hard Tanh function"""
    return T.clip(net_input, -1, 1)


class Model(object):
    '''Abstract model class to hold parameter information'''
    def __init__(self, train_loss=0, train_err=0, train_lossnonzero=0,
                 train_cnt=0):
        self.train_loss = train_loss
        self.train_err = train_err
        self.train_lossnonzero = train_lossnonzero
        self.train_cnt = train_cnt

    def __getstate__(self):
        return (self.train_loss,
                self.train_err,
                self.train_lossnonzero,
                self.train_cnt)

    def __setstate__(self, state):
        (self.train_loss,
         self.train_err,
         self.train_lossnonzero,
         self.train_cnt) = state

    def _get_train_function(self):
        """Each model implements this differently"""
        raise NotImplementedError

    def reset(self):
        """Reset everytime"""
        self.train_loss = 0
        self.train_err = 0
        self.train_lossnonzero = 0
        self.train_cnt = 0


class SSWEModel(Model):
    """Implements the Score-specific word embeddings method"""
    def __init__(self, alpha, window_size, embedding_size, hidden_size,
                 vocab_size, learning_rate=0.01, embedding_learning_rate=0.01,
                 negative_examples=20, batch_size=128,
                 normalize_embeddings=False, activation=htanh, **kwargs):
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = 1
        self.input_size = self.embedding_size * self.window_size
        self.vocab_size = vocab_size
        self.half_window = self.window_size // 2
        self.batch_size = batch_size
        self.alpha = alpha
        self.activation = activation
        self.learning_rate = learning_rate
        self.embedding_learning_rate = embedding_learning_rate
        self.negative_examples = negative_examples
        self.COMPILE_MODE = theano.compile.Mode('c|py', 'fast_run')

        self.ts = self.half_window * embedding_size
        self.te = self.ts + embedding_size

        # embeddings matrix is updated manually
        self.embeddings = np.asarray(
            (np.random.rand(self.vocab_size,
                            self.embedding_size) - 0.5) * 2,
            dtype=FLOAT)

        self.hidden_weights = theano.shared(
            np.asarray(np.random.uniform(
                low=-np.sqrt(6. / (self.input_size + self.hidden_size)),
                high=np.sqrt(6. / (self.input_size + self.hidden_size)),
                size=(self.input_size, self.hidden_size)),
                       dtype=FLOAT)
        )

        self.output_weights = theano.shared(
            np.asarray(np.random.uniform(
                low=-np.sqrt(6. / (self.hidden_size + self.output_size)),
                high=np.sqrt(6. / (self.hidden_size + self.output_size)),
                size=(self.hidden_size, self.output_size)),
                       dtype=FLOAT)
        )

        self.hidden_biases = theano.shared(
            np.asarray(np.zeros((self.hidden_size,)),
                       dtype=FLOAT))

        self.output_biases = theano.shared(
            np.asarray(np.zeros((self.output_size,)),
                       dtype=FLOAT))

        self.output_weights_s = theano.shared(
            np.asarray(np.random.uniform(
                low=-np.sqrt(6. / (self.hidden_size + 1)),
                high=np.sqrt(6. / (self.hidden_size + 1)),
                size=(self.hidden_size, 1)), dtype=FLOAT)
        )

        self.output_biases_s = theano.shared(
            np.asarray(np.zeros((1,)), dtype=FLOAT))

        self.params = [self.hidden_weights, self.hidden_biases,
                       self.output_weights, self.output_biases,
                       self.output_weights_s, self.output_biases_s]

        self.correct_matrix = np.empty([batch_size,
                                        embedding_size * window_size],
                                       dtype=FLOAT)

        self.train_function = self._get_train_function()

        super(SSWEModel, self).__init__(kwargs)

    @staticmethod
    def normalize(embeddings):
        return normalize(embeddings, norm='l2', axis=1)

    def _get_train_function(self):
        correct_inputs = T.matrix(dtype=FLOAT)
        noise_inputs = T.matrix(dtype=FLOAT)
        y_score = T.matrix('y', dtype=FLOAT)

        learning_rate = T.scalar(dtype=FLOAT)

        # correct sequences
        # embeddings -> hidden
        correct_prehidden = T.dot(correct_inputs, self.hidden_weights) + \
            self.hidden_biases
        hidden_c = self.activation(correct_prehidden)
        # hidden -> output
        correct_score = T.dot(hidden_c, self.output_weights) + \
            self.output_biases

        # corrupt sequences
        # embeddings -> hidden
        noise_prehidden = T.dot(noise_inputs, self.hidden_weights) + \
            self.hidden_biases
        hidden_n = T.tanh(noise_prehidden)
        # hidden -> output
        noise_score = T.dot(hidden_n, self.output_weights) + \
            self.output_biases

        # hidden -> score output
        predicted_score = T.dot(hidden_c, self.output_weights_s) + \
            self.output_biases_s
        # MSE
        predicted_loss = T.sum(T.sqr(predicted_score - y_score).mean(axis=-1))

        # hinge loss for the context
        losses = T.sum(T.clip(1 - correct_score + noise_score, 0, 1e999))

        # combined loss
        total_loss = self.alpha * losses + (1 - self.alpha) * \
            predicted_loss

        # gradient descent
        gparams = T.grad(total_loss, self.params)

        dcorrect_inputs = T.grad(total_loss, correct_inputs)
        dnoise_inputs = T.grad(total_loss, noise_inputs)

        updates = [(p, p - learning_rate * gp) for p, gp in zip(self.params,
                                                                gparams)]

        train_function = theano.function([correct_inputs, noise_inputs,
                                          y_score,
                                          learning_rate],
                                         [dcorrect_inputs, dnoise_inputs,
                                          total_loss, losses,
                                          correct_score, noise_score],
                                         mode=self.COMPILE_MODE,
                                         updates=updates)
        return train_function

    def train(self, correct_sequences, y_score):

        for i in xrange(correct_sequences.shape[0]):
            self.correct_matrix[i] = self.embeddings[
                correct_sequences[i]].flatten()
        corrupt_matrix = np.copy(self.correct_matrix)
        noise_sequences = np.copy(correct_sequences)
        avg_loss = []

        for i in xrange(self.negative_examples):
            negs = np.random.choice(self.vocab_size, self.batch_size,
                                    replace=True)
            corrupt_matrix[:, self.ts:self.te] = self.embeddings[negs]
            noise_sequences[:, self.half_window] = negs
            (dcorrect_inputss,
             dnoise_inputss,
             total_loss,
             losses,
             correct_scores,
             noise_scores) = self.train_function(
                 self.correct_matrix,
                 corrupt_matrix,
                 y_score,
                 self.learning_rate)
            avg_loss.append(total_loss)
            self.train_loss += total_loss
            self.train_err += (correct_scores <= noise_scores).sum()
            self.train_lossnonzero += (losses > 0).sum()

            self.reset()

            for index in xrange(len(correct_sequences)):

                correct_sequence = correct_sequences[index].flatten()
                noise_sequence = noise_sequences[index].flatten()

                dcorrect_inputs = dcorrect_inputss[index].reshape((
                    self.window_size, self.embedding_size))
                dnoise_inputs = dnoise_inputss[index].reshape((
                    self.window_size, self.embedding_size))

                for (i, di) in zip(correct_sequence, dcorrect_inputs):
                    self.embeddings[i] -= self.embedding_learning_rate * di
                for (i, di) in zip(noise_sequence, dnoise_inputs):
                    self.embeddings[i] -= self.embedding_learning_rate * di

        self.train_cnt += len(correct_sequences)

        return sum(avg_loss) / float(len(avg_loss))

    def save(self, fname):
        with file(fname, "wb") as fout:
            pkl.dump([self.embeddings,
                      self.hidden_weights,
                      self.hidden_biases,
                      self.output_weights,
                      self.output_biases,
                      self.output_weights_s,
                      self.output_biases_s],
                     fout, -1)
