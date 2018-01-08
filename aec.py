import argparse

from keras.datasets import cifar10, mnist
from keras.layers import Input, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras.regularizers import l2

import numpy as np
import matplotlib.pyplot as plt


class Dataset:
    """Dataset base class"""

    def __init__(self, dataset):
        (self.train, self.train_y), (self.test, self.test_y) = dataset.load_data()

        self.shape = self.train.shape[1:]
        self.instance_size = np.prod(self.shape)

        self.train = self.train.astype('float32') / 255.
        self.test = self.test.astype('float32') / 255.
        self.train = self.train.reshape((len(self.train), self.instance_size))
        self.test = self.test.reshape((len(self.test), self.instance_size))

    def compare(self, original, decoded):
        plt.figure(figsize=(20, 4))
        n = len(original)
        colored = 'gray' if len(self.shape) == 2 else None

        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(original[i].reshape(self.shape), cmap=colored)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded[i].reshape(self.shape), cmap=colored)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.show()


class MNIST(Dataset):
    def __init__(self):
        super().__init__(mnist)


class CIFAR10(Dataset):
    def __init__(self, category=None):
        super().__init__(cifar10)

        if category is not None:
            self.train = np.array([x for x, y in zip(self.train, self.train_y) if y == category])
            self.test = np.array([x for x, y in zip(self.test, self.test_y) if y == category])


class Autoencoder:
    """Autoencoder base class"""

    def __init__(self, dimensions, loss, optimizer, batch_size, dataset, epochs=None):
        self.dimensions = dimensions
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.dataset = dataset
        self.epochs = epochs

        self.autoencoder = None
        self.encoder = None
        self.decoder = None

    def compile(self):
        self.autoencoder.compile(optimizer=self.optimizer, loss=self.loss)

    def train(self, epochs=None):
        self.epochs = epochs or self.epochs
        logs = self.autoencoder.fit(self.dataset.train, self.dataset.train,
                                    epochs=epochs,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    validation_data=(self.dataset.test, self.dataset.test))
        return logs

    def evaluate(self):
        return self.autoencoder.evaluate(self.dataset.test, self.dataset.test, batch_size=256)

    def show(self, n=10):
        original_imgs = self.dataset.test[:n]
        encoded_imgs = self.encoder.predict(original_imgs)
        decoded_imgs = self.decoder.predict(encoded_imgs)

        self.dataset.compare(original_imgs, decoded_imgs)

    def save(self, filename=None):
        filename = '%s-s%s-d%s-l%s-o%s-b%d-e%d' % (self.__class__.__name__, self.dataset.__class__.__name__, self.dimensions,
                                                   self.loss, self.optimizer, self.batch_size, self.epochs)
        np.save(filename, self.autoencoder.get_weights())

    def load(self, filename=None):
        filename = filename or '%s-s%s-d%s-l%s-o%s-b%d-e%d.npy' % (self.__class__.__name__, self.dataset.__class__.__name__, self.dimensions,
                                                                   self.loss, self.optimizer, self.batch_size, self.epochs)
        self.autoencoder.set_weights(np.load(filename))


class SldAutoencoder(Autoencoder):
    """Single-layer dense autoencoder"""

    def __init__(self, dimensions, loss, optimizer, batch_size, dataset, epochs=None):
        super().__init__(dimensions, loss, optimizer, batch_size, dataset, epochs)

        dim = int(dimensions)

        input_img = Input(shape=(self.dataset.instance_size,))
        encoded = Dense(dim, activation='relu')(input_img)
        decoded = Dense(self.dataset.instance_size, activation='sigmoid')(encoded)

        self.autoencoder = Model(input_img, decoded)

        self.encoder = Model(input_img, encoded)

        encoded_input = Input(shape=(dim,))
        decoder_layer = self.autoencoder.layers[-1]
        self.decoder = Model(encoded_input, decoder_layer(encoded_input))

        self.compile()


class SlsAutoencoder(Autoencoder):
    """Single-layer sparse autoencoder"""

    def __init__(self, dimensions, loss, optimizer, batch_size, dataset, epochs=None):
        super().__init__(dimensions, loss, optimizer, batch_size, dataset, epochs)

        dim = int(dimensions)

        input_img = Input(shape=(self.dataset.instance_size,))
        encoded = Dense(dim, activation='relu',
                        activity_regularizer=l2(1e-5))(input_img)
        decoded = Dense(self.dataset.instance_size, activation='sigmoid')(encoded)

        self.autoencoder = Model(input_img, decoded)

        self.encoder = Model(input_img, encoded)

        encoded_input = Input(shape=(dim,))
        decoder_layer = self.autoencoder.layers[-1]
        self.decoder = Model(encoded_input, decoder_layer(encoded_input))

        self.compile()


class DeepAutoencoder(Autoencoder):
    """Deep (multi-layer) autoencoder"""

    def __init__(self, dimensions, loss, optimizer, batch_size, dataset, epochs=None):
        super().__init__(dimensions, loss, optimizer, batch_size, dataset, epochs)

        dims = [int(d) for d in dimensions.split(',')]

        self.encoder = Sequential()
        self.encoder.add(Dense(dims[0], input_dim=self.dataset.instance_size))
        self.encoder.add(LeakyReLU())
        for d in dims[1:]:
            self.encoder.add(Dense(d))
            self.encoder.add(LeakyReLU())

        self.decoder = Sequential()
        self.decoder.add(Dense(dims[-2], input_dim=dims[-1]))
        self.decoder.add(LeakyReLU())
        for d in reversed(dims[:-2]):
            self.decoder.add(Dense(d))
            self.encoder.add(LeakyReLU())
        self.decoder.add(Dense(self.dataset.instance_size, activation='sigmoid'))

        self.autoencoder = Sequential([self.encoder, self.decoder])

        self.compile()


class Coder:
    """Image coder, encodes and decodes a given image using the provided autoencoder"""

    def __init__(self, model):
        self.model = model

    def encode(self, data, output_file):
        encoded = self.model.encoder.predict(np.array([data]))[0].astype('float16').tobytes()
        with open(output_file, 'wb') as f:
            f.write(encoded)

    def decode(self, input_file):
        with open(input_file, 'rb') as f:
            encoded = f.read()
        decoded = self.model.decoder.predict(np.array([np.fromstring(encoded, dtype='float16')]))
        return decoded


DATASETS = {'mnist': MNIST, 'cifar10': CIFAR10}
MODELS = {'sl-dense': SldAutoencoder, 'sl-sparse': SlsAutoencoder, 'deep': DeepAutoencoder}
DEFAULTS = {'sl-dense': '32', 'sl-sparse': '32', 'deep': '128,64,32'}


def load_model_from_file(filename):
    """
        Load model from a file where autoencoder model's weights were saved.

        Expects a particular filename format.
    """
    components = filename.split('-')
    model_name = components[0]
    dataset_name = components[1][1:]
    dimensions = components[2][1:]
    loss_function = components[3][1:]
    optimizer = components[4][1:]
    batch_size = components[5][1:]
    epochs = components[6][1:]

    if '_' in dataset_name:
        dataset_name, category = dataset_name.split('_')
        dataset = globals()[dataset_name](int(category))
    else:
        dataset = globals()[dataset_name]()

    model = globals()[model_name](dimensions, loss_function, optimizer, batch_size, dataset, epochs)

    model.load(filename)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoencoder-based neural network image compression")
    parser.add_argument('-m', '--model', choices=list(MODELS.keys()), default='sl-dense', help='Autoencoder model')
    parser.add_argument('-d', '--model-dimensions', help='Dimensions for the chosen model')
    parser.add_argument('-s', '--dataset', choices=list(DATASETS.keys()), default='mnist', help='Dataset to train/test the autoencoder on')
    parser.add_argument('-c', '--data-category', type=int, help='Category of data from the dataset to train on')
    parser.add_argument('-b', '--batch-size', type=int, default=256, help='Model training batch size')
    parser.add_argument('-e', '--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('-l', '--loss-function', default='binary_crossentropy', help='Loss function for model training')
    parser.add_argument('-o', '--optimizer', default='adadelta', help='Optimizer for model training')
    parser.add_argument('-n', '--example-number', type=int, default=5, help='Number of example images to show (original vs. decompressed)')

    args = parser.parse_args()
    args.model_dimensions = args.model_dimensions or DEFAULTS[args.model]

    dataset = DATASETS[args.dataset]
    dataset = dataset(args.data_category) if args.data_category is not None else dataset()
    model = MODELS[args.model](args.model_dimensions, args.loss_function, args.optimizer, args.batch_size, dataset)

    model.train(args.epochs)
    model.save()
    model.show(args.example_number)
