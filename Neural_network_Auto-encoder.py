import numpy as np
import scipy.misc
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from scipy import linalg
import matplotlib.pyplot as plt
import os

nn_settings = {
    "data_1" : 'C:\Users\Aman Nagar\Desktop\Python_assig\_train',
    "data_2" : 'C:\Users\Aman Nagar\Desktop\Python_assig\_train_2',
    "data_3" : 'C:\Users\Aman Nagar\Desktop\Python_assig\_test',
    "output_layer_neurons": 784,
    "batch_size": 128,
    "epochs": 2,
    "loss": 'mse',
    "metrics": ['accuracy'],
    "optimizer": 'RMSprop'
}

data_settings = {
    "data_1": 'C:\Users\Aman Nagar\Desktop\Python_assig\_train',
    "data_2": 'C:\Users\Aman Nagar\Desktop\Python_assig\_train_2',
    "data_3": 'C:\Users\Aman Nagar\Desktop\Python_assig\_test',
    "data_1_samples": 20000,
    "data_2_samples": 2000,
    "data_3_samples": 100,
    "image_dim": 784
}


class Network(object):

    def __init__(self, network_settings):
        self.__dict__.update(network_settings)
        self.node_wise_error_1 = []
        self.node_wise_error_2 = []

    def build_model(self, n_hidden_neurons):
        model = Sequential()
        model.add(Dense(n_hidden_neurons, input_shape=(data_set.image_dim,), activation='relu'))
        model.add(Dense(self.output_layer_neurons, activation='linear'))
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        return model

    def train(self, model, x_train, y_train, n_epochs):
        print 'x_train.shape: '
        print x_train.shape
        print 'y_train.shape: '
        print y_train.shape
        history = model.fit(x_train, y_train, batch_size=self.batch_size, nb_epoch=n_epochs,
                           verbose=2)
        return history, model

    def clean_memory(self):
        self.node_wise_error_1 = []
        self.node_wise_error_2 = []


class Graph(object):

    def __init__(self):
        print ''

    def plot(self, param1, param2, title):
        import matplotlib.pyplot as plt
        handle_1, = plt.plot(param1, label='train data 20000')
        handle_2, = plt.plot(param2, label='train data 2000')
        plt.legend(handles=[handle_1, handle_2])
        plt.suptitle(title)
        plt.show()

    def plot_grid(self, data, title, grid_width, grid_height):
        fig, axes = plt.subplots(grid_width, grid_height, figsize=(12, 12))
        fig.suptitle(title, fontsize=16)
        for i in range(grid_width * grid_height):
            row, column = divmod(i, 10)
            axes[row, column].imshow(data[i, :].reshape(28, 28), cmap=plt.cm.gray)
            axes[row, column].axis('off')
        plt.axis('off')
        plt.show()


class DataSet(object):

    def __init__(self, settings):
        self.__dict__.update(settings)
        self.x_data_set_1 = np.zeros((784, 20000)),
        self.x_data_set_2 = np.zeros((self.image_dim, self.data_2_samples)),
        self.x_data_set_3 = np.zeros((self.image_dim, self.data_3_samples)),
        self.y_data_set_1 = np.zeros([self.image_dim, self.data_1_samples], dtype=np.int64),
        self.y_data_set_2 = np.zeros([self.image_dim, self.data_2_samples], dtype=np.int64),
        self.y_data_set_3 = np.zeros([self.image_dim, self.data_3_samples], dtype=np.int64),

    def read_image(self, file_name):
        return (scipy.misc.imread(file_name).astype(np.float32) / 255).reshape(-1, 1).flatten()
        '''image = scipy.misc.imread(file_name).astype(np.float32)
        image /= 255
        image = image.reshape(-1, 1)
        return image.flatten()'''

    def read_imageset_1(self):
        print 'read_imageset_1'
        x = np.zeros((784, 20000))
        y = np.zeros([784, 20000], dtype=np.int64)

        pos = 0
        for data in sorted(os.listdir(self.data_1)):
            x[:, pos] = self.read_image(os.path.join(self.data_1, data))
            # self.x_data_set_1[:, pos] = self.read_image(os.path.join(self.data_1, data))
            y[int(data[:1])][pos] = 1
            # self.y_data_set_1[int(data[:1])][pos] = 1
            pos += 1
        self.x_data_set_1 = x.T
        self.y_data_set_1 = y.T

    def read_imageset_2(self):
        print 'read_imageset_2'
        x = np.zeros((784, 2000))
        y = np.zeros([784, 2000], dtype=np.int64)

        pos = 0
        for data in sorted(os.listdir(self.data_2)):
            x[:, pos] = self.read_image(os.path.join(self.data_2, data))
            y[int(data[:1])][pos] = 1
            pos += 1
        self.x_data_set_2 = x.T
        self.y_data_set_2 = y.T

    def read_imageset_3(self):
        print 'read_imageset_3'
        x = np.zeros((784, 100))
        y = np.zeros([784, 100], dtype=np.int64)
        pos = 0
        for data in sorted(os.listdir(self.data_3)):
            x[:, pos] = self.read_image(os.path.join(self.data_3, data))
            y[int(data[:1])][pos] = 1
            pos += 1
        self.x_data_set_3 = x.T
        self.y_data_set_3 = y.T


class PCA(object):

    def __init__(self):
        print ''

    def transform(self, data):
        data -= np.mean(data, axis=0)
        _, _, eigen_vectors = linalg.svd(data, full_matrices=False)
        eigen_vectors = eigen_vectors[:, :100]
        return eigen_vectors


class Tasks(object):

    def __init__(self):
        self.load_data()

    def load_data(self):
        data_set.read_imageset_1()
        data_set.read_imageset_2()
        data_set.read_imageset_3()

    def exec_all_task(self):
        self.exec_task1()
        self.exec_task2()
        self.exec_task3()
        self.exec_task4()
        self.exec_task5()

    def exec_task1(self):
        model = network.build_model(100)
        summary_data1, _ = network.train(model, data_set.x_data_set_1, data_set.y_data_set_1, 50)
        summary_data2, _ = network.train(model, data_set.x_data_set_2, data_set.y_data_set_2, 50)
        graph.plot(summary_data1.history['loss'], summary_data2.history['loss'], "Loss")

    def exec_task2(self):
        network.clean_memory()
        for hidden_neurons in ([20, 40, 60, 80, 100]):
            model = network.build_model(hidden_neurons)
            summary_data1, _ = network.train(model, data_set.x_data_set_1, data_set.y_data_set_1, 50)
            summary_data2, _ = network.train(model, data_set.x_data_set_2, data_set.y_data_set_2, 50)
            network.node_wise_error_1.append(np.mean(summary_data1.history['loss']))
            network.node_wise_error_2.append(np.mean(summary_data2.history['loss']))

        graph.plot(network.node_wise_error_1, network.node_wise_error_2, "No. of Hidden Neurons vs Loss")

    def exec_task3(self):
        model = network.build_model(100)
        _, model = network.train(model, data_set.x_data_set_1, data_set.y_data_set_1, 100)
        model.save('trained_model.h5', overwrite=True)

        weights_1, bias_1 = model.layers[0].get_weights()
        weights_1 = weights_1.T

        _, model = network.train(model, data_set.x_data_set_2, data_set.y_data_set_2, 100)
        weights_2, bias_2 = model.layers[0].get_weights()
        weights_2 = weights_2.T

        graph.plot_grid(weights_1, "Data-1 Weights", 10, 10)
        graph.plot_grid(weights_2, "Data-2 Weights", 10, 10)

    def exec_task4(self):
        graph.plot_grid(data_set.x_data_set_3, "Original Input", 10, 10)
        model = load_model('trained_model.h5')
        output = model.predict(data_set.x_data_set_3)
        graph.plot_grid(output, "Reconstructed Output", 10, 10)

    def exec_task5(self):
        model = load_model('trained_model.h5')
        output = model.predict(data_set.x_data_set_2)

        pca = PCA()
        in_eigen_vectors = pca.transform(data_set.x_data_set_2)
        graph.plot_grid(in_eigen_vectors.T, "Top 100 Eigen Vectors for Input", 10, 10)

        output_eigen_vectors = pca.transform(output)
        graph.plot_grid(output_eigen_vectors.T, "Top 100 Eigen Vectors for Output", 10, 10)


if __name__ == "__main__":
    graph = Graph()
    data_set = DataSet(data_settings)
    network = Network(nn_settings)
    tasks = Tasks()
    tasks.exec_all_task()

