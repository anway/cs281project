from __future__ import absolute_import
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad
from scipy import optimize
import sys

import matplotlib.pyplot as plt

filename = "default.txt" if len(sys.argv) < 2 else sys.argv[1]

def logsumexp(X, axis=1):
    max_X = np.max(X)
    return max_X + np.log(np.sum(np.exp(X - max_X), axis=axis, keepdims=True))

def get_wine_data():
    print("Loading training data...")
    wine_data_file = open('./wines_data/wine.data', 'r')
    num_class_1, num_class_2, num_class_3 = 59, 71, 48
    wines_data = []

    for line in wine_data_file:
        entries = line.split(',')
        wine_type = int(entries[0]) # 1, 2, or 3
        wine_type_one_hot = [1., 0., 0.] if wine_type == 1 else [0., 1., 0.] if wine_type == 2 else [0., 0., 1.]
        wine_features = map(float, entries[1:-1])

        wine_data = wine_features
        wine_data.extend(wine_type_one_hot)
        wines_data.append(wine_data)

    wines_data = np.array(wines_data)
    np.random.shuffle(wines_data)
    features, labels = wines_data[:, :-3], wines_data[:, -3:]

    num_data_pts = len(wines_data)
    train_set_size = 0.8 * num_data_pts
    train_data, train_labels = features[:train_set_size], labels[:train_set_size]
    test_data, test_labels = features[train_set_size:], labels[train_set_size:]

    assert np.sum(wines_data[:, -3]) == num_class_1
    assert np.sum(wines_data[:, -2]) == num_class_2
    assert np.sum(wines_data[:, -1]) == num_class_3
    return num_data_pts, train_data, train_labels, test_data, test_labels


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x)) if x >= 0 else np.exp(x - np.logaddexp(x, 0))

def make_nn_funs(layer_sizes, L2_reg):
    shapes = zip(layer_sizes[:-1], layer_sizes[1:])
    N = sum((m+1)*n for m, n in shapes)

    def unpack_layers(W_vect):
        for m, n in shapes:
            yield W_vect[:m*n].reshape((m,n)), W_vect[m*n:m*n+n]
            W_vect = W_vect[(m+1)*n:]

    def predictions(W_vect, inputs, alpha):
        outputs = 0
        for W, b in unpack_layers(W_vect):
            prev_outputs = outputs
            outputs = np.dot(np.array(inputs), W) + b
            inputs = np.tanh(outputs)
        return sigmoid(alpha) * outputs + (1 - sigmoid(alpha)) * prev_outputs

    def loss(params, X, T):
        W_vect = params[:-1]
        alpha = params[-1]
        log_prior = -L2_reg * np.dot(W_vect, W_vect)
        preds = predictions(W_vect, X, alpha)
        normalised_log_probs = preds - logsumexp(preds)
        log_lik = np.sum(normalised_log_probs * T)
        return -1.0 * (log_prior + log_lik)

    def frac_err(params, X, T):
        W_vect = params[:-1]
        alpha = params[-1]
        percent_wrong = np.mean(np.argmax(T, axis=1) != np.argmax(predictions(W_vect, X, alpha), axis=1))
        return percent_wrong

    return N, predictions, loss, frac_err

def make_batches(N_data, batch_size):
    return [slice(i, min(i+batch_size, N_data))
            for i in range(0, N_data, batch_size)]


if __name__ == '__main__':
    # Network parameters
    layer_sizes = [12, 3, 3]
    L2_reg = 1.0

    # Training parameters
    param_scale = 0.1

    # Load and process wines data
    N_data, train_images, train_labels, test_images, test_labels = get_wine_data()
    batch_size = len(train_images)

    # Make neural net functions
    N_weights, pred_fun, loss_fun, frac_err = make_nn_funs(layer_sizes, L2_reg)

    # Gradient with respect to weights and alpha
    loss_grad_P = grad(loss_fun, 0)

    # Initialize weights
    rs = npr.RandomState(11)
    W = rs.randn(N_weights) * param_scale

    # Initialize alpha
    alpha = 0.

    print("    Train err  |   Test err  |   Alpha")

    f_out = open(filename, 'w')
    f_out.write("    Train err  |   Test err  |   Alpha\n")
    f_out.close()

    def print_perf(params):
        f_out = open(filename, 'a')
        test_perf  = frac_err(params, test_images, test_labels)
        train_perf = frac_err(params, train_images, train_labels)
        print("{0:15}|{1:15}|{2:15}".format(train_perf, test_perf, params[-1]))
        f_out.write("{0:15}|{1:15}|{2:15}\n".format(train_perf, test_perf, params[-1]))
        f_out.close()

    # Minimize with BFGS
    num_iterations = []
    train_errors = []
    test_errors = [] 
    for i in range(0, 100):
        optimize.minimize(loss_fun, np.append(W, alpha), jac=loss_grad_P, method='L-BFGS-B', \
            args=(train_images, train_labels), options={'disp': True}, callback=print_perf)

        training_error = 0.
        test_error = 0.
        with open(filename, 'r') as input_file:
            next(input_file)
            curr_iteration = 0
            for line in input_file:
                data_as_string = line.split('|')
                data = map(float, data_as_string)

                training_error = data[0]
                test_error = data[1]
                curr_iteration += 1
            num_iterations.append(curr_iteration)
            train_errors.append(training_error)
            test_errors.append(test_error)

    print("Average Iteration #: {0:15}".format(sum(num_iterations) / len(num_iterations)))
    print("Average Train Error: {0:15}".format(sum(train_errors) / len(train_errors))) 
    print("Average Test Error: {0:15}".format(sum(test_errors) / len(test_errors)))
