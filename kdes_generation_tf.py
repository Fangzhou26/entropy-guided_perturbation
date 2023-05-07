import os
from multiprocessing import Pool
import dill as pickle
import numpy as np
from scipy.stats import gaussian_kde
from tqdm import tqdm
import gc
from collections import Counter

import tensorflow as tf
from tensorflow.keras.models import Model
# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()
import tensorflow_probability as tfp

from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # set GPU Limits


def _aggr_output(x):
    return [np.mean(x[..., j]) for j in range(x.shape[-1])]


def _get_saved_path(base_path, dtype, layer_names):
    """Determine saved path of ats and pred

    Args:
        base_path (str): Base save path.
        dtype (str): Name of dataset type (e.g., train, test, fgsm, ...).
        layer_names (list): List of layer names.

    Returns:
        ats_path: File path of ats.
        pred_path: File path of pred (independent of layers)
    """

    joined_layer_names = "_".join(layer_names[:5])
    return (
        os.path.join(
            base_path,
            dtype + "_" + joined_layer_names + "_ats" + ".npy",
        ),
        os.path.join(base_path, dtype + "_pred" + ".npy"),
    )


def get_ats(
        model,
        dataset,
        name,
        layer_names,
        save_path=None,
        batch_size=16,
        num_proc=10,
):
    """Extract activation traces of dataset from models.

    Args:
        model (keras models): Subject models.
        dataset (ndarray): Set of inputs fed into the models.
        name (str): Name of input set.
        layer_names (list): List of selected layer names.
        save_path (tuple): Paths of being saved ats and pred.
        batch_size (int): Size of batch when serving.
        num_proc (int): The number of processes for multiprocessing.

    Returns:
        ats (ndarray): Array of (layers, inputs, neuron outputs).
        pred (ndarray): Array of predicted classes.
    """

    outputs = [model.get_layer(layer_name).output for layer_name in layer_names]
    outputs.append(model.output)

    temp_model = Model(inputs=model.input, outputs=outputs)

    prefix = info("[" + name + "] ")
    p = Pool(num_proc)
    print(prefix + "Model serving")
    layer_outputs = temp_model.predict(dataset, batch_size=batch_size, verbose=1)
    pred_prob = layer_outputs[-1]
    pred = np.argmax(pred_prob, axis=1)
    layer_outputs = layer_outputs[:-1]

    print(prefix + "Processing ATs")
    ats = None
    for layer_name, layer_output in zip(layer_names, layer_outputs):
        if layer_output[0].ndim == 3:
            # For convolutional layers
            layer_matrix = np.array(
                p.map(_aggr_output, [layer_output[i] for i in range(len(dataset))])
            )
        else:
            layer_matrix = np.array(layer_output)

        if ats is None:
            ats = layer_matrix
        else:
            ats = np.append(ats, layer_matrix, axis=1)
            layer_matrix = None

    if save_path is not None and name == "train":
        np.save(save_path[0], ats)
        np.save(save_path[1], pred)

    p.close()
    p.join()
    return ats, pred


def _get_train_ats(model, x_train, layer_names, args):
    saved_train_path = _get_saved_path(args.save_path, "train", layer_names)
    if os.path.exists(saved_train_path[0]):
        print(infog("Found saved {} ATs, skip serving".format("train")))
        # In case train_ats is stored in a disk
        train_ats = np.load(saved_train_path[0])
        train_pred = np.load(saved_train_path[1])
    else:
        train_ats, train_pred = get_ats(
            model,
            x_train,
            "train",
            layer_names,
            save_path=saved_train_path,
        )
        print(infog("train ATs is saved at " + saved_train_path[0]))

    return train_ats, train_pred


def _get_kdes(train_ats, class_matrix, args):
    """Kernel density estimation

    Args:
        train_ats (ndarray): List of activation traces in training set.
        class_matrix (dict): List of index of classes.
        args: Keyboard args.

    Returns:
        kdes (list): List of kdes per label if classification task.
        removed_cols (list): List of removed columns by variance threshold.
            To further reduce the computational cost, we filter out neurons
            whose activation values show variance lower than a pre-defined threshold,
        max_kde (list): List of maximum kde values.
        min_kde (list): List of minimum kde values.
    """

    col_vectors = np.transpose(train_ats)
    variances = np.var(col_vectors, axis=1)
    removed_cols = np.where(variances < args.var_threshold)[0]

    kdes = {}
    max_kde = {}
    min_kde = {}
    tot = 0
    for label in tqdm(range(args.num_classes), desc="kde"):
        refined_ats = np.transpose(train_ats[class_matrix[label]])
        refined_ats = np.delete(refined_ats, removed_cols, axis=0)

        tot += refined_ats.shape[1]

        print("refined ats shape: {}".format(refined_ats.shape))

        if refined_ats.shape[0] == 0:
            print(
                warn("all ats were removed by threshold {}".format(args.var_threshold))
            )
            break

        print("refined ats min max {} ; {} ".format(refined_ats.min(), refined_ats.max()))

        kdes[label] = gaussian_kde(refined_ats)
        outputs = kdes[label](refined_ats)
        max_kde[label] = np.max(outputs)
        min_kde[label] = np.min(outputs)
        print("min_kde: %s" % min_kde[label])
        print("max_kde: %s" % max_kde[label])

    print("gaussian_kde(refined_ats) shape[1] sum: {}".format(tot))

    print(infog("The number of removed columns: {}".format(len(removed_cols))))

    return kdes, removed_cols, max_kde, min_kde


def _get_model_output_idx(model, layer_names):
    # return param
    output_idx_map = {}

    # local tmp param
    start = 0
    end = 0
    layer_idx_map = {}

    # mapping layer names to layer
    for layer in model.layers:
        if layer.name in layer_names:
            layer_idx_map[layer.name] = layer
    assert len(layer_names) == len(layer_idx_map)

    # calc each layer output idx
    for layer_name in layer_names:
        layer = layer_idx_map[layer_name]
        name = layer.name
        output_shape = layer.output_shape
        end += output_shape[-1]
        output_idx_map[name] = (start, end)

        start = end

    return output_idx_map


def save_results(fileName, obj):
    dir = os.path.dirname(fileName)
    if not os.path.exists(dir):
        os.makedirs(dir)

    f = open(fileName, 'wb')
    pickle.dump(obj, f)


def _aggr_output_tf(x):
    return [tf.reduce_mean(x[..., j]) for j in range(x.shape[-1])]


def get_ats_tf(
        model,
        dataset,
        name,
        layer_names,
        args
):
    """Extract activation traces of dataset from models.

    Args:
        model (keras models): Subject models.
        dataset (ndarray): Set of inputs fed into the models.
        name (str): Name of input set.
        layer_names (list): List of selected layer names.
        batch_size (int): Size of batch when serving.
        num_proc (int): The number of processes for multiprocessing.

    Returns:
        ats (ndarray): Array of (layers, inputs, neuron outputs).
        pred (ndarray): Array of predicted classes.
    """

    ###### Future improvement to be modified ######
    # saved_path = args.save_path + "ats_tf"
    # print(saved_path,saved_path)
    # if os.path.exists(saved_path):
    #     print(infog("Found saved {} ATs, skip serving".format("test")))
    #     # In case train_ats is stored in a disk
    #     ats = tf.io.read_file(saved_path + "_ats")
    #     pred = tf.io.read_file(saved_path + "_pred")

    #else:
    outputs = [model.get_layer(layer_name).output for layer_name in layer_names]
    outputs.append(model.output)

    temp_model = Model(inputs=model.input, outputs=outputs)

    prefix = info("[" + name + "] ")
    print(prefix + "Model serving")
    layer_outputs = temp_model(dataset)
    pred_prob = layer_outputs[-1]
    pred = pred_prob
    layer_outputs = layer_outputs[:-1]

    print(prefix + "Processing ATs")
    ats = None
    for layer_name, layer_output in zip(layer_names, layer_outputs):
        if layer_output[0].ndim == 3:
            # For convolutional layers
            layer_matrix = [_aggr_output_tf(layer_output[i]) for i in range(dataset.shape[0])]
        else:
            layer_matrix = layer_output

        if ats is None:
            ats = layer_matrix
        else:
            ats = tf.concat((ats, layer_matrix), axis=1)
            layer_matrix = None
        #### Future improvement #####
        # add codes to save tf files to save time

    return ats, pred


def evaluate_tf(points, dataset, kernel):
    """Evaluate the estimated pdf on a set of points.
    Parameters
    ----------
    points : (# of dimensions, # of points)-array
        Alternatively, a (# of dimensions,) vector can be passed in and
        treated as a single point.
    Returns
    -------
    values : (# of points,)-array
        The values at each point.
    Raises
    ------
    ValueError : if the dimensionality of the input points is different than
                 the dimensionality of the KDE.
    """

    d_data, n_data = dataset.shape
    factor = tf.convert_to_tensor(kernel.factor)
    weights = tf.convert_to_tensor(kernel.weights)
    _data_covariance = tfp.stats.covariance(dataset, sample_axis=1, event_axis=0)
    _data_covariance = tf.cast(_data_covariance, dtype=tf.float64)

    try:
        _data_inv_cov = tf.linalg.inv(_data_covariance)

        covariance = _data_covariance * tf.math.pow(factor, 2)
        inv_cov = tf.math.divide(_data_inv_cov, tf.math.pow(factor, 2))
        _norm_factor = tf.math.sqrt(
            tf.linalg.det(2.0 * tf.constant(np.pi, dtype="float64") * covariance) + 1e-10) * tf.cast(n_data, 'float64')
    except:
        _data_covariance = _data_covariance + tf.cast(tf.random.uniform(tf.shape(_data_covariance), minval=0, maxval=1, seed=10) * 1e-10, dtype=tf.float64)
        _data_inv_cov = tf.linalg.inv(_data_covariance)

        covariance = _data_covariance * tf.math.pow(factor, 2)
        inv_cov = tf.math.divide(_data_inv_cov, tf.math.pow(factor, 2))
        _norm_factor = tf.math.sqrt(tf.linalg.det(2.0 * tf.constant(np.pi, dtype="float64") * covariance) + 1e-10) * tf.cast(n_data, 'float64')

    d, m = points.shape

    result = tf.zeros((m,), dtype="float64")

    whitening = tf.transpose(tf.linalg.cholesky(inv_cov))
    scaled_dataset = tf.tensordot(whitening, dataset,axes = 1)
    scaled_points = tf.tensordot(whitening, points, axes = 1)

    if m >= n_data:
        # there are more points than data, so loop over data
        for i in range(n_data):
            diff = scaled_dataset[:, i, tf.newaxis] - scaled_points
            #tdiff = tf.tensordot(inv_cov, diff, axes=1)
            energy = tf.reduce_sum(diff * diff, axis=0) / 2.0
            result = result + weights[i] * tf.math.exp(-energy)
    else:
        # loop over points
        for i in range(m):
            diff = scaled_dataset - scaled_points[:, i, tf.newaxis]
            #tdiff = tf.tensordot(inv_cov, diff, axes=1)
            energy = tf.reduce_sum(diff * diff, axis=0) / 2.0
            result = tf.tensor_scatter_nd_update(result, [[i]], [tf.reduce_sum(tf.math.exp(-energy) * weights, axis=0)])

    result = result / _norm_factor

    return result


def fetch_kdes_tf(models, x_train, x_test, y_train, layer_names, test_name, args):
    # obtain the number of neurons for each layer
    model_output_idx = _get_model_output_idx(models, layer_names)

    # generate feature vectors for each layer on training, validation set
    all_train_ats, train_pred = _get_train_ats(models, x_train, layer_names, args)
    #print("train_ats finished")
    all_test_ats, test_pred = get_ats_tf(models, x_test, test_name, layer_names,args)
    #print("get_ats_tf finished")

    # obtain the input indexes for each class
    class_matrix = {}
    for i, label in enumerate(np.reshape(y_train, [-1])):
        if label not in class_matrix:
            class_matrix[label] = []
        class_matrix[label].append(i)

    pred_labels_test = tf.zeros([len(layer_names), x_test.shape[0], args.num_classes])
    layer_labels_test = tf.zeros([len(layer_names), x_test.shape[0]])
    layer_idx = 0
    for layer_name in layer_names:
        #if int(layer_name[-1]) > 2: break
        # get layer names ats
        (start_idx, end_idx) = model_output_idx[layer_name]
        train_ats = all_train_ats[:, start_idx:end_idx]
        test_ats = all_test_ats[:, start_idx:end_idx]

        # generate kde functions per class and layer
        kdes_file = args.save_path + "/kdes-pack/%s" % layer_name
        file = open(kdes_file, 'rb')
        (kdes, removed_cols, max_kde, min_kde) = pickle.load(file)

        # generate inferred classes for each layer
        kde_values = tf.zeros([args.num_classes, test_ats.shape[0]], dtype="float64")
        indexes = np.array([i for i in range(test_ats.shape[1])])
        # obtain 10 kde values for each test
        for label in range(len(kdes)):
            remain_cols = np.expand_dims(np.delete(indexes, removed_cols), axis=1)
            refined_ats = tf.transpose(test_ats)
            refined_ats = tf.gather_nd(refined_ats, indices=remain_cols)
            refined_ats = tf.cast(refined_ats, 'float64')

            refined_ats_train = tf.transpose(train_ats[class_matrix[label]])
            refined_ats_train = tf.gather_nd(refined_ats_train, indices=remain_cols)
            refined_ats_train = tf.cast(refined_ats_train, 'float64')

            #print("Label:",label)
            kde_values = tf.tensor_scatter_nd_update(kde_values, [[label]], [evaluate_tf(refined_ats, refined_ats_train, kdes[label])])

        kde_values = tf.transpose(kde_values)
        layer_labels = tf.math.argmax(kde_values,axis=1)
        pred_labels = tf.nn.softmax(kde_values, axis=1)

        layer_labels_test = tf.tensor_scatter_nd_update(layer_labels_test, [[layer_idx]], [layer_labels])
        pred_labels_test = tf.tensor_scatter_nd_update(pred_labels_test, [[layer_idx]], [pred_labels])

        layer_idx += 1
    
    # make every row as a test case
    layer_labels_test = tf.transpose(layer_labels_test)
    layer_pred_count = cal_layer_pred_count(layer_labels_test)

    label_num = 10
    prob_sum = tf.reduce_sum(pred_labels_test,axis=0)
    prob_sum = tf.nn.softmax(prob_sum,axis=1)

    del all_train_ats, all_test_ats, train_ats, test_ats, kdes, refined_ats, refined_ats_train, pred_labels
    gc.collect()
    return prob_sum,layer_pred_count

def perturb(models, x_train, x_test, y_train, layer_names, test_name, args):
    loop = 10
    # unique inferred classes for all layers
    layer_pred_count_records = np.zeros((loop,len(x_test)))
    removed_cases = []
    ep = [0.01] * x_test.shape[0]

    for i in tqdm(range(loop)):
        ###### To see the prediction changes, uncomment below ######
        #print("For loop:",i)
        # y_pred_prob = models.predict(x_test, batch_size=16, verbose=1)
        # y_pred_class = np.argmax(y_pred_prob,axis=1)
        # print("predicted_class:",y_pred_class)
        with tf.GradientTape(persistent=True) as tape:
            x_test = tf.Variable(x_test)
            prob_sum,layer_pred_count = fetch_kdes_tf(models, x_train, x_test, y_train, layer_names, test_name, args)
            tape.watch(x_test)
            entropy = cal_entropy(prob_sum)

        layer_pred_count_records[i] = layer_pred_count

        if i == 0: 
            ori_count = layer_pred_count
        else:
            removed_cases.extend([j for j in [i for i in range(x_test.shape[0]) if layer_pred_count[i] > ori_count[i]] if j not in removed_cases])
            
        grads = tape.gradient(entropy,x_test)
        signed_grad = tf.sign(grads)
        if len(removed_cases) == x_test.shape[0]: break
        for i in range(x_test.shape[0]):
            if i not in removed_cases:
                ep[i] +=  0.01
                x_test= tf.tensor_scatter_nd_update(x_test, [[i]], [x_test[i] + ep[i] * signed_grad[i]])         

    return x_test

def cal_entropy(x): 
    n_data,n_class = x.shape
    entropy = tf.zeros((n_data,))
    for i in range(n_data):
        entropy = tf.tensor_scatter_nd_update(entropy, [[i]], [tf.reduce_sum(-tf.math.multiply(x[i],tf.math.log(x[i])))])

    return entropy

def cal_layer_pred_count(x):
    x = x.numpy()
    n_data = len(x)
    count = np.zeros(n_data)
    for i in range(n_data):
        count[i] = len(set(x[i]))
    return count