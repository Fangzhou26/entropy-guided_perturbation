# %%
import argparse
import os

from tensorflow.keras.datasets import cifar10, mnist, fashion_mnist, cifar10, cifar100
from tensorflow.keras.models import load_model
import numpy as np
import random

from kdes_generation_tf import fetch_kdes_tf,perturb
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # set GPU Limits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument("--m", "-m", help="Model", type=str, default="resnet")
    parser.add_argument("--save_path", "-save_path", help="Save path", type=str, default="./tmp/")
    parser.add_argument("--batch_size", "-batch_size", help="Batch size", type=int, default=128)
    parser.add_argument("--var_threshold", "-var_threshold", help="Variance threshold", type=float, default=1e-5)
    parser.add_argument("--num_classes", "-num_classes", help="The number of classes", type=int, default=10)

    args = parser.parse_args()
    args.save_path = args.save_path + args.d + "/" + args.m + "/"
    dir = os.path.dirname(args.save_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    print(args)

    # layer names
    if args.m == "conv":
        layer_names = []
        if args.d == "fmnist" or args.d == "mnist":
            num_layers = 8
        else:
            num_layers = 9
        for i in range(1, num_layers+1):
            layer_names.append("activation_" + str(i))
    elif args.m == "vgg16":
        layer_names = []
        for i in range(1, 16):
            layer_names.append("activation_" + str(i))
    else:
        layer_names = []
        for i in range(1, 20):
            layer_names.append("activation_" + str(i))
        layer_names.append("dense_1")

    # load dataset and models
    x_train_total = x_test = y_train_total = y_test = model = None
    if args.d == "mnist":
        (x_train_total, y_train_total), (x_test, y_test) = mnist.load_data()
        x_train_total = x_train_total.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        num_train = 50000

        print(infog("y_train len:{}".format(len(y_train_total))))
        print(infog("y_test len:{}".format(len(y_test))))
        # Load pre-trained model.
        model = load_model("./models/model_" + args.d + "_" + args.m + ".h5")
        model.summary()

    if args.d == "fmnist":
        (x_train_total, y_train_total), (x_test, y_test) = fashion_mnist.load_data()
        x_train_total = x_train_total.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        num_train = 50000

        print(infog("y_train len:{}".format(len(y_train_total))))
        print(infog("y_test len:{}".format(len(y_test))))
        # Load pre-trained model.
        model = load_model("./models/model_" + args.d + "_" + args.m + ".h5")
        model.summary()

    if args.d == "cifar100":
        (x_train_total, y_train_total), (x_test, y_test) = cifar100.load_data()
        num_train = 40000
        y_train_total = y_train_total.reshape([y_train_total.shape[0]])
        y_test = y_test.reshape([y_test.shape[0]])

        print(infog("y_train len:{}".format(len(y_train_total))))
        print(infog("y_test len:{}".format(len(y_test))))

        model = load_model("./models/model_" + args.d + "_" + args.m + ".h5")
        model.summary()

    if args.d == "cifar100_coarse":
        (x_train_total, y_train_total), (x_test, y_test) = cifar100.load_data(label_mode='coarse')
        num_train = 40000
        y_train_total = y_train_total.reshape([y_train_total.shape[0]])
        y_test = y_test.reshape([y_test.shape[0]])

        print(infog("y_train len:{}".format(len(y_train_total))))
        print(infog("y_test len:{}".format(len(y_test))))

        model = load_model("./models/model_" + args.d + "_" + args.m + ".h5")
        model.summary()

    elif args.d == "cifar10":
        (x_train_total, y_train_total), (x_test, y_test) = cifar10.load_data()
        num_train = 40000
        y_train_total = y_train_total.reshape([y_train_total.shape[0]])
        y_test = y_test.reshape([y_test.shape[0]])

        print(infog("y_train len:{}".format(len(y_train_total))))
        print(infog("y_test len:{}".format(len(y_test))))

        model = load_model("./models/model_" + args.d + "_" + args.m + ".h5")
        model.summary()

    # data pre-processing
    CLIP_MIN = -0.5
    CLIP_MAX = 0.5
    x_train_total = x_train_total.astype("float32")
    x_train_total = (x_train_total / 255.0) - (1.0 - CLIP_MAX)
    x_test = x_test.astype("float32")
    x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

    # split original training dataset into training and validation dataset
    x_train = x_train_total[:num_train]
    y_train = y_train_total[:num_train]
    x_valid = x_train_total[num_train:]
    y_valid = y_train_total[num_train:]


    ###### random sampling from valid set (consistent with RobOT) ######
    ###### see the no. of advs generated by RobOT at 20min timestamp, replace 4020)
    samples = 1000
    label_path = "./tmp/" + args.d + "/" + args.m + "/adv_compare_num/adv_labels_" + str(samples) + ".npy"
    if os.path.exists(label_path):
        print("Found saved labels.")
    else:
        seeds = random.sample(list(range(x_valid.shape[0])), samples) 
        images = x_valid[seeds]
        labels = y_valid[seeds]
        np.save(label_path,labels)

    ###### used to test capacity (how many each iteration) before Generate Ads below #######
    ###### replace 10 with the max capacity #########
    # x_adv = perturb(model, x_train, images[:10], y_train, layer_names,"test", args)
    capacity = 10

#### Usually no use ####
    # compare perturbed results for first x inputs
    # batch_size=16
    # y_pred_prob = model.predict(x_test, batch_size=batch_size, verbose=1)
    # y_pred_class = np.argmax(y_pred_prob,axis=1)

    # y_adv_prob = model.predict(x_adv, batch_size=batch_size, verbose=1)
    # y_adv_class = np.argmax(model.predict(x_adv, batch_size=batch_size, verbose=1),axis=1)
    # #print("y true:",y_test,"y pred:",y_pred_class,"y adv:",y_adv_class)

    # total_count = 0

#### Generate adversarial cases ####
    upper = int(samples / capacity)
    for i in range(1,upper+1):
        adv_path = "./tmp/" + args.d + "/" + args.m + "/adv_compare_num/adv_" + str(i*10) + "_images.npy" 
        print("slice:",i-1,"to",i)
        images_slice = images[ (i-1) * capacity:i * capacity]

        if os.path.exists(adv_path):
            print(infog("Found saved adversarial cases, skip"))  
        else:
            x_adv = perturb(model, x_train, images_slice, y_train, layer_names,"test", args)
            np.save(adv_path, x_adv)