# %%
import argparse
import os
import random
from tensorflow.keras.datasets import cifar10,mnist, fashion_mnist, cifar10, cifar100
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow import keras

from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # set GPU Limits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="cifar10")
    parser.add_argument("--m", "-m", help="Model", type=str, default="conv")
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
    
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    print(y_train.shape)

######## Data concatenation code #############
    samples = 1220 #Adjust accordingly
    capacity = 10  #Adjust accordingly
    upper  = int(samples / capacity)
    path_prefix = "./tmp/" + args.d + "/" + args.m + "/adv_compare_num/adv_"
    #x_labels = np.load(path_prefix + "labels_" + str(samples) + ".npy")
    x_labels = np.load(path_prefix + "labels_6000.npy")[:1220]
    x_labels = keras.utils.to_categorical(x_labels, 10)

    # for saved files
    adv_path = path_prefix + "images_total_" + str(samples) + ".npy"
    if os.path.exists(adv_path):
        x_adv = np.load(adv_path)
    else:
        for i in range(1,upper + 1):
            adv_path = path_prefix + str(i*10) + "_images.npy"
            x_adv_loop = np.load(adv_path)
            if i == 1: x_adv = x_adv_loop
            else: x_adv = np.concatenate((x_adv, x_adv_loop),axis=0)

    print("Shape of advs: ", x_adv.shape,"Shape of labels: ",x_labels.shape)
    #np.save(path_prefix + "images_total_" + str(samples) + ".npy",x_adv)
        

    ##### Analysis on how many predictions are changed #####
    # total_count = 0
    # for i in range(1,13):
    #     adv_path = "./tmp/cifar10/conv/adv_compare_num/adv_" + str(i*100) + ".npy"
    #     print("slice:",i-1,"to",i)
    #     x_test_slice = x_test[ (i-1)*100:i*100,:,:,:]
    #     x_adv= np.load(adv_path)
    
    #     batch_size=16
    #     y_pred_prob = model.predict(x_test_slice, batch_size=batch_size, verbose=1)
    #     y_pred_class = np.argmax(y_pred_prob,axis=1)

    #     y_adv_prob = model.predict(x_adv, batch_size=batch_size, verbose=1)
    #     y_adv_class = np.argmax(model.predict(x_adv, batch_size=batch_size, verbose=1),axis=1)


    #     count = 0
    #     for i in range(100):
    #         if y_pred_class[i] != y_adv_class[i]:
    #             count += 1
    #     print("%d predictions have been changed" % count)

    #     total_count += count


######### Compare to RobOT ###########
    # Load the generated adversarial inputs for Robustness evaluation. 
    if args.d == "cifar10": folder = "CIFAR-10"
    elif args.d == "mnist": folder = "MINIST"
    elif args.d == "fmnist": folder = "FASHION"

    with np.load("../RobOT/" + folder + "/" + args.d + "_" + args.m + "/FGSM_Test.npz") as f:
        fgsm_test, fgsm_test_labels = f['advs'], f['labels']

    with np.load("../RobOT/" + folder + "/" + args.d + "_" + args.m + "/PGD_Test.npz") as f:
        pgd_test, pgd_test_labels = f['advs'], f['labels']

    # Construct dataset to test robustness
    fp_test = np.concatenate((fgsm_test, pgd_test))
    fp_test_labels = np.concatenate((fgsm_test_labels, pgd_test_labels))

    x_train_mix = np.concatenate((x_train, x_adv),axis=0)
    y_train_mix = np.concatenate((y_train, x_labels),axis=0)
    
    print(fp_test.shape,fp_test_labels.shape)
    print(x_train_mix.shape,y_train_mix.shape)

    # Analysis on acc and robustness
    _,robust = model.evaluate(fp_test,fp_test_labels,verbose=0)
    _,acc= model.evaluate(x_test, y_test, verbose=0)

    # Retraining
    model.fit(x_train_mix, y_train_mix, epochs=10, batch_size=64, verbose=0,
                 validation_data=(fp_test, fp_test_labels))

    _, afp = model.evaluate(fp_test, fp_test_labels, verbose=0)
    _, aclean = model.evaluate(x_test, y_test, verbose=0)

    # Robustness improvement
    print(args.d + "-" + args.m + " Original robustness:",robust)
    print(args.d + "-" + args.m + "  Retrained robustness:",afp)
    # Accuracy change
    print(args.d + "-" + args.m + " Original accuracy:",acc)
    print(args.d + "-" + args.m + " Retrained accuracy:",aclean)
