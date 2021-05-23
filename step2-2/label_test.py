
from network import UNet
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm
from shutil import copyfile
import PIL
import csv

def train():

    ## train_path
    train_input_data_path = "./data2/train/X"
    train_label_data_path = "./data2/train/Y"
    train_data_list = [a for a in os.listdir(train_input_data_path)]
    

    ##Hyperparameter
    epochs = 100
    batch_size = 8
    learning_rate = 0.001
    input_shape = [256,256,1]
    output_shape = [256,256,1]
    num_channels = 1
   

    ##make Unet model
    model = UNet(input_shape, output_shape)

    ##cudnn error
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    ##weight 저장
    new_saver = tf.train.Saver()

    ##batch_norm 
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = model.train

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        print(type(model.logits))


if __name__ == "__main__":
    train()