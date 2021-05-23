
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
    train_input_data_path = "./adblock_dataset/train"
    train_label_data_path = "./adblock_dataset/label"
    train_data_list = [a for a in os.listdir(train_input_data_path)]
    

    ##Hyperparameter
    epochs = 500
    batch_size = 8
    learning_rate = 0.001
    input_shape = [256,256,3]
    output_shape = [256,256,3]
    num_channels = 3
    save_file = './Save/adblock_unet.ckpt'

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

        with open('./Save/adblock_unet_loss_acc.csv', 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['epoch','loss','acc','val_loss','val_acc'])

        for Epoch in range(epochs):
            random.shuffle(train_data_list)
            train_list = train_data_list[0:int(len(train_data_list)*0.8)]
            validation_list = train_data_list[int(len(train_data_list)*0.8):]
            train_total_batch = int(len(train_list)/batch_size)
            valid_total_batch = int(len(validation_list)/batch_size)
            avg_loss = 0.
            avg_valid_loss = 0.
            train_acc = 0.
            valid_acc = 0.
            sample_y = 0

            print("Epoch : ", Epoch+1, "\n")

            print("Train Start! \n")
            pbar = tqdm(range(train_total_batch))
            train_idx=0
            for (train_x, train_y), _ in zip(read_generator(train_input_data_path, train_label_data_path, train_list, batch_size, num_channels, (256,256)), pbar):
                pbar.set_description("train loss : %f" % avg_loss)
                hypo_y, loss, _ = sess.run([model.logits, model.loss, optimizer], feed_dict={model.inputs:train_x, model.labels:train_y, model.keep_prob:0.5, model.phase:True})
                avg_loss += loss / train_total_batch
                acc = float((1 - np.sum(abs(np.float32(train_y) - hypo_y)) / (256*256*3*batch_size)) * 100 )
                train_acc += acc
                
                if train_idx == 0:
                    reshaped_input = np.reshape(train_x[0], [256,256,3])
                    reshaped_label = np.reshape(train_y[0], [256,256,3])
                    reshaped_sample = np.reshape(hypo_y[0], [256,256,3])
                    
                    _, ax = plt.subplots(1,3,figsize=(12,4))
                    ax[0].set_axis_off()
                    ax[0].imshow((reshaped_input*255).astype(np.uint8))
                    ax[0].set_title("Input image")
                    ax[1].set_axis_off()
                    ax[1].imshow((reshaped_label*255).astype(np.uint8))
                    ax[1].set_title("Label image")
                    ax[2].set_axis_off()
                    ax[2].imshow((reshaped_sample*255).astype(np.uint8))
                    ax[2].set_title("Created image")
                    plt.savefig("./Save/"+"train_"+str(Epoch)+".png")
                train_idx +=1

            print("Validation Start! \n")
            pbar = tqdm(range(valid_total_batch))
            valid_idx=0
            for (valid_x, valid_y), _ in zip(read_generator(train_input_data_path, train_label_data_path, validation_list, batch_size, num_channels, (256,256)), pbar):
                pbar.set_description("valid")
                ##valid_hypo_y = model.logits.eval(feed_dict={model.inputs:valid_x, model.labels:valid_y, model.keep_prob:1, model.phase:False})
                valid_hypo_y, valid_loss = sess.run([model.logits, model.loss], feed_dict={model.inputs:valid_x, model.labels:valid_y, model.keep_prob:1, model.phase:True})
                avg_valid_loss += valid_loss / valid_total_batch
                v_acc = float((1 - np.sum(abs(np.float32(valid_y) - valid_hypo_y)) / (256*256*3*batch_size)) * 100 )
                valid_acc += v_acc

                if valid_idx == 0:
                    reshaped_input = np.reshape(valid_x[0], [256,256,3])
                    reshaped_label = np.reshape(valid_y[0], [256,256,3])
                    reshaped_sample = np.reshape(valid_hypo_y[0], [256,256,3])

                    _, ax = plt.subplots(1,3,figsize=(12,4))
                    ax[0].set_axis_off()
                    ax[0].imshow((reshaped_input*255).astype(np.uint8))
                    ax[0].set_title("Input image")
                    ax[1].set_axis_off()
                    ax[1].imshow((reshaped_label*255).astype(np.uint8))
                    ax[1].set_title("Label image")
                    ax[2].set_axis_off()
                    ax[2].imshow((reshaped_sample*255).astype(np.uint8))
                    ax[2].set_title("Created image")
                    plt.savefig("./Save/"+"valid_"+str(Epoch)+".png")
                valid_idx +=1

            print("Epoch : %4d | Train_loss : %.8f | Train_acc : %.2f | Valid_loss : %.8f | Valid_acc : %.2f%% \n" %(Epoch+1, avg_loss, (train_acc/train_total_batch), avg_valid_loss, (valid_acc/valid_total_batch)))

            with open('./Save/adblock_unet_loss_acc.csv', 'a', newline='') as f:
                csv_writer = csv.writer(f)
                str_epoch = str(Epoch+1)
                str_loss = str(avg_loss)
                str_acc = str(train_acc/train_total_batch)
                str_val_loss = str(avg_valid_loss)
                str_val_acc = str(valid_acc/valid_total_batch)
                csv_writer.writerow([str_epoch,str_loss,str_acc,str_val_loss,str_val_acc])

            
            if (Epoch+1)%100 == 0:
                print("Save Weight! \n")
                new_saver.save(sess, save_file)
            



def test():
    
    test_input_data_path = "./adblock_dataset/train_test"
    test_label_data_path = "./adblock_dataset/label_test"
    test_data_list = [a for a in os.listdir(test_input_data_path)]

    ##Hyperparameter
    batch_size = 8
    learning_rate = 0.001
    input_shape = [256,256,3]
    output_shape = [256,256,3]
    num_channels = 3
    save_file = './Save/adblock_unet.ckpt'

    ##make Unet model
    model = UNet(input_shape, output_shape)

    ##cudnn error
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    ##weight restore
    restore_saver = tf.train.Saver()

    ##batch_norm 
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = model.train

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        restore_saver.restore(sess, save_file)

        random.shuffle(test_data_list)
        test_list = test_data_list
        test_total_batch = int(len(test_list)/batch_size)
        test_acc = 0.
        file_name_idx = 0
        
        print("Test Start! \n")
        pbar = tqdm(range(test_total_batch))
        for (test_x, test_y), _ in zip(read_generator(test_input_data_path, test_label_data_path, test_list, batch_size, num_channels, (256,256)), pbar):
            pbar.set_description("test")
            ##hypo_y = model.logits.eval(feed_dict={model.inputs:test_x, model.labels:test_y, model.keep_prob:1, model.phase:False})
            hypo_y = sess.run([model.logits], feed_dict={model.inputs:test_x, model.labels:test_y, model.keep_prob:1, model.phase:True})
            acc = float((1 - np.sum(abs(np.float32(test_y) - hypo_y)) / (256*256*3*batch_size)) * 100 )
            test_acc += acc

            reshaped_input = np.reshape(test_x[0], [256,256,3])
            reshaped_label = np.reshape(test_y[0], [256,256,3])
            print(np.array(hypo_y).shape)
            reshaped_sample = np.reshape(hypo_y[0][0], [256,256,3]) ## 해결사항
            _, ax = plt.subplots(1,3,figsize=(12,4))
            ax[0].set_axis_off()
            ax[0].imshow((reshaped_input*255).astype(np.uint8))
            ax[0].set_title("Input image")
            ax[1].set_axis_off()
            ax[1].imshow((reshaped_label*255).astype(np.uint8))
            ax[1].set_title("Label image")
            ax[2].set_axis_off()
            ax[2].imshow((reshaped_sample*255).astype(np.uint8))
            ax[2].set_title("Created image")
            plt.savefig("./Save/"+"result_"+str(file_name_idx)+".png")
            file_name_idx += 1

        print("Test acc : %.2f%% \n" %(test_acc/test_total_batch))
        with open('./Save/adblock_unet_loss_acc.csv', 'a', newline='') as f:
            csv_writer = csv.writer(f)
            str_acc = str(test_acc/test_total_batch)
            csv_writer.writerow(['test','test_acc', str_acc])

    


def read_generator(input_path, label_path, data_list, batch_size, n_channels, dim):
       
    X = np.empty((batch_size, *dim, n_channels))
    Y = np.empty((batch_size, *dim, n_channels))
    idx = 0
    while((batch_size)*(idx)<len(data_list)):
        for i, ID in enumerate(data_list[ (batch_size)*idx : (batch_size)*(idx+1) ]):        
            image_x = np.array(PIL.Image.open(input_path+"/"+ID)).reshape([dim[0],dim[1],n_channels])
            image_y = np.array(PIL.Image.open(label_path+"/"+ID)).reshape([dim[0],dim[1],n_channels])
            image_x = image_x.astype('float32')
            image_y = image_y.astype('float32')
            X[i,] = image_x/255.
            Y[i,] = image_y/255.
    
        idx=idx+1

        yield X, Y






if __name__ == "__main__":
    
    ##train()
    test()
