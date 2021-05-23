import tensorflow as tf
import numpy as np
import PIL
import os
import math
import random
import matplotlib.image as img
import matplotlib.pyplot as plt
from shutil import copyfile
import cv2
from skimage.util import random_noise
from skimage.measure import compare_psnr
import shutil
## image aug for train
def resize_add_noise(train_path, noise_train_path):

    
   
    train_file_list = [a for a in os.listdir(train_path)]
   

    
    for i, file_name in enumerate(train_file_list):
        print("make train noise image", i)

        rd_image = cv2.imread(train_path+"/"+file_name)
        height, width, channel = rd_image.shape
        print("width : ",width, "height : ", height)
        
        ##crop_size = np.random.randint(50,201)
        crop_size = 128
        crop_width = crop_size
        crop_height = crop_size
        print("crop_size", crop_size)

        range_width = (int)(width/crop_width)
        range_height = (int)(height/crop_height)
        print("range width, height : ",range_width, range_height)
        
        mean = 0
        sigma = np.random.uniform(0,50)
        ##sigma = 25
        noise = np.random.normal(mean, sigma, (height, width, channel))
        noise = noise.reshape(height,width,channel)
        noisy = rd_image + noise
        noisy = np.clip(noisy, 0, 255)
        noisy = noisy.astype(np.uint8)

        j = 0
        for w in range(range_width):
            for h in range(range_height):
                box = (w*crop_height, h*crop_width, (w+1)*(crop_height), (h+1)*(crop_width))
                print("box range : ", w*crop_height, h*crop_width, (w+1)*(crop_height), (h+1)*(crop_width))
                
                #original_crop_image = rd_image[box[1]:box[3],box[0]:box[2]]
                noisy_crop_image = noisy[box[1]:box[3],box[0]:box[2]]

                cv2.imwrite(noise_train_path+"/"+str(j)+"_"+file_name, noisy_crop_image)
                #cv2.imwrite(original_train_path+"/"+str(j)+"_"+file_name, original_crop_image)
                j=j+1
    
    

    
def make_resize(train_path, resize_train_path, height_size, width_size):
    
    if not os.path.exists(resize_train_path):
        os.makedirs(resize_train_path)

    train_file_list = [a for a in os.listdir(train_path)]


    for i, file_name in enumerate(train_file_list):
        print("make train resize image", i)

        rd_image = cv2.imread(train_path+"/"+file_name)
        height, width, channel = rd_image.shape
        print("width : ",width, "height : ", height)

        
        resize_image = cv2.resize(rd_image, dsize=(height_size, width_size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(resize_train_path+"/"+file_name, resize_image)





## image naming for train
def naming_img(path, data_path, start_num=0):
    train_data_path = data_path
   
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)


    dir_path = path
    file_list = []
    k = start_num
    file_list = [a for a in os.listdir(dir_path)]
    for filename in file_list:
            rename = '%04d' %k
            print(dir_path+'/'+filename, "=>", str(rename)+'.jpg')
            copyfile(dir_path+'/'+filename, train_data_path+'/'  +str(rename)+'.jpg')
            k = k+1



## image aug for train
def img_aug(file_list_path):

    make_train_path = 'C:\\Users\\KCH\\Desktop\\new_originals'

    train_file_list = [a for a in os.listdir(file_list_path)]
    print('train file list', len(train_file_list))

    
    for i, file_name in enumerate(train_file_list):
        print("make train image", i)

        rd_image = cv2.imread(file_list_path+"/"+file_name)
        height, width, channel = rd_image.shape
        print("width : ",width, "height : ", height)
        
        crop_size = 128
        crop_width = crop_size
        crop_height = crop_size
        print("crop_size", crop_size)

        range_width = (int)(width/crop_width)
        range_height = (int)(height/crop_height)
        print("range width, height : ",range_width, range_height)

        j = 0
        for w in range(range_width):
            for h in range(range_height):
                box = (w*crop_height, h*crop_width, (w+1)*(crop_height), (h+1)*(crop_width))
                
                original_crop_image = rd_image[box[1]:box[3],box[0]:box[2]]

                cv2.imwrite(make_train_path+"/"+str(j)+"_"+file_name, original_crop_image)
                j=j+1 

## make test image
def test_add_noise(test_path, noise_level):
    noise_test_path = test_path +'/test'

    if not os.path.exists(noise_test_path):
        os.makedirs(noise_test_path)

    test_file_list = [a for a in os.listdir(test_path)]
    print(test_file_list)

    
    for i, file_name in enumerate(test_file_list):
        print("make test noise image", i)

        rd_image = cv2.imread(test_path+"/"+file_name)
        height, width, channel = rd_image.shape
        print("width : ",width, "height : ", height)
        
        mean = 0
        ##sigma = np.random.uniform(0,50)
        
        sigma = noise_level
        print('sigma,',sigma)
        noise = np.random.normal(mean, sigma, (height, width, channel))
        noise = noise.reshape(height,width,channel)
        noisy = rd_image + noise
        noisy = np.clip(noisy, 0, 255)
        noisy = noisy.astype(np.uint8)

        cv2.imwrite(noise_test_path+"/"+file_name, noisy)

## image aug for train
def test_img_aug(file_list_path, save_path):


    train_file_list = [a for a in os.listdir(file_list_path)]
    print('train file list', len(train_file_list))

    

    for i, file_name in enumerate(train_file_list):
        print("make image", i)

        rd_image = cv2.imread(file_list_path+"/"+file_name)
        height, width, channel = rd_image.shape
        print("width : ",width, "height : ", height)
        if height == 321:
            box = (0, 0, height-1, width-1)
            original_crop_image = rd_image[box[0]:box[2],box[1]:box[3]]
            cv2.imwrite(save_path+"/"+file_name, original_crop_image)

## random extraction
def random_extract(train_path,label_path,save_train_path,save_label_path,size):
    file_list = [a for a in os.listdir(train_path)]
    print("file list len", len(file_list))
    random.shuffle(file_list)
    save_file_list = file_list[0:size]
    for i, file_name in enumerate(save_file_list):
        print("num :",i)
        shutil.move(train_path+'/'+file_name, save_train_path+'/'+file_name)
        shutil.move(label_path+'/'+file_name, save_label_path+'/'+file_name)

## make test image
def random_noise_add(test_path, noise_level):
    noise_test_path = test_path +'/test'

    if not os.path.exists(noise_test_path):
        os.makedirs(noise_test_path)

    test_file_list = [a for a in os.listdir(test_path)]
    print(test_file_list)

    
    for i, file_name in enumerate(test_file_list):
        print("make test noise image", i)

        rd_image = cv2.imread(test_path+"/"+file_name)
        height, width, channel = rd_image.shape
        print("width : ",width, "height : ", height)
        
        mean = 0
        sigma = noise_level
        print('sigma,',sigma)
        noise = np.random.normal(mean, sigma, (height, width, channel))
        noise = noise.reshape(height,width,channel)
        print('noise',noise)
        noisy = rd_image + noise
        print('noisy',noisy)
        ##noisy = random_noise(rd_image, mode='gaussian', mean=0, var=sigma**2)
        noisy = np.clip(noisy, 0, 255)
        noisy = noisy.astype(np.uint8)
        print('rd_image',rd_image)
        print('noisy',noisy)
        psnr1 = wavelet.measure_psnr(rd_image,noisy)
        psnr2 = compare_psnr(rd_image,noisy,data_range=255)
        with tf.Session() as sess:
            rd_tensor = tf.image.convert_image_dtype(rd_image, tf.float32)
            noisy_tensor = tf.image.convert_image_dtype(noisy, tf.float32)
            psnr3 = tf.image.psnr(rd_tensor, noisy_tensor, max_val=1.0)
            print('noisy_tensor',noisy_tensor.eval())
            print('psnr1, psnr2, psnr3',psnr1,psnr2,psnr3.eval())
        cv2.imwrite(noise_test_path+"/"+file_name, noisy)



if __name__ == "__main__":

    ##random_extract("./adblock_dataset/train","./adblock_dataset/label","./adblock_dataset/train_test","./adblock_dataset/label_test",100)
    ##make_resize("./adblock_dataset/label_original", "./adblock_dataset/label", 256, 256)
    ##make_resize("./adblock_dataset/train_original", "./adblock_dataset/train", 256, 256)
    
    ##naming_img("./adblock_dataset/label_old", "./adblock_dataset/label", 0)

    
