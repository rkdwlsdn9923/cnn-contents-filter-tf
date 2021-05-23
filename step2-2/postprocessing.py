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



##라벨 및 아웃풋의 그레이스케일화->threshold를 이용한 0,1화->resize 원래크기
##->input에 아웃풋에 있는 0의 위치를 찾아 0으로 블락킹한다.

##resize하기 전의 데이터인 label 및 아웃풋으로 acc 및 recall을 구한다.


def grayscale(input_image):
    
    dst = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    return dst

def segmentation(input_image, height, width, threshold):
    output_image = input_image
    for i in range(height):
        for j in range(width):
            if output_image[i][j] <= threshold :
                output_image[i][j] = 0
            else:
                output_image[i][j] = 1
    return output_image


def up_resize(input_image, height, width):
    
    resize_image = cv2.resize(input_image, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
    return resize_image

def blocking(input_image, height, width, block_image, color):
    output_image = input_image
    for i in range(height):
        for j in range(width):
            if block_image[i][j] == 0:
                output_image[i,j,] = color
    return output_image

if __name__ == "__main__":

    image_label = np.array(PIL.Image.open("./test_example/label.jpg")).astype('float32')
    image_output = np.array(PIL.Image.open("./test_example/output.jpg")).astype('float32')
    image_input = np.array(PIL.Image.open("./test_example/input.jpg")).astype('float32')
    print("이미지 크기 label, output, input", image_label.shape, image_output.shape, image_input.shape )
    label = image_label/255.
    output = image_output/255.
    input = image_input/255.
    input_height, input_width, input_channel = image_input.shape

    gray_label = grayscale(label)
    gray_output = grayscale(output)
    print("그레이 이미지 크기 label, output", gray_label.shape, gray_output.shape)
    im = PIL.Image.fromarray((gray_label*255).astype(np.uint8))
    im.save("./test_example/"+"gray_label.jpg")
    im2 = PIL.Image.fromarray((gray_output*255).astype(np.uint8))
    im2.save("./test_example/"+"gray_output.jpg")

    ##for precision, recall
    seg_label = segmentation(gray_label, 256,256,0.5)
    seg_output = segmentation(gray_output, 256,256,0.5)
    print(seg_output)
    im3 = PIL.Image.fromarray((seg_label*255).astype(np.uint8))
    im3.save("./test_example/"+"seg_label.jpg")
    im4 = PIL.Image.fromarray((seg_output*255).astype(np.uint8))
    im4.save("./test_example/"+"seg_output.jpg")

    ## for final output
    resize_output = up_resize(gray_output,input_height, input_width)
    im5 = PIL.Image.fromarray((resize_output*255).astype(np.uint8))
    im5.save("./test_example/"+"resize_output.jpg")

    seg_resize_output = segmentation(resize_output, input_height, input_width, 0.5)
    im6 = PIL.Image.fromarray((seg_resize_output*255).astype(np.uint8))
    im6.save("./test_example/"+"seg_resize_output.jpg")

    block_input = blocking(input, input_height, input_width, seg_resize_output, 1)
    im7 = PIL.Image.fromarray((block_input*255).astype(np.uint8))
    im7.save("./test_example/"+"white_block_input.jpg")
