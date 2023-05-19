import cv2
from glob import glob
import os 
import numpy as np

def r_slash(x) :
    x = x.replace('\\', '/', 10)
    return x

ori_size_path = './cbis-ddsm_total_jpg'
resized_path = './total_jpg_224'



for Each_phase in ['Test', 'Train'] :
    for Each_class in ['Benign', 'Malignant'] :

        Ori_path = ori_size_path + '/'+ Each_phase + '/' + Each_class 
        Trans_path = resized_path + '/' + Each_phase + '/' + Each_class 

        Ori_piced = glob(Ori_path + '/*') 
        Ori_piced = list(map(r_slash, Ori_piced)) 

        for pic_num in range(len(Ori_piced)) :
            each_pic_path = Ori_piced[pic_num]
            ori_pic = cv2.imread(each_pic_path, cv2.IMREAD_COLOR)
            shaped_pic = cv2.resize(ori_pic, dsize=(224,224), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(Trans_path+'/'+each_pic_path.split('/')[-1], shaped_pic)
           


