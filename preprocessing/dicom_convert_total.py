

import os 
import numpy as np
import pydicom
import cv2
from tqdm import tqdm


dicom_folder = './cbis-ddsm_total_dicom'
jpg_folder = './cbis-ddsm_total_jpg'

Ori_set = ['Train', 'Test']
Label_set = ['Malignant', 'Benign']

for set_num in range(len(Ori_set)) :  # Train, Test 중 하나 선택
    i = 0 
    for label_num in range(len(Label_set)) :

        dicom_set_path = dicom_folder + '/' + Ori_set[set_num] + '/' + Label_set[label_num] 
        dicom_set_list = os.listdir(dicom_set_path)
        dicom_img_list = [file for file in dicom_set_list if file.endswith('.dcm')]

        for dcm_num in tqdm(range(len(dicom_img_list))) :

            dcm_file = dicom_set_path + '/' + dicom_img_list[dcm_num]
            print(dcm_file)
            raw_dcm = pydicom.read_file(dcm_file)
            dcm_img = np.array(raw_dcm.pixel_array)

            dcm_jpgimg = dcm_img * (255/65535)

            #print(max(dcm_jpgimg.flatten()), min(dcm_jpgimg.flatten()))
            
            img_name = dicom_img_list[dcm_num][:-4] + '.jpg'
            
            save_path = jpg_folder + '/' + Ori_set[set_num] + '/' + Label_set[label_num]
            
            
            
            
            cv2.imwrite(save_path + '/' + img_name, dcm_jpgimg)
            dcm_jpgimg = cv2.imread(save_path + '/' + img_name)
            cv2.imwrite(save_path + '/' + img_name, dcm_jpgimg)
            
            
            
            print(i)
            i+=1


        