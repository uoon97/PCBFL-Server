import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import ResNet101, ResNet152

import albumentations as A
from augmentation import get_bboxes_list, apply_aug
import yaml, zipfile
from PIL import Image
import PIL, json, sys
import shutil
from distutils.dir_util import copy_tree
# import imagesize
# import ptitprince as pt
from shutil import copyfile

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
import seaborn as sns
import re, time, random
import xml.etree.ElementTree as ET
import torch, cv2, glob
from tqdm.notebook import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix



class pcbDetection:
    
    """
    clf = central, lient1, client2 중 선택.
    input_img_dir = input으로 들어갈 이미지의 directory.
    clf_isLabel = input으로 들어갈 이미지의 label이 있을 경우 적는다.
    
    """
    
    
    def __init__(self, clf = None, yolo = None, input_img_dir = None, clf_isLabel = False):
        
        if clf == None and yolo == None:
            raise ValueError('model을 입력해주세요.')
                

        if clf != None:
            if clf == 'all':
                self.clf_cli = 'PCB_CLF'
                self.clf = load_model(f"{os.path.join(os.getcwd(), 'model_save_PCB_CLF/model_1.h5')}")

            if clf == 'client1':
                self.clf_cli = 'C1_PCB_CLF'
                self.clf = load_model(f"{os.path.join(os.getcwd(), 'model_save_C1_PCB_CLF/model_2.h5')}")

            if clf == 'client2':
                self.clf_cli = 'C2_PCB_CLF'
                self.clf = load_model(f"{os.path.join(os.getcwd(), 'model_save_C2_PCB_CLF/model_3.h5')}")
                
        if yolo != None:
            if yolo == 'all':
                self.yolo_cli = 'PCB'
                self.yolo_model = f"/home/work/KISTI_PCB2/yolov5/runs/train/{self.yolo_cli}_train_results/weights/best.pt"

            if yolo == 'client1':
                self.yolo_cli = 'C1_PCB'
                self.yolo_model = f"/home/work/KISTI_PCB2/yolov5/runs/train/{self.yolo_cli}_train_results/weights/best.pt"

            if yolo == 'client2':
                self.yolo_cli = 'C2_PCB'
                self.yolo_model = f"/home/work/KISTI_PCB2/yolov5/runs/train/{self.yolo_cli}_train_results/weights/best.pt"
                
            if yolo.startswith('federated'):
                self.yolo_cli = yolo
                num_round = self.yolo_cli.split('_')[-1]
                self.yolo_model = f"/home/work/KISTI_PCB2/yolov5/model{num_round}_Izo4FJHxZubxJ9DE.pt"
        
        self.img_dir = input_img_dir
        self.clf_isLabel = clf_isLabel
        self.cname = ['BADPCB', 'GOODPCB']
        self.yoloCheck()
        self.clfCheck()
    
    
    def yoloCheck(self):
        if self.img_dir != None:          
            self.yolo_type = 0
                
        else:
            self.yolo_type = 1 # 기존꺼 이용.
    
    def yoloResult(self):
        if self.yolo_type == 0:
            self.yoloDetect()
        
        if self.yolo_type == 1:
            self.yoloVal()
            self.yoloDetect()
                
    def yoloVal(self):
        if self.yolo_cli.startswith('federated'):
            num_round = self.yolo_cli.split('_')[-1]
            %run yolov5/val.py --data "/home/work/KISTI_PCB2/PCB/PCB.yaml" --weights "{self.yolo_model}" --task 'test' --name 'PCB_result_final(federated)_round{num_round}' --exist-ok
            
        else:
            %run yolov5/val.py --data "/home/work/KISTI_PCB2/{self.yolo_cli}/PCB.yaml" --weights "{self.yolo_model}" --task 'test' --name '{self.yolo_cli}_result_final(central)' --exist-ok
    
    
    # Detect만 clf와 이어서?
    def yoloDetect(self):
        if self.yolo_cli.startswith('federated'):
            num_round = self.yolo_cli.split('_')[-1]
            %run yolov5/detect.py --source "/home/work/KISTI_PCB2/PCB/test/images" --weights "/{self.yolo_model}" --exist-ok --line-thickness 2 --name 'PCB_detect_results(federated)_round{num_round}'
        
        else:
            %run yolov5/detect.py --source "/home/work/KISTI_PCB2/{self.yolo_cli}/test/images" --weights "{self.yolo_model}" --exist-ok --line-thickness 2 --name '{self.yolo_cli}_detect_results(central)'
        

    def clfCheck(self):
        if self.img_dir != None:          
            if self.clf_isLabel == False: 
                self.clf_type = 0
                self.img, _ = self.load_images_and_labels(self.img_dir)
            else: 
                self.clf_type = 1
                self.img, self.clf_label = self.load_images_and_labels(self.img_dir)
                
        else:
            if self.clf_isLabel == True:
                raise ValueError('input_img_dir를 입력해주세요.')
            else: 
                self.clf_type = 2
                self.img, self.clf_label = self.load_images_and_labels(f'./{self.clf_cli}/test')
    
    def clfResult(self):
                
        if self.clf_type == 1:
            print('평가 지표를 출력하며, 추론 결과를 return합니다.')
            self.printLossAcc()
            self.clfReport()
            self.cfmtx()
            self.showFail()
                
        
        if self.clf_type == 2:
            print('시스템 자체에서 정의한 테스트 데이터를 기반으로 결과를 출력하며, 평가 지표를 출력하며, 추론 결과를 return합니다.')
            self.printLossAcc()
            self.clfReport()
            self.cfmtx()
            self.showFail()
                
        return self.predict()
    
    def load_images_and_labels(self, directory):
    
        labels = []
        images = []
        
        for image_name in os.listdir(directory):

            image_path = os.path.join(directory, image_name)
            
            if 'ipynb_checkpoint' in image_path:
                continue
            
            image = tf.keras.preprocessing.image.load_img(image_path, target_size=(416, 416))
            image = tf.keras.preprocessing.image.img_to_array(image)
            images.append(image)
            
            # 파일 이름에 'GOODPCB'가 포함되어 있는지 확인
            if self.clf_type > 0:
                if 'GOODPCB' in image_name:
                    clf_isLabel = 1
                else:
                    clf_isLabel = 0
                labels.append(clf_isLabel)   

        images = np.array(images)
        labels = np.array(labels)
        
        return images, labels

    def predict(self):
        y_pred = self.clf.predict(self.img)
        y_pred = (y_pred > 0.5).astype(int)
        return y_pred

    def printLossAcc(self):
        print()
        print('loss, accuracy 출력')
        print(self.clf.evaluate(self.img, self.clf_label, verbose=2))
        
    def clfReport(self):
        print()
        print('classification Report 출력')
        y_pred = self.predict()
        y_pred = (y_pred > 0.5).astype(int)
        
        class_report = classification_report(self.clf_label, y_pred, target_names=self.cname)
        print(class_report)
        
    def cfmtx(self):
        # 혼동 행렬 생성
        print()
        print('confusion matrix 출력')
        y_pred = self.predict()
        confusion_mtx = confusion_matrix(self.clf_label, y_pred)

        # 혼동 행렬 시각화
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.cname, yticklabels=self.cname)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
    
    def showFail(self):
        print()
        print('예측 실패한 이미지 출력')
        y_pred = self.predict()        
        y_pred_classes = y_pred.flatten()
        incorrect_predictions = np.where(y_pred_classes != self.clf_label)[0]

        plt.figure(figsize=(12, 6))
        for i, idx in enumerate(incorrect_predictions[:10]):
            plt.subplot(2, 5, i + 1)
            plt.imshow(self.img[idx] / 255.0)
            true_label = self.cname[self.clf_label[idx]]
            predicted_label = self.cname[y_pred_classes[idx]]
            plt.title(f'True: {true_label}\nPredicted: {predicted_label}', color='red')
            plt.axis('off')

        plt.tight_layout()
        plt.show()
    
    def image_paths(self, directory):
        image_paths = []

        for image_name in os.listdir(directory):

            image_path = os.path.join(directory, image_name)

            if 'ipynb_checkpoint' in image_path:
                continue

            image_paths.append(image_path)

        return image_paths
    
    def save(self):
        badpcb_idx = np.where(self.predict() == 0)[0]
        
        if self.clf_type > 1:
            img_pths = self.image_paths(f'./{self.clf_cli}/test')
        else:
            img_pths = self.image_paths(self.img_dir) 
        
        shutil.rmtree('/home/work/KISTI_PCB2/result_clf_img/images/', ignore_errors = True)
        os.makedirs('/home/work/KISTI_PCB2/result_clf_img/images/')
        for i, pth in enumerate(img_pths):
            if i in badpcb_idx:
                image = tf.keras.preprocessing.image.load_img(pth)
                image = tf.keras.preprocessing.image.img_to_array(image)
                tf.keras.utils.save_img(f"/home/work/KISTI_PCB2/result_clf_img/images/{pth.split('/')[-1]}", image, file_format = 'png')
                
                
    def detect(self):
        
        shutil.rmtree('/home/work/KISTI_PCB2/result_clf_img/inference/', ignore_errors = True)
        os.makedirs('/home/work/KISTI_PCB2/result_clf_img/inference/')
        
        %run yolov5/detect.py --source "/home/work/KISTI_PCB2/result_clf_img/images" \
                --weights "{self.yolo_model}" \
                --exist-ok \
                --line-thickness 2 \
                --project '/home/work/KISTI_PCB2/result_clf_img/' \
                --name 'inference'