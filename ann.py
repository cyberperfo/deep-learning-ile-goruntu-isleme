"""Mnist veri seti 
0-9 toplam 10 sınıf içerir.
28x28 piksel boyutunda resimler 
grayscale (siyah beyaz) resimler 
60000 eğitim,10000 test verisi içerir.
amacimiz :ann ile bu resimleri tanımak yada sınıflandırmak
Image processing with ANN
histogram eşitleme :konstart iyileştirme 
gaussian blur :gürültü azaltma
canny edge detection :kenar bulma
ANN  ile mnist veri seti sınıflandırma

liblaries: tensorflow,keras,:ann modeli oluşturma ve eğitimi
opencv:cv2 ile image processing,,
 matplotlib veri görselleştirme
"""
#import libraries
import cv2 #open cv
import numpy as np # sayısal işlemler için 
import matplotlib.pyplot as plt #veri görselleştirme
from tensorflow.keras.datasets import mnist #mnist veri seti
from tensorflow.keras.models import Sequential #ann modeli oluşturma
from tensorflow.keras.layers import Dense, Dropout #ann katmanları
#load mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")

#image preprocessing
#ann model creation
#ann model training
#evalaute model performance
