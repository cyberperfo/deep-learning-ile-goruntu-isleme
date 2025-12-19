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
from tensorflow.keras.optimizers import Adam #optimizer
#load mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
#image preprocessing
img=x_train[0]
stages={"original": img}
#histogram equalization:kontrast iyileştirme
hist_eq=cv2.equalizeHist(img)
stages["histogram equalization"]=hist_eq
#gaussian blur#gürültü azaltma
gauss=cv2.GaussianBlur(hist_eq,(5,5),0)
stages["gaussian blur"]=gauss
#canny edge detection#kenar bulma
canny=cv2.Canny(gauss,100,200)
stages["canny edge detection"]=canny
#veri görselleştirme
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
axes=axes.flat
for ax ,(tittle,im) in zip(axes,stages.items()):
    ax.imshow(im,cmap="gray")
    ax.set_title(tittle)
    ax.axis("off")
plt.suptitle("Image Processing Steps",fontsize=16)
plt.tight_layout()
plt.show()
#preprocessing for ANN model
"""histogram eşitleme
gaussian blur
canny edge detection
flattening
normalization
"""
def preprocess_data(x):#x =img
    img_eq=cv2.equalizeHist(x)
    img_blur=cv2.GaussianBlur(img_eq,(5,5),0)
    img_canny=cv2.Canny(img_blur,100,200)   
    feauture=img_canny.flatten()/255.0 #28x28=784# flattening and normalization flaten pixeller arasında ilişkiyi anlar 
    return feauture
num_train=8000 # 10000 -> 8000: CPU yükü için düşürüldü
num_test=1000  # 2000 -> 1000: Hız için düşürüldü
x_train_pp=np.array([preprocess_data(x) for x in x_train[:num_train]])#kütüphane işlemi hızlandırmak için numpy 
y_train_sub=y_train[:num_train]#alt küme ram kapasitesini aşmamak için
x_test_pp=np.array([preprocess_data(x) for x in x_test[:num_test]])#kütüphane işlemi hızlandırmak için numpy
y_test_sub=y_test[:num_test]#alt küme

#ann model creation
model=Sequential([#annler için pratiklik sağlayan kod 
    Dense(64,activation="relu",input_shape=(784,)),# 128 -> 64: İşlem yükü azaltıldı
    Dropout(0.3),# 0.5 -> 0.3: Daha dengeli bir sönümleme
    Dense(32,activation="relu"),# 64 -> 32: Katman küçültüldü
    Dense(10,activation="softmax")#çıkış katmanı 10 sınıf için
])
model.compile(optimizer=Adam(learning_rate=0.001),#compoiler ile esneklik sağlama
              loss="sparse_categorical_crossentropy",#kategorik çapraz entropi kayıp fonksiyonu
                metrics=["accuracy"])#doğruluk metriği
print(model.summary())
#ann model training
history=model.fit(
    x_train_pp,y_train_sub,#
    validation_data=(x_test_pp,y_test_sub),
    epochs=5, # 10 -> 5: Süre kısaltıldı
    batch_size=64, # 32 -> 64: İşlemci verimliliği artırıldı
    verbose=1
)
#evalaute model performance
test_loss,test_accuracy=model.evaluate(x_test_pp,y_test_sub,verbose=0)
print(f"Test Loss: {test_loss:.4f}")
#plot training history
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history["loss"],label="Train Loss")
plt.plot(history.history["val_loss"],label="Val Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history["accuracy"],label="Train Accuracy")
plt.plot(history.history["val_accuracy"],label="Val Accuracy")
plt.title("Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()