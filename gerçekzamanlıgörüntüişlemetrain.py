"""
problem tanımı gerçek zamanlı görüntü işleme ile rakamları sınıflandırmak 
kamera ile kağıtlara yazmış olduğumuz rakamları sınıflandırmaya çalışma 
"""
#import libraries
import tensorflow as tf
from tensorflow.keras import layers, models, make_ndarray
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
#veriyi yükle 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#görselleri ters çevir(netafini al)
"""gerçek hayatta beyaz kağıt üzerine siyah kalemle yazdığımız için görselleri ters çevirmemiz gerekiyor daha doğal ve incenelebilir olması için 
"""
x_train = 255 - x_train
x_test = 255 - x_test
#görüntüyü görselleştir
plt.figure(figsize=(10,10))
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')

plt.tight_layout()#otomatik yerleşim yerlerini düzelten estetik cerrah
plt.show()
#normalizasyon ve reshape
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
#-1 otoamatik mesela 60000 ne 10000 diye ayarla der 28,28 batch boyutu 1 de rgb renk formatı sadece 1 tane 
#eksen sırası önemliye veri tutarlığı kontrol ediliyorsa manuel girmek daha önemli
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0   
#data augmentation
deatagen=ImageDataGenerator(
    rotation_range=10,#rastgele 10 derece döndür döndürme duyarlılığı
    widght_shift_range=0.1,#genişliği %10 sağa sola kaydırma
    height_shift_range=0.1,#yüksekliği %10 yukarı aşağı kaydırma
    zoom_range=0.1#%10 oranında yakınlaştırıp uzaklaştırma
    )
#model oluşturma
model= models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='softmax')
])
#modeli derleme
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#modeli eğitme
batch_size=64
model.fit(deatagen.flow(x_train,y_train,batch_size=batch_size),#flow komutu her epochta augmentasyon ürünlerini rastgeleleştirir
          epochs=10,
          validation_data=(x_test,y_test)#validasyon datası augmentasyon uygulanmaz
          )
          #sınav sorularının değiştirmemek gerekir ve istatiksel tutarlılık sağlanır 
#modeli kaydetme
model.save('mnist_model.h5')#performans ölçümü cartcurt işte 
print("Model kaydedildi.")

