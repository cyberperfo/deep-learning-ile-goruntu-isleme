"""
flowers dataset:
rgb 224x224
CNN ile sınıflandırma modeli oluşturma ve problemi çözme
"""
# Hataları önlemek için sistem ayarları
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

#import libraries
import tensorflow as tf
import tensorflow_datasets as tfds # Hatayı düzelten temel kütüphane
AUTOTUNE = tf.data.AUTOTUNE #veri seti optimize etmek için
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D, 
    Flatten, 
    Dense, 
    Dropout)
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

#veri seti yükleme
# tensorflow.datasets.load yerine tfds.load kullanıldı
(ds_train,ds_val),ds_info=tfds.load(
    'tf_flowers',
    split=['train[:80%]','train[80%:]'],#veri seti yüzde 80 eğitim, yüzde 20 doğrulama
    with_info=True,#veri seti bilgisi
    as_supervised=True,# etiketli veri seti
)

#veri seti görselleştirme 
fig=plt.figure(figsize=(10,10))
for i,(image,label) in enumerate(ds_train.take(9)):
    ax=fig.add_subplot(3,3,i+1)
    ax.imshow(image)
    # ds_info üzerinden etiket ismine erişim
    ax.set_title(ds_info.features['label'].int2str(label))
    ax.axis('off')

plt.tight_layout()
plt.show()     
IMG_SIZE=(180,180)
#data augmentation+processing
def preprocess(image,label):
    """
    resize ,random flip, brightness,contrast,normalization
    """
    image=tf.image.resize(image,IMG_SIZE)#boyutlandırma
    image=tf.image.random_flip_left_right(image)#yatay olarak rastgele çevirme
    image=tf.image.random_brightness(image,0.2)#rastgele parlaklık
    image=tf.image.random_contrast(image,0.8,1.2)#rastgele kontrast
    image=tf.image.random_crop(image,size=(160,160,3))#rastgele crop 
    image=tf.image.resize(image,IMG_SIZE)#tekrar boyutlandırma
    image=image.cast(image,tf.float32)/255.0 #normalization
    return image,label
#veri seti hazırlama
ds_train=(
    ds_train
    .map(preprocess,num_parallel_calls=AUTOTUNE)#ön işleme ve augmentation
    .shuffle(1000)#karıştırma
    .batch(32)#batch boyutu
    .prefetch(AUTOTUNE)#veri setini önceden hazırlama 
          )
ds_val=(
    ds_val
    .map(preprocess,num_parallel_calls=AUTOTUNE)#ön işleme ve augmentation
    .batch(32)#batch boyutu
    .prefetch(AUTOTUNE)#veri setini önceden hazırlama
          )
#model oluşturm
model=Sequential([
        Conv2D(32,(3,3),activation='relu',input_shape=(180,180,3)),#aktivasyon karmaşıklığı çözme
        MaxPooling2D((2,2)),

        Conv2D(64,(3,3),activation='relu'),
        MaxPooling2D((2,2)),#boyut azaltma hesap verimliliği sağlama
        Conv2D(128,(3,3),activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128,activation='relu'),
        Dropout(0.5),#overfiti engelemme
        Dense(5,activation='softmax')#çıkış katmanı ve softmax: olasılık dağılım ile etiketleme
    ])

#callbacks
callbacks=[
    #eğer val loss 5 epoch boyunca iyileşmezse eğitimi durdur ve en iyi ağırlıkları geri yükle
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,restore_best_weights=True),#erken durdurma
    #val loss 2 epoch boyunca iyileşmezse öğrenme oranını azalt ve daha detaylı öğrenmesini sağla
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2,verbose=1,min_lr=1e-6),
    #her epoch sonunda model daha iyi ise kaydolur
    tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True),#model kaydetme ,en iyi model için kaydetme 
]
#derleme
model.compile(  optimizer = Adam(learning_rate=0.001),
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
print(model.summary())
#training
history=model.fit(#öğrenme verisi görseştirmek için historyide saklanır accuacry ıou dice cofficent gibi metrikler için epohtaki değerleri tutarsın 
    #hiper parametre optimizasoyn için öğrenme hızı veya katman sayısı değiştiğinde hangi aayarın en iyi sonuç veridğini anlamak için hsitory ile kıyalama yaparsın 
    ds_train,
    validation_data=ds_val,
    epochs=2,
    callbacks=callbacks,
    verbose=1
)
#değerlendirme
#eğitim geçmişi görselleştirme
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'],label='Eğitim Kaybı')
plt.plot(history.history['val_loss'],label='Doğrulama Kaybı')
plt.xlabel('Epochs')
plt.ylabel('Kaybı')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'],label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'],label='Doğrulama Doğruluğu')
plt.xlabel('Epochs')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()
