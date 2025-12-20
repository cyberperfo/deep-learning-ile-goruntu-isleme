"""
u net ile segmentasyon kutu şeklinde değil nesnenin etrafını saran bir maske ile yapılır.
u net ile segmentasyon yapmak için öncelikle veri seti maskelenmiş olmalıdır
"""
#gerekli kütüphaneler
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
def load_dataset(root,img_size=(128, 128)):
    images = []
    masks = []
    for tile in sorted(os.listdir(root)):#tile parça dosyalardır buda her bir tile dosyasını sıraysıyla dolaştırır
        #hiyeraşik bir şekilde sistemi yormamak adına tile dosyasın aayırıyoruz 
        img_dir=os.path.join(root, tile,"images")#resim dosyalarının olduğu klasör(hangi kutuya bakacağını tanımlıyorsun)igili rafa git
        mask_dir=os.path.join(root, tile,"masks")#mask dosyalarının olduğu klasör
        if not os.path.isdir(img_dir):continue#klasör yoksa atla
        for filename in os.listdir(img_dir):#görüntü dosyalarını dolaştır
            if filename.endswith('.jpg') or filename.endswith('.png'):#sadece jpg veya png dosyalarını al
                img_path = os.path.join(img_dir, filename)#görüntü dosyasının tam yolu(hangi spesifik dosyayı okuyucağını tanımlıyorsun)raftaki kitaba bak
                mask_path = os.path.join(mask_dir, filename.split('.')[0] + '_mask.png')#mask dosyasının tam yolu
                if not os.path.exists(mask_path): continue#mask dosyası yoksa atla
                img=cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)#görüntüyü oku ve renk formatını değiştir
                #open cv genelde bgr formatında okur bunu rgb ye çeviriyoruz
                img=cv2.resize(img,img_size)/255.0#görüntüyü yeniden boyutlandır ve normalize et
                #veri setindeki tüm verileri aynı boyut formatına sokuyoruz ör elma ile armudu
                #aynı ölçekte kıyaslayabilsin 
                #format 0 ile 250 arasında 0 il1 1 e çekip gradyan inişi hızlandıtıp koyaylştırıyoruz
                mask=cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)#maskeyi gri tonlamalı olarak oku
                #amaç hedef ile arka planı ayırt etmektir o yüzden gri tonlamalı okuruz 
                #rgb 3 boyutlu matriks gibi,gri tonlama tek katman buda bellek kullanımı %66 azaltır
                mask=cv2.resize(mask,img_size)#maskeyi yeniden boyutlandır
                mask=np.expand_dims(mask,axis=-1)/255.0#maskeyi normalize et ve boyutunu genişlet
                #boyut genişletme tekrardan aynı formata sokma özellikle loss fonskiyonunda tek katman değilde 3 katman olması şartır
                images.append(img)#görüntüyü listeye ekle
                masks.append(mask)#maskeyi listeye ekle
    return np.array(images),np.array(masks)#görüntü ve maskeleri numpy dizisi olarak döndür
X,y=load_dataset("dataset_path")#veri setini yükle
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=42)#veri setini eğitim ve doğrulama olarak ayır 
print("toplam train örneği:",len(X_train))#eğitim örnek sayısını yazdır
print("toplam val örneği:",len(X_val))#doğrulama örnek sayısını yazdır
def unet_model(input_size=(128,128,3)):
    inputs = keras.Input(input_size)
    # Encoder özellik çıkarımı
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck (en derin seviye)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])#skip connection(özelliklerin korunmasını sağlama bilgi köprüsü)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)#çıkış katmanı sigmoid aktivasyon ile hedefden eminliğin olasılığı 
    #softmax mesela %90 uçak %5 iha %lazer der ama sigmoid sadece %90 uçak der güvenlik skoru işte 
    return keras.Model(inputs, outputs)
model=unet_model()#u net modelini oluştur
unet_model.compile(optimizer='adam',loss='binary_crossentropy')#modeli derle
callbacks=[
    keras.callbacks.ModelCheckpoint("unet_best_model.h5", save_best_only=True),#en iyi modeli kaydet
    keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True),#erken durdurma
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=2,verbose=1,min_lr=1e-6)#öğrenme oranını azalt daha detaylı örğenme 
]

history=model.fit(  #modeli eğit
    X_train,
    y_train,
    validation_data=(X_val,y_val),
    epochs=50,
    batch_size=16,
    callbacks=callbacks
)
#modeli değerlendir
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.legend()
plt.show()
def show_prediction(idx=0):
    img=X_val[idx]
    #doğru maske cevap anahtarı 
    mask_true=y_val[idx].squeeze()#gerçek maskeyi al ve kanalı sıkıştır(mask_true=grand truth)
    #pred mask öğrencinin cevabı ikiside birbibne benziyorsa model doğru segmente ediyor demektir 
    pred_raw=unet_model.predict(img[np.newaxis,...])#model tahmini al ve kanalı sıkıştır 
    #tahmin genelde yüksek formatta çıkar (1,1,1,1) örnek görselleştirmek için boyut küçültüyüruz 
    mask_pred=(pred_raw.squeeze()>0.5).astype('float32')#eşikleme yaparak ikili maske oluştur
    #eşikleme yapmamızın sebebi mühimmatın gri karar ihtiyacı yoktur ya vurucaktrı(1) ya da vurmayacaktır(0)
    #işlem >0.5 diyerek ise beyaz yap değilse siyah yap diyoruz
    #float 32 derin öğrenme modelleri gpu üzerinde genelde float32 ile çalışır
    #bellekte daha az yer kaplar ve işlem hızını artırır
    #sonuçları görselleştir
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title("Girdi Görüntüsü")
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(mask_true,cmap='gray')
    plt.title("Gerçek Maske")
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(mask_pred,cmap='gray')
    plt.title("Tahmin Edilen Maske")
    plt.axis('off')
    plt.show()
show_prediction(idx=0)#0. indeksteki görüntü için tahmin göster    