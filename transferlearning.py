#import libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Densenet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#load data
#data augmentation(reçete yazma)
train_datagen = ImageDataGenerator(
    rescale=1./255,#normalizasyon#255 lik ölçekleri 0 ile 1 arasına indirerek gradyan iniş hızını artırır 
    horizontal_flip=True,#yatay çevirme #yöne bağımlımsızlık kazandırır ve ayırt ediclik 
    rotation_range=20,#döndürme#açı varyasyonu kazandırır
    brightness_range=[0.8,1.2],#parlaklık#farklı aydınlatma koşullarına karşı dayanıklılık
    validation_split=0.1#validation için ayırma#modelin genelleme yeteneğini değerlendirmek için overfiti önler
)
test_datagen = ImageDataGenerator(rescale=1./255)#test için sadece normalizasyon
DATA_DIR = 'chest_xray'  
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_MODE='binary'#sınıflandırma ve 0 ile 1 arası matemtiksel formatlama ör:hedef var yada yok
#eczaneden alma (preprocesing ve generator) 
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),#eğitim verisi
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE,
    subset='training',#eğitim için ayırma amaç valid ile traning ayrımı
    shuffle=True
)
val_gen=train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),#validation için train dizininden ayırma
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE,
    subset='validation',#validation için ayırma
    shuffle=False
)
#test generator de augmetion yok dikkat sadece normalizasyon var
test_gen=test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE,
    shuffle=False
)
#basic visulization
class_names = list(train_generator.class_indices.keys())
images, labels = next(train_generator)
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i])
    plt.set_title(class_names[int(labels[i])])
    plt.axis("off")
plt.tight_layout()    
plt.show()
#transfer learning model tanımlaması 
base_model = Densenet121(
    weights='imagenet',#önceden eğitilmiş ağırlıklar 
    include_top=False,#üst katmanları dahil etme
    input_shape=(*IMG_SIZE, 3)#girdi şekli
    )
base_model.trainable = False#önceden eğitilmiş katmanları dondurma yani base_model train edilmeyecek
x = base_model.output#özellik çıkarımı
x= GlobalAveragePooling2D()(x)#küresel ortalama havuzlama(önceden eğitilmiş özellik haritalarını özetleme)klasik flatten işlemine alternatif daha az paramtre kullanma
x= Dense(128, activation='relu')(x)#yoğun katman ekleme
x= Dropout(0.5)(x)#dropout katmanı ekleme
predictions = Dense(1, activation='sigmoid')(x)#çıktı katmanı ekleme(softmaxin özelleşmiş hali softmax 0.15 ile 0.85 arası çıktı verirken sigmoid 0 ile 1 arası çıktı verirken sigmoid sadece 0.15 yada 0.75 tir der)
model = Model(inputs=base_model.input, outputs=predictions)#model oluşturma
#modelin derlenmesi ve callback 
model.compile(optimizer=Adam(learning_rate=0.0001),#düşük öğrenme hızı
              loss='binary_crossentropy',#ikili çapraz entropi kayıp fonksiyonu
              metrics=['accuracy'])#doğruluk metriği
print(model.summary())
#callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),#erken durdurma
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, min_lr=1e-6),#öğrenme oranını azaltma
    ModelCheckpoint("best_transfer_model.h5", save_best_only=True)#model kaydetme
]
# modelin eğitilmesi ve sonuçların değerlendirilmesi
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_gen,
    callbacks=callbacks
)
pred_probs = model.predict(test_gen,verbose=1)#tahmin olasılıkları
pred_labels = (pred_probs > 0.5).astype("int32").flatten()#tahmin etiketleri
true_labels = test_gen.classes##doğru etiketler
#model performansının görselleştirilmesi
cm = confusion_matrix(true_labels, pred_labels)#doğru ve tahmin etiketleri ile karışıklık matrisi
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names) 
plt.figure(figsize=(8, 8))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()


