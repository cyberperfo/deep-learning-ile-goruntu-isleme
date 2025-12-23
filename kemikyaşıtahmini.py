#import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLronPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanAbsoluteError
#veri seti yükleme ve temizleme 
df=pd.read_csv("boneage-training-dataset.csv")
#klasörde görseli gerçekten var olan resimleri alalım 
#id eşletirme ve temizleme
image_folder= "boneage training-dataset"
available_images=set(os.listdir(image_folder))#ayıklama
available_ids=set(f.replace('.png','') for f in available_images if f.endswith('.png'))#png yazısını hiç bir şey ile değiştir 
#ilede numaralandırcaksın zaten 
df=df[df['id'].astype(str).isin(available_ids)].reset_index(drop=True)#eğer klasörde resim yoksa sil 
#kemik yaşını normalization 
df["bonage"]=df["bonage"]/240.0#normalizyon burda verileri önem sırasına göre sıralar 
#mesela regresyonda ilişki kurmamıza yardmcı oluyordu 
df["path"]=df["id"].apply(lambda x: os.path.join(image_folder,str(x)+".png"))#id initilaztion 
#dinamik dosya yolu ile az önceki hiçliği anlamlandırma tam adresleme idlendirme
print(df.head())
#ilk istatistik gösterme 
plt.hist(df["bonage"]*240,bins=30)
plt.xlabel("Bone Age (months)")
plt.ylabel("Frequency")
plt.title("Distribution of Bone Age")
plt.tight_layout()
plt.show()
#görüntüleri okuma ve ön işleme
def load_images(df,img_size=128):
    images=[]
    valid_indices=[]#sadece başarı ile okunan indisleri tutar
    for i,path in enumerate(df["path"]):#id forward 
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img =cv2.resize(img,(img_size,img_size))#her görüntünün aynı boyutta olması lazım
        img=img/255.0#regresyonumsu ilişki kurmanın ilk ayağı normalizasyon
        images.append(img)
        valid_indices.append(i)
    new_df=df.iloc[valid_indices].reset_index(drop=True)#başarısız resimleri çıkaracak şekilde günceller sadece valid indisleri sikle şeklinde
    return np.array(images).reshape(-1,img_size,img_size,1),new_df["bonage"].values
X,y=load_images(df)
#eğitim ve test setine ayırma
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#veri artırma
datagen=ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)
#model oluşturma
model=Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(128,128,1)),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128,(3,3),activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(1,activation='linear')
])
#model derleme
model.compile(optimizer=Adam(learning_rate=0.001),loss='mean_squared_error',metrics=[MeanAbsoluteError()])
model.summary()
#geri çağırmalar
early_stopping=EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
model_checkpoint=ModelCheckpoint('best_boneage_model.h5',monitor='val_loss',save_best_only=True)
reduce_lr=ReduceLronPlateau(monitor='val_loss',factor=0.5,patience=5,min_lr=1e-6)
#model eğitimi
history=model.fit(
    datagen.flow(X_train,y_train,batch_size=32),
    epochs=100,
    validation_data=(X_test,y_test),
    callbacks=[early_stopping,model_checkpoint,reduce_lr]
)
#model değerlendirme
plt.plot(history.history['loss'],label='Training Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

preds=model.predict(X_test)*240
y_test_rescaled=y_test*240
plt.figure()
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X_test[i].reshape(128,128),cmap='gray')
    plt.title(f'True: {y_test_rescaled[i]:.1f}\nPred: {preds[i][0]:.1f}')
    plt.axis('off')
plt.suptitle('Bone Age Predictions')
plt.tight_layout()
plt.show()


