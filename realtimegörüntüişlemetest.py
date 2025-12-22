#import liblaries
import cv2
import numpy as np
from tensorflow.keras.models import load_model
model=load_model('model.h5')
#kamerayı başlat
cap=cv2.VideoCapture(0)#kameramızın rgb olması bişey değiştirmiyor kendisi otamatik bgr moduna alıyor
print("Kamerayı başlatmak için 's' tuşuna basın.")
while True:
    success,frame=cap.read()
    if not success:
        break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#griye çeviriyoruz bgr dan nedeni malum siyah beyaz işlemler
    #ROİ(region of interest) belirleme
    """tüm ekranı işlemek yerine sadece roiyi işlersek hesaplama tasarufu sağlarız 
    """
    h,w=gray.shape
    box_size=200
    top_left_=(w/2-box_size/2,h/2-box_size/2)
    bottom_right_=(w/2+box_size/2,h/2+box_size/2)
    cv2.rectangle(frame,top_left_,bottom_right_,(0,255,0),2)
    #roi önişleme
    """kameradan gelen ham roi modele uygun değildir o yüzden proprcesing yapılır """
    roi=gray[int(top_left_[1]):int(bottom_right_[1]),int(top_left_[0]):int(bottom_right_[0])]
    roi=cv2.resize(roi,(28,28))#yeni boyutlandırma
    roi=roi.astype('float32')/255.0#normalizasyon
    roi=roi.reshape(1,28,28,1)#modelin beklediği şekle getirme
    #tahmin yap 
    prediction=model.predict(roi,verbose=0)#olasılıklal değerler(0.1,0.4,0.5 gibi)softmax
    """çoklu sınıflandırma yaparken olasılıksal durumdan belirtirken  sofmax kullanılır binary sınıflandırma yaparken sigmoid kullanılır
    """
    digit=np.argmax(prediction)#en yüksek olasılıklı değerin indexi
    #tahmini ekrana yazdır
    cv2.putText(frame,f'Tahmin:{digit}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.imshow('Real-time Digit Recognition',frame)
    key=cv2.waitKey(1)&0xFF
    if key==ord('s'):
        break
cap.release()
cv2.destroyAllWindows()