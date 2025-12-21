"""
poz tahmini ve hareket sınıflandırma
mediapipe: genelde gömülü sistemlerde gerçek zamanlı görüntü işleme için kullanılan açık kaynaklı bir kütüphane
"""
#import gerekli kütüphaneler
import cv2
import mediapipe as mp
import numpy as np
#açı hesaplayan yardımcı fonksiyon
def calculate_angle(a, b, c):
    a = np.array(a)  # Birinci nokta(açılar daha kolay hesaplayabilmek için numpy arrayine dönüştürülür)
    #örnek: a[0]-c[1]  indekleme kolaylığı var diye
    b = np.array(b)  # Orta nokta
    c = np.array(c)  # Üçüncü nokta

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])#örnek 180/pi
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle
#mediapipe modülleri tanımla 
mp_drawing = mp.solutions.drawing_utils#çizim yardımcıları
#tespit edilen eklemler aralarındaki çizgileri çizer
#görsel şölen sunar 
mp_pose = mp.solutions.pose#poz modülü
#görünütüdeki kişilerin eklemlerini bulmasını sağlar 
#video dosyaası yükle
cap = cv2.VideoCapture('path_to_your_video.mp4')
#kural tabanlı poz tahmini gerçkeleştir
counter = 0
stage = None
def classify_pose(knee_angle):#açıya göre poz sınıflandırma
    """
    diz açısınına göre pozlama
    """
    if knee_angle<100:
        return "SQUAT"
    elif 100<=knee_angle<160:
        return "STAND"
    else:
        return "standing"
print(classify_pose(150))  # Örnek kullanım çıktı stand
    
#pose modulunu oluştur ,yaptıklarımızı birleştir 
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():#video açık olduğu sürece devam e
        ret, frame = cap.read()# ret görüntü okuma durumu, frame ise okunan görüntü
        if not ret:# görüntü okunamadıysa döngüden çık
            break

        # Görüntüyü BGR'den RGB'ye dönüştür
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False#gereksiz bellek kopyalamasını önle
        #şuan poz tahmini yapılacaağı için çok katmanlı işlem zaten yükü fazla bide bellekle uğraşmasın
        # Poz tahmini yap
        predicts = pose.process(image)
        #alt yapısı öznitelik çıkarımı ile ısı haritası çıkıyor olasılık dağılıma göre tahmin yapıyor 

        # Görüntüyü tekrar BGR'ye dönüştür
        image.flags.writeable = True# şimdi işlem uygun 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            # landmarks vücüt anahtar noktaları 
            landmarks = predicts.pose_landmarks.landmark

            # Diz açısını hesapla
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,#kalça noktaları
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,#diz noktaları
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,#ayak bileği noktaları
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            knee_angle = calculate_angle(hip, knee, ankle)#3 nokta arasındaki açı

            # Pozu sınıflandır
            pose_class = classify_pose(knee_angle)#açıya göre poz sınıflandırma
            #squad sayacı
            #durum takibi yapıyor ikinci kere if bloğu kullamasının sebi sadece squad olması yetmez
            if pose_class == "SQUAT" and stage == "STAND":
                stage = "SQUAT"
                counter += 1
                print(f'Squat sayısı: {counter}')

            # Ekrana bilgileri yazdır
            cv2.putText(image, f'Pose: {pose_class}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        except:
            pass#hatayı yakala ama bir sonraki frameye geç

        # Sonuçları göster
        mp_drawing.draw_landmarks(image, predicts.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Mediapipe Pose', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()

