"""Yüz ifadesinden duygu tanıma modülü
"""
#import library
import cv2
import mediapipe as mp
import numpy as np
#yardımcı fonksiyonların oluşturulması
#mediapipeline ile face mesh modülü başlat
mp_face_mesh = mp.solutions.face_mesh
face_mesh=mp_face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,
                                refine_landmarks=True,
                                 min_detection_confidence=0.5,min_tracking_confidence=0.5)
#open cv ile video yakalama
cap = cv2.VideoCapture(0)#0 indenks ilk kamera
#yuz messhinden kullanılacak önemli landmark indeklerini al
#insan yüzü zaten simetrik olduğundan tek taraflı noktalar yeterli
#bu şekilde cpuda hesap verimliliği sağlıyoruz
LEFT_EYE =[159,145]
MOUTH =[13,14]#ağzın ortası
MOUTH_LEFT_RİGHT =[61,291]#dudağın solu ve sağı 
LEFT_BROW =[65,52]
#göz açıklığı ve ağız açıklığını metriklerine göre duygu belirle
def detect_emotion(landmarks,image_width,image_height):
    def get_point(index):#landmark noktasını al
        lm=landmarks[index]
        return np.array([int(lm.x*image_width),int(lm.y*image_height)])#ters normalizasyon ve np.array formatı
    """vektörel matematik hız ve bellek ve gelişmiş fonskiyonlardan dolayı nump array  
    ayrıca genelde metrikler mediapipelinede normalize edilmiş şekilde verildiğinden burda ters normalizasyon yapıyoruz"""
    #kas ve göz noktaları 
    brow_point = get_point(65)
    eye_point = get_point(159)
    brow_lift=np.linalg.norm(brow_point - eye_point)#kaş ve göz arasındaki mesafe norm pisagorsal toplama kök
    #kaç kalıklığı 
    #dudak sol ve sağ noktaları
    mouth_left = get_point(13)
    mouth_right = get_point(14)
    mouth_width=np.linalg.norm(mouth_left - mouth_right)

    if brow_lift>15:
        return "Surprised"#şaşırmış
    elif mouth_width>40:
        return "Happy"#mutlu
    else:
        return "Neutral"#nötr
#webcam üzerinden duygu tanıma
while True:#kamera açık kaldığı sürece devam et
    success, image = cap.read()
    if not success:
        break
    rgb_frame=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    results=face_mesh.process(rgb_frame)#prediction

    #ekran boyutları 
    h,w,_=image.shape
    if results.multi_face_landmarks:#görüntüde yüz tespit edildiyse
        for face_landmarks in results.multi_face_landmarks:#eğer kameranın önüdne 3 kişi varsa 3 kez çalışır 1 kez varsa bir keç çalışııe
            
            #duyguyu tespit et
            emotion = detect_emotion(face_landmarks.landmark,w,h)
            #ekrana yazdır
            cv2.putText(image, f'Emotion: {emotion}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            #yüz mesh noktalarını ekrana yazdır
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,#noktların birbirine nasıl bağlanıcağını berlileyen 
                #iskelet haritasıdır 
                landmark_drawing_spec=None,#noktalar üçerinde tek tek daire çizmesini engeller
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=1)#çizgi rengi ve kalınlığı
                )
                #thickness çizgi kalınlığı
    #ekranı göster
                
    cv2.imshow('Emotion Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
