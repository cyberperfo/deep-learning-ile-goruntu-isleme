"""
araçların takibi 
örnek video :https/video
veri seti : https:kaggle/data
1. Veri Seti (Dataset): Modelin Hafızası ve Bilgi Kaynağı
Tracking veri seti, modelin bir nesnenin sadece ne olduğunu değil, o nesnenin zaman içindeki hareket karakteristiğini öğrenmesini sağlar.

Yapısı: Veri seti, bir videodaki her karede (frame) nesnenin nerede olduğunu gösteren etiketlere (ID ve Bbox) sahiptir. Örneğin; 1. karede "ID:0" olan İHA, 50. karede de "ID:0" olarak işaretlenir.

Amacı: Modelin, nesne bir anlığına ağacın arkasına girdiğinde (occlusion) veya kamera sallandığında o nesneyi "başka bir nesne sanmamasını" (ID Switch engelleme) öğretmektir.

Kapsamı: Binlerce farklı hava durumu, ışık açısı ve farklı İHA modellerini içerir.
2. Örnek Video (Inference): Modelin Canlı Sınavı
Örnek video, eğitilmiş bir modelin (YOLO + Tracker) gerçek zamanlı olarak çalıştırıldığı, üzerinde etiket olmayan ham görüntüdür.

Yapısı: İçinde hiçbir koordinat veya kimlik bilgisi yoktur. Model her kareyi o an işler.

Amacı: Modelin öğrendiği "ID takibi" yeteneğini pratikte sergilemesidir. Model burada kendi tahminlerini yapar; "Bence bu nesne 120 numaralı hedeftir ve şu yöne gidiyor" der.

Sonuç: Ekranda nesnenin peşinden giden bir kutu ve üzerinde "UAV #1" gibi bir yazı görürsün. Bu, modelin başarısının görsel kanıtıdır.
"""
from unittest import result
from ultralytics import YOLO
import cv2
#veri seti yükleme komutları
#bla bla bla
#yolo modelini yükle
model=YOLO('yolov8n.pt')#küçük model
#video yolu
video_path='data/videos/uav_tracking.mp4'
cap=cv2.VideoCapture(video_path)#video yakalama:video kaynağı ile bağlantı kurma ve frame frame veri okumasını sağlayan yapı 
#output video ayarları
widhth=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps=int(cap.get(cv2.CAP_PROP_FPS))
out=cv2.VideoWriter('output/tracked_video.mp4',cv2.VideoWriter_fourcc(*'mp4v'),fps,(widhth,height))#video yazma objesi
#frameleri birleştirerek video objesi oluşturur
while cap.isOpened():#video açık olduğu sürece
    sucess,frame=cap.read()#frame oku
    if not sucess: #frame okunamadıysa döngüden çık
        break
    #yolo ile tracking yap
    result=model(
        frame,
        persist=True, #takip idlerinin aynı nesne içinde devam etmesini sağlar
        conf=0.5, #güven eşiği
        iou=0.3, #iou eşiği iki farklı kutunun örtüşerek hedef tespit etmesi skoru kümelerin kesişimi /birleşimi gibi gibi
        tracker='bytetrack.yaml', #tracker yapılandırma dosyası
        classes=[0] #sınıf filtresi sadece insanları takip et yada araba herneyse artık
    )
    cv2.imshow('YOLOv8 Video Tracking',frame)#görüntüyü ekranda göster
    out.write(frame)#görüntyü output olarak kaydet 
    if cv2.waitKey(1)&0xFF==27:#esc tuşu ile çıkış
        break
cap.release()#video kaynağını serbest bırak
out.release()#output video kaynağını serbest bırak
cv2.destroyAllWindows()#tüm OpenCV pencerelerini kapat