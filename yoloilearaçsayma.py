"""
yolo tracking adlı ilk dosyada amacımız trackingi öğrenmekti yüzeyseldi burda bir şart ekleyip istatistik çıkarıyoruz
"""
#import libraries
import cv2
import numpy as np
from ultralytics import YOLO
#yardımcı fonksiyonlar
def get_line_side(x,y,line_start,line_end):
    """Noktanın çizginin hangi tarafında olduğunu belirler
    Pozitif değer bir tarafı negatif değer diğer tarafı gösterir
    """ 
    return (line_end[0]-line_start[0])*(y-line_start[1])-(line_end[1]-line_start[1])*(x-line_start[0])
#modeli yükle
model=YOLO('yolov8n-counting.pt')#önceden eğitilmiş model
#video yakalama
cap=cv2.VideoCapture(0)#kamerayı başlat

#çizgi tanımlama
success, frame = cap.read()
"""
ilk başlangıçta frame ve success tanımlamanın sebebi ilk anda görüntü yakalancak ki preprocesing yapılabilsin görüntü reshepa yapılailsin 
kanal çıkartılabilsin o kadar çizgi çizcen yani"""
if not success:
    exit("Kamera açılamadı")

frame=cv2.resize(frame,(800,600))#farklı kameralardan farklı görüntü gelebilir o yüzden stabil olacak şekilde görünütyü standardize ediyoruz 
#ayrıca görüntüyü küçülterek fps artırır bellek verimliliği artar 
frame_height, frame_width = frame.shape[:2]#çizgi çizdirirken renge gerke olmaz o yüzden kanalı çıkarıyoruz height ve widght alıyoruz
#çapraz çizgi tanımlama
line_start = (int(frame_width * 0.1), int(frame_height * 0.5))
line_end = (int(frame_width * 0.9), int(frame_height * 0.5))
#obje sayacı
counts={'car':0,'bus':0,'truck':0,'motorcycle':0,'bicycle':0}
count_ids=set()#sayılmış idleri tutmak için
object_id_count=0#her obje için benzersiz id atamak için
object_last_side={}#her objenin son tarafını tutmak için
#yolo ile araç sayımı 
while True:
    """
    süreklilik aşamasında frame ve succes yapman zaten elzem olan birşey """
    success, frame = cap.read()
    if not success:
        exit("video açılamadı")
    frame=cv2.resize(frame,(800,600))
    #tracking ile araç takibi 
    results=model.track(frame,conf=0.4,iou=0.5,tracker='bytetrack.yaml',persist=True)
    # eğer takip edilen nesne varsa
    if results[0].boxes.id is not None:
       ids=results[0].boxes.id.int().tolist()#tüm idleri almış olduk
       classes=results[0].boxes.cls.int().tolist()#tüm classları almış olduk
       xyxy=results[0].boxes.xyxy#kordinatlar 

       for i ,box in enumerate(xyxy):#o ana kadarki tespit edilen her kutuyu işleme alıyoruz enumarate ile veri indeksleme 
           cls_id=classes[i]
           track_id=ids[i]
           """
           clas id ve track id ile kimlik eşleştrimesi yapıyoruz clas id türünü track id eşsizliğini temsil eder"""
           #sadece belirli sınıfları say
           class_name=model.names[cls_id]#sayıdan kelimeye dönüşüm örnek print(class_name) diyince sayı yerine isim çıksın 
           if class_name not in counts:#sadece belirli sınıfları say
                continue
           x1,y1,x2,y2=map(int,box)
           cx=int((x1+x2)/2)#çizgiyi geçip geçmeme hesabını koca kutu ile hesap etmek zordur en garanti ve kolay yöntem merkez seçilir
           cy=int((y1+y2)/2)

           current_side=get_line_side(cx,cy,line_start,line_end)
           previous_side=object_last_side.get(track_id,None)#eğer öncesi yoksa es geçer öncesi varsa sonrakine refernas sağlar
           object_last_side[track_id]=current_side
           if previous_side is not None and previous_side != current_side:#eğer araç ilk kez geçiyorsa ve previous ile current farklıysa
               #kesinlikle taraf değişimi olmuştur çünkü ikisinden bir aralarında değişsede daima bir poztif bir negatif değeri temsil eder
               if track_id not in count_ids:#tekrakere aynı aracı saymayı engellemek için görüntü hataları durumunda bile hata yapmamak için
                   #ön kontrol yapıyoruz
                   counts[class_name]+=1
                   count_ids.add(track_id)
            #kutu çizimi ve etiketleme
           cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
           cv2.putText(frame,f"{class_name} ID:{track_id}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
           cv2.circle(frame,(cx,cy),5,(255,0,0),-1)    
    #çapraz çizgi çizimi
    cv2.line(frame,line_start,line_end,(0,0,255),2)
    #matematik ve görleştirme zamanları farklıdır line star ve line end dediğimiz ilk safhada zaten matematiksel olarak line yi soyut bir şekilde ayarlamıştık trackinge engel değil 
    #şimdi burda görselleştirmek daha doğru 
    #sayaçları göster
    y_offset=30
    """
    zaten daha önce syacın matemetiği yapıldı mantık sırası olarak burda şimdi görsellşetiriliyor"""
    for cls,count in counts.items():
        text=f"{cls}: {count}"
        cv2.putText(frame,text,(10,y_offset),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        y_offset+=30
    cv2.imshow("araç takip ve sayim",frame)
    if cv2.waitKey(1) & 0xFF==ord('s'):
        break

cap.release()
cv2.destroyAllWindows()