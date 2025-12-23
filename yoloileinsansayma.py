#import libraries
import cv2
import numpy as np
from ultralytics import YOLO
#model tanımlama
model=YOLO('yolov8n.pt')#önceden eğitilmiş model
cap=cv2.VideoCapture(0)#kamerayı başlat
sucsess, frame = cap.read()
if not sucsess:
    exit("Kamera açılamadı")
frame=cv2.resize(frame,(800,600))
frame_height, frame_width = frame.shape[:2]
line_x=int(frame_width/2)#dikey çizgi x koordinatı
#7.sınıf matematiği x doğrsunu x ekseninden ikiye böl x=2 noktsaından geçen y eksenine paralel bir doğru gibi 
offset=10#çizgi kalınlığı
#sayaçlar
giren=0
çıkan=0
counted_ids=set()#sayılmış idleri tutmak için
person_last_x={}#her kişinin son x koordinatını tutmak için
while True:
    succsess, frame = cap.read()
    if not succsess:
        break
    frame=cv2.resize(frame,(800,600))#yeniden boyutlandırma
    #result içinde bir sürü obje olabilir ,yani birden fazla öğe track edilebilir
    results=model.track(frame,conf=0.4,iou=0.5,tracker='bytetrack.yaml',persist=True)
    if results[0].boxes.id is not None:
        ids= results[0].boxes.id.int().tolist()#tüm idleri almış olduk
        classes= results[0].boxes.cls.int().tolist()#tüm classları almış olduk
        xyxy=results[0].boxes.xyxy#kordinatlar
        for i, box in enumerate(xyxy):
            cls=classes[i]
            track_id=ids[i]
            class_name=model.names[cls]

            if class_name!='person':
                continue#sadece insanları al 
            #kordinatları bulunan insanların merkezini bulalım 
            x1,y1,x2,y2=map(int,box)
            cx=int((x1+x2)/2)
            cy=int((y1+y2)/2)
            previous_x=person_last_x.get(track_id,None)#garanti eşitleme durum güncellenirken bile previus last olacak yeni duruma refernas olmak için 
            person_last_x[track_id]=cx
            if previous_x is not None:#daha önce bu kişinin x koordinatı varsa
                if previous_x > line_x >= cx:
                    #bu adamın eski konumu çizginin sağındaysa yani değeri büyükse yeni konumu çizginin solunda yani değeri küçükse bu eleman konum değiştirmiştir
                    if track_id not in counted_ids:#çıkan sayımı 
                        çıkan += 1
                        counted_ids.add(track_id)
                    elif previous_x < line_x <= cx:
                        if track_id not in counted_ids:#giren sayımı
                            giren += 1
                            counted_ids.add(track_id)
            #kutuları çizdirme 
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.circle(frame,(cx,cy),5,(255,0,0),-1)
            cv2.putText(frame,f'ID:{track_id}',(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
                     
    #dikey ayrım çizgisi
    cv2.line(frame,(line_x,0),(line_x,frame_height),(0,255,255),2)
    #sayaçları ayarla 
    cv2.putText(frame,f'Giren: {giren}',(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.putText(frame,f'Çıkan: {çıkan}',(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    

    cv2.imshow("awm yolu takibi",frame)
    if cv2.waitKey(1) & 0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()

         
