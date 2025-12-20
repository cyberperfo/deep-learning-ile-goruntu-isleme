"""
2. Validation/Test (Doğrulama/Test) Aşaması: "Sınav ve Denetim"
Eğitim bittikten sonra (veya her epoch sonunda), modelin hiç görmediği resimlerle performansı ölçülür.

Görmediği Veri: Modele, eğitim setinde yer almayan "Validation" resimleri verilir. Bu, mühimmatın daha önce hiç gitmediği bir arazide hedefi tanıyıp tanıyamayacağını ölçmek gibidir.

Metrik Ölçümü: Modelin başarısı sadece "doğru/yanlış" diye değil, senin de tabloda gördüğün mAP (Mean Average Precision), Precision ve Recall değerleriyle ölçülür.

Fitne Kontrolü (Overfitting Check): Eğer model eğitim setinde çok başarılı ama test setinde çok başarısızsa, "ezberlemiş" (overfit olmuş) demektir. Bu durumda eğitim durdurulur veya parametreler değiştirilir.
"""
import cv2
from ultralytics import YOLO
model=YOLO('yolov8n.pt')#küçük model
#test edilecek görselin yüklenmesi
image_path='data/images/bus.jpg'
image=cv2.imread(image_path)#görseli yükle
#image tahmini
result=model(image_path)#prediction
print(result)
#kutu çizimi
for box in result.boxes:
    x1,y1,x2,y2=map(int,box.xyxy[0])#kutu koordinatları
    conf=float(box.conf[0])#güven skoru
    cls_id=int(box.cls[0])#sınıf idsi
    label=f'{model.names[cls_id]} {conf:.2f}'#detection label 
    #kutu çizimi
    cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)#yeşil kutu
    cv2.putText(image,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)#label yazısı
#sonucu göster
cv2.imshow('YOLOv8 Detection',image)
cv2.waitKey(0)#bir tuşa basılana kadar bekle zamansal hakimiyet
cv2.destroyAllWindows()#tüm pencereleri kapat
#sonucu kaydet
output_path='output/detected_image.jpg'
cv2.imwrite(output_path,image)
print(f'Detected image saved to {output_path}')
