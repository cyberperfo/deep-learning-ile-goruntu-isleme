""""
1. Train (Eğitim) Aşaması: "Öğrenme ve Ezber Bozma"
Eğitim aşamasında model, etiketlediğin resimlere bakarak nesneleri tanımayı öğrenir.

İleri Besleme (Forward Pass): Model, gürültülü (henüz eğitilmemiş) ağırlıklarıyla resme bakar ve "Burada bir İHA var" diye tahmin yapar.

Hata Hesaplama (Loss Calculation): Modelin tahmini ile senin çizdiğin gerçek kutu (Ground Truth) karşılaştırılır. Aradaki fark; box_loss (kutu konumu hatası) ve cls_loss (yanlış nesne tanıma hatası) olarak hesaplanır.

Geri Yayılım (Backpropagation): tf.GradientTape veya PyTorch'un otomatik türev mekanizmasıyla, bu hatayı azaltmak için hangi "vidayı" (ağırlığı) ne kadar sıkması gerektiğini hesaplar.

Optimizasyon: Senin kodundaki Momentum ve Weight Decay burada devreye girer; hataları sarsıntısız bir şekilde düzeltir ve modelin ağırlıklarını günceller."""
"""
eo sensor:kamera radar lidar vb
otonom araçlarda envivoment tanımlama
plan program : veri bulma veri yükleme train test
"""
from ultralitics import YOLO
#modeli seç
model=YOLO('yolov8n.pt')#küçük model
model.train(
    data='coco128.yaml',#dataset örnek data
    #içinde nesnelerin etiketlerini ve nesne belirmtek için patlhleri belirtiyorsun
    epochs=10,
    imgsz=640,
    batch=16,
    name='yolo8n-coco128',#output folder name
    lr0=0.001,
    optimizer='SGD',
    weight_decay=0.0005,#ağırlık cezası overfitinigi önler ağırlık azlatar her küçük detaya sert karar vermez 
    #weight decay:ağırlıklara oynar overfitingi engeller ve sade kalmasını sağlar
    #lr decay: adım sayısını azaltır epoch ilerledikçe öğrenme hızını azaltır hedefe tam ve sarsıntısız yaklaşmayı sağlar
    #momentum: önceki adımları dikkate alır ani değişiklikleri engeller gradyanlara etki eder sarsıntısız ilerleme engelleri aşma
    momentum=0.9,
    patience=5,#erken durdurma için sabır sayısı
    workers=4,#kernel tarafından ayarlanan iş parçacığı sayısı cpu performansını artırır
    device='cpu',
    save=True,#modelleri kaydet
    save_period=5,#her 5 epochta bir kaydet
    val=True,#valüdasyon deneme sınavı
    verbose=True#ayrıntılı çıktı
)
"""
epoch   gpu_mem       box_loss    cls_loss    dfl_loss   Instances       Size   
box_loss: kutu kaybı, nesne tespiti için kutu konumlarının doğruluğunu ölçer 0.1-0.3 arası iyi
cls_loss: sınıf kaybı, modelin nesne sınıflarını doğru tanımlama yeteneğini ölçer 1 in altı iyi
dfl_loss: dağıtılmış odaklı kayıp, modelin kutu konumlarını daha hassas tahmin etmesini sağlar" 0.5-1 arası iyi
kutu konumu: nesnenin görüntü içindeki lokasyonu ve boyutu
Instances: her bir eğitim örneğinde bulunan nesne sayısı
Size: giriş görüntü boyutu
"""