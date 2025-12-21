"""
problem tanımı: resim-> açıklama yapmak 
BLIP Model 
huggingface :yapay zekanın githubu 
otomatik altyazı üretme 
blıp model
-multimodal:birden fazl veri tipini aynı anda işleyebilme(görsel,ses,metin,sensör verisi)
caption üretme:altyazı  
input =image output=text
-mİmari
  -görsel encoder :(vision transformer) görüntüyü özellik vektörüne çevirir
    -text decoder : (transformer tabanlı dil modeli )özellik vektöründen metin üretir
    -cross attention mekanizması : görsel ve metinsel bilgiyi birleştirir
"""
#import libraries
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import torch
#model yükle    
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")#görseli modele uygun girdi tensörlerine çevirir
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")#görsel girdiden metin üreten caption sürümü
#test görseli indir
img_url = "https://raw.githubusercontent.com/salesforce/BLIP/main/examples/data/image1.jpg"
image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")# 
#görseli tensöre çevirme
inputs = processor(image, return_tensors="pt")#görseli normalize et ve pytorch tensörüne çevir
#caption metni üretimi
with torch.no_grad():#yanlızca inference yapılacak o yüzden gradyan hesaplama kapalı
    """inference: önceden eğitilmiş modelin hiç görmediği veriler üzerine tahmin yapması sonuç karar üretmek 
       train:sıfırdan neyin ne olduğunu öğrenme süreci ağırlıkları bulmak 
       test: modelin öğrendiklerini değerlendirme süreci başarıyı ölçmek 
    """
    out = model.generate(**inputs)#görsel tensöründen caption üret
    #cross atention burda gerçekleşenn gizli mekanizma
#tokenleri insan okuması için düzenle
caption = processor.decode(out[0], skip_special_tokens=True)
print("Generated Caption:", caption)

