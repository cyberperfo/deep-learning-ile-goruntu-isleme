"""
tokenizasyon metni paraçalara ayırma
"""
#import libraries
from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel
from PIL import Image #görseli işlemek için pıl bu işe yarar 
import requests
import torch
#model ,processor ve tokenizer yükle
model=VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")#görselden metin üreten model
#encoder vision transformer (ViT) decoder gpt2
#VIT Processor: görseli modele uygun girdi tensörlerine çevirir
processor=ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
#AutoTokenizer:
tokenizer=AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")#gpt2 decoder tarafından oluşturlan tokenleri metne çevirmek için 
#token modelin ürettiği sayısal kodlar 
#görsel url and request
img_url="https://images.unsplash.com/photo-1516116216624-53e697fedbe2"
image=Image.open(requests.get(img_url,stream=True).raw).convert("RGB")#görseli aç ve rgb ye çevir
#görseli tensöre çevirme
pixel_values=processor(images=image,return_tensors="pt").pixel_values#görseli normalize et ve pytorch tensörüne çevir
#modeli uygun cihaza gönder 
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
pixel_values=pixel_values.to(device)
#modeli çalıştıralım 
model.generate(pixel_values,max_length=16)
#sonuçları ekrana yazdır 
caption= tokenizer.decode(output[0],skip_special_tokens=True)
#tokenizör ilk başta initilza edildi burda forward yapıldı 
print("Generated Caption:",caption)
