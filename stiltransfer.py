"""
content :temel özellikleri korunan resim 
stil : sanatsal özellikler aktarılan resim 
"""
#import libraries
import torch
import torch.nn as nn
from torchvision import models,transforms
from PIL import Image
import matplotlib.pyplot as plt 
from tqdm import tqdm
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#resim yükleme ve ön işleme
def load_image(image_path,transform=None,max_size=None,shape=None):
    image=Image.open(image_path).convert("RGB")
    if shape is not None:
        size=shape#stil ve içerik aynı 
    else:
        size =max(image.size)#uzun kenarı al 
        if size>max_size:#fazla uzun ise kırp 
            size=max_size 
    #dönüşümler 
    in_transform=transforms.Compose([
        transforms.Resize((size,size)),#convert ile stili aynı boyuta getirip çakışmayı önleme
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485,0.456,0.406),
            std=(0.229,0.224,0.225))
    ])
    image=in_transform(image)[:3,:,:].unsqueeze(0)
    """resim olmaktan çıkarıp pytorchun midsine tam oturacak şekilde sayılı veri paketine dönüştürür
    [::] bu kanal filtreleme işlemi gerekli kanalları hesaba çek 
    unsquezee batch boyutu ekleme yeni bir kanal 1 ekler mesela yada 2 bu kaçtane resim olduğuunn temsilidir 
    """
    return image.to(device)
#görseli ekrana düzgün gösterme 
def im_convert(tensor):#ters domalizasyon 
    image=image*torch.tensor([0.229,0.224,0.225]).view(3,1,1)#std ile çarpma konstrastı geri getirir
    image=image+torch.tensor([0.485,0.456,0.406]).view(3,1,1)#mean ile toplama parlaklığı geri getirir
    image=image.clamp(0,1)#sınırlandırma 0-1 arası
    return image.permute(1,2,0).numpy()#eksenleri düzeltme ve numpy formatına getirme kütüphane uyşmazlığını gidermek için 
def gram_matrix(tensor):#gram matrisi ile bir katmandaki farklı özelliklerin korelasyonunu ölçeriz
    #kanal sayısı, yükseklik, genişlik
    """
    (C,H,W) -> (C,H*W)=A, AxA^T=Gram Matrix
    """
    _,d,h,w=tensor.size()
    tensor=tensor.view(d,h*w)#kanal sayısını koruyup diğerlerini tek bir boyutta toplama 
    gram=torch.mm(tensor,tensor.t())#matris çarpımı 
    return gram
#öznitelik çıkarıcı model (vgg19)
class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet,self).__init__()
        self.vgg=models.vgg19(pretrained=True).features[:21]#ilk 21 katmanı alıyoruz
        #batch normazliayon istastikleri süreki günceller bunu dondurarak zaten daha önceden eğitlmiş ağırlıkları kullnarak kolay
        # kolay tahmin yapabiliyoz 
        for param in self.vgg.parameters():
            param.requires_grad=False#ağırlıkları donduruyoruz
            #sebep ise önceden eğitilmiş ağırlıklar zaten var işimize engel olmasın diye yapıyoruz
            self.layers={
                '0':'conv_0',#stil katmanı
                '5':'conv_5',
                '10':'conv_10',
                '19':'conv_19',#içerik katmanı
                '21':'conv_21'
            } 
    def forward(self,x):
        features={}
        for name,layer in self.vgg._modules.items():#katman katman ilerleme
            x=layer(x)
            if name in self.layers:#özellikleri yakalalama
                #sadece belirlediğimiz katmanların çıktısını alıyoruz
                # çünkü bize output lazım değil alt duraklaardkaki stil üst duraklardaki içerik özellikleri lazım 
                features[self.layers[name]]=x
        return features
    #stil transfer döngüsü tamamla
    def run_style_transfer(content_img,
                           style_img,
                           steps=2000,
                           style_weight=1e6,#stil kaybı katsayısı
                           content_weight=1#içerik kaybı katsayısı
                           ):
        #hedef tensörü içerik görselinden kopylaya ve optimize edilcek değişşken yapalım
        target=content_img.clone().requires_grad_(True).to(device)
        #modeli başlat
        model=VGGNet().to(device)
        #optimizatör tanımla
        optimizer=torch.optim.Adam([target],lr=0.003)
        for step in tqdm(range(steps)):
            target_features=model(target)
            content_features=model(content_img)
            style_features=model(style_img)
            #içerik kaybı#tek bir parçadır for a gerek yok 
            content_loss=torch.mean((target_features['conv_19']-content_features['conv_19'])**2)
            #stil kaybı#farklı farklı katmanlardan oluşan parçadır o yüzden for kullanıırız
            style_loss=0
            for layer in ["conv_0","conv_5","conv_10","conv_19"]:#stil weight 
                target_feature=target_features[layer]
                style_features=style_features[layer]
                target_gram=gram_matrix(target_feature)
                style_gram=gram_matrix(style_features)
                layer_loss=torch.mean((target_gram-style_gram)**2)
                style_loss+=layer_loss
            #toplam kayıp
            total_loss=content_weight*content_loss+style_weight*style_loss
            #optimizasyon adımı
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if step % 100 ==0:#her 100 adımda bir kaybı yazdır
                print("Toplam Kayıp: ",total_loss.item())
        return target
    
#uygulama
content=load_image("content.jpg",max_size=400).to(device)
style=load_image("style.jpg",shape=content.shape[-2:]).to(device)
output=VGGNet.run_style_transfer(content,style)
#sonucu göster
plt.figure(figsize=(10,5)) 
plt.subplot(1,3,1)
plt.title("İçerik Resmi")
plt.imshow(im_convert(content.cpu().squeeze(0)))
plt.axis("off")
plt.subplot(1,3,2)
plt.title("Stil Resmi")
plt.imshow(im_convert(style.cpu().squeeze(0)))
plt.axis("off")
plt.subplot(1,3,3)
plt.title("Stil Transferli Resim")
plt.imshow(im_convert(output.cpu().squeeze(0)))
plt.axis("off")
plt.show()
#sonucu kaydet                