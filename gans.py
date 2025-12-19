#import libraries
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.dataserts import fashion_mnist
BUFFER_SIZE = 60000
BATCH_SIZE = 256
NOISE_DIM = 100#genratöre verilcek gürültü vektörünün boyutu(karıncalandırma)
IMG_SHAPE = (28, 28, 1)#fashion mnist resim boyutu
EPOCHS = 50
#veri seti yükle
(train_images, _), (_, _) = fashion_mnist.load_data()#ganlar denetimsizdir o yüzden etiketlere almadık 
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')#float32 normalizasyon hazırlığ
#gradyanların hassasiyeti için float32 kullanılır ve gpuda yüksek performnaslı matrix çarpımları için
#standart veri tipi budur
#reshape ise ganlar genelde cnn tabanlıdır fashion mnist 28x28 gri tonlamalı resimlerden oluşur bu yüzden 4d tensöre çevirdik
#ve cnnler kanal tiplidir ona uygun hale getirdik 
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(256)
#generator modeli tanımla
def make_generator_model():
    model = tf.keras.Sequential([
    layers.Dense(7*7*256, use_bias=False, input_shape=(NOISE_DIM,)),#ilk katman gürültü vektörünü alır 
    layers.BatchNormalization(),
    layers.LeakyReLU(),#negatif değerler için küçük eğim sağlar(yumuşatma)

    layers.Reshape((7, 7, 256)),#tek boyutlu vektörü 7x7x256 boyutunda bir tensöre dönüştürür
    #cnnlerin çalışabilmesi için en boy kanal gerekleidir uzaysal boyut kazanıdırı

    layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),#ilk Conv2DTranspose katmanı 7x7x256 tensörünü 7x7x128 tensörüne dönüştürür
    #standart bir cnn tam tersini yapar boyut güçültür özellik özetler buda boyut büyültür ve özellik öğrenir
    layers.BatchNormalization(),
    layers.LeakyReLU(),

    layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.LeakyReLU(),

    layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'),
    #ensonda tanh kullanılmasının sebebi:
    #görüntü piksellerini 0 ile 1 e sıkıştırmak yerine -1 ile 1 arasında sıkıştırarak verinin merkezini sıfır noktası yapar buda 
    #gradyanların hem pozitif hemde negatif olmasını sağlar ve eğitim sürecini hızlandırır 
    ])    
    return model
#disciminator modeli tanımla
def make_discriminator_model():
    model = tf.keras.Sequential([
    layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=IMG_SHAPE),#max pooling yerine stride kullanıldı 
    #çünkü max pooling bazı durumlarda bilgi kaybına yol açabilir o yüzden stride tercih edildi
    #bu stride convulation hem kendi filterisini öğrendi(hangi piksellerin önemli olduğunu) hemde bilgi kaybını minimize eder
    layers.LeakyReLU(),
    layers.Dropout(0.3),#overfittingi önlemek için dropout kullanıldı
    #kedinin(discrimanator)ün gözlerini bağlaki fare(genaratör) kaçmayı ve kendini geliştirmeyi öğrensin
    layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
    layers.LeakyReLU(),
    layers.Dropout(0.3),

    layers.Flatten(),#3d yi düzleştir
    layers.Dense(1),#binary sınıflandırma real/fake
    ])
    return model
#kayıp fonksiyonu tanımla optimizer tanımla
def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)#gerçek resimler için kayıp
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)#sahte resimler için kayıp
    total_loss = real_loss + fake_loss
    return total_loss
def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)#genaratör sahte görüntüyü 1 e çevirecek
generator=make_generator_model()
discriminator=make_discriminator_model()
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
#yardımcı fonksiyonlar
seed = tf.random.normal([16, NOISE_DIM])#normal dağılım kullanarak rasthele 16 görüntü (karıncaalandırma)
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)#modeli test modunda çalıştır
    fig = plt.figure(figsize=(4, 4))
    #16 tane görüntünün içinde dolaşarak subplot halinde 4e 4 lük çizdirdik
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i, :, :, 0] +1)/2, cmap='gray')#görüntüyü [0,1] aralığına getir
        plt.axis('off')
    if not os.path.exists('images'):#klasör kontrolü yoksa oluştur
        os.makedirs('images')    
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()
#eğitim fonksiyonları tanımla
def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])#gürültü vektörü oluştur
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:#kara kutu kayıt cihazı 
                #içinde yapılan tüm matematiksel işlemler kaydedilir gen tape ve disc tape adında iki tane oluşturduk
                #hikmeti gan gibi karışık modellerde standart model.fit() yetersiz olduğu için el yordamı(sistematik yöntem kullanmadan 
                #kişisel deneyim ve sezgiye dayanarak)
                # ile yönetmek zorundayız
                generated_images = generator(noise, training=True)#gürültüden sahte görüntü oluştur
                real_output = discriminator(image_batch, training=True)#gerçek görüntüleri değerlendir
                fake_output = discriminator(generated_images, training=True)#sahte görüntüleri değerlendir
                gen_loss = generator_loss(fake_output)#genaratör kaybı hesapla
                disc_loss = discriminator_loss(real_output, fake_output)#disciminator kaybı hesapla
                #gradyan aslında hata pusulasıdır modelin nereye gitmesi gerektiğini gösterir
                #hata hesapla gradyan hesapla ve uygula yani optimize et 
            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)#genaratörün gradyanlarını hesapla
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)#disciminatorün gradyanlarını hesapla
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        #her epoch sonunda görüntü oluştur ve kaydet
        generate_and_save_images(generator, epoch + 1, seed)
        print ('Epoch {} completed'.format(epoch+1))
        generate_and_save_images(generator, epochs, seed)#reküskif şeklinde her epochtan sonra genratör birazda eğitilmiş şekilde gönderme
#eğitimi başlat
train(train_dataset, EPOCHS)
#ganlar birşey tahmin etmek için değil yeni veri üretmek için kullanılır bu yüzden evaluate fonksiyonu yoktur

