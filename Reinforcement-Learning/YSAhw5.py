import numpy as np
import random
import math
import matplotlib.pyplot as plt
import os
import xlwt
from xlwt import Workbook 
from xlrd import open_workbook

os.system('cls||clear')

class ACE():
    # Uyarlanabilir eleştiri bileşeni. Ödüle ve sistem durumlarına göre uyarlanabilir
    # karar bileşenine (ASE) hata işaretini verir. Sistem, bu işarete göre duvarlara
    # çarpmaması ve çubuğu düşürmemesi için kendini ayarlamaktadır.
    def __init__(self, dimension, lambda_val, gamma, dif_step):
        # ACE içerisinde kullanılacak değişkenler atanır.
        self.dimension = dimension # Durum değişkeni boyutu.
        self.lambda_val = lambda_val # Bozulma oranı.
        self.gamma = gamma # Hata işaretindeki benzerliği belirlemeye yarayan sabit.
        self.dif_step = dif_step # Açık euler için kullanılacak olan adım boyutu.

        self.reset()

    def reset(self):
        self.v_weights = np.zeros((self.dimension, 1), dtype=float) # ACE ağırlıkları.
        # Başlangıç ağırlıkları [-0.5,0.5] aralığında rastgele sayılardır.
        self.v_weights[:,0] = np.random.rand(self.dimension)*0.1 - 0.05 

        # Giriş işaretinin seçilebilirliğini gösterecek terim. Başlangıçta 0'dır.
        self.xhat_selection = np.zeros((self.dimension, 1), dtype=float)
        
        # Giriş, aktivasyon (p), önceki aktivasyon(p(k-1)) ve ödül beklentisi için
        # değişkenler oluşturulur. Başlangıçta 0'lardır.
        self.data = np.zeros(self.dimension)
        self.p_activation = 0
        self.prev_p_activation = 0
        self.r_int_reinf = 0

    def activate(self, data, reinf):

        # İleri yol, ACE aktivasyonu ve beklenti hatasının hesaplanması.
        self.p_activation = np.dot(np.transpose(self.v_weights), data)
        self.r_int_reinf = reinf + self.gamma*self.p_activation - self.prev_p_activation

        # Ağırlık ve seçilebilirlik teriminin güncellenmesi.
        self.v_weights = self.v_weights + self.dif_step*(reinf + gamma*self.p_activation - self.prev_p_activation)*self.xhat_selection
        self.xhat_selection = self.lambda_val*self.xhat_selection + (1 - self.lambda_val)*data
        self.prev_p_activation = self.p_activation
        
        # Beklenti hatasını döndürür.
        return self.r_int_reinf

class ASE():
    # Uyarlanabilir karar bileşeni. Durum değişkenlerine ve ödül beklentisine göre karar verir. Bu karar
    # durumun değişiminde kullanılacaktır.
    def __init__(self, dimension, alpha, delta, activation_type, dif_step):
        # ASE içerisinde kullanılacak değişkenler atanır.
        self.dimension = dimension # Durum değişkeni boyutu.
        self.alpha = alpha # Ağırlıkların değişme oranı.
        self.delta = delta # Bozulma sabiti.
        self.activation_type = activation_type # ASE'nin kullanacağı aktivasyon fonksiyonu.
        # Aktivasyon fonksiyonu için perceptron (signum) ve tanh fonksiyonları tanımlıdır.
        self.dif_step = dif_step # Açık euler için kullanılacak olan adım boyutu.

        self.reset()

    def reset(self):
        self.w_weights = np.zeros((self.dimension, 1), dtype=float) # ASE ağırlıkları.
        # Başlangıç ağırlıkları [-0.5,0.5] aralığında rastgele sayılardır.
        self.w_weights[:,0] = np.random.rand(self.dimension)*0.1 - 0.05

        # Seçilen davranış ve sistemin cevabı arasındaki ilişkiyi belirten terim. Başlangıçta 0'dır.
        self.e_relevance = np.zeros((self.dimension, 1), dtype=float)

        # Giriş, ve aktivasyon (y) için değişkenler oluşturulur. Başlangıçta 0'lardır.
        self.data = np.zeros(self.dimension)
        self.y_activation = 0

    def activate(self, data, int_reinf, noise):

        # İleri yol, ASE aktivasyonu.
        self.y_activation = self.activation_function(np.dot(np.transpose(self.w_weights), data) + noise)

        # Ağırlık ve uygunluk teriminin güncellenmesi.
        self.w_weights = self.w_weights + self.alpha*int_reinf*self.e_relevance
        self.e_relevance = self.delta*self.e_relevance + (1 - self.delta)*self.y_activation*data

        # Aktivasyonu döndürür.
        return self.y_activation

    def activation_function(self, x):
        # Aktivasyon fonksiyonu olarak tanh ve perceptron (signum) tanımlıdır. Fakat, bu fonksiyonların,
        # Tez_Kuyumcu.pdf'te belirtildiği gibi, y eksenine göre simetrileri alınmıştır.
        if self.activation_type == "tanh":
            if np.isinf((np.exp(x) + np.exp(-x))):
                # Kodda, x çok büyük veya çok küçük değerler alabiliyor. numpy.exp fonksiyonu
                # "overflow" hatası verdiğinde "t" değeri "nan" oluyor. Bu durumda, x değeri
                # zaten çok büyük veya çok küçük olduğu için tanh(x) fonksiyonu -1 veya 1 değerine
                # sahip. Bu nedenle "t"nin değerinin ölçülemediği durumlarda, değerin atanması için 
                # perceptronda da kullanılan algoritma kullanılır. Bu işlem "t" hesaplanırken paydanın
                # sonsuzla karşılaştırılmasıyla yapılarak, hatadan önce saptanır.
                return -np.sign(-x)
            t = (np.exp(-x) - np.exp(x))/(np.exp(-x) + np.exp(x))
            return t
        elif self.activation_type == "perceptron":
            return np.sign(-x)
        else:
            return -x

class CarStick():
    # Araba ve çubuk sistemi.
    def __init__(self, gravity, m_car, m_stick, l_stick, fs_ground, fs_car, dif_step, theta1_init, theta2_init, x1_init, x2_init):
        # Problemle ilgili fiziksel sabitler atanır.
        self.gravity = gravity # Yerçekimi ivmesi.
        self.m_car = m_car # Arabanın kütlesi.
        self.m_stick = m_stick # Çubuğun kütlesi.
        self.l_stick = l_stick # Çubuğun uzunluğu.
        self.fs_ground = fs_ground # Araba-yer arasındaki sürtünme sabiti.
        self.fs_car = fs_car # Çubuk-araba arasında sürtünme sabiti.
        self.dif_step = dif_step # Açık euler için kullanılacak olan adım boyutu.
        self.m1 = self.m_car + self.m_stick # Araba ve çubuğun toplam kütlesi.

        self.reset(theta1_init, theta2_init, x1_init, x2_init)

    def reset(self, theta1_init, theta2_init, x1_init, x2_init):
        # Sistemdeki araba ve çubuğun başlangıç koşulları.
        self.theta1 = theta1_init
        self.theta2 = theta2_init
        self.x1 = x1_init
        self.x2 = x2_init
 
    def activate(self, y):
        # İleri yol, sistemdeki kuvvetin ve f1,f2 terimlerinin hesaplanması.
        F = 10*np.sign(y)
        f1 = self.m_stick*self.l_stick*(self.theta2**2)*math.sin(self.theta1) - self.fs_ground*np.sign(self.x2)
        f2 = self.gravity*math.sin(self.theta1) + math.cos(self.theta1)*((-F - f1)/self.m1) - self.fs_car*self.theta2/(self.m_stick*self.l_stick)

        # Araba konumu ve çubuk açısının sonraki durumlarının hesaplanması.
        new_theta1 = self.theta1 + self.dif_step*self.theta2
        new_theta2 = self.theta2 + self.dif_step*f2/(self.l_stick*(4/3 - self.m_stick*(math.cos(self.theta1)**2)/self.m1))
        new_x1 = self.x1 + self.dif_step*self.x2
        new_x2 = self.x2 + self.dif_step*((F + (f1 - self.m_stick*self.l_stick*f2)*math.cos(self.theta1))/self.m1)

        # Sistem durumunun güncellenmesi.
        self.update(new_theta1, new_theta2, new_x1, new_x2)

        # Ödül fonksiyonu. Açı büyüklüğü 8 dereceden küçük veya araba konumu 1.5 metreden küçükse 
        # sistem ödül verir, bu sınırların dışındayken ceza verir.
        if abs(new_theta1) < 8 or abs(new_x1) < 1.5:
            reinf = 1
        else:
            reinf = -1

        # Bir sonraki iterasyonda veri olarak sistemin yeni durumu sunulur. Ödülle birlikte sistem durumu çekilir.
        return [np.array([new_theta1, new_theta2, new_x1, new_x2], dtype=float).reshape(4,1), reinf]

    def update(self, new_theta1, new_theta2, new_x1, new_x2):
        # Sistem durumunun güncellenmesi.
        self.theta1 = new_theta1
        self.theta2 = new_theta2
        self.x1 = new_x1
        self.x2 = new_x2

# Elde edilen sonuçları excel dosyasına kaydeden fonksiyon.
def saveResults(results, savefolder, filename):
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    filedir = os.path.join(savefolder,filename)
    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet1')
    col = 0
    row = 0
    sheet1.write(row, col, 'Test Numarası')
    sheet1.write(row, col+1, 'Alpha')
    sheet1.write(row, col+2, 'Delta')
    sheet1.write(row, col+3, 'Theta1_init')
    sheet1.write(row, col+4, 'Theta2_init')
    sheet1.write(row, col+5, 'X1_init')
    sheet1.write(row, col+6, 'X2_init')
    sheet1.write(row, col+7, 'Başarısız Olunan İterasyon')
    sheet1.write(row, col+8, 'Beklenti Hata Performansı')
    row += 1
    for result in results:
        for i in range(len(result)):
            sheet1.write(row, col+i, result[i])
        row +=1 

    wb.save(filedir) 

def activate_system(lambda_val, gamma, alpha, delta, theta1_init, theta2_init, x1_init, x2_init, activation_type, epoch, title, savefolder):
    # Araba-çubuk sisteminin başladığı fonksiyon.

    # Sistem için sabitler atanır. Sabitler Tez_Kuyumcu.pdf'ten alınmıştır.
    dimension = 4 # Durum değişkenleri boyutu = 4.
    gravity = -9.8 # Yerçekimi ivmesi = -9.8 m/s^2.
    m_car = 1 # Araba kütlesi = 1 kg.
    m_stick = 0.1 # Çubuk kütlesi = 0.1 kg.
    l_stick = 0.5 # Çubuk uzunluğu = 0.5 m.
    fs_ground = 5e-4 # Araba-yer arasındaki sürtünme katsayısı = 0.0005.
    fs_car = 2e-6 # Çubuk-araba arasındaki sürtünme katsayısı = 0.000002.
    dif_step = 0.005 # Açık euler çözümünde kullanılacak adım büyüklüğü = 0.005.

    # Karar bileşeni, eleştiri bileşeni ve araba-çubuk sistemi verilen parametrelerle oluşturulur.
    my_ACE = ACE(dimension, lambda_val, gamma, dif_step)
    my_ASE = ASE(dimension, alpha, delta, activation_type, dif_step)
    my_CarStick = CarStick(gravity, m_car, m_stick, l_stick, fs_ground, fs_car, dif_step, theta1_init, theta2_init, x1_init, x2_init)

    # Simülasyon başlatılır.
    sim_result = start_sim(my_ACE, my_ASE, my_CarStick, epoch, title, savefolder)

    # Simülasyon sonuçları döndürülür.
    return sim_result

def start_sim(ACE, ASE, CarStick, epoch, title, savefolder):
    y_delta = []
    y_state = []
    
    # Başlangıçtaki durum değişkenleri çekilir ve veri olarak birleştirilir.
    theta1 = CarStick.theta1 
    theta2 = CarStick.theta2
    x1 = CarStick.x1
    x2 = CarStick.x2
    data = np.array([theta1, theta2, x1, x2], dtype=float).reshape(4,1)

    # Başlangıç ödülü 0 alınır.
    reinf = 0

    # Sistem performansının ölçütleri olarak simülasyonun sürdüğü iterasyon sayısı ve 
    # beklenti hatasının düşük olduğu seri iterasyon sayısı ile ilgili değişkenler atanır.
    lastiter = epoch
    delta_performance = 0
    performance_iter = 0
    simdone = False
    
    # Simülasyon, araba çarpmadıkça veya çubuk düşmedikçe, "epoch" iterasyon devam edecektir.
    for i in range(epoch):

        # 0 medyan ve 0.1 standart sapmalı normal dağılımlı rastgele bir gürültü tanımlanır.
        noise = np.random.normal(0, 0.1)
        
        # Öncelikle eleştiri bileşeni çalıştırılarak durum değişkenlerine göre
        # ödül beklentisi bulunur.
        int_reinf = ACE.activate(data, reinf)

        # Bulunan ödül bileşeni, gürültü ve durum değişkenleri kullanılarak karar bileşeni
        # çalıştırılır. Karar bileşeni çıkışı arabaya uygulanacak kuvveti belirler.
        y_activation = ASE.activate(data, int_reinf, noise)

        # Karar bileşeni çıkışı ile sistem çalıştırılır. Uygulanan kuvvet ve sistemin durumuna
        # göre, sistemin yeni durumu ve eleştiri bileşenine iletilecek ödül bulunur.
        [data, reinf] = CarStick.activate(y_activation)

        # Beklenti hatası hesaplanır.
        delta = int_reinf - reinf

        # Araba 2.4 metrenin dışında veya çubuk 12 dereceden daha eğik ise simülasyonu durdurma kararı alınır.
        if (abs(CarStick.x1) > 2.4 or abs(CarStick.theta1) > 12):
            simdone = True

        if abs(delta) < 0.5:
            performance_iter += 1

        if (abs(delta) >= 0.5 and performance_iter > 0) or i + 1 == epoch or simdone == True:
            if performance_iter > delta_performance:
                # Beklenti hatasının 0.5'ten küçük olduğu ardışık iterasyon sayısı kaydedilir.
                # Simülasyon boyunca en uzun süre bu aralıkta kalınan sayı, simülasyonun 
                # beklenti hata performansını belirler.
                delta_performance = performance_iter
            performance_iter = 0

        # Simülasyon sonunda çizilmek üzere beklenti hatası ve durum değişkenleri bir listede kaydedilir.
        y_delta.append(delta)
        y_state.append([CarStick.theta1, CarStick.theta2, CarStick.x1, CarStick.x2])

        # Simülasyonu durdurma kararı alındıysa son iterasyon kaydedilir ve simülasyon durdurulur.
        if simdone == True:
            # Simülasyonun sürdüğü iterasyon sayısı kaydedilir.
            lastiter = i + 1
            break

    # Simülasyon sonucunda elde edilen beklenti hatası ve durum değişkenleri çizilir ve kaydedilir.
    x = np.arange(1, lastiter + 1)
    y_state = np.array(y_state, dtype=float).reshape(lastiter,4)
    
    plt.figure()
    plt.plot(x, np.array(y_delta, dtype=float).reshape(lastiter))
    plt.ylim(-3, 3)   
    plt.xlim(0,epoch)
    plt.title(title)
    plt.savefig(savefolder + '_Hata.png')
    # plt.show()
    plt.close()

    fig, axs = plt.subplots(4)
    fig.suptitle('Durum Değişkenlerinin Değişimi')
    axs[0].plot(x, y_state[:,0])
    axs[0].set_title('Çubuk Açısı')
    axs[1].plot(x, y_state[:,1])
    axs[1].set_title('Çubuk Açısının Hızı')
    axs[2].plot(x, y_state[:,2])
    axs[2].set_title('Araba Konumu')
    axs[3].plot(x, y_state[:,3])
    axs[3].set_title('Araba Hızı')
    fig.text(0.5, 0.04, 'İterasyon', ha='center')
    fig.savefig(savefolder + '_Durum.png')
    plt.close()

    # Simülasyonun sürdüğü iterasyon sayısı ve beklenti hata performansı döndürülür.
    return [lastiter, delta_performance]

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Sistemin kullanacağı lambda ve gamma değişkenleri Tez_Kuyumcu.pdf'te belirtildiği gibi alınır.
lambda_val = 0.8
gamma = 0.95

# Sistemin başlangıç koşulları farklı testler için değiştirilebilir.
theta1_init = 1
theta2_init = 0
x1_init = 1
x2_init = 0

# Kullanılacak aktivasyon fonksiyonu ve simülasyonun maksimum iterasyon sayısı belirlenir.
# Tanh ve perceptron (signum) fonksiyonları tanımlıdır.
activation_type = "tanh"
epoch = 500

# Sonuçların kaydedileceği klasör oluşturulur.
resultFile = os.path.join(__location__, "Sonuclar")
if not os.path.exists(resultFile):
    os.makedirs(resultFile)

# Tez_Kuyumcu.pdf'te değeri verilmeyen fakat sistemin kullandığı iki sabit vardır. Bunlar "delta" (bozulma oranı)
# ve "alpha"dır (ağırlıkların değişme oranı). Bu değerlerin bu problemde optimal değerlerinin bulunması için test
# düzenlenir. Testte değerlerin 0, 0.2, 0.4, 0.6 ve 0.8 değerleri aldığı 10 farklı simülasyon yapılır. Bu simülasyonlar
# sonucunda elde edilen simülasyon iterasyon sayısı ve beklenti hata performansları bir excel dosyasında kaydedilir.
# Ayrıca her farklı değer çifti için sonuncu testte elde edilen durum değişkenleri ve beklenti hatası çizdirilir.
testnumber = 1
resultList = []
for i in range(5):
    for j in range(5):
        for k in range(10):
            # Test edilecek olan alpha ve delta değerleri atanır.
            alpha = i*0.2
            delta = j*0.2

            # Çizimlerin kaydedileceği klasör belirlenir.
            savefolder = os.path.join(resultFile, "alpha0" + str(int(10*alpha)) + "_delta0" + str(int(10*delta)))

            # Çizimlerin başlıkları belirlenir.
            title = "alpha = " + str(alpha) + ", delta = " + str(delta)

            # Sistem aktive edilir ve simülasyon başlatılır.
            result = activate_system(lambda_val, gamma, alpha, delta, theta1_init, theta2_init, x1_init, x2_init, activation_type, epoch, title, savefolder)
            # Simülasyon sonucu elde edilen sonuçlar yazdırılır.
            print("delta:%.1f ---- alpha:%.1f ---- başarısız olunan iterasyon:%d ---- beklenti hata performansı:%d" % (delta, alpha, result[0], result[1]))

            # Sonuçlar kaydedilmek üzere bir listeye eklenir ve bir sonraki teste geçilir.
            resultList.append([testnumber, alpha, delta, theta1_init, theta2_init, x1_init, x2_init, result[0], result[1]])
            testnumber += 1

# Test sonuçları kaydedilir.
saveResults(resultList, resultFile, 'Test_Sonuclari.xls')