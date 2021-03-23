import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
import xlwt
from xlwt import Workbook 
from xlrd import open_workbook
import time

os.system('cls||clear')

#Nöron yapısı.
class Neuron():
    def __init__(self, u_dim, x_dim, y_dim, learning_rate = 1e-2, activation_type = "tanh"):
        # Nöronun ilk değerleri atanır.
        self.u_dim = u_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.learning_rate = learning_rate
        self.activation_type = activation_type

        self.reset()

    def reset(self):
        # Nöronun ilk ağırlıkları [-0.5,0.5] arasında rastgele seçilir. 
        # Nöronla ilgili içerik katmanı, çıkış katmanı ve giriş katmanı ağırlıkları tek nöronda saklanır.
        self.w_u = np.zeros((1, self.u_dim), dtype=float)
        for i in range(self.u_dim):
            self.w_u[0,i] = np.random.rand()*0.1 - 0.05

        self.w_x = np.zeros((1, self.x_dim), dtype=float)
        for i in range(self.x_dim):
            self.w_x[0,i] = np.random.rand()*0.1 - 0.05

        self.w_y = np.zeros((self.y_dim, 1), dtype=float)
        for i in range(self.y_dim):
            self.w_y[i,0] = np.random.rand()*0.1 - 0.05

        # Ağırlıkların önceki değerleri momentum için kullanılacaktır.
        self.w_u_prev = self.w_u
        self.w_x_prev = self.w_x
        self.w_y_prev = self.w_y

        self.data = np.zeros((self.u_dim, 1), dtype=float)

    def forward(self, data, prevx):
        # İleri yolda girilen veri (u(k)) ve önceki çıkış değerlerini (x(k-1)) kullanarak 
        # tek nörondaki çıkış değerini (xi(k)) elde eder.
        self.data = np.array(data).reshape((self.u_dim, 1))
        self.lin_comb = np.dot(self.w_u,self.data) + np.dot(self.w_x,prevx)
        self.activation = self.activation_function(self.lin_comb)
        return self.activation

    def activation_function(self, data):
        # Aktivasyon fonksiyonunun kullanıldığı fonksiyon. Türev bilgisi de elde edilir.
        if self.activation_type == "tanh":
            y = tanh(data)
            self.activation_derivative = y[1]
            return y[0]
        elif self.activation_type == "sigmoid":
            y = sigmoid(data)
            self.activation_derivative = y[1]
            return y[0]
        else:
            return data

    def updateN(self, w_u, w_x, w_y):
        # Nöron ağırlıklarının güncellenir.
        # Momentum kullanılır. 
        # Momentumun çıkarılması için 0'a eşitlenmesi yeterlidir.
        momentum = 0.9
        moment_u = momentum*(self.w_u - self.w_u_prev)
        moment_x = momentum*(self.w_x - self.w_x_prev)
        moment_y = momentum*(self.w_y - self.w_y_prev)
        self.w_u_prev = self.w_u
        self.w_x_prev = self.w_x
        self.w_y_prev = self.w_y
        self.w_u = w_u + moment_u
        self.w_x = w_x + moment_x
        self.w_y = w_y + moment_y
        
# Eğitimde girişe önceki veriler girileceği için bağlı liste oluşturulur.
class YNode():
    # Bağlı listede indis, gerçek değer, tahmin edilen değer ve önceki değer bulunur.
    def __init__(self, index, yprev = None, ypred = random.random()):
        self.index = index
        self.pred = ypred
        # İndis 0'dan büyükse linkler ve gerçek değer atanır. 0 indisi için değer 0 olacak, önceki değer
        # kendisini gösterecektir. Böylece nedensel bir sistem tanımlanıp başlangıç koşulu olarak 0 seçilir.
        if index > 0:
            self.links(yprev)
            self.targetfunc()

    def links(self, yprev):
        self.prev = yprev

    # Bir önceki ve iki önceki değerler hedef fonksiyona girilir ve gerçek değer elde edilir.
    def targetfunc(self):
        self.val = targetfunction(self.prev.val, self.prev.prev.val)

    # Değer ve linkler bu fonksiyonla da atanabilir.
    def setnode(self, val, yprev):
        self.val = val   
        self.prev = yprev

# Tanh aktitasyon fonksiyonu. Türev bilgisi de bu fonksiyonla elde edilir.
def tanh(x):
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    dt=1-t**2
    return t,dt

# Sigmoid aktitasyon fonksiyonu. Türev bilgisi de bu fonksiyonla elde edilir.
def sigmoid(x):
    s=1/(1+math.exp(-x))
    ds=s*(1-s)  
    return s,ds

# Sonuçları kaydetme fonksiyonu.
def saveResults(results, savefolder, filename):
    # Girilen 'savefolder' adresine 'filename' excel dosyasını kaydeder.
    # Excel dosyası içinde kodun sonundaki testlerin sonuçları bulunur.
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    filedir = os.path.join(savefolder,filename)
    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet1')
    col = 0
    row = 0
    sheet1.write(row, col, 'Öğrenme Hızı') 
    sheet1.write(row, col+1, 'Eğitim İterasyonu')
    sheet1.write(row, col+2, 'Eğitim Süresi')
    sheet1.write(row, col+3, 'Hata Oranı')
    result = results[0]
    row += 1
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            sheet1.write(row, col+j, result[i][j])
        row +=1 
    sheet1.write(row, col, 'Gizli Katman Nöron Sayısı')
    sheet1.write(row, col+1, 'Eğitim İterasyonu')
    sheet1.write(row, col+2, 'Eğitim Süresi')
    sheet1.write(row, col+3, 'Hata Oranı')
    result = results[1]
    row += 1
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            sheet1.write(row, col+j, result[i][j])
        row +=1  
    wb.save(filedir) 

# Ağın ve verinin karşılaştırıldığı grafiği çizen fonksiyon.
def plotcomparison(trainList, testList, maxval, minval, savefolder):
    # Ağı, eğitim ve test kümesini ayrı ayrı ve karşılaştırarak çizer.
    # Figürleri 'savefolder' adresine kaydeder.
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)

    # Verileri çizerken her veri düğümü ve önce gelen veri düğümünü alır. Bu iki düğümün gerçek
    # değerleri arasında aralıklı bir dizi oluşturur. Bu şekilde figür çizilir.
    plt.figure()
    for node in trainList:
        x = np.linspace(node.prev.index, node.index - 0.01, 100)
        m = (node.val - node.prev.val)/0.99
        normalized_y = node.prev.val + (x - node.prev.index)*m
        y = minval + normalized_y*(maxval - minval)
        y = y.reshape(100)
        plt.plot(x, y, color = 'blue', label='Gerçek Değer')
    plt.title("Yaklaşılan fonksiyon (Eğitim kümesi)")
    plt.savefig(os.path.join(savefolder, 'Gercek_Egitim_Kumesi.png'))
    plt.close()

    # Ağın çiziminde de benzer bir şekilde iki ardışık düğümün tahmini değerleri arasında
    # aralıklı bir dii oluşturularak çizdirilir. Bu tahmini değer eğitilmiş ağ ile test sonucu
    # elde edilir.
    plt.figure()
    for node in trainList:
        x = np.linspace(node.prev.index, node.index - 0.01, 100)
        m = (node.pred - node.prev.pred)/0.99
        normalized_y = node.prev.pred + (x - node.prev.index)*m
        # Çizilen y değeri geri ölçeklendirilir. 'maxval' ve 'minval' veri ölçeklendirilirken
        # verinin maksimum ve minimumlarının saklandığı değişkenlerdir ve geri ölçeklendirmede
        # kullanılır.
        y = minval + normalized_y*(maxval - minval)
        y = y.reshape(100)
        plt.plot(x, y, color = 'red')
    plt.title("Tahmin edilen fonksiyon (Eğitim kümesi)")
    plt.savefig(os.path.join(savefolder, 'Tahmini_Egitim_Kumesi.png'))
    plt.close()

    # İki çizim aynı anda uygulanıp karşılaştırma çizilir.
    plt.figure()
    for node in trainList:
        x = np.linspace(node.prev.index, node.index - 0.01, 100)
        m = (node.pred - node.prev.pred)/0.99
        normalized_y = node.prev.pred + (x - node.prev.index)*m
        y = minval + normalized_y*(maxval - minval)
        y = y.reshape(100)
        plt.plot(x, y, color = 'red', label='Tahmin edilen')
        m = (node.val - node.prev.val)/0.99
        normalized_y = node.prev.val + (x - node.prev.index)*m
        y = minval + normalized_y*(maxval - minval)
        y = y.reshape(100)
        plt.plot(x, y, color = 'blue', label='Gerçekte olan')
    plt.title("İki fonksiyonun karşılaştırması (Eğitim kümesi)")
    plt.savefig(os.path.join(savefolder, 'Egitim_Kumesi_Karsilastirma.png'))
    plt.close()

    # Aynı işlemler test kümesi için de yapılır.
    plt.figure()
    for node in testList:
        x = np.linspace(node.prev.index, node.index - 0.01, 100)
        m = (node.val - node.prev.val)/0.99
        normalized_y = node.prev.val + (x - node.prev.index)*m
        y = minval + normalized_y*(maxval - minval)
        y = y.reshape(100)
        plt.plot(x, y, color = 'blue')
    plt.title("Yaklaşılan fonksiyon (Test kümesi)")
    plt.savefig(os.path.join(savefolder, 'Gercek_Test_Kumesi.png'))
    plt.close()

    plt.figure()
    for node in testList:
        x = np.linspace(node.prev.index, node.index - 0.01, 100)
        m = (node.pred - node.prev.pred)/0.99
        normalized_y = node.prev.pred + (x - node.prev.index)*m
        y = minval + normalized_y*(maxval - minval)
        y = y.reshape(100)
        plt.plot(x, y, color = 'red')
    plt.title("Tahmin edilen fonksiyon (Test kümesi)")
    plt.savefig(os.path.join(savefolder, 'Tahmini_Test_Kumesi.png'))
    plt.close()

    plt.figure()
    for node in testList:
        x = np.linspace(node.prev.index, node.index - 0.01, 100)
        m = (node.pred - node.prev.pred)/0.99
        normalized_y = node.prev.pred + (x - node.prev.index)*m
        y = minval + normalized_y*(maxval - minval)
        y = y.reshape(100)
        plt.plot(x, y, color = 'red', label='Tahmin edilen')
        m = (node.val - node.prev.val)/0.99
        normalized_y = node.prev.val + (x - node.prev.index)*m
        y = minval + normalized_y*(maxval - minval)
        y = y.reshape(100)
        plt.plot(x, y, color = 'blue', label='Gerçekte olan')
    plt.title("İki fonksiyonun karşılaştırması (Test kümesi)")
    plt.savefig(os.path.join(savefolder, 'Test_Kumesi_Karsilastirma.png'))
    plt.close()

    # Ardından y(k) ve y(k-1) gerçek ve tahmini değerleri çekilir. Bu değerler kullanılarak
    # durum portreleri çizdirilir ve kaydedilir. 
    # Veriler çekilirken geri ölçeklendirme uygulanmaktadır.
    trainingList_x = []
    trainingList_y = []
    for node in trainList:
        trainingList_x.append(float(minval + node.prev.val*(maxval - minval)))
        trainingList_y.append(float(minval + node.val*(maxval - minval)))

    trainingListPred_x = []
    trainingListPred_y = []
    for node in trainList:
        trainingListPred_x.append(float(minval + node.prev.pred*(maxval - minval)))
        trainingListPred_y.append(float(minval + node.pred*(maxval - minval)))


    testingList_x = []
    testingList_y = []
    for node in testList:
        testingList_x.append(float(minval + node.prev.val*(maxval - minval)))
        testingList_y.append(float(minval + node.val*(maxval - minval)))

    testingListPred_x = []
    testingListPred_y = []
    for node in testList:
        testingListPred_x.append(float(minval + node.prev.pred*(maxval - minval)))
        testingListPred_y.append(float(minval + node.pred*(maxval - minval)))

    plt.figure()
    plt.plot(trainingListPred_x, trainingListPred_y, color = 'red')
    plt.title("Tahmini değerlerin durum portesi (Eğitim kümesi)")
    plt.savefig(os.path.join(savefolder, 'Egitim_Kumesi_Tahmini_Durum_Portresi.png'))
    plt.close()

    plt.figure()
    plt.plot(trainingList_x, trainingList_y, color = 'blue')
    plt.title("Gerçek değerlerin durum portesi (Eğitim kümesi)")
    plt.savefig(os.path.join(savefolder, 'Egitim_Kumesi_Gercek_Durum_Portresi.png'))
    plt.close()

    plt.figure()
    plt.plot(trainingListPred_x, trainingListPred_y, color = 'red', label='Tahmin edilen')
    plt.plot(trainingList_x, trainingList_y, color = 'blue', label='Gerçekte olan')
    plt.title("Tahmini ve gerçek değerlerin durum portrelerinin karşılaştırılması (Eğitim kümesi)")
    plt.savefig(os.path.join(savefolder, 'Egitim_Kumesi_Durum_Portresi_Karsilastirma.png'))
    plt.close()
    
    plt.figure()
    plt.plot(testingListPred_x, testingListPred_y, color = 'red')
    plt.title("Tahmini değerlerin durum portesi (Test kümesi)")
    plt.savefig(os.path.join(savefolder, 'Test_Kumesi_Tahmini_Durum_Portresi.png'))
    plt.close()

    plt.figure()
    plt.plot(testingList_x, testingList_y, color = 'blue', label='Gerçekte olan')
    plt.title("Gerçek değerlerin durum portesi (Test kümesi)")
    plt.savefig(os.path.join(savefolder, 'Test_Kumesi_Gercek_Durum_Portresi.png'))
    plt.close()

    plt.figure()
    plt.plot(testingListPred_x, testingListPred_y, color = 'red', label='Tahmin edilen')
    plt.plot(testingList_x, testingList_y, color = 'blue', label='Gerçekte olan')
    plt.title("Tahmini ve gerçek değerlerin durum portrelerinin karşılaştırılması (Test kümesi)")
    plt.savefig(os.path.join(savefolder, 'Test_Kumesi_Durum_Portresi_Karsilastirma.png'))
    plt.close()
    # plt.show()

# Yaklaşılması istenen fonksiyon.
def targetfunction(pval, ppval):
    # e_k değeri -0.0005-0.0005 arasında rastgele bir sayıdır.
    e_k = random.random()*0.001 - 0.0005
    val = pval*(0.8 - math.exp(-(pval**2))) - ppval*(0.3 + math.exp(-(pval**2))) + 0.1*math.sin(math.pi*pval) + e_k
    return val

# Test fonksiyonu.
def testNetwork(nodeList, network, prevx = 0):
    # Girilen bir veri kümesi (bu problem için düğüm kümesi) ile girilen ağ test edilir.
    # Test aşaması eğitimde ileri yol aşamasıyla aynıdır. Veriler ağa girilir ve çıkış elde edilir.
    # Elde edilen çıkışla hata hesaplanır. Hata tüm verilerde hesaplanıp bu hatanın ortalaması alınır.
    # Test esnasında ağın ağırlıkları güncellenmez.

    # Test kümesi, eğitimden hemen sonra geleceği için eğitimdeki son x(k) değerleri çekilir ve
    # testin başlangıcında x(k-1) olarak kullanılır.
    if not isinstance(prevx, np.ndarray):
        prevx = np.zeros((x_dim,1), dtype=float) + 0.5
    currx = np.zeros(x_dim, dtype=float).reshape(x_dim,1)
    Eort = 0
    for node in nodeList:
        j = 0
        data = np.array([node.prev.prev.prev.prev.prev.val, 
                        node.prev.prev.prev.prev.val, 
                        node.prev.prev.prev.val, 
                        node.prev.prev.val, 
                        node.prev.val], dtype=float).reshape(dim, 1)
        for neuron in network:
            x_j = neuron.forward(data, prevx)
            currx[j] = x_j
            if j == 0:
                w_y = neuron.w_y
            else:
                w_y = np.concatenate((w_y, neuron.w_y), axis=1)
            j += 1
        curry = np.dot(w_y, currx)
        e = node.val - curry
        E = (e**2)/2
        Eort += E
        node.pred = curry
        prevx = currx
    Eort = (Eort/len(nodeList)).reshape(1)
    return Eort, prevx

# Güncelleme fonksiyonu.
def update(network, error, prevx):
    # Ağırlıkların güncellendiği fonksiyon. Nöronlardan ağırlıklar çekilip güncellendikten sonra tekrar nöronlara yüklenir.
    init_step = 0
    for neuron in network:
        # Döngünün ilk aşamasında yeni matrisler oluşturulur. u, x, ve y ağırlıkları, 
        # gizli katman nöronlarının çıkışları (x), aktivasyon fonksiyonlarının türevleri,
        # ağa girilen veriler (u) ve ağın öğrenme hızı çekilir.
        if init_step == 0:
            w_u = neuron.w_u
            w_x = neuron.w_x
            w_y = neuron.w_y
            x_vec = np.array(neuron.activation, dtype=float).reshape(1,1)
            act_derivative_vec = np.array(neuron.activation_derivative, dtype=float).reshape(1,1)

            # Ağa girilen veriler ve öğrenme hızı yalnız ilk aşamada çekilir.
            u_vec = neuron.data
            learning_rate = neuron.learning_rate

            init_step += 1
        else:
            # Döngünün devamında ağırlıklar ve diğer değerler çekilmeye devam edilip bulunan matrisle birleştirilir.
            w_u = np.concatenate((w_u, neuron.w_u), axis=0)
            w_x = np.concatenate((w_x, neuron.w_x), axis=0)
            w_y = np.concatenate((w_y, neuron.w_y), axis=1)
            x_vec = np.concatenate((x_vec, np.array(neuron.activation, dtype=float).reshape(1,1)), axis=0)
            act_derivative_vec = np.concatenate((act_derivative_vec, np.array(neuron.activation_derivative, dtype=float).reshape(1,1)), axis=0)

    # Yeni ağırlıklar elde edilir.
    new_w_y = w_y + learning_rate * np.dot(error,np.transpose(x_vec))
    new_w_x = w_x + learning_rate * (np.dot(np.transpose(w_y),error) * np.dot(act_derivative_vec,np.transpose(prevx)))
    new_w_u = w_u + learning_rate * (np.dot(np.transpose(w_y),error) * np.dot(act_derivative_vec,np.transpose(u_vec)))
    
    i = 0
    # Yeni ağırlıklar, vektör olarak çekilerek nöronlara yerleştirilir ve güncelleme gerçekleşir.
    for neuron in network:
        w_u = np.array(new_w_u[i,:], dtype=float).reshape(1, dim)
        w_x = np.array(new_w_x[i,:], dtype=float).reshape(1, x_dim)
        w_y = np.array(new_w_y[:,i], dtype=float).reshape(y_dim, 1)

        neuron.updateN(w_u, w_x, w_y)

        i += 1

# Eğitim aşaması fonksiyonu.
def educationprocess(network, nodeList, epoch, eth, shuffle = True):
    prevx = np.zeros((x_dim,1), dtype=float) + 0.5
    currx = np.zeros(x_dim, dtype=float).reshape(x_dim,1)
    Eortprev = 0
    last_iter = epoch
    # 'epoch' iterasyonuna kadar eğitim yapılır.
    for i in range(epoch):
        Eort = 0
        # Eğitim yapılırken listenin karıştırılıp karıştırılmaması belirlenir. Diğer ağlarda her iterasyonda
        # eğitim kümesinin karıştırılması sağlıklı iken Elman ağında 'context unit' ile ağın eğitildiği bir önceki
        # veri önem taşımaktadır. Bu sebeple ağa girilen verinin sıralaması önem kazanmaktadır. Bu sadece 
        # spekülasyon olup karıştırılması ve karıştırılmaması durumunda test edilecektir.
        if shuffle == True:
            random.shuffle(nodeList)

        # Eğitim kümesi bu problem için düğümlerden oluşur.
        for node in nodeList:
            j = 0
            # Düğümlerden bir önceki düğümlerin değerleri, karesi ve ondan da bir önceki düğümün değeri çekilir.
            # Ağa girilecek veri bu şekilde atanır. u(k) = [y(k-2), y(k-1), y(k-1)^2]
            data = np.array([node.prev.prev.prev.prev.prev.val, 
                            node.prev.prev.prev.prev.val, 
                            node.prev.prev.prev.val, 
                            node.prev.prev.val, 
                            node.prev.val], dtype=float).reshape(dim, 1)
            # Ardından ağdaki gizli katman nöron çıkışları (x(k)) bulunur.
            for neuron in network:
                x_j = neuron.forward(data, prevx)
                currx[j] = x_j
                
                # Çıkış katmanı ağırlıkları da çekilip bir matriste toplanır.
                if j == 0:
                    w_y = neuron.w_y
                else:
                    w_y = np.concatenate((w_y, neuron.w_y), axis=1)
                j += 1

            # Ağ çıkışı ve hata bulunur.

            curry = np.dot(w_y, currx)
            node.pred = curry
            e = node.val - curry
            E = (e**2)/2

            # Bulunan hata ile ağ güncellenir. 
            update(network, e, prevx)

            # Hata, ortalama hata için kaydedilir. Gizli katman çıkışı (x(k)) 'context unit' olarak
            # kullanılmak üzere (x(k-1)) kaydedilir.
            Eort += E
            prevx = currx
        # Oralama hata bulunur.
        Eort = Eort/len(nodeList)
        # print("İterasyon ", i+1, " hata oranı: %", "%.7f" % (100*Eort[0][0]))
        # Ortalama hata ile seçilen hata limiti karşılaştırılır. Hata, önceki hataya göre yeterince değişmemiş veya
        # seçilen hata limitinin altına düşmüşse eğitim sonlandırılır.
        if (abs(Eortprev - Eort) < (Eort*eth)) or (Eort < eth):
            last_iter = i+1
            break
        Eortprev = Eort
        # print("İter:", i, "Hata:", Eort)
    # Eğitim sonunda son iterasyon ve son iterasyondaki ortalama hata döndürülür.
    return [last_iter, Eort]

# Ağ oluşturma fonksiyonu.
def generate_network(learning_rate, activation_type):
    network = []
    # Seçilen öğrenme hızı ve aktivasyon fonksiyonu ile yeni bir ağ oluşturulur.
    # Elman ağında aktivasyon fonksiyonunun bulunduğu tek aşama x(k) değerinin bulunduğu aşamadır.
    # Bu sebeple yalnızca gizli katmanda nöron oluşturulup diğer katmanların ağırlıkları da
    # bu nöronda saklanabilir.
    for i in range(x_dim):
        # x_dim, y_dim ve dim sırayla gizli katmandaki nöron sayısı, ağ çıkışı boyutu ve girişteki
        # verinin boyutu olup küresel değişkenlerdir. Fonksiyonun dışında belirlenir.
        new_Neuron = Neuron(dim, x_dim, y_dim, learning_rate, activation_type)
        network.append(new_Neuron)
    return network

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
# Girilen veri boyutu 3 seçilir. Eğitim için çıkış boyutu 1, gizli katman nöron sayısı 5 seçilir.
dim = 5
y_dim = 1
x_dim = 10

# Veriler ikinci ödevdekiyle aynı şekilde oluşturulur.
trainList = []
# Başlangıç koşulu oluşturulur ve y(0) = 0 seçilir.
yinit = YNode(0)
yinit.setnode(0.1, yinit)
yinit.pred = 0.1
yprev = yinit
# y(1)'den y(99)'a kadar veri oluşturulup eğitim kümesine atanır.
for i in range(1,100):
    y = YNode(i, yprev)
    yprev = y
    trainList.append(yprev)

# Test listesinde ise y(100)'den y(299)'a kadar olan değerler vardır.
testList = []
for i in range(100,300):
    y = YNode(i, yprev)
    yprev = y
    testList.append(yprev)

# Verilerin maksimum ve minimum değerleri elde edilir.
maxval = 0
minval = 0
for node in trainList:
    if node.val > maxval:
        maxval = node.val
    if node.val < minval:
        minval = node.val

# Veriler 0-1 aralığında ölçeklenir.
for node in trainList:
    node.val = (node.val - minval)/(maxval - minval)
for node in testList:
    node.val = (node.val - minval)/(maxval - minval)

# Hata limiti ve maksimum iterasyon sayısı ikinci ödevdekiyle 
# karşılaştırmak amacıyla aynı değerler alınır.
epoch = 1000
eth = 1e-4

# Öğrenme hızı 0.1 ve aktivasyon fonksiyonu sigmoid seçilir ve ağ oluşturulur.
network = generate_network(0.1, "sigmoid")

# Eğitim başlatılır.
start_time = time.time()
result = educationprocess(network, trainList[:], epoch, eth)
print("Eğitim süresi: %.3f saniye" % (time.time() - start_time))
print("Eğitim %d adımda yüzde %.7f hata oranı ile tamamlandı" % (result[0], 100*result[1]))

# Eğitilen ağ ile test yapılır. Test, hem eğitim hem test kümesi ile yapılmaktadır. Bunun sebebi
# veri düğümü içinde bulunan 'pred' değeridir. Veriler ve ağın tahmini çizilirken bu değer kullanılmaktadır.
# Ağ test aşamasında herhangi bir veri için çıkışı tahmin etmektedir.
edresult = testNetwork(trainList, network)
result = testNetwork(testList, network, edresult[1])
print("x = 100-299 için test edildi ve ortalama yüzde %.7f hata elde edildi." % (100*result[0][0]))
network.clear()
# Ağ çıkışı, eğitim kümesi verileri ve test kümesi verileri çizdirilip 'Liste_Karistirildiginda' klasörüne kaydedilir.
plotcomparison(trainList, testList, maxval, minval, os.path.join(__location__, 'Liste_Karistirildiginda'))
plt.close()

# Eğitim tekrar gerçekleştirilir. Bu sefer değişen tek şey veri kümesinin eğitim sırasında karıştırılmamasıdır.
# Bunun, ağın 'context unit'leri daha iyi kullanmasını sağlayacağı düşünülmektedir.
network = generate_network(0.1, "sigmoid")

start_time = time.time()
result = educationprocess(network, trainList, epoch, eth, shuffle = False)
print("Eğitim süresi: %.3f saniye" % (time.time() - start_time))
print("Eğitim %d adımda yüzde %.7f hata oranı ile tamamlandı." % (result[0], 100*result[1]))

edresult = testNetwork(trainList, network)
result = testNetwork(testList, network, edresult[1])
print("x = 100-299 için test edildi ve ortalama yüzde %.7f hata elde edildi." % (100*result[0][0]))
network.clear()
# Yine eğitilen ağ ile testler yapılır ve sonuçlar 'Liste_Karistirilmadiginda' klasörüne kaydedilir.
savefolder = os.path.join(__location__, 'Liste_Karistirilmadiginda')
plotcomparison(trainList, testList, maxval, minval, savefolder)
plt.close()

epoch = 3000

# Ağ, farklı öğrenme hızlarıyla eğitilip 5 iterasyonun ortalaması şeklinde sonuçlar alınır ve kaydedilir.
# Test edilen öğrenme hızları (0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999) şeklindedir.
# Ağın diğer değerleri (boyutlar, eth, epoch) önceki eğitimlerle aynıdır.
lrateT = (0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999)
learning_rate_results = np.zeros((len(lrateT),4), dtype=float)
for i in range(len(lrateT)):
    print("Öğrenme hızı %d. testi" % (i+1))
    eduiter = 0
    testacc = 0
    edutime = 0
    for j in range(5):
        print("İterasyon: %d" % (j+1))
        network = generate_network(lrateT[i], "sigmoid")

        start_time = time.time()
        result = educationprocess(network, trainList, epoch, eth, shuffle = False)
        edutime += time.time() - start_time
        eduiter += result[0]

        edresult = testNetwork(trainList, network)
        result = testNetwork(testList, network, edresult[1])

        network.clear()

        testacc += result[0][0]
    learning_rate_results[i,0] = lrateT[i]
    learning_rate_results[i,1] = eduiter/5
    learning_rate_results[i,2] = edutime/5
    learning_rate_results[i,3] = testacc/5

# Bu sefer öğrenme hızı 0.1 olarak sabit tutulup gizli katmandaki nöron sayısının değişimine göre
# test gerçekleştirilir ve benzer şekilde sonuçlar kaydedilir.
hncountT = (5, 10, 15, 20, 25, 30, 35, 40)
neuron_count_results = np.zeros((len(hncountT),4), dtype=float)
for i in range(len(hncountT)):
    print("Nöron sayısı %d. testi" % (i+1))
    eduiter = 0
    testacc = 0
    edutime = 0
    x_dim = hncountT[i]
    for j in range(5):
        print("İterasyon: %d" % (j+1))
        network = generate_network(0.1, "sigmoid")

        start_time = time.time()
        result = educationprocess(network, trainList, epoch, eth, shuffle = False)
        edutime += time.time() - start_time
        eduiter += result[0]

        edresult = testNetwork(trainList, network)
        result = testNetwork(testList, network, edresult[1])

        network.clear()

        testacc += result[0][0]
    neuron_count_results[i,0] = hncountT[i]
    neuron_count_results[i,1] = eduiter/5
    neuron_count_results[i,2] = edutime/5
    neuron_count_results[i,3] = testacc/5
# Elde edilen sonuçlar toplanıp 'Sonuclar' klasöründe 'Test_Sonuclari.xls' adı altında kaydedilir.
test_results = [learning_rate_results, neuron_count_results]
saveResults(test_results, 'Sonuclar', 'Test_Sonuclari.xls')


