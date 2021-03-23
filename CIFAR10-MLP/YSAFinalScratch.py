import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
import os
import pickle
import xlwt
from xlwt import Workbook 
from xlrd import open_workbook
import time

os.system('cls||clear')

class Neuron():
    # Nöron yapısı.
    def __init__(self, dims, learning_rate = 1e-2, dropout_rate = 1, momentum_rate = 0.9, bias = True, activation_type = "sigmoid"):
        # Nöronun parametreleri atanır. Bu parametrelerde giriş boyutu, öğrenme hızı, 
        # dropout oranı, momentum oranı, bias eklenip eklenmeyeceği ve aktivasyon fonksyionu bulunur.
        self.dims = dims
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.momentum_rate = momentum_rate
        self.bias = bias
        self.activation_type = activation_type

        # Nöronun başlangıç değerleri atanır.
        self.reset()

    def reset(self):
        # Nöronda ağırlık vektörü oluşturulur. Ayrıca momentum için kullanılacak, 
        # önceki ağırlıkların saklanacağı bir vektör oluşturulur.
        if self.bias is True:
            self.biasval = 1
            self.weights = np.zeros((1, self.dims + 1), dtype=float)
            self.weights[0,:] = np.random.rand(self.dims + 1)*0.1 - 0.05
            self.prevweights = np.zeros((1, self.dims + 1), dtype=float)
        else:
            self.weights = np.zeros((1, self.dims), dtype=float)
            self.weights[0,:] = np.random.rand(self.dims)*0.1 - 0.05
            self.prevweights = np.zeros((1, self.dims), dtype=float)
        # Giriş verisi, lineer kombinasyon ve aktivasyon için boş değişkenler atanır.
        self.data = np.zeros(self.dims)
        self.lin_comb = 0
        self.activation = 0
        # İnaktiflik durumu da "False" olup nöron aktif olarak başlatılır.
        self.dropped = False

    def forward(self, data):
        # Gizli katmanlar için ileri yol işlemi. Giriş verisi düzenlenir ve biasla birlikte vektör haline getirilir.
        if self.bias is True:
            self.data = np.concatenate((data, np.array([self.biasval],dtype=float).reshape(1,1)), axis = 0)
        else:
            self.data = np.array(data)
        w = self.weights
        # Ardından ağırlıklarla çarpılarak lineer kombinasyon elde edilir.
        self.lin_comb = np.dot(w, self.data)
        # if np.isinf(self.lin_comb):
        #     print("lincomb inf")
        # if np.isnan(self.lin_comb):
        #     print("lincomb nan")
        # Aktivasyon fonksiyonu ile de nöron çıkışı elde edilir.
        self.activation = self.activation_function(self.lin_comb)
        return self.activation

    def out_preforward(self, data):
        # Çıkış katmanı için ileri yolun başlangıç işlemi.
        # Veri düzenlenir ve bias değeri eklenir.
        if self.bias is True:
            self.data = np.concatenate((data, np.array([self.biasval],dtype=float).reshape(1,1)), axis = 0)
        else:
            self.data = np.array(data)
        w = self.weights
        # Lineer kombinasyon elde edilir.
        self.lin_comb = np.dot(w, self.data)
        # if np.isinf(self.lin_comb):
        #     print("lincomb inf")
        # if np.isnan(self.lin_comb):
        #     print("lincomb nan")
        # Katman boyunca toplanmak üzere, lineer kombinasyonun eksponansiyeli döndürülür.
        return np.exp(self.lin_comb)

    def out_postforward(self, exp_sum):
        # Çıkış katmanı için ileri yolun bitiş işlemi.
        # Toplam eksponansiyel değer ile ağ çıkışı elde edilir.
        self.activation = np.exp(self.lin_comb)/exp_sum
        return self.activation

    def activation_function(self, data):
        # Farklı aktivasyon fonksiyonları ve lokal gradyan hesaplanmasında
        # kullanılmak üzere fonksiyonların türevlerinin elde edilir.
        if self.activation_type == "tanh":
            y = tanh(data)
            self.activation_derivative = y[1]
            return y[0]
        elif self.activation_type == "sigmoid":
            y = sigmoid(data)
            self.activation_derivative = y[1]
            return y[0]
        elif self.activation_type == "relu":
            y = relu(data)
            self.activation_derivative = y[1]
            return y[0]
        elif self.activation_type == "lrelu":
            y = lrelu(data)
            self.activation_derivative = y[1]
            return y[0]
        else:
            return data

    def update(self, grad):
        # Nöronun ağırlıklarının güncellendiği fonksiyon.
        momentumterm = self.momentum_rate*(self.weights - self.prevweights)
        self.prevweights = self.weights
        # print(self.weights.shape)
        self.weights += ((self.learning_rate*grad[0])*self.data).reshape(1, self.weights.size) + momentumterm
        # print(self.weights.shape)

def tanh(x):
    # Tanjant hiperbolik aktivasyon fonksiyonu.
    if np.isinf((np.exp(x) + np.exp(-x))):
        return np.sign(x),0
    t = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    dt = 1-t**2
    return t,dt

def sigmoid(x):
    # Lojistik aktivasyon fonksiyonu.
    if np.isinf((np.exp(x) + np.exp(-x))):
        if x < 0:
            s = 0
        else:
            s = 1
        return s,0
    s = 1/(1 + np.exp(-x))
    ds = s*(1 - s)  
    return s,ds

def relu(x):
    # ReLU aktivasyon fonksiyonu.
    if x > 0:
        r = x
        dr = 1
    else:
        r = 0
        dr = 0
    return r,dr

def lrelu(x):
    # Leaky ReLU aktivasyon fonksiyonu.
    if x > 0:
        r = x
        dr = 1
    else:
        r = 0.01*x
        dr = 0.01
    return r,dr

def loadList(file):
    # Listeden veriler çekilir. Bu fonksiyon verisetinin belirttiği "unpickle"
    # fonksiyonunun modifiye edilmiş halidir. Başlıca değişiklikler; sınıf bilgisinin
    # [0-9] şeklinde değil de 10 boyutlu, sınıf indisinde 1 geri kalanında 0 olan bir
    # vektöre çevrilmesi, verilerin bir numpy dizisi yerine liste halinde çekilmesidir.
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
        # print(batch.keys()) 
    features = batch[b'data']
    labels = np.array(batch[b'labels'])
    labels = labels.reshape(labels.size,1)
    # print(labels.shape)
    # print(features.shape)
    init = 0
    for label in labels:
        if init == 0:
            labelArr = np.zeros((1,outdim))
            labelArr[0,label] = 1
            init = 1
        else:
            tempArr = np.zeros((1,outdim))
            tempArr[0,label] = 1
            labelArr = np.concatenate((labelArr, tempArr),axis=0)
    dataArr = np.concatenate((features, labelArr), axis=1)

    print(batch[b'batch_label'], " loaded")
    return dataArr.tolist()

def saveResults(results, savefolder, filename):
    # Elde edilen sonuçlar belirtilen excel dosyasında yazdırılır.
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    filedir = os.path.join(savefolder,filename)
    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet1')
    col = 0
    row = 0
    sheet1.write(row, col, 'Test Number')
    sheet1.write(row, col+1, 'Neuron Count (Hidden Layer 1)')
    sheet1.write(row, col+2, 'Neuron Count (Hidden Layer 2)')
    sheet1.write(row, col+3, 'Learning Rate')
    sheet1.write(row, col+4, 'Momentum Rate')
    sheet1.write(row, col+5, 'Dropout Rate')
    sheet1.write(row, col+6, 'Activation Function')
    sheet1.write(row, col+7, 'Education Iteration')
    sheet1.write(row, col+8, 'Education Time')
    sheet1.write(row, col+9, 'Education Time-per-iteration')
    sheet1.write(row, col+10, 'Education Error')
    sheet1.write(row, col+11, 'Test Prediction Error')
    sheet1.write(row, col+12, 'Test Cross Entropy Error')
    row += 1
    for result in results:
        for i in range(len(result)):
            sheet1.write(row, col+i, result[i])
        row +=1 

    wb.save(filedir) 

def plot_conf_mat(mat, savefolder, show = False):
    # "Confusion Matrix" çiziminin yapıldığı fonksyion.
    # X ve Y'deki indislerin belirttiği sınıfların isimleri bir listede toplanır.
    label = [" ", "Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
    # 12.8x9.6 inçlik bir çizim açılır.
    fig = plt.figure(figsize=(12.8, 9.6))
    ax = fig.add_subplot(111)
    # Çizimde "tick"lerdeki yazıların boyutları okunabilmesi için 10 punto seçilir.
    ax.tick_params(axis='both', labelsize=10)
    # "Blues" renk haritası kullanılarak matris çizdirilir.
    cax = ax.matshow(mat, interpolation='nearest', cmap=plt.cm.Blues)
    # "tick"lerdeki etiketler atanır.
    ax.set_xticklabels(label)
    ax.set_yticklabels(label)
    # Renk barı çizdirilir.
    plt.colorbar(cax)
    # Eksenlerin etiketleri yazdırılır.
    plt.title("Confusion Matrix")
    ax.set_ylabel('Truth')
    ax.set_xlabel('Predicted')
    ax.xaxis.set_label_position('top') 
    # Matris içindeki renklerin değerleri, hücrelere yazdırılır.
    for (i, j), z in np.ndenumerate(mat):
        ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center', fontsize=8)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # show = True seçilmesi durumunda eğitim sonrasında matris figür olarak çizdirilir. 
    # Aksi takdirde sadece kaydedilir.
    if show == True:
        plt.show()
    # Figür kaydedilir.
    plt.savefig(os.path.join(savefolder, 'Confusion_Matrix.png'))
    plt.close()

def create_2h_network(data_size, l1_size, l2_size, lo_size, learning_rate, dropout_rate, momentum_rate, bias, function):
    # İki gizli katmana sahip bir çok katmanlı algılayıcı ağı üreten fonksiyon.
    l1_List = []
    l2_List = []
    lo_List = []
    # Öğrenme hızı, biasın eklenip eklenmemesi, momentum oranı, ağdaki nöron sayısı gibi argümanlar kullanılır.
    # Dropout oranı ve aktivasyon fonksiyonu gizli katmanlarda kullanılacaktır ve çıkış katmanında sabittir.
    for i in range(l1_size):
        l1_List.append(Neuron(data_size, learning_rate, dropout_rate, momentum_rate, bias, function))
    for i in range(l2_size):
        l2_List.append(Neuron(l1_size, learning_rate, dropout_rate, momentum_rate, bias, function))
    for i in range(lo_size):
        lo_List.append(Neuron(l2_size, learning_rate, 1, momentum_rate, bias))
    # Oluşturulan katmanlar bir liste halinde döndürülür.
    Layers = np.array([l1_List, l2_List, lo_List])
    return Layers

def local_grad(list1, list2, nextgrad):
    # Lokal gradyanın hesaplandığı fonksiyon.
    ncount = len(list1)
    act_derivative = np.zeros((ncount, 1), dtype=float)
    grad = np.zeros((ncount, 1), dtype=float)
    # Ağırlık matrisinin transpozu alınır ve bias değerlerinin ağırlıkları çıkarılır.
    wrow = (list2[0].weights.size) - 1
    outcount = len(list2)
    wmat = np.zeros((wrow, outcount), dtype=float)
    for i in range(outcount):
        wmat[:, i] = list2[i].weights[0, 0:wrow]
    # Aktivasyonların türevleri bir vektör halinde çekilir.
    for i in range(ncount):
        act_derivative[i] = list1[i].activation_derivative
    # Lokal gradyan hesaplanırken sonraki katmanın lokal gradyanı da kullanılır.
    # grad(k) = [wT.grad(k+1)] x f'(v(k)) şeklinde ifade edilebilir.
    grad = np.multiply(np.dot(wmat, nextgrad), act_derivative)
    return grad

def local_grad_out(y, clsdata):
    # Softmax ve cross entropy birlikte kullanıldığında elde edilen gradyan değeri.
    return (clsdata - y)

def testdata(data, clsdata, network):
    # Eğitimde gerçekleştirilen ileri yol ve hata hesaplaması burada da yapılır.
    # Tüm ağ değerlendirildiği için tüm nöronlar hep aktiftir.
    y1 = np.zeros((len(network[0]), 1), dtype=float)
    y2 = np.zeros((len(network[1]), 1), dtype=float)
    yo = np.zeros((len(network[2]), 1), dtype=float)
    i1 = 0
    i2 = 0
    io = 0
    for Neuron in network[0]:
        y1[i1] = Neuron.forward(data)
        i1 += 1
    for Neuron in network[1]:
        y2[i2] = Neuron.forward(y1)
        i2 += 1

    exp_sum = 0
    for Neuron in network[2]:
        exp_sum += Neuron.out_preforward(y2)
    for Neuron in network[2]:
        yo[io] = Neuron.out_postforward(exp_sum)
        io += 1
    E = 0
    for i in range(yo.size):
        E -= clsdata[i]*np.log(yo[i,0])

    return yo, E.item(0)

def testprocess(dataList, network, dim, conf_mat):
    # Test aşamasında ileri yol izlenir ve tahmin edilen sınıfla gerçek sınıf karşılaştırılır.
    # Tahmin edilen ve gerçek değerlerin bulunduğu bir matris oluşturulur. Sonuç olarak hatalı tahmin sayısı,
    # test kümesinin büyüklüğü, hata oranı ve oluşturulan "Confusion Matrix" matrisi döndürülür.
    Errorcount = 0
    totalcount = 0
    random.shuffle(dataList)
    Loss = 0
    for data in dataList:
        imdata = np.array(data[:dim],dtype=float).reshape(dim,1)
        clsdata = np.array(data[dim:],dtype=float).reshape(len(data) - dim,1)
        # İleri yol gerçekleştirilip hata ve ağ çıkışı elde edilir.
        [pred, E] = testdata(imdata, clsdata, network)
        # Ağ çıkışındaki maksimum değer 1 olacak şekilde ölçeklenir.
        pred = pred/np.amax(pred, axis = 0)
        # Ağ çıkışında ve sınıf verisinde 1 değerinin bulunduğu indis kaydedilir.
        for i in range(pred.size):
            if pred[i] == float(1):
                pred_ind = i
            if clsdata[i] == float(1):
                cls_ind = i
        # "Confusion Matrix" içinde bu iki indis kullanılarak eşleşen hücre artırılır.
        conf_mat[cls_ind, pred_ind] += 1
        Loss += E
        # Eğer tahmin edilen ve gerçek sınıfın indisleri uyuşmuyorsa hata sayısına bir eklenir.
        if pred_ind != cls_ind:
            Errorcount += 1
        totalcount += 1
    Loss = Loss/len(dataList)
    result = np.array([Errorcount, totalcount, Loss, conf_mat])
    return result

def educate(data, network, clsdata):
    # Tek bir veri için eğitimin gerçekleştiği fonksiyon.
    # Eğitim sırasında kullanılacak iterasyon değişkenleri
    # ve katman çıkışları dizileri oluşturulur.
    y1 = np.zeros((len(network[0]), 1), dtype=float)
    y2 = np.zeros((len(network[1]), 1), dtype=float)
    yo = np.zeros((len(network[2]), 1), dtype=float)
    i1 = 0
    i2 = 0
    io = 0
    # Gizli katmanlar için ileri yol izlenir ve katman çıkışı elde edilir.
    # Dropout oranına göre nöronların inaktif olma ihtimali bulunmaktadır.
    # Dropout oranı kesin bir oran vermemekte, [0,1) aralığında seçilen bir
    # sayının bu orandan büyük olup olmaması gözlenecektir.
    # İnaktif bir nöronun çıkışı 0 alınır ve inaktif olduğu kaydedilir.
    # Çıkış katmanında dropout oranı her zaman 1'dir, tüm nöronlar hep aktiftir.
    for Neuron in network[0]:
        if np.random.rand() > Neuron.dropout_rate:
            y1[i1] = 0
            Neuron.dropped = True
            Neuron.activation_derivative = 0
            i1 += 1
            continue
        y1[i1] = Neuron.forward(data)
        i1 += 1
    for Neuron in network[1]:
        if np.random.rand() > Neuron.dropout_rate:
            y2[i2] = 0
            Neuron.dropped = True
            Neuron.activation_derivative = 0
            i2 += 1
            continue
        y2[i2] = Neuron.forward(y1)
        i2 += 1

    # Çıkış katmanında softmax fonksiyonu kullanılacağı için öncelikle tüm nöronların
    # w*x + b = v değerleri, yani lineer kombinasyonları hesaplanmalıdır. Bu değerlerin ekponansiyelleri
    # toplanacak ve nöron çıkışı için kullanılacaktır. Bu nedenle ileri yol iki parçaya ayrılmıştır.
    # İlk parçada lineer kombinasyonlar bulunup ekponansiyelleri toplamı elde edilir.
    exp_sum = 0
    for Neuron in network[2]:
        exp_sum += Neuron.out_preforward(y2)
    # İkinci parçada ise bu toplam kullanılarak nöron çıkışları hesaplanır ve kaydedilir.
    for Neuron in network[2]:
        yo[io] = Neuron.out_postforward(exp_sum)
        io += 1

    # Hata hesaplanması için cross entropy kullanılır. Ağın mükemmel bir şekilde sınıflandırması
    # durumunda doğru sınıfların çıkışlarında 1 görüleceği için hata E = ln(1) = 0 olacaktır.
    # Örnek olarak başlangıç durumunda ise yaklaşık 0.1 çıkışları olacağı için E = ln(0.1) = 2.3 olacaktır.
    E = 0
    for i in range(yo.size):
        E -= clsdata[i]*np.log(yo[i,0])

    # İleri yolun ardından gradyanlar için boş diziler açılır.
    grad1 = np.zeros((len(network[0]), 1), dtype=float)
    grad2 = np.zeros((len(network[1]), 1), dtype=float)
    grado = np.zeros((len(network[2]), 1), dtype=float)
    
    # Lokal gradyanlar hesaplanır. Gizli katmanlarda gradyan hesabı ortalama karesel hata kullanıldığı
    # geriye yönelim durumuyla aynıdır. Çıkış katmanında ise işlem kolaylığı sağlayan cross entropy ve softmax
    # fonksiyonlarının birlikte kullanıldığında elde edilen bir gradyan hesaplaması kullanılır.
    grado = local_grad_out(yo, clsdata)
    grad2 = local_grad(network[1], network[2], nextgrad = grado)
    grad1 = local_grad(network[0], network[1], nextgrad = grad2)

    # Hesaplanan lokal gradyanlara göre nöronların ağırlıkları güncellenir. 
    # İnaktif nöronlar güncelleme aşamasında atlanır ve bir sonraki iterasyon için tekrar aktif olurlar.
    i1 = 0
    i2 = 0
    io = 0
    for Neuron in network[0]:
        if Neuron.dropped == True:
            Neuron.dropped = False
            i1 += 1
            continue
        Neuron.update(grad1[i1])
        i1 += 1
    for Neuron in network[1]:
        if Neuron.dropped == True:
            Neuron.dropped = False
            i2 += 1
            continue
        Neuron.update(grad2[i2])
        i2 += 1
    for Neuron in network[2]:
        Neuron.update(grado[io])
        io += 1
    
    # Elde edilen hata döndürülür.
    return E.item(0)

def educationprocess(dataList, network, epoch, errorthreshold, dim, savefolder, testiter):
    # Eğitim işleminin gerçekleştiği fonksiyon.
    # Öncelikle sonuç elde etmek ve çizim yapılabilmesi için boş listeler ve diziler atanır.
    y = []
    result = np.zeros(2)
    result[0] = epoch
    # Eğitimde en iyi performansı gösteren ağın çıkması sağlanacaktır. Bunun için başlangıçtaki
    # en iyi hatayı gösterecek "bestE"; yüksek bir değer, bu durum için 999, atanır.
    # "bestE_streak" ise ağın bulunan minimuma yakın şekilde eğitilmesi için kullanılır.
    # 8 iterasyonda ağ gelişmemişse en düşük hatayı elde ettiği duruma geri döner ve eğitime
    # devam eder.
    bestE_streak = 0
    bestE = 999
    best_network = network[:]
    for i in range(epoch):
        # Her iterasyonun başında eğitim kümesi karıştırılır.
        random.shuffle(dataList)
        Eort = 0
        # Eğitim kümesindeki her veri için ileri yol ve geri yol gerçekleşir.
        for data in dataList:
            # Giriş verisi "imdata" ve sınıf verisi "clsdata" ayrılır ve numpy dizisi haline getirilir.
            imdata = np.array(data[:dim],dtype=float).reshape(dim,1)
            clsdata = np.array(data[dim:],dtype=float).reshape(len(data) - dim,1)
            # Veri için ağ eğitilir.
            E = educate(imdata, network, clsdata)
            # Tüm hatalar toplanır. Tüm veriler için eğitim gerçekleştiğinde hataların ortalaması alınır.
            Eort += E
        Eort = Eort/len(dataList)
        # Eğer ortalama hata ağın en iyi durumundaki hatadan daha iyiyse yeni en iyi ağ yapısı olarak kaydedilir.
        if Eort < bestE:
            if bestE == 999:
                print("%d iteration, %.4f error" % (i + 1, Eort))
            else:
                print("%d iteration, %.4f error, improved from %.4f error" % (i + 1, Eort, bestE))
            best_network = network[:]
            bestE_streak = 0
            bestE = Eort
        # Aksi takdirde ise, 8 kere ağ gelişmemişse, en iyi olduğu duruma geri döner ve eğitime devam eder.
        else:
            if bestE_streak < 8:
                print("%d iteration, %.4f error" % (i + 1, Eort))
                bestE_streak += 1
            else:
                print("%d iteration, %.4f error, reverted to best performing network with %.4f error" % (i + 1, Eort, bestE))
                network = best_network[:]
                bestE_streak = 0
        # Eğitim sırasındaki hata değişimi çizdirilmek üzere kaydedilir.
        y.append(Eort)
        # Ortalama hata, istenen hata limitinin altına düşmüşse eğitim durdurulur.
        if Eort < errorthreshold:
            result[0] = i + 1
            break
    # Eğitim bittikten sonra test edilmek üzere en iyi durumdaki ağ yapısı seçilir.
    network = best_network[:]

    # Eğitim sırasındaki ortalama hata değişimi çizdirilir.
    plt.figure()
    yshape = int(round(result[0]))
    x = np.arange(1, yshape + 1)
    plt.plot(x, np.array(y).reshape(yshape), color = 'blue')
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.title("Error during education")
    plt.savefig(os.path.join(savefolder, 'Test' + str(testiter) + '.png'))
    plt.close()

    # Elde edilen en iyi hata döndürülür.
    result[1] = bestE
    return result

def normalize_List(dataList, dim):
    # Veriler RGB değerlere sahip 32x32'lik resimlerin pixel bilgilerinden oluşur.
    # RGB değerleri 0-255 aralığında olup işlem yapılırken overflow yaşanmaması için
    # 0-1 aralığında ölçeklenir.
    for data in dataList:
        data[:dim] = [x / 255 for x in data[:dim]]
    return dataList

def batchset(dataList, batch_size, dim, class_count):
    # Veri kümesini daha küçük kümelere ayıran fonksiyon. "batch_size" değeri
    # tek bir sınıftan kaç verinin alınacağını belirler. Girilen bir veri listesinden
    # her sınıftan "batch_size" sayısında veri yeni bir listeye eklenip döndürülür.
    newList = []
    batchArr = np.zeros(class_count)
    random.shuffle(dataList)
    for data in dataList:
        clsdata = np.array(data[dim:],dtype=float).reshape(class_count,1)
        for i in range(class_count):
            if clsdata[i] == float(1) and batchArr[i] < batch_size:
                newList.append(data)
                batchArr[i] += 1
    return newList

# Kodun bulunduğu adres, girişteki veri boyutu ve çıkış boyutu atanır.
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
dataFile = os.path.join(__location__, 'Data')
dim = 3072
outdim = 10

# Eğitim ve test kümeleri liste halinde çekilir. Bu işlem sırasında 32x32'lik renkli resimler 
# tek vektör olarak ve [0,1] aralığında ölçeklenerek listelere çekilir. Ek olarak veri setinde
# resim sınıfları [0-9] aralığındaki sayılardan oluşmaktadır. Bu sayılar, (10,1)'lik bir vektörde
# sınıf indisi 1 olup geri kalanlarının 0 olduğu bir vektöre dönüştürülür ve verinin sonuna eklenir.
trainList1 = normalize_List(loadList(os.path.join(dataFile, 'data_batch_1')), dim)
testList = normalize_List(loadList(os.path.join(dataFile, 'test_batch')), dim)

# batch_size: her sınıftan kaç resmin eğitim ve test veri setlerinde bulundurulacağı bilgisi.
# Eğitim kümesinde 10000 resim olduğu için çok uzun süre eğitim gerçekleşir. Ayrıca eğitim sırasında
# ağın bir sınıfa uzun süre rastlamaması ihtimali de bulunmaktadır. Çok büyük bir eğitim kümesi kullanıldığında
# Eğitim yavaşlar ve verimsizleşir. Bu nedenle her sınıftan batch_size = 200 resim bulundurmak üzere toplam 
# 2000 resimden oluşan bir eğitim kümesi oluşturulur.
batch_size = 200
trainList = batchset(trainList1[:], batch_size, dim, outdim)

# Maksimum iterasyon sayısı ve eğitimin durdurulacağı hata limiti seçilir.
epoch = 50
eth = 1e-2

# Sonuçların toplanacağı klasör atanır.
resultfolder = os.path.join('Scratch', 'Results')

# Öğrenme hızının farklı değerleri için eğitim ve test gerçekleştirilir. Her test 5 kere gerçekleştirilip
# yorumlanmak üzere sonuçların ortalaması alınacaktır. Her testin sonucu kaydedilir.
# Seçilen öğrenme hızları ve {0.0001, 0.001, 0.01, 0.1} şeklindedir. 0.1 öğrenme hızından sonrasında ağın eğitilmediği
# gözlemlenmiştir. 0.0001 durumunda da ağın fazla yavaş eğitilip yüksek lokal minimum noktalarından kurtulamadığı görülür.
results = []
lrateT = (0.0001, 0.001, 0.01, 0.1)
for i in range(4):
    # Ağ özellikleri atanır. Gizli katman nöron sayıları, öğrenme hızı, dropout oranı, momentum oranı, gizli katmanların
    # kullanacağı aktivasyon fonksiyonu seçilir. 
    # Çıkış katmanı Softmax fonksiyonu kullanıp gizli katmanlar ReLU kullanacaktır.
    # Öğrenme hızı, dropout oranı ve momentum oranının standart değerleri sırasıyla 0.01, 0.8 ve 0.8'dir. Oranların değiştiği
    # testlerde diğer oranlar bu sayılarda sabit tutulmuştur.
    # Dropout rate katmanda eğitim sırasında aktif tutulan nöron oranının bilgisini verir. drate = 0.8, nöronların yaklaşık
    # %20'sinin inaktif olduğunu söyler.
    h1n = 128
    h2n = 32
    lrate = lrateT[i]
    drate = 0.8
    mrate = 0.8
    act_func = 'relu'
    # Ağın performansının incelenmesi için "Confusion matrix" matrisi atanır.
    conf_mat = np.zeros((outdim, outdim), dtype=float)

    # Test sonuçlarının kaydedileceği klasör oluşturulur.
    savefolder = os.path.join(resultfolder, 'Learning_Rate_' + str(lrate))
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)

    # Her test 3 kez tekrarlanıp sonuçları kaydedilir.
    for j in range(3):
        # Daha önce belirlenmiş değerlerle iki gizli katmanlı ağ oluşturulur. 
        network = create_2h_network(dim, h1n, h2n, outdim, lrate, drate, mrate, True, act_func)

        # Eğitim başlatılır ve eğitim süresi, hata oranı gibi sonuçlar kaydedilir ve yazdırılır.
        start_time = time.time()
        result = educationprocess(trainList, network, epoch, eth, dim, savefolder, j)
        ed_time = time.time() - start_time
        print("Education time: %f" % ed_time)
        print("Training is over in %d steps with %.4f error with the best performing network." % (result[0], result[1]))

        # Eğitilen ağ ile test kümesindeki resimlerin tahmin edilmesi sağlanır. Doğru ve tahmin edilen sınıflar,
        # yanlış tahmin oranı değerleri kaydedilir.
        testresult = testprocess(testList, network, dim, conf_mat)
        print("Test is over with %.4f error" % testresult[2])
        print("Made %d wrong predictions out of %d patterns" % (testresult[0], testresult[1]))
        testpredictionerror = testresult[0]/testresult[1]
        
        # Eğitim ve test sonuçları sonuç listesine eklenir.
        results.append([j+1, h1n, h2n, lrate, mrate, drate, act_func, result[0], ed_time, ed_time/result[0], result[1], testpredictionerror, testresult[2]])
    
    # "Confusion Matrix" içindeki değerlerin 0-1 aralığına alınması sağlanır. Bu değerler
    # aynı zamanda birçok testin ortalamasıdır. Daha sonra bu grafik çizdirilirve kaydedilir.
    # Kod çalışırken çizimin görülmesi için plot_conf_mat() fonksiyonuna show = True argümanı eklenmelidir. 
    conf_mat = conf_mat/(len(testList)*5)
    plot_conf_mat(conf_mat, savefolder)
# Test sonuçları kaydedilir.
saveResults(results, resultfolder, 'Results_LRate.xls')

# Momentumun farklı değerleri için, öğrenme hızı 0.01 ve dropout oranı 0.8'ken 
# momentuma {0, 0.2, 0.4, 0.6, 0.8, 1} kümesinden veriler atanarak ağ eğitilir ve test edilir.
# mrate = 0 değeri momentumun olmaması durumuna denk gelmektedir. Ağ yapısı, eğitim ve test aşamaları
# öğrenme hızı testinden farksızdır.
results = []
mrateT = (0, 0.2, 0.4, 0.6, 0.8, 1)
for i in range(6):
    h1n = 128
    h2n = 32
    lrate = 0.01
    drate = 0.8
    mrate = mrateT[i]
    act_func = 'relu'
    conf_mat = np.zeros((outdim, outdim), dtype=float)
    
    savefolder = os.path.join(resultfolder, 'Momentum_Rate_' + str(mrate))
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)

    for j in range(3):
        network = create_2h_network(dim, h1n, h2n, outdim, lrate, drate, mrate, True, act_func)

        start_time = time.time()
        result = educationprocess(trainList, network, epoch, eth, dim, savefolder, j)
        ed_time = time.time() - start_time
        print("Education time: %f" % ed_time)
        print("Training is over in %d steps with %.4f error with the best performing network." % (result[0], result[1]))

        testresult = testprocess(testList, network, dim, conf_mat)
        print("Test is over with %.4f error" % testresult[2])
        print("Made %d wrong predictions out of %d patterns" % (testresult[0], testresult[1]))
        testpredictionerror = testresult[0]/testresult[1]
        
        results.append([j+1, h1n, h2n, lrate, mrate, drate, act_func, result[0], ed_time, ed_time/result[0], result[1], testpredictionerror, testresult[2]])
    
    conf_mat = conf_mat/(len(testList)*5)
    plot_conf_mat(conf_mat, savefolder)
saveResults(results, resultfolder, 'Results_MRate.xls')

# Dropout oranının farklı değerleri için, öğrenme hızı 0.01 ve momentum oranı 0.8'ken 
# dropout oranına {0.5, 0.6, 0.7, 0.8, 0.9, 1} kümesinden veriler atanarak ağ eğitilir ve test edilir.
# drate = 1 değeri inaktif nöronun olmaması durumuna denktir. Ağ yapısı, eğitim ve test aşamaları
# önceki testlerden farksızdır.
results = []
drateT = (0.5, 0.6, 0.7, 0.8, 0.9, 1)
for i in range(6):
    h1n = 128
    h2n = 32
    lrate = 0.01
    drate = drateT[i]
    mrate = 0.8
    act_func = 'relu'
    conf_mat = np.zeros((outdim, outdim), dtype=float)
    
    savefolder = os.path.join(resultfolder, 'Dropout_Rate_' + str(drate))
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)

    for j in range(3):
        network = create_2h_network(dim, h1n, h2n, outdim, lrate, drate, mrate, True, act_func)

        start_time = time.time()
        result = educationprocess(trainList, network, epoch, eth, dim, savefolder, j)
        ed_time = time.time() - start_time
        print("Education time: %f" % ed_time)
        print("Training is over in %d steps with %.4f error with the best performing network." % (result[0], result[1]))

        testresult = testprocess(testList, network, dim, conf_mat)
        print("Test is over with %.4f error" % testresult[2])
        print("Made %d wrong predictions out of %d patterns" % (testresult[0], testresult[1]))
        testpredictionerror = testresult[0]/testresult[1]
        
        results.append([j+1, h1n, h2n, lrate, mrate, drate, act_func, result[0], ed_time, ed_time/result[0], result[1], testpredictionerror, testresult[2]])
    
    conf_mat = conf_mat/(len(testList)*5)
    plot_conf_mat(conf_mat, savefolder)
saveResults(results, resultfolder, 'Results_DRate.xls')
