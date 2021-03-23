import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib import interactive
from mpl_toolkits.mplot3d import Axes3D
import os
import pickle
import xlwt
from xlwt import Workbook 
from xlrd import open_workbook
import time

os.system('cls||clear')

# Nöron sınıfı.
class NeuronCls():
    # Her nöron oluşturulduğunda (x,y) indisine, başlangıç öğrenme hızına, 
    # w boyutuna ve kendinden sonra gelen nöronun göstergesine sahiptir.
    def __init__(self, x, y, dimension, learning_rate):
        self.x = x
        self.y = y
        self.init_learning_rate = learning_rate
        self.dimension = dimension
        self.nextNeuron = self

        self.reset()
    
    # Nöron ilk oluşturulduğunda ağırlık vektör değerleri (-0.1,0.1) aralığında
    # uniform dağılımlı bir şekilde atanır.
    def reset(self):
        self.weights = np.zeros((1, self.dimension), dtype=float)
        for i in range(self.dimension):
            self.weights[0][i] = 0.2*np.random.rand() - 0.1

    # Ağ oluşturulurken nöronlar birbirine bağlanır.
    def next(self, Neuron):
        self.nextNeuron = Neuron

    # Eğitim esnasında kazanan nöronun (x,y) indislerine göre ağdaki tüm nöronlar güncellenir.
    def update(self, winner_x, winner_y, dev, data, learning_adaptive):
        key = (abs(self.x - winner_x), abs(self.y - winner_y))
        # Güncelleme esnasında kazanana uzaklığa ve eğitim iterasyonuna göre bir komşuluk etkisi hesaplanır.
        # Bu etki normal dağılımlıdır ve özdüzenleme aşamasında her iterasyonda daralmaktadır. 
        dist = distDict[key]
        self.hdist = math.exp(-(dist**2)/(2*(dev**2)))
        # Öğrenme hızı da özdüzenleme aşamasında her iterasyonda azalmaktadır. Adaptif bir terimin başlangıçtaki
        # öğrenme hızıyla çarpımıyla bulunur.
        learning_rate = self.init_learning_rate*learning_adaptive
        # print(self.weights, self.x, self.y)
        # Ağırlıklar, veriye yaklaşacak şekilde güncellenir. Kazanan nöron ve yakınındakiler daha fazla yaklaşmaktadırlar.
        self.weights = self.weights + learning_rate*self.hdist*(data - self.weights)
        # print(self.weights, self.x, self.y)

# Veriyi kaydetip yüklemek için kullanılan fonksiyonlar.
def saveData(Data, File):
    with open(File, 'wb') as f:
        pickle.dump(Data, f)

def loadList(File):
    DataList = []
    with open(File, 'rb') as f:
        DataList = pickle.load(f)
    return DataList

# Nöronların kazandığı veri sınıflarını excel dosyasına kaydeden fonksiyon.
def saveResults(result, xdim, ydim, filedir):
    if os.path.exists(filedir):
        wb = open_workbook(filedir)
        sheet1 = wb.sheet_by_index(0)
    else:
        wb = Workbook()
        sheet1 = wb.add_sheet('Sheet1')
    sheet1.write(0, 0, 'Nöron (x,y)') 
    sheet1.write(0, 1, 'Setosa')
    sheet1.write(0, 2, 'Versicolor')
    sheet1.write(0, 3, 'Virginica')
    for i in range(xdim):
        for j in range(ydim):
            row = i*ydim + j + 1
            sheet1.write(row, 0, str((i,j)))
            for k in range(3):
                sheet1.write(row, (k+1), result[i,j][k])
    wb.save(filedir) 

# Ağ ve verileri çizdirme fonksiyonları.
def plotnetworkanddata(network, xdim, ydim, xind, yind, xlabel, ylabel, title, dList1, dList2 = [], dList3 = [], single_list = True):
    # Ağ çizdirme ve veri çizdirme fonksiyonlarının birleşimi.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    if single_list == False:
        for data in dList1:
            ax.scatter(data[xind], data[yind], s=5, c="red")
        for data in dList2:
            ax.scatter(data[xind], data[yind], s=5, c="blue")
        for data in dList3:
            ax.scatter(data[xind], data[yind], s=5, c="green")
    else:
        for data in dList1:
            ax.scatter(data[xind], data[yind], s=5, c="magenta")
    
    for Neuron in network:
        weights = Neuron.weights[0]
        ax.scatter(weights[xind], weights[yind], s=10, c="black")
    for Neuron in network:
        weights = Neuron.weights[0]
        nextNeuron = Neuron.nextNeuron
        next_weights1 = nextNeuron.weights[0]

        xcord1 = np.linspace(weights[xind], next_weights1[xind], 5)
        ycord1 = np.linspace(weights[yind], next_weights1[yind], 5)

        nextNeuron = Neuron
        for i in range(ydim):
            nextNeuron = nextNeuron.nextNeuron
        next_weights2 = nextNeuron.weights[0]

        xcord2 = np.linspace(weights[xind], next_weights2[xind], 5)
        ycord2 = np.linspace(weights[yind], next_weights2[yind], 5)
        if Neuron.y != ydim - 1:
            ax.plot(xcord1, ycord1, linewidth=1.2, c='0.12')
        if Neuron.x != xdim - 1:
            ax.plot(xcord2, ycord2, linewidth=1.2, c='0.12')


    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def plot2ddata(title, xind, yind, xlabel, ylabel, dList1, dList2 = [], dList3 = [], single_list = True):
    # Verilerin girilen indislerini iki boyutlu düzlemde çizdirir. Eğer üç farklı veri listesi girilmişse 
    # bu verileri kırmızı, mavi ve yeşil renkte çizdirir. Tek liste girilmişse tüm verileri mor/pembe renkte çizdirir.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    if single_list == False:
        for data in dList1:
            ax.scatter(data[xind], data[yind], s=5, c="red")
        for data in dList2:
            ax.scatter(data[xind], data[yind], s=5, c="blue")
        for data in dList3:
            ax.scatter(data[xind], data[yind], s=5, c="green")
    else:
        for data in dList1:
            ax.scatter(data[xind], data[yind], s=5, c="magenta")
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def plotnetwork(network, xdim, ydim, xind, yind, xlabel, ylabel, title):
    # Önceki problemdeki ağ çizdirme fonksiyonuyla hemen hemen aynı şekilde çalışır. Önce ağdaki nöronları,
    # daha sonra nöronlar arasındaki bağları çizdirir.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    for Neuron in network:
        weights = Neuron.weights[0]
        ax.scatter(weights[xind], weights[yind], s=10, c="black")
    for Neuron in network:
        weights = Neuron.weights[0]
        nextNeuron = Neuron.nextNeuron
        next_weights1 = nextNeuron.weights[0]

        xcord1 = np.linspace(weights[xind], next_weights1[xind], 5)
        ycord1 = np.linspace(weights[yind], next_weights1[yind], 5)

        nextNeuron = Neuron
        for i in range(ydim):
            nextNeuron = nextNeuron.nextNeuron
        next_weights2 = nextNeuron.weights[0]

        xcord2 = np.linspace(weights[xind], next_weights2[xind], 5)
        ycord2 = np.linspace(weights[yind], next_weights2[yind], 5)
        if Neuron.y != ydim - 1:
            ax.plot(xcord1, ycord1, linewidth=1.2, c='0.12')
        if Neuron.x != xdim - 1:
            ax.plot(xcord2, ycord2, linewidth=1.2, c='0.12')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)   

def testing_process(network, dList, dim):
    # Test aşamasında öncelikle nöronların x ve y kordinatları anahtar olarak kullanılıp bir sözlük oluşturulur.
    # Bu sözlükte her nöronun hangi tip veride kaç kere kazandığı bilgisi olacaktır.
    winnerDict = {}
    for Neuron in network:
        # Nöronun x ve y koordinatları anahtar olarak alınır.
        key = (Neuron.x, Neuron.y)
        # Nöronun hangi sınıftan veride kaç kere kazandığı ise değerler olacaktır.
        # Bu değerler sırayla setosa, versicolor ve virginica çiçeklerini ifade eder.
        winnerDict[key] = [0, 0, 0]

    for data in dList:
        winner = 999
        winner_x = 0
        winner_y = 0
        for Neuron in network:
            current = 0
            # Kazananın belirlenmesi için veri ile nöronların ağırlıkları arasındaki uzaklık ölçülür ve en düşük olan seçilir.
            for i in range(len(Neuron.weights[0])):
                current += (Neuron.weights[0][i] - data[i])**2
            # Kazananın indisleri kaydedilir.
            if current < winner:
                winner = current
                winner_x = Neuron.x
                winner_y = Neuron.y
        # Kazanan nöronun kazandığı sınıftaki değeri artar. Test kümesi oluşturulurken farklı çiçekler için
        # tüm verilere farklı değerler eklenmiştir. Verinin sonuna setosa için 0, versicolor için 1 ve virginica
        # için 2 eklenmiştir. Bu değerler, kazanan nöronun hangi sınıftaki değerini artıracağını seçerken kullanılır.
        winner_key = (winner_x, winner_y)
        winner_val = winnerDict[winner_key]
        winner_cls = int(data[dim])
        winner_val[winner_cls] = winner_val[winner_cls] + 1
        winnerDict[winner_key] = winner_val
    return winnerDict

def education(data, network, dev, learning_adaptive):
    # Eğitim için öncelikle kazanan belirlenir. Kazananın indislerine göre ağdaki nöronlar güncellenir.
    winner = 999
    winner_x = 0
    winner_y = 0
    for Neuron in network:
        current = 0
        # Kazananın belirlenmesi için veri ile nöronların ağırlıkları arasındaki uzaklık ölçülür ve en düşük olan seçilir.
        for i in range(len(Neuron.weights[0])):
            current += (Neuron.weights[0][i] - data[i])**2
        # Kazananın indisleri kaydedilir.
        if current < winner:
            winner = current
            winner_x = Neuron.x
            winner_y = Neuron.y 

    #     print("kazanan nöron uzaklığı", winner)
    #     print("kazanan nöron indisleri, x: %d, y: %d" % (winner_x, winner_y))
    #     print("______________________________________________________")
    # print("kazanan indisler", winner_x, winner_y)

    # Kazananın indislerine göre ağdaki her nöron güncellenir. Bu işlem için veri, güncel iterasyondaki komşuluk
    # etkisi ve adaptif öğrenme hızı da kullanılır.
    for Neuron in network:
        # print("Nöron indisleri: x: %d, y: %d" % (Neuron.x, Neuron.y))
        # print("Güncelleme öncesi ağırlıklar:", Neuron.weights[0])
        Neuron.update(winner_x, winner_y, dev, data, learning_adaptive)
        # print("Güncelleme sonrası ağırlıklar:", Neuron.weights[0])
        # print("______________________________________________________")

def education_process(network, dList, epoch, conv_epoch, xdim, ydim, tcons1, tcons2, current_result_folder):
    
    plot2ddata("Ölçeklenme sonrası eğitim kümesi", 0, 1, "Çanak yaprak uzunluğu", "Çanak yaprak genişliği", trainingList_normal)
    plt.savefig(current_result_folder + '/fig7.png')
    plot2ddata("Ölçeklenme sonrası eğitim kümesi", 2, 3, "Taç yaprak uzunluğu", "Taç yaprak genişliği", trainingList_normal)
    plt.savefig(current_result_folder + '/fig8.png')
    
    plotnetwork(network, xdim, ydim, 0, 1, "Çanak yaprak uzunluğu", "Çanak yaprak genişliği", "Eğitim öncesi Kohonen Ağı")
    plt.savefig(current_result_folder + '/fig9.png')
    # plt.show()
    plt.close()
    plotnetwork(network, xdim, ydim, 2, 3, "Taç yaprak uzunluğu", "Taç yaprak genişliği", "Eğitim öncesi Kohonen Ağı")
    plt.savefig(current_result_folder + '/fig10.png')
    # plt.show()
    plt.close()
    # Özdüzenleme aşaması. "epoch" iterasyonu kadar sürecektir.
    start_time = time.time()
    for i in range(epoch):
        # Her iterasyonun başında eğitim listesi karıştırılır.
        random.shuffle(dList)
        # dev: Eğitimin bulunduğu iterasyondaki komşuluk etkisi standart sapması. Her iterasyonda eksponansiyel
        # olarak küçülmekte olup sonraki iterasyonların ağ üzerindeki etksini azaltır.
        # learning_adaptive: Adaptif öğrenme hızı. Öğrenme hızı da her iterasyonda eksponansiyel olarak azalmaktadır.
        dev = math.exp(-(i/tcons1))    
        learning_adaptive = math.exp(-(i/tcons2))

        # Listedeki her veri için ağ eğitime girer.
        for data in dList:
            education(data, network, dev, learning_adaptive)
    ordering_time = (time.time() - start_time)
    print("Özdüzenleme aşaması süresi: %f" % ordering_time)

    plotnetwork(network, xdim, ydim, 0, 1, "Çanak yaprak uzunluğu", "Çanak yaprak genişliği", "Özdüzenleme aşamasından sonra Kohonen Ağı")
    plt.savefig(current_result_folder + '/fig11.png')
    # plt.show()
    plt.close()
    plotnetwork(network, xdim, ydim, 2, 3, "Taç yaprak uzunluğu", "Taç yaprak genişliği", "Özdüzenleme aşamasından sonra Kohonen Ağı")
    plt.savefig(current_result_folder + '/fig12.png')
    # plt.show()
    plt.close()

    plotnetworkanddata(network, xdim, ydim, 0, 1, "Çanak yaprak uzunluğu", "Çanak yaprak genişliği", "Özdüzenleme sonrası ağ ve eğitim kümesi verileri", dList)
    plt.savefig(current_result_folder + '/fig13.png')
    # plt.show()
    plt.close()
    plotnetworkanddata(network, xdim, ydim, 2, 3, "Taç yaprak uzunluğu", "Taç yaprak genişliği", "Özdüzenleme sonrası ağ ve eğitim kümesi verileri", dList)
    plt.savefig(current_result_folder + '/fig14.png')
    # plt.show()
    plt.close()

    # Yakınsama aşamasında ise komşuluk etkisi ve öğrenme hızı sabitken ağ eğitilir. Haykin, kitabında bu aşama
    # için nöron sayısının 500 katı iterasyon sürmesi gerektiğini söylemiştir.
    dev = math.exp(-((epoch-1)/tcons1))    
    learning_adaptive = math.exp(-((epoch-1)/tcons2))
    start_time2 = time.time()
    for i in range(round(conv_epoch*len(network))):
        random.shuffle(dList)
        for data in dList:
            education(data, network, dev, learning_adaptive)
    convergence_time = (time.time() - start_time2)
    print("Yakınsama aşaması süresi: %f" % convergence_time)
    print("Eğitim süresi: %f" % (ordering_time + convergence_time))

    plotnetwork(network, xdim, ydim, 0, 1, "Çanak yaprak uzunluğu", "Çanak yaprak genişliği", "Yakınsama aşamasından sonra Kohonen Ağı")
    plt.savefig(current_result_folder + '/fig15.png')
    # plt.show()
    plt.close()
    plotnetwork(network, xdim, ydim, 2, 3, "Taç yaprak uzunluğu", "Taç yaprak genişliği", "Yakınsama aşamasından sonra Kohonen Ağı")
    plt.savefig(current_result_folder + '/fig16.png')
    # plt.show()
    plt.close()

    return [ordering_time, convergence_time]

# Geri ölçekleme fonksyionu. Verileri, ölçekli halinde minimum ve maksimum değerleri normal_min ve normal_max iken 
# bu değerleri minL ve maxL değerlerine çeker.
def denormalize_data(data, dim, normal_min, normal_max, minL, maxL):
    for i in range(dim):
        data[i] = minL[i] + (maxL[i] - minL[i])*(data[i]-normal_min[i])/(normal_max[i]-normal_min[i])
    return data

# Ölçekleme fonksiyonu. Minimum ve maksimum değerleri normal_min ve normal_max olmak üzere
# veriyi ölçekler.
def normalize_List(dList, dim, normal_min, normal_max):
    maxL = np.zeros(dim, dtype=float)
    minL = np.zeros(dim, dtype=float)
    for data in dList:
        for i in range(dim):
            if data[i] > maxL[i]:
                maxL[i] = data[i]
            if data[i] < minL[i]:
                minL[i] = data[i]
    for data in dList:
        for i in range(dim):
            data[i] = normal_min[i] + (normal_max[i] - normal_min[i])*(data[i] - minL[i])/(maxL[i] - minL[i])
    return [dList, maxL, minL]

# Ağ oluşturma fonksiyonu.
def generate_network(xdim, ydim, dimension, learning_rate):
    network = []
    prevNeuron = None
    # Her x ve y değeri için bir nöron oluşturup bu nöronları birbirine bağlar.
    for i in range(xdim):
        for j in range(ydim):
            # Önce y değerleri sonra x değerleri artarak nöron oluşturur. Yani (x,y) nöronundan
            # sonra (x,y+1) nöronu, daha sonra (x,y+2) nöronu oluşturur. 
            newNeuron = NeuronCls(i, j, dimension, learning_rate)
            network.append(newNeuron)
            # Yeni nöron oluşturduktan sonra bir önceki nörona sonraki nöron olarak atanır. Bu çizimi kolaylaştırmak için
            # Her nöronun birbirine bağlanmasını sağlayacaktır. İlk nörondan önce bir nöron olmadığı için işlem pas geçilir.
            if prevNeuron != None:
                prevNeuron.next(newNeuron)
            # Önceki nörona sonraki olarak atandıktan sonra, yeni oluşturulan nöron önceki nöron olarak atanır.
            prevNeuron = newNeuron
    return network

# Uniform gürültü oluşturma fonksiyonu.
def generate_uniform_noise(dimension, size):
    # Girilen boyutta, girilen büyüklük sayısında veri oluşturur. Bu veriler uniform dağılımlıdır.
    # Gürültüyle eğitmek için denendi, asıl fonksiyonda kullanılmıyor.
    dList = []
    for i in range(0, size):
        data = np.zeros(dimension, dtype=float)
        for j in range(0, dimension):
            data[j] = np.random.rand()
        dList.append(data)
    return dList

# Normal dağılımlı veri oluşturma fonksiyonu.
def generate_normal_data(dimension, size, mean, deviation):
    dList = []
    # Girilen boyutta, girilen sayı kadar veri oluşturur. Bu verilerin medyanları ve standart
    # sapmaları da fonksiyona girilen büyüklüklerdedir. Daha sonra verilerden oluşan listeyi döndürür.
    for i in range(size):
        data = np.zeros(dimension, dtype=float)
        for j in range(dimension):
            data[j] = np.random.normal(mean[j], deviation)
        dList.append(data)
    return dList


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
dataFile = os.path.join(__location__, 'p2data/iris.data')
dim = 4

irisfile = open(dataFile, 'r') 
Lines = irisfile.readlines() 

setosaL = []
versicolorL = []
virginicaL = []
for line in Lines:
    rawdata = line.split(',')
    if len(rawdata) < 5:
        break

    data = np.zeros(dim, dtype=float)
    data[0] = rawdata[0]
    data[1] = rawdata[1]
    data[2] = rawdata[2]
    data[3] = rawdata[3]    

    # Çiçeğin sınıfına göre farklı listelere dağıtılır.
    if rawdata[4].rstrip("\n") == "Iris-setosa":
        setosaL.append(data)
    elif rawdata[4].rstrip("\n") == "Iris-versicolor":
        versicolorL.append(data)
    elif rawdata[4].rstrip("\n") == "Iris-virginica":
        virginicaL.append(data)
    else: 
        continue

# Eğitim ve test sırasında kullanılacak veri kümelerinin dağılımı.
trainingFile = os.path.join(__location__, 'p2data/Training.pkl')
testingFile = os.path.join(__location__, 'p2data/Testing.pkl')
if os.path.exists(trainingFile):
    trainingList = loadList(trainingFile)
    testingList = loadList(testingFile)
else:
    trainingList = []
    testingList = []

    for i in range(0, len(setosaL)):
        if i < len(setosaL)/2:
            trainingList.append(setosaL[i])
        else:
            testdata = np.concatenate([setosaL[i], [0]])
            testingList.append(testdata)
    for i in range(0, len(versicolorL)):
        if i < len(versicolorL)/2:
            trainingList.append(versicolorL[i])
        else:
            testdata = np.concatenate([versicolorL[i], [1]])
            testingList.append(testdata)
    for i in range(0, len(virginicaL)):
        if i < len(virginicaL)/2:
            trainingList.append(virginicaL[i])
        else:
            testdata = np.concatenate([virginicaL[i], [2]])
            testingList.append(testdata)
        
    saveData(trainingList, trainingFile)
    saveData(testingList, testingFile)

# Eğitimin süreceği toplam iterasyon sayısı Haykin'in "Neural Networks - A Comprehensive Foundation" kitabındaki tavsiyesi 
# özdüzenlemenin 1000, yakınsama aşamasının toplam nöron sayısının 500 katı olduğu yönündedir. Bu problem için 500 özdüzenleme iterasyonu 
# seçilecek, yakınsama iterasyonu da toplam nöron sayısının, seçilen iterasyonun yarısıyla çarpımı olarak seçilecek.
# Ardından, komşuluk etkisinin ve adaptif öğrenme hızının zamanla eksponansiyel olarak azalması için zaman sabitleri belirlenir. 
# Bu değerler aynı kitabın 474-475 sayfalarında Haykin'in söylediği şekilde alınır. 
epoch = 500
conv_epoch = epoch/2

# 6'ya 6 boyutlu toplam 36 nörondan oluşan, başlangıçtaki öğrenme hızı 0.1 olan bir ağ oluşturulacak.
xdim = 6
ydim = 6
learning_rate = 0.1

tcons1 = epoch/math.log(math.sqrt(xdim**2 + ydim**2))
tcons2 = epoch

# Eğitimde kullanmak üzere, uzaklıkların bulunduğu bir kütüphane oluşturulur.
distDict = {}
for i in range(xdim):
    for j in range(ydim):
        dist = math.sqrt(i**2 + j**2)
        # 0.1 standart sapmaya sahip normal dağılımı. Farklı nöronların komşuluk derecesini
        # ve kazanan nörona göre ağırlıklarının ne kadar değişeceğini belirler.
        distkey = (i, j)
        distDict[distkey] = dist

current_result_folder = 'p2results'
# Verilerin üç boyutlu uzayda gösterimi.
plot2ddata("Veriler", 0, 1, "Çanak yaprak uzunluğu", "Çanak yaprak genişliği", setosaL, versicolorL, virginicaL, single_list = False)
plt.savefig(current_result_folder + '/fig1.png')
# plt.show()
plt.close()
plot2ddata("Veriler", 2, 3, "Taç yaprak uzunluğu", "Taç yaprak genişliği", setosaL, versicolorL, virginicaL, single_list = False)
plt.savefig(current_result_folder + '/fig2.png')
# plt.show()
plt.close()

# Test kümesi verileri.
plot2ddata("Test kümesi", 0, 1, "Çanak yaprak uzunluğu", "Çanak yaprak genişliği", testingList)
plt.savefig(current_result_folder + '/fig3.png')
# plt.show()
plt.close()
plot2ddata("Test kümesi", 2, 3, "Taç yaprak uzunluğu", "Taç yaprak genişliği", testingList)
plt.savefig(current_result_folder + '/fig4.png')
# plt.show()
plt.close()

# Eğitim kümesi verileri.
plot2ddata("Eğitim kümesi", 0, 1, "Çanak yaprak uzunluğu", "Çanak yaprak genişliği", trainingList)
plt.savefig(current_result_folder + '/fig5.png')
# plt.show()
plt.close()
plot2ddata("Eğitim kümesi", 2, 3, "Taç yaprak uzunluğu", "Taç yaprak genişliği", trainingList)
plt.savefig(current_result_folder + '/fig6.png')
# plt.show()
plt.close()

normalized_training = normalize_List(trainingList[:], dim, np.zeros(dim), np.ones(dim))
trainingList_normal = normalized_training[0]
maxL_training = normalized_training[1]
minL_training = normalized_training[2]

# Ölçeklenen eğitim kümesi verileri.
plot2ddata("Ölçeklenme sonrası eğitim kümesi", 0, 1, "Çanak yaprak uzunluğu", "Çanak yaprak genişliği", trainingList_normal)
plt.savefig(current_result_folder + '/fig7.png')
# plt.show()
plt.close()
plot2ddata("Ölçeklenme sonrası eğitim kümesi", 2, 3, "Taç yaprak uzunluğu", "Taç yaprak genişliği", trainingList_normal)
plt.savefig(current_result_folder + '/fig8.png')
# plt.show()
plt.close()

# Belirlenen değerlerle eğitim başlatılır.
total_ordering_time = 0
total_convergence_time = 0
for j in range(3):
    print("test: %d" % (j+1))

    # Eğitilecek ağ, verilen değerlere göre oluşturulur ve eğitim başlatılır.
    network = generate_network(xdim, ydim, dim, learning_rate)
    [ordering_time, convergence_time] = education_process(network, trainingList_normal, epoch, conv_epoch, xdim, ydim, tcons1, tcons2, current_result_folder)
    total_ordering_time += ordering_time
    total_convergence_time += convergence_time

    # Kohonen Ağı'nın ağırlıklarının geri ölçeklemesi. Ağırlıklar verinin orjinal boyutlarına döner.
    for Neuron in network:
        Neuron.weights[0] = denormalize_data(Neuron.weights[0], dim, np.zeros(dim), np.ones(dim), minL_training, maxL_training)
    plotnetwork(network, xdim, ydim, 0, 1, "Çanak yaprak uzunluğu", "Çanak yaprak genişliği", "Geri ölçekleme sonrası ağ")
    plt.savefig(current_result_folder + '/fig17.png')
    # plt.show()
    plt.close()
    plotnetwork(network, xdim, ydim, 2, 3, "Taç yaprak uzunluğu", "Taç yaprak genişliği", "Geri ölçekleme sonrası ağ")
    plt.savefig(current_result_folder + '/fig18.png')
    # plt.show()
    plt.close()

    plotnetworkanddata(network, xdim, ydim, 0, 1, "Çanak yaprak uzunluğu", "Çanak yaprak genişliği", "Ağ ve test kümesi verileri", testingList)
    plt.savefig(current_result_folder + '/fig19.png')
    # plt.show()
    plt.close()
    plotnetworkanddata(network, xdim, ydim, 2, 3, "Taç yaprak uzunluğu", "Taç yaprak genişliği", "Ağ ve test kümesi verileri", testingList)
    plt.savefig(current_result_folder + '/fig20.png')
    # plt.show()
    plt.close()

    plotnetworkanddata(network, xdim, ydim, 0, 1, "Çanak yaprak uzunluğu", "Çanak yaprak genişliği", "Ağ ve tüm veriler", setosaL, versicolorL, virginicaL, single_list = False)
    plt.savefig(current_result_folder + '/fig21.png')
    # plt.show()
    plt.close()
    plotnetworkanddata(network, xdim, ydim, 2, 3, "Taç yaprak uzunluğu", "Taç yaprak genişliği", "Ağ ve tüm veriler", setosaL, versicolorL, virginicaL, single_list = False)
    plt.savefig(current_result_folder + '/fig22.png')
    # plt.show()
    plt.close()

    result = testing_process(network, testingList, dim)

    resultFile = current_result_folder + '/irispred.xls'
    saveResults(result, xdim, ydim, resultFile)

    network.clear()

timefile = open(current_result_folder + '/egitim_sureleri.txt',"a")
timefile.truncate(0)
L = ["Özdüzenleme süresi: %f\n" % (total_ordering_time/3),"Yakınsama süresi: %f\n" % (total_convergence_time/3), "Eğitim süresi: %f" % (total_ordering_time/3 + total_convergence_time/3)]
timefile.writelines(L)
timefile.close()  