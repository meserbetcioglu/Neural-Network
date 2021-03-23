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

# Verilerin çizimini yapan fonksiyonlar
def plotnetworkanddata(network, xdim, ydim, title, dList1, dList2 = [], dList3 = [], single_list = True):
    # Ağ ve veri çizen fonksiyonların birleşimi.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    if single_list == False:
        for data in dList1:
            ax.scatter(data[0], data[1], data[2], s=5, c="red", depthshade=True)
        for data in dList2:
            ax.scatter(data[0], data[1], data[2], s=5, c="blue", depthshade=True)
        for data in dList3:
            ax.scatter(data[0], data[1], data[2], s=5, c="green", depthshade=True)
    else:
        for data in dList1:
            ax.scatter(data[0], data[1], data[2], s=5, c="magenta", depthshade=True)
    
    for Neuron in network:
        weights = Neuron.weights[0]
        ax.scatter(weights[0], weights[1], weights[2], s=10, c="black", depthshade=True)
    for Neuron in network:
        weights = Neuron.weights[0]
        nextNeuron = Neuron.nextNeuron
        next_weights1 = nextNeuron.weights[0]

        xcord1 = np.linspace(weights[0], next_weights1[0], 5)
        ycord1 = np.linspace(weights[1], next_weights1[1], 5)
        zcord1 = np.linspace(weights[2], next_weights1[2], 5)

        nextNeuron = Neuron
        for i in range(ydim):
            nextNeuron = nextNeuron.nextNeuron
        next_weights2 = nextNeuron.weights[0]

        xcord2 = np.linspace(weights[0], next_weights2[0], 5)
        ycord2 = np.linspace(weights[1], next_weights2[1], 5)
        zcord2 = np.linspace(weights[2], next_weights2[2], 5)
        if Neuron.y != ydim - 1:
            ax.plot(xcord1, ycord1, zcord1, linewidth=1.2, c='0.12')
        if Neuron.x != xdim - 1:
            ax.plot(xcord2, ycord2, zcord2, linewidth=1.2, c='0.12')


    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

def plot3ddata(title, dList1, dList2 = [], dList3 = [], single_list = True):
    # Girilen listedeki tüm verileri scatter plot şeklinde çizdirir. Tek liste girilmişse mor/pembe bir renkte
    # çizim yapar, üç farklı liste girilmişse listedeki verileri kırmızı, mavi ve yeşil renklerle çizer.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    if single_list == False:
        for data in dList1:
            ax.scatter(data[0], data[1], data[2], s=5, c="red", depthshade=True)
        for data in dList2:
            ax.scatter(data[0], data[1], data[2], s=5, c="blue", depthshade=True)
        for data in dList3:
            ax.scatter(data[0], data[1], data[2], s=5, c="green", depthshade=True)
    else:
        for data in dList1:
            ax.scatter(data[0], data[1], data[2], s=5, c="magenta", depthshade=True)
        
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

def plotnetwork(network, xdim, ydim, title):
    # Ağ çizimini yapan fonksiyon. Öncelikle ağdaki nöronları ağırlıklarına göre scatter plot şeklinde çizdirir.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    for Neuron in network:
        weights = Neuron.weights[0]
        ax.scatter(weights[0], weights[1], weights[2], s=10, c="black", depthshade=True)
    # Ardından yanındaki nöronlarla olan bağını çizer. Bunun için bir nörondan iki farklı çizgi çizdirilir.
    for Neuron in network:
        # Birinci çizgi hemen sonraki nörona çizilir. (x,y) nöronundan (x,y+1) nöronuna olan çizgidir.
        weights = Neuron.weights[0]
        nextNeuron = Neuron.nextNeuron
        next_weights1 = nextNeuron.weights[0]

        xcord1 = np.linspace(weights[0], next_weights1[0], 5)
        ycord1 = np.linspace(weights[1], next_weights1[1], 5)
        zcord1 = np.linspace(weights[2], next_weights1[2], 5)

        # İkincisi ise y boyutu kadar sonraki nöronla arasındaki çizgidir. (x,y) nöronundan (x+1,y) nöronuna
        # olan çizgidir. Bunun sebebi, nöronlar sıralanırken önce y değerlerine göre, sonra x değerlerine göre sıralanmalarıdır.
        nextNeuron = Neuron
        for i in range(ydim):
            nextNeuron = nextNeuron.nextNeuron
        next_weights2 = nextNeuron.weights[0]

        xcord2 = np.linspace(weights[0], next_weights2[0], 5)
        ycord2 = np.linspace(weights[1], next_weights2[1], 5)
        zcord2 = np.linspace(weights[2], next_weights2[2], 5)
        # x ve y boyutlarına göre sınırda olan nöronlardan çizgi çizdirilmez.
        if Neuron.y != ydim - 1:
            ax.plot(xcord1, ycord1, zcord1, linewidth=1.2, c='0.12')
        if Neuron.x != xdim - 1:
            ax.plot(xcord2, ycord2, zcord2, linewidth=1.2, c='0.12')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')    

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
    plotnetwork(network, xdim, ydim, "Eğitim öncesi Kohonen Ağı")
    plt.savefig(current_result_folder + '/fig5.png')
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

    plotnetwork(network, xdim, ydim, "Özdüzenleme aşamasından sonra Kohonen Ağı")
    plt.savefig(current_result_folder + '/fig6.png')
    # plt.show()
    plt.close()

    plotnetworkanddata(network, xdim, ydim, "Özdüzenleme sonrası ağ ve eğitim kümesi verileri", dList)
    plt.savefig(current_result_folder + '/fig7.png')
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
    plotnetwork(network, xdim, ydim,  "Yakınsama aşamasından sonra Kohonen Ağı")
    plt.savefig(current_result_folder + '/fig8.png')
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
dataFile1 = os.path.join(__location__, 'p1data/Data1.pkl')
dataFile2 = os.path.join(__location__, 'p1data/Data2.pkl')
dataFile3 = os.path.join(__location__, 'p1data/Data3.pkl')
dim = 3

# Önceden oluşturulmuş verileri çeker. Eğer veri bulunmuyorsa
# 3 boyutlu, normal dağılımlı üç farklı noktalar kümesi oluşturur.
# Her kümede 200 nokta vardır. Kümelerde noktalar normal dağılımlıdır.
if os.path.exists(dataFile1):
    dList1 = loadList(dataFile1)
else:
    # Birinci kümede medyan (2,1,2) ve standart sapma 0.4'tür. 
    dList1 = generate_normal_data(dim, 200, [2, 1, 2], 0.4)
    saveData(dList1, dataFile1)

if os.path.exists(dataFile2):
    dList2 = loadList(dataFile2)
else:
    # ikincide medyan (2,1,1) ve standart sapma 0.1'dir. 
    dList2 = generate_normal_data(dim, 200, [2, 1, 1], 0.1)
    saveData(dList2, dataFile2)

if os.path.exists(dataFile3):
    dList3 = loadList(dataFile3)
else:
    # Üçüncü kümede medyan (1,1,2) ve standart sapma 0.3'tür.
    dList3 = generate_normal_data(dim, 200, [1, 1, 2], 0.3)
    saveData(dList3, dataFile3)

# Eğitim ve test sırasında kullanılacak veri kümelerinin dağılımı.
trainingFile = os.path.join(__location__, 'p1data/Training.pkl')
testingFile = os.path.join(__location__, 'p1data/Testing.pkl')
if os.path.exists(trainingFile):
    trainingList = loadList(trainingFile)
    testingList = loadList(testingFile)
else:
    trainingList = []
    testingList = []

    for i in range(0, len(dList1)):
        if i < len(dList1)/2:
            trainingList.append(dList1[i])
        else:
            testingList.append(dList1[i])
    for i in range(0, len(dList2)):
        if i < len(dList2)/2:
            trainingList.append(dList2[i])
        else:
            testingList.append(dList2[i])
    for i in range(0, len(dList3)):
        if i < len(dList3)/2:
            trainingList.append(dList3[i])
        else:
            testingList.append(dList3[i])
        
    saveData(trainingList, trainingFile)
    saveData(testingList, testingFile)

# Farklı iterasyon sayıları için ağ eğitilir ve sonuçlar kaydedilir.
epochT = (50, 250, 500, 750, 1000)
for k in range(len(epochT)):
    print("iteration: %d" % (k+1))
    
    # Eğitimin süreceği toplam iterasyon sayısı Haykin'in "Neural Networks - A Comprehensive Foundation" kitabındaki tavsiyesi 
    # özdüzenlemenin 1000, yakınsama aşamasının toplam nöron sayısının 500 katı olduğu yönündedir. Bu problem için 50, 250, 500, 750 ve 1000 
    # özdüzenleme iterasyonu seçilecek, yakınsama iterasyonu da toplam nöron sayısının, seçilen iterasyonun yarısıyla çarpımı olarak seçilecek.
    # Ardından, komşuluk etkisinin ve adaptif öğrenme hızının zamanla eksponansiyel olarak azalması için zaman sabitleri belirlenir. 
    # Bu değerler aynı kitabın 474-475 sayfalarında Haykin'in söylediği şekilde alınır. 
    epoch = epochT[k]
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

    current_test = str(epoch)
    current_result_folder = 'p1results/epoch' + current_test
    # Verilerin üç boyutlu uzayda gösterimi.
    plot3ddata("Veriler", dList1, dList2, dList3, single_list = False)
    plt.savefig(current_result_folder + '/fig1.png')
    # plt.show()
    plt.close()

    # Test kümesi verileri.
    plot3ddata("Test kümesi", testingList)
    plt.savefig(current_result_folder + '/fig2.png')
    # plt.show()
    plt.close()

    # Eğitim kümesi verileri.
    plot3ddata("Eğitim kümesi", trainingList)
    plt.savefig(current_result_folder + '/fig3.png')
    # plt.show()
    plt.close()

    normalized_training = normalize_List(trainingList, dim, np.zeros(dim), np.ones(dim))
    trainingList_normal = normalized_training[0]
    maxL_training = normalized_training[1]
    minL_training = normalized_training[2]

    # Ölçeklenen eğitim kümesi verileri.
    plot3ddata("Ölçeklenme sonrası eğitim kümesi", trainingList_normal)
    plt.savefig(current_result_folder + '/fig4.png')
    # plt.show()
    plt.close()

    # noise = generate_uniform_noise(dim, 100)
    # for i in range(0, len(noise)):
    #     trainingList.append(noise[i])

    # # Gürültü eklendikten sonra eğitim kümesi verileri.
    # plot3ddata("Gürültülü eğitim kümesi", trainingList)
    # # plt.show()
    # plt.close()

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

        # # Eğitim kümesindeki verilerin ve Kohonen Ağı'nın ağırlıklarının geri ölçeklemesi. Veriler orjinal boyutlarına döner.
        # for data in trainingList:
        #     data = denormalize_data(data, dim, np.zeros(dim), np.ones(dim), minL_training, maxL_training)

        for Neuron in network:
            Neuron.weights[0] = denormalize_data(Neuron.weights[0], dim, np.zeros(dim), np.ones(dim), minL_training, maxL_training)

        plotnetwork(network, xdim, ydim, "Geri ölçekleme sonrası ağ")
        plt.savefig(current_result_folder + '/fig9.png')
        # plt.show()
        plt.close()

        plotnetworkanddata(network, xdim, ydim, "Ağ ve test kümesi verileri", testingList)
        plt.savefig(current_result_folder + '/fig10.png')
        # plt.show()
        plt.close()

        plotnetworkanddata(network, xdim, ydim, "Ağ ve tüm veriler", dList1, dList2, dList3, single_list = False)
        plt.savefig(current_result_folder + '/fig11.png')
        # plt.show()
        plt.close()

        network.clear()

    timefile = open(current_result_folder + '/egitim_sureleri.txt',"a")
    timefile.truncate(0)
    L = ["Özdüzenleme süresi: %f\n" % (total_ordering_time/3),"Yakınsama süresi: %f\n" % (total_convergence_time/3), "Eğitim süresi: %f" % (total_ordering_time/3 + total_convergence_time/3)]
    timefile.writelines(L)
    timefile.close()  

# Farklı başlangıç öğrenme hızları için ağ eğitilir ve sonuçlar kaydedilir.
learning_rateT = (0.001, 0.01, 0.1, 0.3, 0.5)
for k in range(len(learning_rateT)):
    print("iteration: %d" % (k+1))

    # Eğitimin süreceği toplam iterasyon sayısı Haykin'in "Neural Networks - A Comprehensive Foundation" kitabındaki tavsiyesi 
    # özdüzenlemenin 1000, yakınsama aşamasının toplam nöron sayısının 500 katı olduğu yönündedir. Bu problem için 250 özdüzenleme iterasyonu seçilecek, 
    # yakınsama iterasyonu da toplam nöron sayısının, seçilen iterasyonun yarısıyla çarpımı olarak seçilecek.
    # Ardından, komşuluk etkisinin ve adaptif öğrenme hızının zamanla eksponansiyel olarak azalması için zaman sabitleri belirlenir. 
    # Bu değerler aynı kitabın 474-475 sayfalarında Haykin'in söylediği şekilde alınır. 
    epoch = 250
    conv_epoch = epoch/2

    # 6'ya 6 boyutlu toplam 36 nörondan oluşan, başlangıçtaki öğrenme hızı 0.1, 0.3, 0.5, 0.7 ve 0.9 olan ağlar oluşturulacak ve karşılaştırılacak.
    xdim = 6
    ydim = 6
    learning_rate = learning_rateT[k]

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

    current_test = str(learning_rate)
    current_result_folder = 'p1results/init_learning_rate' + current_test
    # Verilerin üç boyutlu uzayda gösterimi.
    plot3ddata("Veriler", dList1, dList2, dList3, single_list = False)
    plt.savefig(current_result_folder + '/fig1.png')
    # plt.show()
    plt.close()

    # Test kümesi verileri.
    plot3ddata("Test kümesi", testingList)
    plt.savefig(current_result_folder + '/fig2.png')
    # plt.show()
    plt.close()

    # Eğitim kümesi verileri.
    plot3ddata("Eğitim kümesi", trainingList)
    plt.savefig(current_result_folder + '/fig3.png')
    # plt.show()
    plt.close()

    normalized_training = normalize_List(trainingList, dim, np.zeros(dim), np.ones(dim))
    trainingList_normal = normalized_training[0]
    maxL_training = normalized_training[1]
    minL_training = normalized_training[2]

    # Ölçeklenen eğitim kümesi verileri.
    plot3ddata("Ölçeklenme sonrası eğitim kümesi", trainingList)
    plt.savefig(current_result_folder + '/fig4.png')
    # plt.show()
    plt.close()

    # noise = generate_uniform_noise(dim, 100)
    # for i in range(0, len(noise)):
    #     trainingList.append(noise[i])

    # # Gürültü eklendikten sonra eğitim kümesi verileri.
    # plot3ddata("Gürültülü eğitim kümesi", trainingList)
    # # plt.show()
    # plt.close()

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

        # # Eğitim kümesindeki verilerin ve Kohonen Ağı'nın ağırlıklarının geri ölçeklemesi. Veriler orjinal boyutlarına döner.
        # for data in trainingList:
        #     data = denormalize_data(data, dim, np.zeros(dim), np.ones(dim), minL_training, maxL_training)

        for Neuron in network:
            Neuron.weights[0] = denormalize_data(Neuron.weights[0], dim, np.zeros(dim), np.ones(dim), minL_training, maxL_training)

        plotnetwork(network, xdim, ydim, "Geri ölçekleme sonrası ağ")
        plt.savefig(current_result_folder + '/fig9.png')
        # plt.show()
        plt.close()

        plotnetworkanddata(network, xdim, ydim, "Ağ ve test kümesi verileri", testingList)
        plt.savefig(current_result_folder + '/fig10.png')
        # plt.show()
        plt.close()

        plotnetworkanddata(network, xdim, ydim, "Ağ ve tüm veriler", dList1, dList2, dList3, single_list = False)
        plt.savefig(current_result_folder + '/fig11.png')
        # plt.show()
        plt.close()

        network.clear()

    timefile = open(current_result_folder + '/egitim_sureleri.txt',"a")
    timefile.truncate(0)
    L = ["Özdüzenleme süresi: %f\n" % (total_ordering_time/10),"Yakınsama süresi: %f\n" % (total_convergence_time/10), "Eğitim süresi: %f" % (total_ordering_time/10 + total_convergence_time/10)]
    timefile.writelines(L)
    timefile.close()  

# Farklı nöron sayıları için ağ eğitilir ve sonuçlar kaydedilir.
network_dimT = (4, 5, 6, 7, 8)
for k in range(len(network_dimT)):
    print("iteration: %d" % (k+1))

    # Eğitimin süreceği toplam iterasyon sayısı Haykin'in "Neural Networks - A Comprehensive Foundation" kitabındaki tavsiyesi 
    # özdüzenlemenin 1000, yakınsama aşamasının toplam nöron sayısının 500 olduğu yönündedir. Bu problem için 500 özdüzenleme iterasyonu seçilecek, 
    # yakınsama iterasyonu da toplam nöron sayısının, seçilen iterasyonun yarısıyla çarpımı olarak seçilecek.
    # Ardından, komşuluk etkisinin ve adaptif öğrenme hızının zamanla eksponansiyel olarak azalması için zaman sabitleri belirlenir. 
    # Bu değerler aynı kitabın 474-475 sayfalarında Haykin'in söylediği şekilde alınır. 
    epoch = 250
    conv_epoch = epoch/2

    # 6'ya 6 boyutlu toplam 36 nörondan oluşan, başlangıçtaki öğrenme hızı 0.1, 0.3, 0.5, 0.7 ve 0.9 olan ağlar oluşturulacak ve karşılaştırılacak.
    xdim = network_dimT[k]
    ydim = network_dimT[k]
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

    current_test = str(network_dimT[k])
    current_result_folder = 'p1results/network_dim' + current_test
    # Verilerin üç boyutlu uzayda gösterimi.
    plot3ddata("Veriler", dList1, dList2, dList3, single_list = False)
    plt.savefig(current_result_folder + '/fig1.png')
    # plt.show()
    plt.close()

    # Test kümesi verileri.
    plot3ddata("Test kümesi", testingList)
    plt.savefig(current_result_folder + '/fig2.png')
    # plt.show()
    plt.close()

    # Eğitim kümesi verileri.
    plot3ddata("Eğitim kümesi", trainingList)
    plt.savefig(current_result_folder + '/fig3.png')
    # plt.show()
    plt.close()

    normalized_training = normalize_List(trainingList[:], dim, np.zeros(dim), np.ones(dim))
    trainingList_normal = normalized_training[0]
    maxL_training = normalized_training[1]
    minL_training = normalized_training[2]

    # Ölçeklenen eğitim kümesi verileri.
    plot3ddata("Ölçeklenme sonrası eğitim kümesi", trainingList)
    plt.savefig(current_result_folder + '/fig4.png')
    # plt.show()
    plt.close()

    # noise = generate_uniform_noise(dim, 100)
    # for i in range(0, len(noise)):
    #     trainingList.append(noise[i])

    # # Gürültü eklendikten sonra eğitim kümesi verileri.
    # plot3ddata("Gürültülü eğitim kümesi", trainingList)
    # # plt.show()
    # plt.close()

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

        # # Eğitim kümesindeki verilerin ve Kohonen Ağı'nın ağırlıklarının geri ölçeklemesi. Veriler orjinal boyutlarına döner.
        # for data in trainingList:
        #     data = denormalize_data(data, dim, np.zeros(dim), np.ones(dim), minL_training, maxL_training)

        for Neuron in network:
            Neuron.weights[0] = denormalize_data(Neuron.weights[0], dim, np.zeros(dim), np.ones(dim), minL_training, maxL_training)

        plotnetwork(network, xdim, ydim, "Geri ölçekleme sonrası ağ")
        plt.savefig(current_result_folder + '/fig9.png')
        # plt.show()
        plt.close()

        plotnetworkanddata(network, xdim, ydim, "Ağ ve test kümesi verileri", testingList)
        plt.savefig(current_result_folder + '/fig10.png')
        # plt.show()
        plt.close()

        plotnetworkanddata(network, xdim, ydim, "Ağ ve tüm veriler", dList1, dList2, dList3, single_list = False)
        plt.savefig(current_result_folder + '/fig11.png')
        # plt.show()
        plt.close()

        network.clear()
    
    timefile = open(current_result_folder + '/egitim_sureleri.txt',"a")
    timefile.truncate(0)
    L = ["Özdüzenleme süresi: %f\n" % (total_ordering_time/10),"Yakınsama süresi: %f\n" % (total_convergence_time/10), "Eğitim süresi: %f" % (total_ordering_time/10 + total_convergence_time/10)]
    timefile.writelines(L)
    timefile.close()  

