import numpy as np
import matplotlib.pyplot as plt
import random
import math
import json
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from operator import attrgetter
import pickle
import os
from scipy.interpolate import griddata


class DataPoints:
    #DataPoints obje sınıfında koordinat, anahtar, veri sınıfı mevcut.
    def __init__(self, Coordinate, Key, Domain): 
        self.Coordinate = Coordinate
        self.Key = Key
        self.Domain = Domain

    def setDomain(self, Domain):
        self.Domain = Domain

    def setCoordinate(self, Coordinate):
        self.Coordinate = Coordinate

#Data dosyasını File lokasyonuna kaydeder.
def saveData(Data, File):
    with open(File, 'wb') as f:
        pickle.dump(Data, f)

#File lokasyonundaki listeyi çeker.
def loadList(File):
    DataList = []
    with open(File, 'rb') as f:
        DataList = pickle.load(f)
    return DataList

#Yeni verilerin oluşturulduğu fonksiyon
def newData(epoch):
    DataList = []
    for i in range(epoch): 
        #x1 ve x2'yi soruda verilen aralıklarda rastgele sayılardan oluşturur. Daha sonra bias ekleyip veri koordinatına geçirir.
        x1 = random.random()
        x2 = np.pi*random.random()/2
        Coord = [x1, x2, 1]
        Coord = np.around(Coord, decimals = 9)
        #Kordinatlarla yeni bir veri listesi oluşturur. Yeni verilerin anahtarları iterasyon sayısına eşit olacak ve birbirinden farklı olacaktır.
        DataList.append(DataPoints(Coord, i, "None"))  
    return DataList
    
#Normal dağılımlı rastgele bir ağırlık vekötrü atar.
def getrandomw(mean, scale):
    w = np.random.normal(mean, scale, 3)
    return w

#Eğitim fonksiyonu
def educate(objList, w, c):
    print("İlk ağırlık vektörü:", w)
    count = 1
    countT = -1
    i = 0
    Eortprev = 0
    EortTprev = 0
    Eort = 0
    w = np.array(w)
    while i < 1:
        E = 0
        for obj in objList:
            if GetDomain(obj) == "E":
                Coord = GetCoordinate(obj)
                yd = f(Coord[0], Coord[1])/5
                v = np.dot(w, Coord)
                y = phi(v)
                e = yd - y
                E += (1/2)*(e**2)
                phiDerivative = np.sinh(v)/(abs(np.sinh(v)/np.cosh(v))*(np.cosh(v)**3))
                w = w + e*phiDerivative*Coord
        Eort = E/len(objList)
        EortR = Eort
        if abs(Eort - Eortprev) < Eort/(10**12):
            i +=1

        if i == 1:
            E = 0
            for obj in objList:
                if GetDomain(obj) == "T":
                    yd = f(Coord[0], Coord[1])/5
                    v = np.dot(w, Coord)
                    y = phi(v)
                    e = yd - y
                    E += (1/2)*(e**2)
            Eort = E/len(objList)
            if abs(Eort - EortTprev) < Eort/(10**12):
                i +=1
            else:
                countT += 1
                
            EortTprev = Eort
        print("İterasyon:",count,"\tOrtalama hata:", EortR, "\tAğırlık vektörü:", w)
        count += 1
        Eortprev = Eort
        Eort = 0
    if countT > 0:
            print("Başarısız Test Sayısı:", countT)
    result = [w, count, EortR]
    return result

def phi(v):
    return abs(np.tanh(v))

def f(x1, x2):
    return 3*x1 + 2*np.cos(x2)

def clamp(x): 
  return max(0, min(x, 255))

def rgb2hex(color):
    """Converts a list or tuple of color to an RGB string

    Args:
        color (list|tuple): the list or tuple of integers (e.g. (127, 127, 127))

    Returns:
        str:  the rgb string
    """
    return f"#{''.join(f'{hex(c)[2:].upper():0>2}' for c in color)}"

def distDomain(objList):
    for obj in objList:
        if GetKey(obj)%2 == 0:
            obj.setDomain("E")
        else:
            obj.setDomain("T")

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
dataFile = os.path.join(__location__, 'DataQ3.pkl')
max_epoch = 100

GetCoordinate = attrgetter('Coordinate')
GetKey = attrgetter('Key')
GetDomain = attrgetter('Domain')

if os.path.exists(dataFile):
    objList = loadList(dataFile)
    
else:
    objList = newData(max_epoch)
    distDomain(objList)
    saveData(objList, dataFile)

w1 = np.array([1, 1, 1])
Inf = educate(objList, w1, 0.5)
print("Son ağırlık vektörü:", Inf[0], "\tİterasyon sayısı:", Inf[1], "\tSon iterasyonda ortalama hata:", Inf[2])

plt.figure()
plt.subplot(131)
legx = np.linspace(-1,1,80)
legy = np.linspace(0,5,200)
for i in range(legy.size):
    for j in range(legx.size):
        x = legx[j]
        y = legy[i]
        colorrgb = (clamp(int(255 - 40*y)), 0, 0)
        color = rgb2hex(colorrgb)
        plt.plot(x, y, 'o', color=color, markersize=12)
    
plt.title('legend')

plt.subplot(132)
x1 = []
x2 = []
for obj in objList:
    x = GetCoordinate(obj)
    yd = f(x[0], x[1])
    colorrgb = (clamp(int(255 - 40*yd)), 0, 0)
    color = rgb2hex(colorrgb)
    plt.plot(x[0], x[1], 'o', color=color, markersize=8)
plt.title('f(x1, x2) ile oluşturulan veri seti')

plt.subplot(133)
for obj in objList:
    x = GetCoordinate(obj)
    a = np.dot(Inf[0], x)
    y = 5*phi(a)
    colorrgb = (clamp(int(255 - 40*y)), 0, 0)
    color = rgb2hex(colorrgb)
    plt.plot(x[0], x[1], 'o', color=color, markersize=8)
plt.title('phi(v) ile eğitilen veri seti')
plt.show()

print("\n\n\n\n")


dataFile2 = os.path.join(__location__, 'DataQ3_2.pkl')

if os.path.exists(dataFile2):
    objList = loadList(dataFile2)
    
else:
    objList = newData(max_epoch*100)
    distDomain(objList)
    saveData(objList, dataFile2)
w1 = np.array([1, 1, 1])
Inf = educate(objList, w1, 0.5)
print("Son ağırlık vektörü:", Inf[0], "\tİterasyon sayısı:", Inf[1], "\tSon iterasyonda ortalama hata:", Inf[2])

plt.figure()
plt.subplot(131)
legx = np.linspace(-1,1,80)
legy = np.linspace(0,5,200)
for i in range(legy.size):
    for j in range(legx.size):
        x = legx[j]
        y = legy[i]
        colorrgb = (clamp(int(255 - 40*y)), 0, 0)
        color = rgb2hex(colorrgb)
        plt.plot(x, y, 'o', color=color, markersize=12)
    
plt.title('legend')

plt.subplot(132)
x1 = []
x2 = []
for obj in objList:
    x = GetCoordinate(obj)
    yd = f(x[0], x[1])
    colorrgb = (clamp(int(255 - 40*yd)), 0, 0)
    color = rgb2hex(colorrgb)
    plt.plot(x[0], x[1], 'o', color=color, markersize=8)
plt.title('f(x1, x2) ile oluşturulan veri seti')

plt.subplot(133)
for obj in objList:
    x = GetCoordinate(obj)
    a = np.dot(Inf[0], x)
    y = 5*phi(a)
    colorrgb = (clamp(int(255 - 40*y)), 0, 0)
    color = rgb2hex(colorrgb)
    plt.plot(x[0], x[1], 'o', color=color, markersize=8)
plt.title('phi(v) ile eğitilen veri seti')
plt.show()
