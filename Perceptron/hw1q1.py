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

os.system('cls||clear')

class DataPoints:
    #DataPoints obje sınıfında koordinat, anahtar, veri sınıfı ve küme bilgileri mevcut.
    def __init__(self, Coordinate, Key, DataClass, Domain): 
        self.Coordinate = Coordinate
        self.Key = Key
        self.DataClass = DataClass
        self.Domain = Domain

    #
    def setCoordinate(self, Coordinate):
        self.Coordinate = Coordinate

    def setClass(self, DataClass):
        self.DataClass = DataClass
    
    def setDomain(self, Domain):
        self.Domain = Domain
    
    #Verinin koordinat vektörünün büyüklüğünü veren fonksiyon.
    def CoordinateMag(self):
        return np.linalg.norm(self.Coordinate)

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

#File lokasyonundaki sözlüğü çeker.
def loadDict(File):
    DataDict = {}
    with open(File, 'rb') as f:
        DataDict = pickle.load(f)
    return DataDict

#"epoch" kez yeni beş boyutlu koordinat oluşturur. 
def newData(epoch):
    DataList = []
    for i in range(epoch): 
        #Koordinatlar (-1,1) aralığında noktadan sonra 4 basamaklı sayılardan oluşmakta. Uniform dağılımla seçiliyorlar.
        Coord = 10*np.random.random_sample(5,) - 5
        Coord = np.around(Coord, decimals = 4)
        #Kordinatlarla yeni bir veri listesi oluşturur. Yeni verilerin anahtarları iterasyon sayısına eşit olacak ve birbirinden farklı olacaktır.
        DataList.append(DataPoints(Coord, i, 0, "0"))  
    return DataList

#Girilen listeden koordinatın beşinci boyut elemanı ve anahtar bilgilerini çeken bir sözlük oluşturur.
def newDict(List):
    d = {}
    for obj in List: 
        a = GetCoordinate(obj)
        k = GetKey(obj)
        d[k] = a[4]
    #Oluşturulan sözlük lineer ayrıştırılabilirlik için kullanılacak.
    return d

#Eğitim ve test kümelerine rastgele dağıtım gerçekleştirir.
def distDomain(List):
    random.shuffle(List)
    shuffle = random.sample(range(0, len(List)), 40)
    testArr = np.split(shuffle, [15])[0]
    educationArr = np.split(shuffle, [15])[1]
    for obj in List:
        if obj.Key in testArr:
            obj.setDomain("test")
        elif obj.Key in educationArr:
            obj.setDomain("education")
        else:
            continue

#Eğitim ve Test kümesi arasında eleman alışverişinde bulunur.
def balanceDomain(List):
    Class1Education = []
    Class2Education = []
    Class1Test = []
    Class2Test = []

    for obj in List:
        if GetDataClass(obj) == 1:
            if GetDomain(obj) == "education":
                Class1Education.append(obj)
            elif GetDomain(obj) == "test":
                Class1Test.append(obj)
            else:
                print("None domain object detected.")
        elif GetDataClass(obj) == -1:
            if GetDomain(obj) == "education":
                Class2Education.append(obj)
            elif GetDomain(obj) == "test":
                Class2Test.append(obj)
            else:
                print("None domain object detected.")
        else:
            print("None class object detected.")
    
    #(0,1) arasında normal dağılımlı rastgele bir c sayısı seçer. Daha sonra bu sayıyı kümelerin eleman sayılarıyla karşılaştırır.
    c = np.random.normal(0.5, 0.1, 1)

    #Eğer birinci veri sınıfına ait eğitim kümesi elemanlarının sayısı, ikinciye ait olanlardan fazlaysa c'nin aşağıdaki değerden küçük olma olasılığı fazladır.
     
    if c < len(Class1Education)/(len(Class1Education)+len(Class2Education)):
        #Bu durumda eğitim kümesinden birince sınıfa ait bir eleman çıkarılıp test kümesine aktarılır.
        item = random.choice(Class1Education)
        item.setDomain("test")
        #Küme sayılarının aynı kalması için test kümesinden ikinci sınıfa ait bir eleman çıkarılıp eğitim kümesine aktarılır.
        item = random.choice(Class2Test)
        item.setDomain("education")
        #Aksi takdirde ise aynı işlem sınıflar değiştirilerek gerçekleştirilir.
    else:
        item = random.choice(Class2Education)
        item.setDomain("test")
        item = random.choice(Class1Test)
        item.setDomain("education")
        #Sonuçta ortaya çıkan kümelerin öncekilerden daha iyi olma garantisi yoktur ancak c için verdiğimiz koşul 
        #ortaya çıkan kümelerin daha iyi olması ihtimalini yükseltir.

#Testin gerçekleştiği fonksiyon
def test(List, w):
    #Eğitim fonksiyonuyla hemen hemen aynı algoritmaya sahip. Aralarındaki fark test fonksiyonunun ağırlığı değiştirmeyip yalnızca hata sayısına bakması.
    testList = []
    errorCount = 0
    for obj in List:
        if GetDomain(obj) == "test":
            testList.append(obj)
    
    for obj in testList:
        biasCoord = np.concatenate((GetCoordinate(obj),[1]))
        v = np.dot(w, biasCoord)
        if v > 0:
            y = 1
        else:
            y = -1
        yd = GetDataClass(obj)
        if yd != y:
            errorCount += 1
    #Test sonucu elde edilen hata sayısını döndürür.
    result = errorCount
    return result 

#Normal dağılımlı rastgele bir ağırlık vekötrü atar.
def getrandomw(mean, scale):
    w = np.random.normal(mean, scale, 6)
    return w

#Veri sınıflarını dağıtan fonksiyon
def distClass(epoch, Dict, List, Lin):
    #Lineerizasyon gerçekleşmesi için Lin argümanı doğru olmalıdır.
    if Lin == True:
        totalDict = Dict
        minDict = {}
        #Verilen sözlüğü kullanarak verilerin koordinatlarının beşinci boyut elemanlarını büyüklüklerine göre ikiye ayırır.
        #Bu lineer ayrıştırılmanın kesinlikle mümkün olmasını sağlar.
        for i in range(round(epoch/2)):
            key_min = min(totalDict.keys(), key=(lambda k: totalDict[k]))
            minDict[key_min] = totalDict[key_min]
            del totalDict[key_min]
        for obj in List:
            k = GetKey(obj)
            #Minimum sözlüğündeki değerlerin veri sınıfını 1'e, diğerlerini -1'e atar.
            if k in minDict.keys():
                obj.setClass(1)
            else:
                obj.setClass(-1)

    #Lin argümanı doğru değilse girilen liste çekilir, karıştırılır ve sınıflara dağıtılır.
    else:
        totalDict = Dict
        random.shuffle(List)
        for i in range(len(List)):
            if i in range(round(len(List)/2)):
                List[i].setClass(1)
            else:
                List[i].setClass(-1)
        random.shuffle(List)

#Verilerin kümelere dağılımlarını test eden fonksiyon
def domainTest(List):
    data1 = 0
    data2 = 0
    t1 = 0
    t2 = 0
    e1 = 0
    e2 = 0
    for obj in List:
        print(" ",obj.DataClass, end='')
        if obj.DataClass == 1:
            data1 += 1
            if obj.Domain == "test":
                t1 += 1
            else:
                e1 += 1
        elif obj.DataClass == -1:
            data2 += 1
            if obj.Domain == "test":
                t2 += 1
            else:
                e2 += 1
    print("\nE1 count:", e1, "\tT1 count:", t1, "\nE2 count:", e2, "\tT2 count:", t2)

#Verilerin küme bilgisini "domainFile" lokasyonunda saklar.
def extractDomain(List, domainFile):
    domainId = {}
    for obj in List:
        k = GetKey(obj)
        dom = GetDomain(obj)
        domainId[k] = dom
    saveData(domainId, domainFile)
    return domainId

#Verilerin küme bilgisini "domainFile" lokasyonundan çeker.
def insertDomain(List, domainFile):
    for obj in List:
        k = GetKey(obj)
        domainId = loadDict(domainFile)
        obj.setDomain(domainId[k])

#Listedeki verileri sıralayan fonksiyon
def sortList(List, mod):
    newList = List
    #Verileri, koordinat vektörlerinin büyüklükleri azalacak şekilde sıralar. 
    if mod == "-1":
        newList = sorted(newList, key=lambda x: x.CoordinateMag, reverse=True)
    #Verileri, koordinat vektörlerinin büyüklükleri artacak şekilde sıralar. 
    elif mod == "1":
        newList = sorted(newList, key=lambda x: x.CoordinateMag, reverse=False)
    return newList

#Eğitim fonksiyonu
def educate(List, w, eSpeed, maxiter):
    educateList = []
    final = []
    totalEdError = 0
    
    #Geçerli bir ağırlık vektörü verilmemişse rastgele bir ağırlık vektörü seçer.
    if w.size != 6:
        w = np.array(getrandomw(0, 1))
        #Eğitime başlanılan ağırlığın çıktısını verir.
        print("First weight:", np.around(w, decimals = 4))
    firstw = w
    finaliter = -1
    #Verilerin içinden eğitim kümesine ait elemanları çeker.
    for obj in List:
        if GetDomain(obj) == "education":
            educateList.append(obj)
    
    for i in range(maxiter):
        changeCount = 0
        #Eğitim iterasyonu terasyon döngüsü
        for obj in educateList:
            #Bias terimini verinin koordinatına ekler.
            biasCoord = np.concatenate((GetCoordinate(obj),[1]))

            #Perceptron eğitim algoritması
            v = np.dot(w, biasCoord)
            if v > 0:
                y = 1
            else:
                y = -1
            yd = GetDataClass(obj)
            w = w + (eSpeed/2)*(yd - y)*biasCoord
            w = np.around(w, decimals = 4)

            #Değişim gerçekleştiyse bir iterasyondaki değişim sayısına bir ekler.
            if yd != y:
                changeCount += 1
        totalEdError += changeCount
        #Eğitim sırasında her iterasyondaki hata sayısını verir.
        #print("Error count on ", i + 1, "th iteration is: ", changeCount)

        #Eğitim hatasız ise eğitimi durdurur.
        if changeCount == 0:
            finaliter = i+1
            finalw = w
            break
    #Sonuç olarak başlanılan ağırlık vektörü, son eğitimdeki iterasyon sayısı, son ağırlık vektörü ve son iterasyondaki değişim sayısını verir.
    final = [firstw, finaliter, finalw, totalEdError]
    return final
    
#Eğitim ve test işlemlerinin gerçekleştiği fonksiyon
def process(objList, w, c, maxiter, domainFile):
    i = 0
    failedTestCount = 0
    totalEdError = 0
    wantedInfo = []
    w = np.array(w)

    if w.size == 6:
        #Eğitime başlanılan ağırlığın çıktısını verir.
        print("First weight:", np.around(w, decimals = 4))

    #Önceden belirli bir eğitim-test kümesi dağılımı yapar. Yukarıda belirtilen domainFile1'deki eğitim-test kümesini çeker.
    if domainFile != None:
        if os.path.exists(domainFile):
            insertDomain(objList, domainFile)
        else: 
            distDomain(objList)
            extractDomain(objList, domainFile)
    #Eğer önceden belirli bir eğitim-test kümesi yoksa, girilen datayı dağıtır. Bu sayede kümeleri belirtilmeyen her veri rastgele kümelere dağıtılır.
    else:
        distDomain(objList)

    while i < 1:
        #Eğitim ve test kümelerinin eleman sayılarını gösterir. Debug amaçlı
        """domainTest(objList)"""

        #Eğitimi başlatır ve ilk ağırlık, eğitimin sonlandığı iterasyon sayısı, son ağırlığı ve her iterasyon için iterasyondaki hata sayısını verir.
        final = educate(objList, w, c, maxiter)

        #Her iterasyonda toplam hatayı verir. Karşılaştırmada kullanılacak.
        totalEdError = final[3]

        #Verilerin test kümesinden geçememesi durumu için w1 ağırlığını eğitilen ağırlığa eşitler. Sonraki eğitimde ilk ağırlık olarak bu ağırlık kullanılır.
        w = final[2]

        #Eğitim belirlenen iterasyonda sonuçlanmamışsa eğitilemediğine dair hata verir. Bu büyük ihtimalle verinin lineer ayrıştırılamaz olduğunu gösterir.
        if final[1] == -1:
            print("Education failed and likely impossible.")
            break

        #Eğitilen ağırlık test fonksiyonuna girilir. Burada ağırlık değişmez ve dönüt olarak hata veren veri sayısı alınır.
        errorCount = test(objList, final[2])
        print("Tested error count of ", failedTestCount + 1, "th education is ", errorCount)

        #Eğer test sonrası hata alınmadıysa döngüden çıkar.
        if errorCount == 0:
            i += 1

        #Testten geçemediyse veri balanceDomain fonksiyonuna girilir ve test kümesi ve eğitim kümesi birer eleman değişir. Bu değişim iki sınıfın da elemanı olacak şekilde
        #ağırlık koyularak yapılır ancak yine de rastgele bir değişimdir. Ayrıca kümeler değiştiği için iterasyon başına hatayı sıfırlar.
        else:
            balanceDomain(objList)
            failedTestCount += 1
            if failedTestCount > maxiter:
                final[1] = -1
                print("Education failed and likely impossible.")
                break
    #Test-Eğitim kümesi değişim sayısı    
    print("Failed Test Count:", failedTestCount)

    #Eğitim sonucu elde edilen eğitim-test kümesi elemanları çekilir ve domainFile dosyasına kaydedilir.  
    if domainFile != None:
        extractDomain(objList, domainFile)

    #Eğer eğitim tamamlanmamışsa 
    if len(final) == 4:
        print("Educated on: ", final[1], "th iteration.")
        print("Final weight: ", final[2])
    iterAmount = final[1]
    avgErrorPerIter = totalEdError/iterAmount
    magFirst = np.linalg.norm(final[0])
    magFinal = np.linalg.norm(final[2])
    magRatio = magFinal/magFirst
    wantedInfo = [failedTestCount, avgErrorPerIter, iterAmount, magFirst, magFinal, magRatio]
    return wantedInfo

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
dataFile = os.path.join(__location__, 'Data.pkl')
max_epoch = 40

GetCoordinate = attrgetter('Coordinate')
GetKey = attrgetter('Key')
GetDataClass = attrgetter('DataClass')
GetDomain = attrgetter('Domain')

if os.path.exists(dataFile):
    objList = loadList(dataFile)
    allDict = newDict(objList)
    
else:
    objList = newData(max_epoch)
    allDict = newDict(objList)
    distClass(max_epoch, allDict, objList, True)
    distDomain(objList)
    saveData(objList, dataFile)

# #1'lerden oluşan bir ağırlık vektörü.
# w1 = [1, 1, 1, 1, 1, 1]

# #Önceden atanmış bir test-eğitim kümesi. 
# #w1 = [1, 1, 1, 1, 1, 1] için 22 iterasyonda eğitilmekte ve son ağırlık w = [1.5964 0.8325 2.1872 -1.2123 -21.1314 23].
# domainFile1 = os.path.join(__location__, 'DomainEx1.pkl')
# domainFile2 = os.path.join(__location__, 'DomainEx2.pkl')
# testnum = 1
# #objList listesindeki verileri w1 = [1 1 1 1 1 1] ilk ağırlığı ve c = 1 öğrenme hızı ile, maksimum 80 iterasyon sürecek şekilde eğiten fonksiyon. 
# #DomainEx1.pkl dosyasındaki eğitim-test kümesi dağılımlarını çekip verilere işler.
# print("\n\nTest number:", testnum, "_"*150, "\n")
# testnum += 1
# insertDomain(objList, domainFile1)
# extractDomain(objList, domainFile2)
# Inf = process(objList, w1, 1, max_epoch*2, domainFile2)
# print("_"*165, "\n")

#objList listesindeki verileri rastgele bir ilk ağırlıkla ve c = 1 öğrenme hızı ile, maksimum 80 iterasyon sürecek şekilde eğiten fonksiyon. 
#domainFile kısmı "None" olduğu için eğitim sonucunda eğitim-test kümesi dağılımını kaydetmeyecek.
# print("\n\nTest number:", testnum, "_"*150, "\n")
# testnum += 1
# Inf = process(objList, [], 1, max_epoch*2, None)
# print("_"*165, "\n")

# domainFile3 = os.path.join(__location__, 'DomainEx3.pkl')
# domainFile4 = os.path.join(__location__, 'DomainEx4.pkl')
# # İlk olarak rastgele seçilen bir w = w3 ile aynı veri üzerinde öğrenme hızı değiştirilerek: 
# # Başarısız test sayısı 
# # Son eğitimdeki iterasyon başına düşen ortalama hata sayısı
# # Son eğitimdeki iterasyon sayısı 
# # Son eğitimdeki başlangıç ağırlık vektörünün boyutu
# # Son eğitimdeki bitiş ağırlık vektörünün boyutu
# # Son eğitimdeki bitiş ve başlangıç ağırlık vektörlerinin oranı 
# # karşılaştırılır. Bu 40 kez, c = (0.1,4) aralığında 0.1 aralıklarla gerçekleştirilir.
# w3 = getrandomw(0, 1)
# cTestResults = []
# resultFileC = os.path.join(__location__, 'SpeedResult.json')
# for i in range(40):

#     #Rastgele dağıtılmış DomainEx3 dağılımını kullanır.
#     #DomainEx3'ün değişmemesi için bu dağılım DomainEx4'ye aktarılır.
#     insertDomain(objList, domainFile3)
#     extractDomain(objList, domainFile4)
#     #Elde ettiğimiz değerleri cTestResults'ta toplayıp döngü sonunda SpeedResult.json adlı dosyaya kaydederiz.
#     c = i/10 + 0.1
#     Inf = process(objList, w3, c, max_epoch*2, domainFile4)
#     Inf.append(c)
#     cTestResults.append(Inf)

# for i in range(39):
#     w3 = getrandomw(0, 1)
#     for result in cTestResults:
#         insertDomain(objList, domainFile3)
#         extractDomain(objList, domainFile4)
#         c = result[6]
#         Inf = process(objList, w3, c, max_epoch*2, domainFile4)
#         Inf.append(c)
#         for j in range(len(result)):
#             if Inf[2] != -1:
#                 result[j] = (Inf[j] + result[j])/2
# with open(resultFileC, 'w') as json_file:
#     json.dump(cTestResults, json_file)
    


# domainFile5 = os.path.join(__location__, 'DomainEx5.pkl')
# domainFile6 = os.path.join(__location__, 'DomainEx6.pkl')
# # Daha sonra hız c = 1 ve aynı eğitim-test kümesiyle başlayıp farklı ağırlık vektörleriyle: 
# # Başarısız test sayısı 
# # Son eğitimdeki iterasyon başına düşen ortalama hata sayısı
# # Son eğitimdeki iterasyon sayısı 
# # Son eğitimdeki başlangıç ağırlık vektörünün boyutu
# # Son eğitimdeki bitiş ağırlık vektörünün boyutu
# # Son eğitimdeki bitiş ve başlangıç ağırlık vektörlerinin oranı 
# # karşılaştırılır. Bu 40 kez, her döngüde farklı w değerleri alınarak gerçekleştirilir. 
# # Sonuç w vektörlerinin büyüklüklerine göre karşılaştırılacaktır.
# wTestResults = []
# resultFileW = os.path.join(__location__, 'WeightResult.json')

# for i in range(40):
#     w4 = getrandomw(0, 1)
#     #Rastgele dağıtılmış DomainEx4 dağılımını kullanır.
#     #DomainEx5'in değişmemesi için bu dağılım DomainEx6'ya aktarılır.
#     insertDomain(objList, domainFile5)
#     extractDomain(objList, domainFile6)
#     #Elde ettiğimiz değerleri wTestResults'ta toplayıp döngü sonunda WeightResult.json adlı dosyaya kaydederiz.
#     Inf = process(objList, w4, 1, max_epoch*2, domainFile6)
#     Inf.append(np.linalg.norm(w4))
#     if Inf[2] != -1:
#         wTestResults.append(Inf)

# with open(resultFileW, 'w') as json_file:
#     json.dump(wTestResults, json_file)

# domainFile7 = os.path.join(__location__, 'DomainEx7.pkl')
# domainFile8 = os.path.join(__location__, 'DomainEx8.pkl')
# # Son olarak aynı eğitim-test kümesi ve ağırlıkla başlayıp 
# # eğitim kümesi vektörlerin büyüklüklerine göre farklı şekilde sıralandığında: 
# # Başarısız test sayısı 
# # Son eğitimdeki iterasyon başına düşen ortalama hata sayısı
# # Son eğitimdeki iterasyon sayısı 
# # Son eğitimdeki başlangıç ağırlık vektörünün boyutu
# # Son eğitimdeki bitiş ağırlık vektörünün boyutu
# # Son eğitimdeki bitiş ve başlangıç ağırlık vektörlerinin oranı 
# # karşılaştırılır. Bu 40 kez c = (0.1,4) aralığında 0.1 değişirken eğitim kümesi 
# # elemanları büyükten küçüğe ve küçükten büyüğe sıralıyken 40 kez test edilir ve karşılaştırılır.
# w4 = getrandomw(0, 1)
# sortTestResults = []
# resultFileSort = os.path.join(__location__, 'SortResult.json')
# for i in range(40):
#     #Rastgele dağıtılmış DomainEx6 dağılımını kullanır.
#     #DomainEx7'nin değişmemesi için bu dağılım DomainEx8'e aktarılır.
#     insertDomain(objList, domainFile7)
#     extractDomain(objList, domainFile8)
#     #Elde ettiğimiz değerleri sortTestResults'ta toplayıp döngü sonunda SortResult.json adlı dosyaya kaydederiz.
#     c = i/10 + 0.1
#     sort = "descending"
#     descList = sortList(objList, -1)
#     Inf = process(descList, w4, c, max_epoch*2, domainFile8)
#     Inf.append(c)
#     Inf.append(sort)
#     if Inf[2] != -1:
#         sortTestResults.append(Inf)

#     insertDomain(objList, domainFile7)  
#     extractDomain(objList, domainFile8)
#     sort = "ascending"
#     ascList = sortList(objList, 1)
#     Inf = process(ascList, w4, c, max_epoch*2, domainFile8)
#     Inf.append(c)
#     Inf.append(sort)
#     if Inf[2] != -1:
#         sortTestResults.append(Inf)

# for i in range(39):
#     for result in sortTestResults:
#         insertDomain(objList, domainFile7)
#         extractDomain(objList, domainFile8)
#         c = result[6]
#         Inf = process(objList, w4, c, max_epoch*2, domainFile8)
#         Inf.append(c)
#         Inf.append(sort)
#         for j in range(len(result)-1):
#             if Inf[2] != -1:
#                 result[j] = (Inf[j] + result[j])/2
# with open(resultFileSort, 'w') as json_file:
#     json.dump(sortTestResults, json_file)

nonLinFile = os.path.join(__location__, 'NonLinearData.pkl')

if os.path.exists(nonLinFile):
    nonLinList = loadList(nonLinFile)
    nonLinDict = newDict(nonLinList)
    
else:
    nonLinList = newData(max_epoch)
    nonLinDict = newDict(nonLinList)
    distDomain(nonLinList)
    distClass(max_epoch, nonLinDict, nonLinList, 0)
    saveData(nonLinList, nonLinFile)

#Sorunun ikinci kısmı için lineer ayrıştırılamaz bir küme oluşturulur. Kodun bu kısmı bilinmeyen bir hatadan dolayı çalışmamakta. 
#Verilerin kümeleri karışıyor ve eğitim kümesi anında eğitiliyor.
Inf = process(nonLinList, [], 1, max_epoch*2, nonLinFile)
domainTest(nonLinList)
#Verilerin son halini kaydeder
saveData(nonLinList, nonLinFile)
saveData(objList, dataFile)
