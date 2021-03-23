import tensorflow as tf
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
import seaborn as sns
import os
import pickle
import xlwt
from xlwt import Workbook 
from xlrd import open_workbook
import time

os.system('cls||clear')

def Create_Train_Test_Network(lrate, mrate, drate, epoch, h1n, h2n, train_images, train_labels, test_images, test_labels, test_iter, savefolder):
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(1 - drate),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(1 - drate),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    opt = tf.keras.optimizers.SGD(learning_rate=lrate, momentum=mrate)

    start_time = time.time()
    model.compile(optimizer=opt,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    ed_time = time.time() - start_time

    history = model.fit(train_images, train_labels, epochs=epoch, 
                        validation_data=(test_images, test_labels))
    
    plt.plot(history.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(savefolder, str(test_iter) + '_Loss.png'))
    plt.close()

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print("Test accuracy: ", test_acc)

    y_pred = np.argmax(model.predict(test_images), axis=1)
    y_true = test_labels

    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12.8, 9.6))
    sns.heatmap(confusion_mtx, xticklabels=class_names, yticklabels=class_names, 
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Truth')
    plt.savefig(os.path.join(savefolder, str(test_iter) + '_Conf_Mtx.png'))
    plt.close()

    results = [test_iter, h1n, h2n, lrate, mrate, drate, 'relu', epoch, ed_time, ed_time/epoch, test_acc, test_loss]
    return results

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

def batchset_Arr(dataArr, labelArr, batch_size, class_count):
    # Veri kümesini daha küçük kümelere ayıran fonksiyon. "batch_size" değeri
    # tek bir sınıftan kaç verinin alınacağını belirler. Girilen bir veri listesinden
    # her sınıftan "batch_size" sayısında veri yeni bir listeye eklenip döndürülür.
    newdataArr = np.zeros((int(batch_size*class_count), 32, 32, 3), dtype=np.float64)
    newlabelArr = np.zeros((int(batch_size*class_count), 1), dtype=np.uint8)
    batchArr = np.zeros(class_count)
    i = 0
    for data in dataArr:
        j = labelArr[i]
        if batchArr[j] < batch_size:
            newdataArr[i, :, :, :] = data
            newlabelArr[i, 0] = labelArr[i, 0]
            batchArr[j] += 1
            i += 1
    return newdataArr, newlabelArr


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

class_names = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

train_images = train_images / 255.0
test_images = test_images / 255.0
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
batch_size = 200
train_images, train_labels = batchset_Arr(train_images[:], train_labels[:], batch_size, 10)

epoch = 50
h1n = 128
h2n = 32

resultfolder = os.path.join('TF', 'Results')
results = []
lrateT = (0.0001, 0.001, 0.01, 0.1)
for i in range(4):
    lrate = lrateT[i]
    mrate = 0.8
    drate = 0.8
    savefolder = os.path.join(resultfolder, 'Learning_Rate_' + str(lrate))
    for j in range(3):
        result = Create_Train_Test_Network(lrate, mrate, drate, epoch, h1n, h2n, train_images, train_labels, test_images, test_labels, j, savefolder)
        results.append(result)
saveResults(results, resultfolder, 'Results_LRate.xls')

results = []
mrateT = (0, 0.2, 0.4, 0.6, 0.8, 1)
for i in range(6):
    lrate = 0.01
    mrate = mrateT[i]
    drate = 0.8
    savefolder = os.path.join(resultfolder, 'Momentum_Rate_' + str(mrate))
    for j in range(3):
        result = Create_Train_Test_Network(lrate, mrate, drate, epoch, h1n, h2n, train_images, train_labels, test_images, test_labels, j, savefolder)
        results.append(result)
saveResults(results, resultfolder, 'Results_MRate.xls')

results = []
drateT = (0.5, 0.6, 0.7, 0.8, 0.9, 1)
for i in range(6):
    lrate = 0.01
    mrate = 0.8
    drate = drateT[i]
    savefolder = os.path.join(resultfolder, 'Dropout_Rate_' + str(drate))
    for j in range(3):
        result = Create_Train_Test_Network(lrate, mrate, drate, epoch, h1n, h2n, train_images, train_labels, test_images, test_labels, j, savefolder)
        results.append(result)
saveResults(results, resultfolder, 'Results_DRate.xls')
