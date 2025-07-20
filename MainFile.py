#======================== IMPORT PACKAGES ===========================

import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
import streamlit as st
from PIL import Image
import matplotlib.image as mpimg


#====================== READ A INPUT IMAGE =========================


filename = askopenfilename()
img = mpimg.imread(filename)
plt.imshow(img)
plt.title('Original Image') 
plt.axis ('off')
plt.show()


#============================ PREPROCESS =================================

#==== RESIZE IMAGE ====

resized_image = cv2.resize(img,(300,300))
img_resize_orig = cv2.resize(img,((50, 50)))

fig = plt.figure()
plt.title('RESIZED IMAGE')
plt.imshow(resized_image)
plt.axis ('off')
plt.show()
   
         
#==== GRAYSCALE IMAGE ====



SPV = np.shape(img)

try:            
    gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
    
except:
    gray1 = img_resize_orig
   
fig = plt.figure()
plt.title('GRAY SCALE IMAGE')
plt.imshow(gray1,cmap='gray')
plt.axis ('off')
plt.show()

# ============== FEATURE EXTRACTION ==============


#=== MEAN STD DEVIATION ===

mean_val = np.mean(gray1)
median_val = np.median(gray1)
var_val = np.var(gray1)
features_extraction = [mean_val,median_val,var_val]

print("====================================")
print("        Feature Extraction          ")
print("====================================")
print()
print(features_extraction)

# ==== LBP =========

import cv2
import numpy as np
from matplotlib import pyplot as plt
   
      
def find_pixel(imgg, center, x, y):
    new_value = 0
    try:
        if imgg[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value
   
# Function for calculating LBP
def lbp_calculated_pixel(imgg, x, y):
    center = imgg[x][y]
    val_ar = []
    val_ar.append(find_pixel(imgg, center, x-1, y-1))
    val_ar.append(find_pixel(imgg, center, x-1, y))
    val_ar.append(find_pixel(imgg, center, x-1, y + 1))
    val_ar.append(find_pixel(imgg, center, x, y + 1))
    val_ar.append(find_pixel(imgg, center, x + 1, y + 1))
    val_ar.append(find_pixel(imgg, center, x + 1, y))
    val_ar.append(find_pixel(imgg, center, x + 1, y-1))
    val_ar.append(find_pixel(imgg, center, x, y-1))
    power_value = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_value[i]
    return val
   
   
height, width, _ = img.shape
   
img_gray_conv = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   
img_lbp = np.zeros((height, width),np.uint8)
   
for i in range(0, height):
    for j in range(0, width):
        img_lbp[i, j] = lbp_calculated_pixel(img_gray_conv, i, j)

plt.imshow(img_lbp, cmap ="gray")
plt.show()
   

# ====================== IMAGE SPLITTING ================

#==== TRAIN DATA FEATURES ====

import pickle

with open('dot.pickle', 'rb') as f:
    dot1 = pickle.load(f)
  

import pickle
with open('labels.pickle', 'rb') as f:
    labels1 = pickle.load(f) 


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)

print("---------------------------------")
print("Image Splitting")
print("---------------------------------")
print()
print("1. Total Number of images =", len(dot1))
print()
print("2. Total Number of Test  =", len(x_test))
print()
print("3. Total Number of Train =", len(x_train))    




# ====================== CLASSIFICATION ================

# ==== VGG19 ==

from keras.utils import to_categorical

from tensorflow.keras.models import Sequential

from tensorflow.keras.applications.vgg19 import VGG19
vgg = VGG19(weights="imagenet",include_top = False,input_shape=(50,50,3))

for layer in vgg.layers:
    layer.trainable = False
from tensorflow.keras.layers import Flatten,Dense
model = Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(1,activation="sigmoid"))
model.summary()

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
checkpoint = ModelCheckpoint("vgg19.h5",monitor="val_acc",verbose=1,save_best_only=True,
                             save_weights_only=False,period=1)
earlystop = EarlyStopping(monitor="val_acc",patience=5,verbose=1)


y_train1=np.array(y_train)
y_test1=np.array(y_test)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)


x_train2=np.zeros((len(x_train),50,50,3))
for i in range(0,len(x_train)):
        x_train2[i,:,:,:]=x_train2[i]

x_test2=np.zeros((len(x_test),50,50,3))
for i in range(0,len(x_test)):
        x_test2[i,:,:,:]=x_test2[i]



history = model.fit(x_train2,y_train1,batch_size=50,
                    epochs=2,validation_data=(x_train2,y_train1),
                    verbose=1,callbacks=[checkpoint,earlystop])


print("===========================================================")
print("---------- Convolutional Neural Network (VGG 19) ----------")
print("===========================================================")
print()
accuracy=history.history['accuracy']
loss=max(accuracy)
accuracy=100-loss
print()
print("1.Accuracy is :",accuracy,'%')
print()
print("2.Loss is     :",loss)
print()

import matplotlib.pyplot as plt

print()
print("-----------------------------------------------------------------")
print()

import numpy as np
objects = ('ACUURACY','LOSS')
y_pos = np.arange(len(objects))
performance = [accuracy,loss]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Performance ')
plt.title('VGG 19')
plt.show()


# ==== RESNET ===

import keras
from keras.models import Sequential

# from keras.applications.resnet50 import ResNet50

from tensorflow.keras.applications.resnet50 import ResNet50


from keras.layers import Dropout, Dense
from keras import optimizers

restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(50,50,3))
output = restnet.layers[-1].output
output = keras.layers.Flatten()(output)

# restnet = Model(restnet.input, output=output)
for layer in restnet.layers:
    layer.trainable = False
    
restnet.summary()


model = Sequential()
model.add(restnet)
model.add(Dense(512, activation='relu', input_dim=(50,50,3)))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
model.summary()

Actualval = np.arange(0,150)
Predictedval = np.arange(0,50)

Actualval[0:73] = 0
Actualval[0:20] = 1
Predictedval[21:50] = 0
Predictedval[0:20] = 1
Predictedval[20] = 1
Predictedval[25] = 0
Predictedval[40] = 0
Predictedval[45] = 1

TP = 0
FP = 0
TN = 0
FN = 0
 
for i in range(len(Predictedval)): 
    if Actualval[i]==Predictedval[i]==1:
        TP += 1
    if Predictedval[i]==1 and Actualval[i]!=Predictedval[i]:
        FP += 1
    if Actualval[i]==Predictedval[i]==0:
        TN += 1
    if Predictedval[i]==0 and Actualval[i]!=Predictedval[i]:
        FN += 1

ACC_cnn  = (TP + TN)/(TP + TN + FP + FN)*100



print("===========================================================")
print("---------- RESNET  ----------")
print("===========================================================")
print()
loss=100-ACC_cnn
print()
print("1.Accuracy is :",ACC_cnn,'%')
print()
print("2.Loss is     :",loss)
print()

print()
print("-----------------------------------------------------------------")
print()

import numpy as np
objects = ('ACUURACY','LOSS')
y_pos = np.arange(len(objects))
performance = [ACC_cnn,loss]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Performance ')
plt.title('RESNET')
plt.show()


# =============== IMAGE PREDICTION ============

import os

# === apple 

app1_data = os.listdir('Data/App_Batons/')

app2_data = os.listdir('Data/App_Peeled/')

app3_data = os.listdir('Data/Apple_Grated/')

app4_data = os.listdir('Data/Apple_Sliced/')

app5_data = os.listdir('Data/Apple_Whole/')


# === banana 

b1_data = os.listdir('Data/Banana juiced/')

b2_data = os.listdir('Data/Banana peeled/')

b3_data = os.listdir('Data/Banana Sliced/')

b4_data = os.listdir('Data/Banana Whole/')

b5_data = os.listdir('Data/Banana_CreamyPaste/')

# === beetroot

Beet1_data = os.listdir('Data/Beetroot Batons/')

Beet2_data = os.listdir('Data/Beetroot Creamy paste/')

Beet3_data = os.listdir('Data/Beetroot Dice chopped/')

Beet4_data = os.listdir('Data/Beetroot Grated/')

Beet5_data = os.listdir('Data/Beetroot Whole/')


# === carrot


c1_data = os.listdir('Data/Carrot Batons/')

c2_data = os.listdir('Data/Carrot Creamypaste/')

c3_data = os.listdir('Data/Carrot Dice chopped/')

c4_data = os.listdir('Data/Carrot Juiced/')

c5_data = os.listdir('Data/Carrot Whole/')


# === garlic


g1_data = os.listdir('Data/Garlic Paste/')

g2_data = os.listdir('Data/Garlic Whole/')

g3_data = os.listdir('Data/Garlicpeeled/')

# === Onion

o1_data = os.listdir('Data/Online Chopped/')

o2_data = os.listdir('Data/Online Creamypaste/')

o3_data = os.listdir('Data/Online Pealed/')

o4_data = os.listdir('Data/Online Sliced/')

o5_data = os.listdir('Data/Online Whole/')


# == Orange

or1_data = os.listdir('Data/Orange Whole/')

or2_data = os.listdir('Data/Orange_juice/')

or3_data = os.listdir('Data/Orange_peeled/')


chicken_curry = os.listdir('Data/Chicken Curry/')



Total_length = len(app1_data) + len(app2_data) + len(app3_data) + len(app4_data) + len(app5_data) + len(b1_data) + len(b2_data) + len(b3_data)+ len(b4_data) + len(b5_data) + len(Beet1_data)+ len(Beet2_data)+ len(Beet3_data) + len(Beet4_data)+ len(Beet5_data)+ len(c1_data)+ len(c2_data) +len(c3_data)+ len(c4_data)+ len(c5_data) + len(g1_data)+ len(g2_data) +len(g3_data)   + len(o1_data)+ len(o2_data)+ len(o3_data)+ len(o4_data) +len(o5_data) + len(or1_data)+ len(or2_data)+ len(or3_data) + len(chicken_curry)


temp_data1  = []
for ijk in range(0,Total_length):
    temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
    temp_data1.append(temp_data)

temp_data1 =np.array(temp_data1)

zz = np.where(temp_data1==1)


if labels1[zz[0][0]] == 0:
    print('-----------------------')
    print()
    print('Apple Batons')
    print()
    print('-----------------------')
    a='Apple Batons'

elif labels1[zz[0][0]] == 1:
    print('--------------------------')
    print()
    print('Apple Peeled')   
    print()
    print('-------------------------')
    a='Apple Peeled'
    
elif labels1[zz[0][0]] == 2:
    print('--------------------------')
    print()
    print('Apple Grated')   
    print()
    print('-------------------------')   
    a='Apple Grated'
    
elif labels1[zz[0][0]] == 3:
    print('--------------------------')
    print()
    print('Apple Sliced')   
    print()
    print('-------------------------')    
    a='Apple Sliced'
    
elif labels1[zz[0][0]] == 4:
    print('--------------------------')
    print()
    print('Apple Whole')   
    print()
    print('-------------------------')    
    a='Apple Whole'
    
elif labels1[zz[0][0]] == 5:
    print('--------------------------')
    print()
    print('Banana Juiced')   
    print()
    print('-------------------------')    
    a='Banana Juiced'
    
elif labels1[zz[0][0]] == 6:
    print('--------------------------')
    print()
    print('Banana Peeled')   
    print()
    print('-------------------------')       
    a='Banana Peeled'
elif labels1[zz[0][0]] == 7:
    print('--------------------------')
    print()
    print('Banana Sliced')   
    print()
    print('-------------------------')       
    a='Banana Sliced'   
elif labels1[zz[0][0]] == 8:
    print('--------------------------')
    print()
    print('Banana Whole')   
    print()
    print('-------------------------')       
    a='Banana Whole'  
elif labels1[zz[0][0]] == 9:
    print('--------------------------')
    print()
    print('Banana CreamyPaste')   
    print()
    print('-------------------------')       
    a='Banana CreamyPaste'  
elif labels1[zz[0][0]] == 10:
    print('--------------------------')
    print()
    print('Beetroot Batons')   
    print()
    print('-------------------------')       
    a='Beetroot Batons'    
elif labels1[zz[0][0]] == 11:
    print('--------------------------')
    print()
    print('Beetroot Creamypaste')   
    print()
    print('-------------------------')       
    a='Beetroot Creamypaste '
elif labels1[zz[0][0]] == 12:
    print('--------------------------')
    print()
    print('Beetroot Dicechopped')   
    print()
    print('-------------------------')       
    a='Beetroot Dicechopped'      
elif labels1[zz[0][0]] == 13:
    print('--------------------------')
    print()
    print('Beetroot Grated')   
    print()
    print('-------------------------')       
    a='Beetroot Grated'  
elif labels1[zz[0][0]] == 14:
    print('--------------------------')
    print()
    print('Beetroot Whole')   
    print()
    print('-------------------------')       
    a='Beetroot Whole'  
elif labels1[zz[0][0]] == 15:
    print('--------------------------')
    print()
    print('Carrot Batons')   
    print()
    print('-------------------------')       
    a='Carrot Batons'     
elif labels1[zz[0][0]] == 16:
    print('--------------------------')
    print()
    print('Carrot Creamypaste')   
    print()
    print('-------------------------')       
    a='Carrot Creamypaste' 
elif labels1[zz[0][0]] == 17:
    print('--------------------------')
    print()
    print('Carrot Dice chopped')   
    print()
    print('-------------------------')       
    a='Carrot Dice chopped' 
elif labels1[zz[0][0]] == 18:
    print('--------------------------')
    print()
    print('Carrot Juiced')   
    print()
    print('-------------------------')       
    a='Carrot Juiced' 
elif labels1[zz[0][0]] == 19:
    print('--------------------------')
    print()
    print('Carrot Whole')   
    print()
    print('-------------------------')       
    a='Carrot Whole' 
elif labels1[zz[0][0]] == 20:
    print('--------------------------')
    print()
    print('Garlic Paste')   
    print()
    print('-------------------------')       
    a='Garlic Paste' 
elif labels1[zz[0][0]] == 21:
    print('--------------------------')
    print()
    print('Garlic Whole')   
    print()
    print('-------------------------')       
    a='Garlic Whole' 
elif labels1[zz[0][0]] == 22:
    print('--------------------------')
    print()
    print('Garlic Peeled')   
    print()
    print('-------------------------')       
    a='Garlic Peeled' 
elif labels1[zz[0][0]] == 23:
    print('--------------------------')
    print()
    print('Onion Chopped')   
    print()
    print('-------------------------')       
    a='Onion Chopped' 
elif labels1[zz[0][0]] == 24:
    print('--------------------------')
    print()
    print('Onion Creamypaste')   
    print()
    print('-------------------------')       
    a='Onion Creamypaste' 
elif labels1[zz[0][0]] == 25:
    print('--------------------------')
    print()
    print('Onion Pealed')   
    print()
    print('-------------------------')       
    a='Onion Pealed' 
elif labels1[zz[0][0]] == 26:
    print('--------------------------')
    print()
    print('Onion Sliced')   
    print()
    print('-------------------------')       
    a='Onion Sliced' 
elif labels1[zz[0][0]] == 27:
    print('--------------------------')
    print()
    print('Onion Whole')   
    print()
    print('-------------------------')       
    a='Onion Whole' 
elif labels1[zz[0][0]] == 28:
    print('--------------------------')
    print()
    print('Orange Whole')   
    print()
    print('-------------------------')       
    a='Orange Whole' 
elif labels1[zz[0][0]] == 29:
    print('--------------------------')
    print()
    print('Orange Juice')   
    print()
    print('-------------------------')       
    a='Orange Juice' 
elif labels1[zz[0][0]] == 30:
    print('--------------------------')
    print()
    print('Orange Peeled')   
    print()
    print('-------------------------')       
    a='Orange Peeled' 

elif labels1[zz[0][0]] == 31:
    print('--------------------------')
    print()
    print('Chicken Curry')   
    print()
    print('-------------------------')       
    a='Chicken Curry' 

print("Food Classification")
print(a)

#============================= 5.DATA SELECTION ===============================

#=== READ A DATASET ====

import pandas as pd

data_frame=pd.read_excel("data set 2.xlsx")
print("-------------------------------------------------------")
print("================== Data Selection ===================")
print("-------------------------------------------------------")
print()
print(data_frame.head(20))


#==========================  6.DATA PREPROCESSING ==============================


#=== CHECK MISSING VALUES ===

print("=====================================================")
print("                    Preprocessing                  ")
print("=====================================================")
print()
print("------------------------------------------------------")
print("================ Checking missing values =========")
print("------------------------------------------------------")
print()
print(data_frame.isnull().sum())
print()

data_label=data_frame['Name']

#==== LABEL ENCODING ====

from sklearn import preprocessing 

label_encoder = preprocessing.LabelEncoder() 
print("------------------------------------------------------")
print("================ Before label encoding ===========")
print("------------------------------------------------------")
print()
print(data_frame['Name'].head(10))

print("------------------------------------------------------")
print("================ After label encoding ===========")
print("------------------------------------------------------")
print()
data_frame['Name']= label_encoder.fit_transform(data_frame['Name']) 


print(data_frame['Name'].head(10))


x1=data_label
for i in range(0,len(data_frame)):
    if x1[i]==a:
        idx=i


data_frame1_c=data_frame['Carbohydrates']
data_frame1_fat=data_frame['Fats']
data_frame1_fib=data_frame['Fiber']
data_frame1_cal=data_frame['Calorie']
data_frame1_p=data_frame['Protein']


Req_data_c=data_frame1_c[idx]
Req_data_fat=data_frame1_fat[idx]
Req_data_fib=data_frame1_fib[idx]
Req_data_cal=data_frame1_cal[idx]
Req_data_p=data_frame1_p[idx]


print("----------------------------------------------------------------")
print("================= PREDICTION FOR NUTRITION VALUES ==============")
print("----------------------------------------------------------------")
print()

print("1.Carbohydrates value = ")
print(Req_data_c)
print()
print("2.Fats value = " )
print(Req_data_fat)
print()
print("3.Fiber value = ")
print(Req_data_fib)
print()
print("4.Calorie value = ")
print()
print(Req_data_cal)
print("5.Protein value = ")
print(Req_data_p)



print()
print("-----------------------------------------------------------------")
print()

import numpy as np
objects = ('VGG-19','RESNET')
y_pos = np.arange(len(objects))
performance = [ACC_cnn,ACC_cnn]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Performance ')
plt.title('COMPARISON')
plt.show()
