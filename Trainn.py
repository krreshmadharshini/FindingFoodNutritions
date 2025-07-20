import os 
import cv2
import numpy as np
import matplotlib.image as mpimg

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






dot1= []
labels1 = []
for img in app1_data:
        # print(img)
        img_1 = mpimg.imread('Data/App_Batons/' + "/" + img)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(0)

        
for img in app2_data:
    try:
        img_2 = mpimg.imread('Data/App_Peeled/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(1)
    except:
        None

for img in app3_data:
    try:
        img_2 = mpimg.imread('Data/Apple_Grated'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(2)
    except:
        None
        
        
for img in app4_data:
    try:
        img_2 = mpimg.imread('Data/Apple_Sliced/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(3)
    except:
        None


        
for img in app5_data:
    try:
        img_2 = mpimg.imread('Data/Apple_Whole/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(4)
    except:
        None

############        
for img in b1_data:
    try:
        img_2 = mpimg.imread('Data/Banana juiced/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(5)
    except:
        None
        
for img in b2_data:
    try:
        img_2 = mpimg.imread('Data/Banana peeled/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(6)
    except:
        None

for img in b3_data:
    try:
        img_2 = mpimg.imread('Data/Banana Sliced/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(7)
    except:
        None

for img in b4_data:
    try:
        img_2 = mpimg.imread('Data/Banana Whole/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(8)
    except:
        None

        
for img in b5_data:
    try:
        img_2 = mpimg.imread('Data/Banana_CreamyPaste/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(9)
    except:
        None

# =======
        
for img in Beet1_data:
    try:
        img_2 = mpimg.imread('Data/Beetroot Batons/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(10)
    except:
        None

for img in Beet2_data:
    try:
        img_2 = mpimg.imread('Data/Beetroot Creamy paste/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(11)
    except:
        None

for img in Beet3_data:
    try:
        img_2 = mpimg.imread('Data/Beetroot Dice chopped/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(12)
    except:
        None

for img in Beet4_data:
    try:
        img_2 = mpimg.imread('Data/Beetroot Grated/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(13)
    except:
        None

for img in Beet5_data:
    try:
        img_2 = mpimg.imread('Data/Beetroot Whole/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(14)
    except:
        None

# ===========
        
for img in c1_data:
    try:
        img_2 = mpimg.imread('Data/Carrot Batons/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(15)
    except:
        None

for img in c2_data:
    try:
        img_2 = mpimg.imread('Data/Carrot Creamypaste/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(16)
    except:
        None

for img in c3_data:
    try:
        img_2 = mpimg.imread('Data/Carrot Dice chopped/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(17)
    except:
        None

for img in c4_data:
    try:
        img_2 = mpimg.imread('Data/Carrot Juiced/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(18)
    except:
        None

for img in c5_data:
    try:
        img_2 = mpimg.imread('Data/Carrot Whole/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(19)
    except:
        None

# ========
        
for img in g1_data:
    try:
        img_2 = mpimg.imread('Data/Garlic Paste/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(20)
    except:
        None

for img in g2_data:
    try:
        img_2 = mpimg.imread('Data/Garlic Whole/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(21)
    except:
        None

for img in g3_data:
    try:
        img_2 = mpimg.imread('Data/Garlicpeeled/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(22)
    except:
        None


# =========
        
for img in o1_data:
    try:
        img_2 = mpimg.imread('Data/Online Chopped/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(23)
    except:
        None

for img in o2_data:
    try:
        img_2 = mpimg.imread('Data/Online Creamypaste/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(24)
    except:
        None

for img in o3_data:
    try:
        img_2 = mpimg.imread('Data/Online Pealed/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(25)
    except:
        None

for img in o4_data:
    try:
        img_2 = mpimg.imread('Data/Online Sliced/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(26)
    except:
        None

for img in o5_data:
    try:
        img_2 = mpimg.imread('Data/Online Whole/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(27)
    except:
        None
    
    
# ======
        
for img in or1_data:
    try:
        img_2 = mpimg.imread('Data/Orange Whole/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(28)
    except:
        None
        
for img in or1_data:
    try:
        img_2 = mpimg.imread('Data/Orange_juice/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(29)
    except:
        None        
        
for img in or1_data:
    try:
        img_2 = mpimg .imread('Data/Orange_peeled/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(30)
    except:
        None           

        
for img in chicken_curry:
    try:
        img_2 = mpimg .imread('Data/Chicken Curry/'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(31)
    except:
        None   



import pickle
with open('dot.pickle', 'wb') as f:
    pickle.dump(dot1, f)
    
with open('labels.pickle', 'wb') as f:
    pickle.dump(labels1, f)        
        
