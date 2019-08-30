#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import numpy as np
from IPython.display import Image
import glob
import io
#import matplotlib.pyplot as plt
import time
import cv2
from imutils import paths
from skimage import io
import imutils
from skimage import feature
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
#%matplotlib inline


# In[44]:


Start = time.time
smalldata = "/data/cmpe255-sp19/data/pr2/traffic-small/train/"


# In[45]:


labellistfile = "/data/cmpe255-sp19/data/pr2/traffic-small/train.labels"


# In[46]:


labeldf = pd.read_csv(labellistfile, header=None)


# In[47]:


labeldf = labeldf.rename(index=str, columns={0: "label"})


# In[48]:


labeldf.head()


# In[49]:


#labeldf.groupby('label').size().plot.bar()


# In[50]:


#Image(filename=smalldata+'000902.jpg',width=500)


# In[51]:


import os
image_list = sorted(os.listdir(smalldata))


# In[52]:


len(image_list)


# In[53]:


labeldf['name'] = image_list


# In[54]:


labeldf.head()


# In[55]:


image1 = cv2.imread(smalldata+'000902.jpg')


# In[56]:


image1_rs = cv2.resize(image1, (64, 64))


# In[57]:


#plt.imshow(image1_rs)


# In[58]:


gray1 = cv2.cvtColor(image1_rs, cv2.COLOR_BGR2GRAY)
edged1 = imutils.auto_canny(gray1)


# In[59]:


#plt.imshow(gray1)


# In[60]:


#plt.imshow(edged1)


# In[61]:


logo1 = cv2.resize(edged1, (64, 64))


# In[62]:


#plt.imshow(logo1)


# In[63]:


##Histogram features:

img = cv2.imread(smalldata+'000902.jpg',0)


# In[64]:


hist = cv2.calcHist([img],[0],None,[16],[0,16])


# In[65]:


#plt.hist(img.ravel(),16,[0,16]); plt.show()


# In[66]:


hist.T.shape
type(hist)


# In[67]:


histdf=pd.DataFrame(hist)


# In[68]:


histdf.shape


# In[69]:


H1 = feature.hog(logo1, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")


# In[70]:


type(H1)


# In[71]:


len(H1)


# In[72]:


image2 = cv2.imread(smalldata+'000007.jpg')


# In[73]:


#plt.imshow(image2)


# In[74]:


image2_rs = cv2.resize(image2, (64, 64))
gray2 = cv2.cvtColor(image2_rs, cv2.COLOR_BGR2GRAY)
edged2 = imutils.auto_canny(gray2)
# logo2 = cv2.resize(edged2, (100, 100))
H2 = feature.hog(edged2, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")


# In[75]:


histlist= hist.tolist()
H2list= H2.tolist()


# In[76]:


H3=histlist+H2list


# In[77]:


#plt.imshow(edged2)


# In[78]:


labels = []
fdf = pd.DataFrame()


# In[ ]:





# In[ ]:





# In[79]:


data = []
labels = []
finaldf = pd.DataFrame()


# In[80]:


# Extract features of Training Data
print("Derive features of Small Data")
for imagePath in paths.list_images(smalldata):
    make = imagePath.split("/")[-1]
    labels.append(make)
    image = cv2.imread(imagePath)
    hist = cv2.calcHist([image],[0,1,2], None, [8,8,8],[0,256,0,256,0,256])
    hist = cv2.normalize(hist,hist).flatten()
    #resized = cv2.resize(image, (64, 64))
    #gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    #edged = imutils.auto_canny(gray)
    #H = feature.hog(edged, orientations=9, pixels_per_cell=(10, 10),cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
    data.append(hist)


# In[41]:


len(data[2])


# In[42]:


finaldf['name'] = labels
finaldf['data'] = data


# In[43]:


resultdf = pd.merge(finaldf,labeldf, on='name')


# In[ ]:


resultdf.head()


# In[ ]:


#Small Data feature vector
datadf = pd.DataFrame(resultdf.data.tolist())
labeldf1 = resultdf['label']


# In[ ]:


datadf.head()


# In[42]:


testdata = "/data/cmpe255-sp19/data/pr2/traffic-small/test/"
testlabel = "/data/cmpe255-sp19/data/pr2/traffic-small/test.labels"

testlabeldf = pd.read_csv(testlabel, header=None)
testlabeldf = testlabeldf.rename(index=str, columns={0: "label"})

test_image_list = sorted(os.listdir(testdata))
testlabeldf['name'] = test_image_list
test =[]
label_test =[]
finaldf_test = pd.DataFrame()


# In[ ]:


# Extract features of Test Data
print("Derive features of Small Test Data")
for imagePath in paths.list_images(testdata):
    make = imagePath.split("/")[-1]
    label_test.append(make)
    image = cv2.imread(imagePath)
    hist1 = cv2.calcHist([image],[0,1,2], None, [8,8,8],[0,256,0,256,0,256])
    hist1 = cv2.normalize(hist1,hist1).flatten()
    #resized = cv2.resize(image, (64, 64))
    #gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    #edged = imutils.auto_canny(gray)
    #H1 = feature.hog(edged, orientations=9, pixels_per_cell=(10, 10),cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
    test.append(hist1)


# In[ ]:


finaldf_test['name'] = label_test
finaldf_test['data'] = test
resultdf_test = pd.merge(finaldf_test,testlabeldf, on='name')
datadf_test = pd.DataFrame(resultdf.data.tolist())
labeldf1_test = resultdf['label']


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=5)


# In[ ]:


knn.fit(datadf,labeldf1)


# In[ ]:


pred = knn.predict(datadf_test)


# In[ ]:


print(classification_report(labeldf1_test,pred))


# In[81]:


#Process the Big Data file images
bigdatatrain = "/data/cmpe255-sp19/data/pr2/traffic/train/"
bigdatalabels = "/data/cmpe255-sp19/data/pr2/traffic/train.labels"


# In[82]:


bigdatalabeldf = pd.read_csv(bigdatalabels, header=None)
bigdatalabeldf = bigdatalabeldf.rename(index=str, columns={0: "label"})
bigdata_image_list = sorted(os.listdir(bigdatatrain))
bigdatalabeldf['name'] = bigdata_image_list
bigdata =[]
label_bigdata =[]
finaldf_bigdata = pd.DataFrame()


# In[ ]:


# Extract features of Big Data
print("Derive features of Big Data")
for imagePath in paths.list_images(testdata):
    make = imagePath.split("/")[-1]
    label_bigdata.append(make)
    image = cv2.imread(imagePath)
    hist2 = cv2.calcHist([image],[0,1,2], None, [8,8,8],[0,256,0,256,0,256])
    hist2 = cv2.normalize(hist2,hist2).flatten()
    #resized = cv2.resize(image, (64, 64))
    #gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    #edged = imutils.auto_canny(gray)
    #H1 = feature.hog(edged, orientations=9, pixels_per_cell=(10, 10),cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
    bigdata.append(hist2)


# In[ ]:


finaldf_bigdata['name'] = label_bigdata
finaldf_bigdata['data'] = bigdata
resultdf = pd.merge(finaldf_bigdata,bigdatalabeldf, on='name')
bigdata = pd.DataFrame(resultdf.data.tolist())
label_bigdata = resultdf['label']


# In[ ]:


#Process the Big Test file images
bigdatatest = "/data/cmpe255-sp19/data/pr2/traffic/test/"
#bigTest_image_list = sorted(os.listdir(bigdatatest))
Testdata =[]
label_Testdata =[]
finaldf_Testdata = pd.DataFrame()


# In[ ]:


# Extract features of Big Test Data
print("Derive features of Big Test Data")
for imagePath in paths.list_images(bigdatatest):
    make = imagePath.split("/")[-1]
    label_Testdata.append(make)
    image = cv2.imread(imagePath)
    hist2 = cv2.calcHist([image],[0,1,2], None, [8,8,8],[0,256,0,256,0,256])
    hist2 = cv2.normalize(hist2,hist2).flatten()
    #resized = cv2.resize(image, (64, 64))
    #gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    #edged = imutils.auto_canny(gray)
    #H1 = feature.hog(edged, orientations=9, pixels_per_cell=(10, 10),cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
    Testdata.append(hist2)


# In[ ]:


finaldf_Testdata = finaldf_Testdata.sort_values('name')
finaldf_Testdata['name'] = label_Testdata
finaldf_Testdata['data'] = Testdata
TestDatadf = pd.DataFrame(finaldf_Testdata.data.tolist())


# In[ ]:


print("Apply Classification Algorithm for Big Data and Test Prediction")
knn.fit(datadf,labeldf1)
pred = knn.predict(datadf_test)


# In[ ]:


print("Write the output file")
    with open('255_pro2_output_4thApr.txt', 'w') as f:
        for item in pred:
            f.write("%s\n" % item)


# In[ ]:


End = time.time
print("Time to run the program")
print(End-Start)

