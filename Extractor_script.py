import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import os

path = "/home/abhaylal/Desktop/Projects/GAIT/Paper-2/dataFinal"


folder = os.listdir(path)
imports = []
for i in folder:
    images = os.listdir(path + f"/{i}")
    sub_images = []
    for j in images:
        image = cv2.imread(path + f"/{i}" + f"/{j}")
        #image = cv2.resize(image, (480,360))
        sub_images.append(image)
    imports.append(sub_images)
    print(f'folder {i} completed')
print("Import Complete")
"""
def preprocessing(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img/255
        return img
    """
def preprocessing(img):
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img/255
    return img

def preprocessor(images):
    imgFinal = []
    for i in images:
        img = []
        for j in i:
            if isinstance(j, str):
                img.append(preprocessing(cv2.imread(j, cv2.IMREAD_GRAYSCALE)))
            else:
                print("Skipping image:", j, "because it is not a filename string.")
        imgFinal.append(img)
    return imgFinal


"""
def preprocessor(images):
        imgFinal = []
        for i in images:
            img = []
            for j in i:
                img.append(preprocessing(np.asarray(j)))
            imgFinal.append(img)
        return imgFinal
"""
def medianCalc(images):
    median = []
    for i in images:
        if len(i) == 0:
            median.append(None)
            continue
        median_i = []
        for j in i:
            if len(j) == 0:
                median_i.append(None)
                continue
            median_j = []
            for k in j:
                if len(k) == 0:
                    median_j.append(None)
                else:
                    median_j.append(statistics.median(k))
            median_i.append(statistics.median(median_j))
        median.append(statistics.median(median_i))
    return median


def Extractor(images, threshold):
            Res = []
            medianCall = medianCalc(images)
            for i in images:
                subRes = []
                count = 0
                for j in i:
                    subRes2 = []
                    for k in j:
                        if k > medianCall[count]/threshold:
                            subRes2.append(0)
                        else:
                            subRes2.append(k)
                    subRes.append(subRes2)
                count+=1
                Res.append(subRes)
            return np.asarray(Res)

preprocessorRes = preprocessor(imports)
res = Extractor(preprocessorRes,1.85)

path = "/home/abhaylal/Desktop/Projects/GAIT/Paper-2/fin/"
if not os.path.exists(path):
    os.makedirs(path)
count1=1
for i in res:
    count_2 = 0
    if not os.path.exists(path + f"/{count1}"):
            os.makedirs('/home/abhaylal/Desktop/Projects/GAIT/Paper-2/fin/'+str(count1))
    for j in i:
            cv2.imwrite(path + f"/{count1}" + f"/{count1}_{count_2}.jpg",255*j)
            count_2 += 1
    count1+=1


