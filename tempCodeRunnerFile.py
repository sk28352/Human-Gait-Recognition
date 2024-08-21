def preprocessing(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img/255
        return img
    
def preprocessor(images):
        imgFinal = []
        for i in images:
            img = []
            for j in i:
                img.append(preprocessing(np.asarray(j)))
            imgFinal.append(img)
        return imgFinal