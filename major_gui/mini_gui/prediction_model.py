import pickle
import cv2
import numpy as np

def Nodule_prediction(img_url):
    #print(img_url)
    ct_img = cv2.imread(img_url,0)
    ct_img = cv2.resize(ct_img,(32,32))
    image = np.array(ct_img).flatten()

    loaded_model = pickle.load(open("B:\\major_project_2\\svm_nodule.sav", 'rb'))
    result = loaded_model.predict([image])
    if result[0] == 0 :
        return True
    else:
        return False    

def Cancer_prediction(img_url):
    #print(img_url)
    ct_img = cv2.imread(img_url,0)
    ct_img = cv2.resize(ct_img,(32,32))
    image = np.array(ct_img).flatten()

    loaded_model = pickle.load(open("B:\\major_project_2\\svm_cancer.sav", 'rb'))
    result = loaded_model.predict([image])
    if result[0] == 0 :
        return True
    else:
        return False    

