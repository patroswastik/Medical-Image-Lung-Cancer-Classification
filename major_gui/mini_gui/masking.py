

from PIL import Image
import os
from sklearn.cluster import KMeans
from skimage import measure
from skimage.transform import resize
import pydicom
import csv
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage
import SimpleITK as sitk
import numpy as np
import skimage as sk
import skimage.morphology as morph
from scipy import ndimage






def make_lungmask(img, display=False):
    row_size= img.shape[0]
    col_size = img.shape[1]
    
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean
    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
    # We don't want to accidentally clip the lung.

    eroded = morph.erosion(thresh_img,np.ones([3,3]))
    dilation = morph.dilation(eroded,np.ones([8,8]))

    labels = measure.label(dilation) # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 0

    #
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    #
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morph.dilation(mask,np.ones([10,10])) # one last dilation

    """if (display):
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask*img, cmap='gray')
        ax[2, 1].axis('off')
        
        plt.show()"""
    return mask*img


# In[30]:


def resample(image,dicom_file,new_spacing=[1,1]):
    # Determine current pixel spacing
    SliceThickness = dicom_file.SliceThickness
    PixelSpacing=dicom_file.PixelSpacing
    spacing = map(float, (list(PixelSpacing)))
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    return image

directory = "B:\\major_project_2\\major_gui"


def pre_process_img(fileloc):
    directory = "B:\\major_project_2\\major_gui"
    if fileloc.endswith(".dcm"):
        ds1=pydicom.filereader.dcmread(directory+fileloc)
        lung_image = ds1.pixel_array*ds1.RescaleSlope+ds1.RescaleIntercept
        resample_image = resample(lung_image,ds1)
    else:
        resample_image = cv2.imread(os.path.join(directory,fileloc))
    final_image = make_lungmask(resample_image)
    
    result_image = Image.fromarray((final_image * 255).astype(np.uint8))
    
    
    #print (fileloc)    
    tmp = directory+"\\static\\masked_image\\"+fileloc[7:-4]
    tmp+=".jpg"
    #print ("temp is    :    ",tmp)
    result_image.save(tmp)
    return tmp
    
    







    

        

