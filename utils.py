import os, sys
import pandas as pd

from skimage import io
import numpy as np
import shutil
import matplotlib.pyplot as plt



def write_images(l_file, image_path, target_path):
    if not os.path.exists(target_path):
        os.mkdir(target_path)
        
    landmarks_frame = pd.read_csv(l_file)
    
    for i in range(len(landmarks_frame.iloc[:, 0])):
        img_name = landmarks_frame.iloc[i, 0]
        img = io.imread(os.path.join(image_path, img_name))
        
        io.imsave(img_name, img)
        shutil.move(img_name, target_path + "/" + img_name)
       

    #write_images("CAT_00.csv", '../../NonlinearAFM_origin/cat-dataset/CAT_00/', 'img')
    
    
def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=50, marker='.', c='r')
    plt.pause(0.001)
    
    
def prints_obj(points, fileName, mesh=None):
    """
    input:
        points--(1256, 3)
        mesh--(2484, 3)
    """
    f = open(fileName,'a')
    
    for i in range(1256):
        f.write("v "+str(points[i, 0])+" "+str(points[i, 1])+" "+str(points[i, 2])+"\n")
    
    if mesh.any() != None:
        mesh = mesh+1
        for i in range(mesh.shape[0]):
            f.write("f "+str(mesh[i, 0])+" "+str(mesh[i, 1])+" "+str(mesh[i, 2])+"\n")
    
    f.close()