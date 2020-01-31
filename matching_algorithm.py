#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, sys
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from config import FLAGS

sys.path.append("../face3d")
import face3d
from face3d import mesh

def main():
    csv_file = FLAGS['csv_file']
    img_folder = FLAGS['img_folder']
    lamb = FLAGS['lamb']
    
    landmarks_index_3D = [589, 775, 780, 716, 73, 136, 120, 1104, 1094, 463, 453, 561, 581]
    
    landmarks_frame = pd.read_csv(csv_file)
    Scaler = []
    Angles = []
    Rotation = []
    Translation =[]
    Shape = []
    Shape_origin = []
    for n in range(len(landmarks_frame)):
        meanFace = np.load("meanFace.npy")
        s_i = np.load("PCA.npy")
        triangles = np.load('mesh.npy')

        img_name = landmarks_frame.iloc[n, 0]
        landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        img_size = io.imread(os.path.join(img_folder, img_name)).shape

        print(n)
        print('Image name: {}'.format(img_name))
        print('Landmarks shape: {}'.format(landmarks.shape))
        print('First 4 Landmarks: {}'.format(landmarks[:4]))
        print('Image size: {}'.format(img_size))

        landmarks = landmarks[2:]
        landmarks[:, 1] = landmarks[:, 1]*(-1) + img_size[0] #shape: (13, 2)

        index = []
        if 0 in landmarks:
            for i in range(landmarks.shape[0]):
                if 0 in landmarks[i]:
                    index.append(i)
            landmarks = np.delete(landmarks, index, axis=0)
            kpt_index = [i-1 for i in landmarks_index_3D]
            kpt_index = [kpt_index[i] for i in range(len(kpt_index)) if i not in index]

        else:
            kpt_index = [i-1 for i in landmarks_index_3D]

        sp = np.zeros((49, 1), dtype=np.float32)

        shapeMU = meanFace.reshape(-1, 3)[kpt_index].reshape(-1, 1) #shape: (39, 1)
        shapePC = s_i.reshape(-1, 3, 49)[kpt_index].reshape(-1, 49) #shape: (39, 49)

        #shapeEV, expression = np.zeros((3, landmarks.shape[0]))
        shapeEV = np.load("shapeEV.npy")
        expression = np.zeros((3, landmarks.shape[0]))

        for i in range(10):   
            X = shapeMU + np.dot(shapePC, sp) #shape: (39, 1)

            P = mesh.transform.estimate_affine_matrix_3d22d(X.reshape(-1, 3), landmarks)

            s, R, t = mesh.transform.P2sRt(P)

            sp = face3d.morphable_model.fit.estimate_shape(landmarks.T, shapeMU, shapePC, shapeEV, expression,
                                                          s, R, t[:2], lamb)
            angles = mesh.transform.matrix2angle(R)

        S = meanFace.T + np.dot(s_i.T, sp)
        S_ = mesh.transform.similarity_transform(S.reshape(-1, 3), s, R, t)
        S_[:, 2] = S_[:, 2] - min(S_[:, 2])

        Scaler.append(s)
        Rotation.append(R)
        Angles.append(angles)
        Translation.append(t)
        Shape.append(S_)
        Shape_origin.append(S)
        
    
    path = "YOU CREATED"
    folder = os.path.exists(path)
 
    if not folder:                   
        os.makedirs(path)
    
    os.chdir(path)
    np.save("transformed_Shape.npy", Shape)
    np.save("generated_Shape.npy", Shape_origin)
    np.save("Scaler.npy", Scaler)
    np.save("Rotation.npy", Rotation)
    np.save("Translation.npy", Translation)
    np.save("Angles.npy", Angles)    
    
if __name__ == "__main__":
    main()

