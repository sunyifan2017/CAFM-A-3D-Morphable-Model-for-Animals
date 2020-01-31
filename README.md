# CAFM: A 3D Morphable Model for Animals

We present Cat-like Animals Facial Model (CAFM) -- a 3D Morphable Model (3DMM) constructed from 50 samples, including lion, tiger, puma, American Shorthair, Abyssinian cat, etc. To the best of our knowledge, CAFM is the first animal morphable model ever constructed. 

The constructed cat-like animal face dataset is released at [Google Drive](https://drive.google.com/drive/folders/1Ar_Wi6QpXxixJiIw_Qju5OWJStAb4TCv?usp=sharing) containing pairs of 2D face images with 15 landmarks and 3D face meshes with projection parameters. 
DATA you have downloaded includes:
- Folder `imgs/`: 1706 images of cat (we will released the rest images later);
- `CAT_00.csv`: contains the 15 landmarks of each images;
- `transformed_Shape.npy`: the 3D meshes of cat images(transformed);
- `Angles.npy`: the Rotation angles;
- `Rotation.npy`: the Rotation matrix;
- `Scaler.npy`: the Scaler parameters;
- `Translation.npy`: the Translation parameters;

FILE Introduction:
- `meanFace.npy`: is the mean shape of cat-like animals;
- `PCA.npy`: is the shape base of PCA;
- `shapeEV.npy`: is the standard deviation;
- `mesh.npy`: is the face of mesh.

## CAFM:
If you wanna generate a specific 3D mesh,

```python
import numpy as np

meanFace = np.load("meanFace.npy")
s_i = np.load("PCA.npy")
triangles = np.load('mesh.npy')

sp = np.zeros((49, 1), dtype=np.float32)
S = meanFace.T + np.dot(s_i.T, sp) # you can set for yourself.
```

## Matching the Morphable Model to Images
First, you should Download [face3d](https://github.com/YadiraF/face3d) which is a remarkable open source code.

In terminal:

>`>`python matching_algorithm.py

Generated 3D mesh will be saved in a folder called "YOU CREATED", named "Transformed _Shape.npy".
