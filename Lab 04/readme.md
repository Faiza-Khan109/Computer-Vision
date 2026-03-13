**Image Enhancement for Computer Vision Applications**

This repository contains Python implementations of several image enhancement techniques used in computer vision.



These techniques improve image quality under challenging environmental conditions.



**Enhancement Methods**

Contrast Stretching

Expands the dynamic range of pixel intensities.



**Formula:**



I\_new = ((I - I\_min)/(I\_max - I\_min)) \* 255



**Gamma Transformation**

A nonlinear transformation used to control brightness.



**Formula:**



I\_out = 255 \* (I\_in/255)^gamma



**Histogram Equalization**

Redistributes pixel intensity values to improve global contrast.



**CLAHE**

Contrast Limited Adaptive Histogram Equalization enhances local contrast while preventing noise amplification.



**Images Used in the Project**

X-ray medical images

Low-light surveillance images

Foggy traffic scenes

Satellite thermal images

Underwater photographs



**Output**

Each script displays:



Original Image

Enhanced Image

Histogram Analysis (where applicable)

Visualization is performed using Matplotlib.

