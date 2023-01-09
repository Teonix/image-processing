# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 11:17:22 2021

@author: Teonix
"""
#Spyder 4.1.5

# ΕΙΣΟΔΟΣ ΒΙΒΛΙΟΘΗΚΩΝ
import cv2
import os
import numpy as np

# ΣΥΝΑΡΤΗΣΗ ΠΡΟΣΘΗΚΗΣ ΓΚΑΟΥΣΙΑΝΟΥ ΘΟΡΥΒΟΥ
def gauss(image):
     row,col,ch= image.shape
     mean = 0
     var = 400
     sigma = var**0.5
     gauss = np.random.normal(mean,sigma,(row,col,ch))
     gauss = gauss.reshape(row,col,ch)
     noisy = image + gauss
     return noisy

directory = r'' # ΔΗΛΩΣΗ ΤΟΥ ΜΟΝΟΠΑΤΙΟΥ ΠΟΥ ΘΑ ΑΠΟΘΗKEΥΤΟΥΝ ΟΙ ΕΙΚΟΝΕΣ

os.chdir(directory) # ΕΠΙΛΟΓΗ ΤΟΥ ΜΟΝΟΠΑΤΙΟΥ ΠΟΥ ΦΤΙΑΞΑΜΕ ΠΙΟ ΠΑΝΩ

### ΕΡΩΤΗΜΑ 1 ###

img = cv2.imread('spider-man.jpg') # ΕΙΣΟΔΟΣ ΤΗΣ ΕΙΚΟΝΑΣ spider-man ΣΤΗ ΜΕΤΑΒΛΗΤΗ IMG
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # ΜΕΤΑΤΡΟΠΗ ΤΗΣ ΕΙΚΟΝΑΣ ΣΕ GRAYSCALE

# ΕΜΦΑNΙΣΗ ΤΩΝ ΕΙΚΟΝΩΝ ORIGINAL,GRAYSCALE
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL) 
cv2.imshow("Original Image", img) 
cv2.resizeWindow("Original Image", 700, 393) 

cv2.namedWindow("Grayscale Image", cv2.WINDOW_NORMAL) 
cv2.imshow("Grayscale Image", img_gray) 
cv2.resizeWindow("Grayscale Image", 700, 393) 
cv2.imwrite("spideyGrayscale.jpg", img_gray) 

cv2.waitKey() 
cv2.destroyWindow("Original Image")
cv2.destroyWindow("Grayscale Image") 


### ΕΡΩΤΗΜΑ 2 ###

# ΕΦΑΡΜΟΓΗ ΤΟΥ CANNY EDGES ΣΤΗΝ ΕΙΚΟΝΑ
canny_edges = cv2.Canny(img_gray, 10, 10, edges=None, apertureSize=5)

# ΕΜΦΑNΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ ΜΕ CANNY EDGES
cv2.namedWindow("Canny Edges Image", cv2.WINDOW_NORMAL)
cv2.imshow('Canny Edges Image', canny_edges)
cv2.resizeWindow("Canny Edges Image", 700, 393) 
cv2.imwrite("spideyCanny.jpg", canny_edges) 

cv2.waitKey()  
cv2.destroyWindow("Canny Edges Image") 


### ΕΡΩΤΗΜΑ 3 ###

# ΕΙΣΑΓΩΓΗ ΓΚΑΟΥΣΙΑΝΟΥ ΘΟΡΥΒΟΥ ΜΕ ΤΗ ΒΟΗΘΕΙΑ ΤΗΣ ΣΥΝΑΡΤΗΣΗΣ
gauss_noise=gauss(img)
gauss_noise = np.floor(np.abs(gauss_noise)).astype('uint8')
gauss_noise_gray = cv2.cvtColor(gauss_noise, cv2.COLOR_BGR2GRAY)

# ΕΜΦΑNΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ ΜΕ ΓΚΑΟΥΣΙΑΝΟ ΘΟΡΥΒΟ
cv2.namedWindow("Gauss Noise Image", cv2.WINDOW_NORMAL)
cv2.imshow("Gauss Noise Image", gauss_noise_gray)
cv2.resizeWindow("Gauss Noise Image", 700, 393) 
cv2.imwrite("spideyGaussNoise.jpg", gauss_noise_gray) 

cv2.waitKey()  
cv2.destroyWindow("Gauss Noise Image") 


### ΕΡΩΤΗΜΑ 4 ###

# ΑΦΑΙΡΕΣΗ ΤΟΥ ΘΟΡΥΒΟΥ ΜΕ ΤΟ ΦΙΛΤΡΟ BILLATERAL
blur = cv2.bilateralFilter(gauss_noise_gray,9,75,75)

# ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ ΜΕΤΑ ΤΗΝ ΑΦΑΙΡΕΣΗ ΤΟΥ ΘΟΡΥΒΟΥ
cv2.namedWindow("Bilateral Filter", cv2.WINDOW_NORMAL)
cv2.imshow("Bilateral Filter", blur )
cv2.resizeWindow("Bilateral Filter", 700, 393)
cv2.imwrite("spideyBilateralFilter.jpg", blur)
 
cv2.waitKey()  
cv2.destroyWindow("Bilateral Filter") 


### ΕΡΩΤΗΜΑ 5 ###

# ΕΦΑΡΜΟΓΗ ΤΟΥ CANNY EDGES ΣΤΗ ΔΙΟΡΘΩΜΕΝΗ ΕΙΚΟΝΑ
canny_edges_cleared = cv2.Canny(blur, 10, 10, edges=None, apertureSize=5)

# ΕΜΦΑNΙΣΗ ΤΗΣ ΔΙΟΡΘΩΜΕΝΗΣ ΕΙΚΟΝΑΣ ΜΕ CANNY EDGES
cv2.namedWindow("Canny Edges Filtered Image", cv2.WINDOW_NORMAL)
cv2.imshow('Canny Edges Filtered Image', canny_edges_cleared)
cv2.resizeWindow("Canny Edges Filtered Image", 700, 393) 
cv2.imwrite("spideyCannyBilateral.jpg", canny_edges_cleared)
 
cv2.waitKey() 
cv2.destroyWindow("Canny Edges Filtered Image") 


### ΕΡΩΤΗΜΑ 6 ###

# ΥΠΟΛΟΓΙΣΜΟΣ ΤΗΣ ΔΙΑΦΟΡΑΣ ΤΩΝ ΔΥΟ ΕΙΚΟΝΩΝ ΜΕ CANNY(CORRECTED ΚΑΙ ORIGINAL)
difference = cv2.absdiff(canny_edges, canny_edges_cleared)

# ΕΥΡΕΣΗ ΤΩΝ ΜΗ ΜΗΔΕΝΙΚΩΝ ΣΤΟΙΧΕΙΩΝ
num_diff = cv2.countNonZero(difference)

# ΤΥΠΩΣΗ ΤΟΥ ΑΠΟΤΕΛΕΣΜΑΤΟΣ
print("\n")
print("The score of the two canny images(corrected and original) is: {} \n".format(num_diff))


### ΕΡΩΤΗΜΑ 7α-β ###

kernel = np.ones((3,3),np.uint8)

# ΕΦΑΡΜΟΓΗ THRESHOLDING
th,img = cv2.threshold(gauss_noise_gray, 120,255, cv2.THRESH_BINARY)

# EROSION
erosion = cv2.erode(img, kernel, iterations = 1)

# ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ ΜΕ EROSION
cv2.namedWindow("Erosion", cv2.WINDOW_NORMAL)
cv2.imshow('Erosion', erosion)
cv2.resizeWindow("Erosion", 700, 393) 
cv2.imwrite("spideyErosion.jpg", erosion) 

# DILATION
dilation = cv2.dilate(img, kernel, iterations = 1)

# ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ ΜΕ DILATION
cv2.namedWindow("Dilation", cv2.WINDOW_NORMAL)
cv2.imshow('Dilation', dilation)
cv2.resizeWindow("Dilation", 700, 393)
cv2.imwrite("spideyDilation.jpg", dilation) 

cv2.waitKey()
cv2.destroyWindow("Erosion")
cv2.destroyWindow("Dilation") 


### ΕΡΩΤΗΜΑ 7γ ###

# ΥΠΟΛΟΓΙΣΜΟΣ ΤΗΣ ΔΙΑΦΟΡΑΣ ΤΩΝ ΔΥΟ ΕΙΚΟΝΩΝ ΜΕ CANNY ΚΑΙ EROSION
difference1 = cv2.absdiff(canny_edges, erosion)

# ΕΥΡΕΣΗ ΤΩΝ ΜΗ ΜΗΔΕΝΙΚΩΝ ΣΤΟΙΧΕΙΩΝ
num_diff1 = cv2.countNonZero(difference1)

print("The score of the two images(canny and erosion) is: {} \n".format(num_diff1))

# ΥΠΟΛΟΓΙΣΜΟΣ ΤΗΣ ΔΙΑΦΟΡΑΣ ΤΩΝ ΔΥΟ ΕΙΚΟΝΩΝ ΜΕ CANNY ΚΑΙ DILATION
difference2 = cv2.absdiff(canny_edges, dilation)

# ΕΥΡΕΣΗ ΤΩΝ ΜΗ ΜΗΔΕΝΙΚΩΝ ΣΤΟΙΧΕΙΩΝ
num_diff2 = cv2.countNonZero(difference2)

print("The score of the two images(canny and dilation) is: {} \n".format(num_diff2))