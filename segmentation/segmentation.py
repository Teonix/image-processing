# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 11:43:44 2021

@author: Teonix
"""
#Spyder 4.1.5

# ΕΙΣΟΔΟΣ ΒΙΒΛΙΟΘΗΚΩΝ
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from sklearn.cluster import MeanShift, estimate_bandwidth, AgglomerativeClustering, MiniBatchKMeans
from skimage.color import label2rgb
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

directory1 = r'' # ΔΗΛΩΣΗ ΤΟΥ ΜΟΝΟΠΑΤΙΟΥ 1 ΠΟΥ ΘΑ ΑΠΟΘΗKEΥΤΟΥΝ ΟΙ ΕΙΚΟΝΕΣ
directory2 = r'' # ΔΗΛΩΣΗ ΤΟΥ ΜΟΝΟΠΑΤΙΟΥ 2 ΠΟΥ ΘΑ ΑΠΟΘΗKEΥΤΟΥΝ ΟΙ ΕΙΚΟΝΕΣ
directory3 = r'' # ΔΗΛΩΣΗ ΤΟΥ ΜΟΝΟΠΑΤΙΟΥ 3 ΠΟΥ ΘΑ ΑΠΟΘΗKEΥΤΟΥΝ ΟΙ ΕΙΚΟΝΕΣ
directory4 = r'' # ΔΗΛΩΣΗ ΤΟΥ ΜΟΝΟΠΑΤΙΟΥ 4 ΠΟΥ ΘΑ ΑΠΟΘΗKEΥΤΟΥΝ ΟΙ ΕΙΚΟΝΕΣ
directory5 = r'' # ΔΗΛΩΣΗ ΤΟΥ ΜΟΝΟΠΑΤΙΟΥ 5 ΠΟΥ ΘΑ ΑΠΟΘΗKEΥΤΟΥΝ ΟΙ ΕΙΚΟΝΕΣ

img = cv2.imread('hulk.jpg') # ΕΙΣΟΔΟΣ ΤΗΣ ΕΙΚΟΝΑΣ HULK ΣΤΗ ΜΕΤΑΒΛΗΤΗ IMG
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # ΜΕΤΑΤΡΟΠΗ ΤΗΣ ΕΙΚΟΝΑΣ HULK ΣΕ RGB
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # ΜΕΤΑΤΡΟΠΗ ΤΗΣ ΕΙΚΟΝΑΣ HULK ΣΕ HSV
rgbPlusHsv_img= np.concatenate((rgb_img, hsv_img ), axis=2) # ΕΝΩΣΗ ΤΗΣ ΕΙΚΟΝΑΣ RGB ΜΕ ΤΗΝ HSV

os.chdir(directory1) # ΕΠΙΛΟΓΗ ΤΟΥ ΜΟΝΟΠΑΤΙΟΥ 1 ΠΟΥ ΦΤΙΑΞΑΜΕ ΠΙΟ ΠΑΝΩ

# ΕΜΦΑNΙΣΗ ΤΩΝ ΕΙΚΟΝΩΝ ORIGINAL,RGB,HSV 
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL) 
cv2.imshow("Original Image", img) 

cv2.namedWindow("RGB Image", cv2.WINDOW_NORMAL) 
cv2.imshow("RGB Image", rgb_img) 
cv2.imwrite("hulkRGB.jpg", rgb_img) 

os.chdir(directory2) # ΕΠΙΛΟΓΗ ΤΟΥ ΜΟΝΟΠΑΤΙΟΥ 2 ΠΟΥ ΦΤΙΑΞΑΜΕ ΠΙΟ ΠΑΝΩ

cv2.namedWindow("HSV Image", cv2.WINDOW_NORMAL) 
cv2.imshow("HSV Image", hsv_img) 
cv2.imwrite("hulkHSV.jpg", hsv_img) 

cv2.waitKey() 
cv2.destroyWindow("Original Image") 
cv2.destroyWindow("RGB Image") 
cv2.destroyWindow("HSV Image") 


### ΕΡΩΤΗΜΑ 1 ###

#RGB 3D SCATTER PLOT
r, g, b = cv2.split(rgb_img) 
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")
pixel_colors = rgb_img.reshape((np.shape(rgb_img)[0]*np.shape(rgb_img)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()
axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
plt.show()


### ΕΡΩΤΗΜΑ 2 ###

#HSV 3D SCATTER PLOT
h, s, v = cv2.split(hsv_img)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")
axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()


### ΕΡΩΤΗΜΑ 3 ###

#BLUE CHANNEL
blue = rgb_img.copy()
blue=blue[:, :, 0]

#GREEN CHANNEL
green = rgb_img.copy()
green=green[:, :, 1]

#RED CHANNEL
red = rgb_img.copy()
red=red[:, :, 2]

os.chdir(directory1) # ΕΠΙΛΟΓΗ ΤΟΥ ΜΟΝΟΠΑΤΙΟΥ 1 ΠΟΥ ΦΤΙΑΞΑΜΕ ΠΙΟ ΠΑΝΩ

# ΕΜΦΑNΙΣΗ ΤΩΝ 3 ΚΑΝΑΛΙΩΝ RGB
cv2.namedWindow("Blue-RGB", cv2.WINDOW_NORMAL) 
cv2.imshow("Blue-RGB", blue) 
cv2.imwrite("hulkBLUE.jpg", blue) 

cv2.namedWindow("Green-RGB", cv2.WINDOW_NORMAL) 
cv2.imshow("Green-RGB", green) 
cv2.imwrite("hulkGREEN.jpg", green) 

cv2.namedWindow("Red-RGB", cv2.WINDOW_NORMAL) 
cv2.imshow("Red-RGB", red) 
cv2.imwrite("hulkRED.jpg", red) 

cv2.waitKey()  
cv2.destroyWindow("Blue-RGB") 
cv2.destroyWindow("Green-RGB")
cv2.destroyWindow("Red-RGB") 

#HUE CHANNEL
hue = hsv_img.copy()
hue=hue[:, :, 0]

#SATURATION CHANNEL
saturation = hsv_img.copy()
saturation=saturation[:, :, 1]

#VALUE CHANNEL
value = hsv_img.copy()
value=value[:, :, 2]

os.chdir(directory2) # ΕΠΙΛΟΓΗ ΤΟΥ ΜΟΝΟΠΑΤΙΟΥ 2 ΠΟΥ ΦΤΙΑΞΑΜΕ ΠΙΟ ΠΑΝΩ

# ΕΜΦΑNΙΣΗ ΤΩΝ 3 ΚΑΝΑΛΙΩΝ HSV
cv2.namedWindow("Hue-HSV", cv2.WINDOW_NORMAL) 
cv2.imshow("Hue-HSV", hue) 
cv2.imwrite("hulkHUE.jpg", hue) 

cv2.namedWindow("Saturation-HSV", cv2.WINDOW_NORMAL) 
cv2.imshow("Saturation-HSV", saturation) 
cv2.imwrite("hulkSATURATION.jpg", saturation) 

cv2.namedWindow("Value-HSV", cv2.WINDOW_NORMAL) 
cv2.imshow("Value-HSV", value) 
cv2.imwrite("hulkVALUE.jpg", value) 

cv2.waitKey() 
cv2.destroyWindow("Hue-HSV") 
cv2.destroyWindow("Saturation-HSV")
cv2.destroyWindow("Value-HSV") 


### ΕΡΩΤΗΜΑ 4α-5(ΕΜΦΑΝΙΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ) ###

# RGB SEGMENTATION
# ΚΑΝΟΝΙΚΟΠΟΙΗΣΗ ΤΗΣ RGB ΕΙΚΟΝΑΣ
norm_imgRGB = cv2.normalize(rgb_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

originShape = norm_imgRGB.shape

# ΜΕΤΑΤΡΟΠΗ ΤΗΣ ΕΙΚΟΝΑΣ ΣΕ ΠΙΝΑΚΑ ΔΙΑΣΤΑΣΗΣ [nb of pixels in originImage, 3]
# ΒΑΣΙΣΜΕΝΟ ΣΤΑ 3 ΚΑΝΑΛΙΑ R,G,B
flatImgRGB=np.reshape(norm_imgRGB, [-1, 3])

# MeanShift ΜΕΘΟΔΟΣ
# ΥΠΟΛΟΓΙΣΜΟΣ ΤΟΥ BANDWIDTH ΤΟΥ ΑΛΓΟΡΙΘΜΟΥ
bandwidth = estimate_bandwidth(flatImgRGB, quantile=0.1, n_samples=100)
ms = MeanShift(bandwidth = bandwidth, bin_seeding=True)

# ΕΚΤΕΛΕΣΗ ΤΟΥ MeanShift ΣΤΗΝ FlatImage
print('Using MeanShift algorithm, it takes time!')
ms.fit(flatImgRGB)
labelsRGB_MeanShift=ms.labels_
 
# ΕΥΡΕΣΗ ΚΑΙ ΕΜΦΑΝΙΣΗ ΤΩΝ ΑΡΙΘΜΩΝ ΤΩΝ ΣΥΣΤΑΔΩΝ
labels_unique = np.unique(labelsRGB_MeanShift)
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)

# ΕΜΦΑΝΙΣΗ ΤΗΣ ΚΑΤΑΤΜΗΜΕΝΗΣ ΕΙΚΟΝΑΣ ME MeanShift
segmentedImg = np.reshape(labelsRGB_MeanShift, originShape[:2]) 
segmentedImg = label2rgb(segmentedImg) * 255 

os.chdir(directory3) # ΕΠΙΛΟΓΗ ΤΟΥ ΜΟΝΟΠΑΤΙΟΥ 3 ΠΟΥ ΦΤΙΑΞΑΜΕ ΠΙΟ ΠΑΝΩ

cv2.namedWindow("MeanShiftSegmentsRGB", cv2.WINDOW_NORMAL)
cv2.imshow("MeanShiftSegmentsRGB",segmentedImg.astype(np.uint8))
cv2.imwrite("MeanShiftSegmentsRGB.jpg", segmentedImg.astype(np.uint8)) 


# K-Means ΜΕΘΟΔΟΣ
print('Using kmeans algorithm, it is faster!')
km = MiniBatchKMeans(n_clusters = n_clusters_)
km.fit(flatImgRGB)
labelsRGB_kmeans = km.labels_

# ΕΜΦΑΝΙΣΗ ΤΗΣ ΚΑΤΑΤΜΗΜΕΝΗΣ ΕΙΚΟΝΑΣ ΜΕ K-Means
segmentedImg = np.reshape(labelsRGB_kmeans, originShape[:2])
segmentedImg = label2rgb(segmentedImg) * 255 

cv2.namedWindow("kmeansSegmentsRGB", cv2.WINDOW_NORMAL)
cv2.imshow("kmeansSegmentsRGB",segmentedImg.astype(np.uint8))
cv2.imwrite("kmeansSegmentsRGB.jpg", segmentedImg.astype(np.uint8)) 

cv2.waitKey() 
cv2.destroyWindow("MeanShiftSegmentsRGB") 
cv2.destroyWindow("kmeansSegmentsRGB") 


### ΕΡΩΤΗΜΑ 4β-5(ΕΜΦΑΝΙΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ) ###

# HSV SEGMENTATION
# ΚΑΝΟΝΙΚΟΠΟΙΗΣΗ ΤΗΣ HSV ΕΙΚΟΝΑΣ
norm_imgHSV = cv2.normalize(hsv_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
originShape = norm_imgHSV.shape

# ΜΕΤΑΤΡΟΠΗ ΤΗΣ ΕΙΚΟΝΑΣ ΣΕ ΠΙΝΑΚΑ ΔΙΑΣΤΑΣΗΣ [nb of pixels in originImage, 3]
# ΒΑΣΙΣΜΕΝΟ ΣΤΑ 3 ΚΑΝΑΛΙΑ H,S,V
flatImgHSV=np.reshape(norm_imgHSV, [-1, 3])


# MeanShift ΜΕΘΟΔΟΣ
# ΥΠΟΛΟΓΙΣΜΟΣ ΤΟΥ BANDWIDTH ΤΟΥ ΑΛΓΟΡΙΘΜΟΥ
bandwidth = estimate_bandwidth(flatImgHSV, quantile=0.1, n_samples=100)
ms = MeanShift(bandwidth = bandwidth, bin_seeding=True)

# ΕΚΤΕΛΕΣΗ ΤΟΥ MeanShift ΣΤΗΝ FlatImage
print('Using MeanShift algorithm, it takes time!')
ms.fit(flatImgHSV)
labelsHSV_MeanShift=ms.labels_

# ΕΥΡΕΣΗ ΚΑΙ ΕΜΦΑΝΙΣΗ ΤΩΝ ΑΡΙΘΜΩΝ ΤΩΝ ΣΥΣΤΑΔΩΝ
labels_unique = np.unique(labelsHSV_MeanShift)
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)

# ΕΜΦΑΝΙΣΗ ΤΗΣ ΚΑΤΑΤΜΗΜΕΝΗΣ ΕΙΚΟΝΑΣ ME MeanShift
segmentedImg = np.reshape(labelsHSV_MeanShift, originShape[:2]) 
segmentedImg = label2rgb(segmentedImg) * 255 

os.chdir(directory4) # ΕΠΙΛΟΓΗ ΤΟΥ ΜΟΝΟΠΑΤΙΟΥ 4 ΠΟΥ ΦΤΙΑΞΑΜΕ ΠΙΟ ΠΑΝΩ

cv2.namedWindow("MeanShiftSegmentsHSV", cv2.WINDOW_NORMAL)
cv2.imshow("MeanShiftSegmentsHSV",segmentedImg.astype(np.uint8))
cv2.imwrite("MeanShiftSegmentsHSV.jpg", segmentedImg.astype(np.uint8)) 

# K-Means ΜΕΘΟΔΟΣ
print('Using kmeans algorithm, it is faster!')
km = MiniBatchKMeans(n_clusters = n_clusters_)
km.fit(flatImgHSV)
labelsHSV_kmeans = km.labels_

# ΕΜΦΑΝΙΣΗ ΤΗΣ ΚΑΤΑΤΜΗΜΕΝΗΣ ΕΙΚΟΝΑΣ ΜΕ K-Means
segmentedImg = np.reshape(labelsHSV_kmeans, originShape[:2])
segmentedImg = label2rgb(segmentedImg) * 255 

cv2.namedWindow("kmeansSegmentsHSV", cv2.WINDOW_NORMAL)
cv2.imshow("kmeansSegmentsHSV",segmentedImg.astype(np.uint8))
cv2.imwrite("kmeansSegmentsHSV.jpg", segmentedImg.astype(np.uint8)) 

cv2.waitKey()
cv2.destroyWindow("MeanShiftSegmentsHSV") 
cv2.destroyWindow("kmeansSegmentsHSV") 


### ΕΡΩΤΗΜΑ 4γ-5(ΕΜΦΑΝΙΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ) ###

# RGB+HSV SEGMENTATION
# ΚΑΝΟΝΙΚΟΠΟΙΗΣΗ ΤΗΣ RGB+HSV ΕΙΚΟΝΑΣ
norm_img_RGBHSV = cv2.normalize(rgbPlusHsv_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
originShape = norm_img_RGBHSV.shape

# ΜΕΤΑΤΡΟΠΗ ΤΗΣ ΕΙΚΟΝΑΣ ΣΕ ΠΙΝΑΚΑ ΔΙΑΣΤΑΣΗΣ [nb of pixels in originImage, 6]
# ΒΑΣΙΣΜΕΝΟ ΣΤΑ 6 ΚΑΝΑΛΙΑ R,G,B,H,S,V
flatImgRGBHSV=np.reshape(norm_img_RGBHSV, [-1, 6])


# MeanShift ΜΕΘΟΔΟΣ
# ΥΠΟΛΟΓΙΣΜΟΣ ΤΟΥ BANDWIDTH ΤΟΥ ΑΛΓΟΡΙΘΜΟΥ
bandwidth = estimate_bandwidth(flatImgRGBHSV, quantile=0.1, n_samples=100)
ms = MeanShift(bandwidth = bandwidth, bin_seeding=True)

# ΕΚΤΕΛΕΣΗ ΤΟΥ MeanShift ΣΤΗΝ FlatImage
print('Using MeanShift algorithm, it takes time!')
ms.fit(flatImgRGBHSV)
labelsRGBHSV_MeanShift=ms.labels_

# ΕΥΡΕΣΗ ΚΑΙ ΕΜΦΑΝΙΣΗ ΤΩΝ ΑΡΙΘΜΩΝ ΤΩΝ ΣΥΣΤΑΔΩΝ
labels_unique = np.unique(labelsRGBHSV_MeanShift)
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)

# ΕΜΦΑΝΙΣΗ ΤΗΣ ΚΑΤΑΤΜΗΜΕΝΗΣ ΕΙΚΟΝΑΣ ME MeanShift
segmentedImg = np.reshape(labelsRGBHSV_MeanShift, originShape[:2]) 
segmentedImg = label2rgb(segmentedImg) * 255 

os.chdir(directory5) # ΕΠΙΛΟΓΗ ΤΟΥ ΜΟΝΟΠΑΤΙΟΥ 5 ΠΟΥ ΦΤΙΑΞΑΜΕ ΠΙΟ ΠΑΝΩ

cv2.namedWindow("MeanShiftSegmentsRGB+HSV", cv2.WINDOW_NORMAL)
cv2.imshow("MeanShiftSegmentsRGB+HSV",segmentedImg.astype(np.uint8))
cv2.imwrite("MeanShiftSegmentsRGB+HSV.jpg", segmentedImg.astype(np.uint8)) 

# K-Means ΜΕΘΟΔΟΣ
print('Using kmeans algorithm, it is faster!')
km = MiniBatchKMeans(n_clusters = n_clusters_)
km.fit(flatImgRGBHSV)
labelsRGBHSV_kmeans = km.labels_

# ΕΜΦΑΝΙΣΗ ΤΗΣ ΚΑΤΑΤΜΗΜΕΝΗΣ ΕΙΚΟΝΑΣ ΜΕ K-Means
segmentedImg = np.reshape(labelsRGBHSV_kmeans, originShape[:2])
segmentedImg = label2rgb(segmentedImg) * 255 

cv2.namedWindow("kmeansSegmentsRGB+HSV", cv2.WINDOW_NORMAL)
cv2.imshow("kmeansSegmentsRGB+HSV",segmentedImg.astype(np.uint8))
cv2.imwrite("kmeansSegmentsRGB+HSV.jpg", segmentedImg.astype(np.uint8)) 

cv2.waitKey()
cv2.destroyWindow("MeanShiftSegmentsRGB+HSV")
cv2.destroyWindow("kmeansSegmentsRGB+HSV") 


### ΕΡΩΤΗΜΑ 6 ###

print("\n")
# ΕΥΡΕΣΗ ΚΑΙ ΤΥΠΩΣΗ ΤΟΥ SILHOUETTE SCORE ΓΙΑ ΤΗΝ ΚΑΤΑΤΜΗΜΕΝΗ RGB ΕΙΚΟΝΑ ΜΕ MeanShift ΑΛΓΟΡΙΘΜΟ
score_RGB_MeanShift = silhouette_score(flatImgRGB, labelsRGB_MeanShift, metric='euclidean')
print("The silhouette score of the RGB MeanShift algorithm is: {}".format(score_RGB_MeanShift))

print("\n")
# ΕΥΡΕΣΗ ΚΑΙ ΤΥΠΩΣΗ ΤΟΥ SILHOUETTE SCORE ΓΙΑ ΤΗΝ ΚΑΤΑΤΜΗΜΕΝΗ RGB ΕΙΚΟΝΑ ΜΕ K-Means ΑΛΓΟΡΙΘΜΟ
score_RGB_kmeans = silhouette_score(flatImgRGB, labelsRGB_kmeans, metric='euclidean')
print("The silhouette score of the RGB K-Means algorithm is: {}".format(score_RGB_kmeans))

print("\n")
# ΕΥΡΕΣΗ ΚΑΙ ΤΥΠΩΣΗ ΤΟΥ SILHOUETTE SCORE ΓΙΑ ΤΗΝ ΚΑΤΑΤΜΗΜΕΝΗ HSV ΕΙΚΟΝΑ ΜΕ MeanShift ΑΛΓΟΡΙΘΜΟ
score_HSV_MeanShift = silhouette_score(flatImgHSV, labelsHSV_MeanShift, metric='euclidean')
print("The silhouette score of the HSV MeanShift algorithm is: {}".format(score_HSV_MeanShift))

print("\n")
# ΕΥΡΕΣΗ ΚΑΙ ΤΥΠΩΣΗ ΤΟΥ SILHOUETTE SCORE ΓΙΑ ΤΗΝ ΚΑΤΑΤΜΗΜΕΝΗ HSV ΕΙΚΟΝΑ ΜΕ K-Means ΑΛΓΟΡΙΘΜΟ
score_HSV_kmeans = silhouette_score(flatImgHSV, labelsHSV_kmeans, metric='euclidean')
print("The silhouette score of the HSV K-Means algorithm is: {}".format(score_HSV_kmeans))

print("\n")
# ΕΥΡΕΣΗ ΚΑΙ ΤΥΠΩΣΗ ΤΟΥ SILHOUETTE SCORE ΓΙΑ ΤΗΝ ΚΑΤΑΤΜΗΜΕΝΗ RGB+HSV ΕΙΚΟΝΑ ΜΕ MeanShift ΑΛΓΟΡΙΘΜΟ
score_RGBHSV_MeanShift = silhouette_score(flatImgRGBHSV, labelsRGBHSV_MeanShift, metric='euclidean')
print("The silhouette score of the RGB+HSV MeanShift algorithm is: {}".format(score_RGBHSV_MeanShift))

print("\n")
# ΕΥΡΕΣΗ ΚΑΙ ΤΥΠΩΣΗ ΤΟΥ SILHOUETTE SCORE ΓΙΑ ΤΗΝ ΚΑΤΑΤΜΗΜΕΝΗ RGB+HSV ΕΙΚΟΝΑ ΜΕ K-Means ΑΛΓΟΡΙΘΜΟ
score_RGBHSV_kmeans = silhouette_score(flatImgRGBHSV, labelsRGBHSV_kmeans, metric='euclidean')
print("The silhouette score of the RGB+HSV K-Means algorithm is: {}".format(score_RGBHSV_kmeans))