# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 22:01:26 2020

@author: Teonix
"""
#Spyder 4.1.5

# ΕΙΣΟΔΟΣ ΒΙΒΛΙΟΘΗΚΩΝ
import cv2
import os
import numpy as np
import random
import skimage.measure

# ΚΑΤΑΣΚΕΥΗ ΣΥΝΑΡΤΗΣΗΣ SALT & PEPPER
def sp_noise(image,prob):
    output = np.zeros(image.shape,np.uint8) # ΓΕΜΙΣΜΑ ΜΕΤΑΒΛΗΤΗΣ ΤΟΥ ΑΠΟΤΕΛΕΣΜΑΤΟΣ ΜΕ 0
    thres = 1 - prob # ΥΠΟΛΟΓΙΣΜΟΣ ΤΟΥ THRESHVALUE ΑΦΑΙΡΩΝΤΑΣ ΤΗΝ ΠΙΘΑΝΟΤΗΤΑ ΠΟΥ ΕΔΩΣΕ Ο ΧΡΗΣΤΗΣ
    for i in range(image.shape[0]): # ΛΟΥΠΑ ΣΤΗΝ 1 ΔΙΑΣΤΑΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
        for j in range(image.shape[1]): # ΛΟΥΠΑ ΣΤΗΝ 2 ΔΙΑΣΤΑΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
            rdn = random.random() # ΔΗΜΙΟΥΡΓΙΑ ΤΗΣ ΜΕΤΑΒΛΗΤΗΣ ΠΟΥ ΠΑΙΡΝΕΙ ΤΥΧΑΙΑ ΤΙΜΗ ΑΠΟ 0 ΕΩΣ 1
            if rdn < prob: # ΑΝ Η ΤΥΧΑΙΑ ΤΙΜΗ ΕΙΝΑΙ ΜΙΚΡΟΤΕΡΗ ΑΠΟ ΤΗΝ ΠΙΘΑΝΟΤΗΤΑ ΠΟΥ ΕΔΩΣΕ Ο ΧΡΗΣΤΗΣ
                output[i][j] = 0 # ΤΟ ΣΥΓΚΕΚΡΙΜΕΝΟ PIXEL ΓΙΝΕΤΑΙ ΙΣΟ ΜΕ TH TIMH 0
            elif rdn > thres: # ΑΝ Η ΤΥΧΑΙΑ ΤΙΜΗ ΕΙΝΑΙ ΜΕΓΑΛΥΤΕΡΗ ΑΠΟ ΤΟ THRESHVALUE
                output[i][j] = 255 # ΤΟ ΣΥΓΚΕΚΡΙΜΕΝΟ PIXEL ΓΙΝΕΤΑΙ ΙΣΟ ΜΕ TH TIMH 255
            else:
                output[i][j] = image[i][j] # ΑΛΛΙΩΣ ΤΟ PIXEL ΠΑΡΑΜΕΝΕΙ ΙΔΙΟ
    return output # ΕΠΙΣΤΡΟΦΗ ΤΟΥ ΑΠΟΤΕΛΕΣΜΑΤΟΣ

directory1 = r'' # ΔΗΛΩΣΗ ΤΟΥ 1ου ΜΟΝΟΠΑΤΙΟΥ ΠΟΥ ΘΑ ΑΠΟΘΗKEΥΤΟΥΝ ΟΙ ΕΙΚΟΝΕΣ
directory2 = r'' # ΔΗΛΩΣΗ ΤΟΥ 2ου ΜΟΝΟΠΑΤΙΟΥ ΠΟΥ ΘΑ ΑΠΟΘΗKEΥΤΟΥΝ ΟΙ ΕΙΚΟΝΕΣ ΠΟΥ ΑΦΟΡΟΥΝ ΤΟ ΘΟΡΥΒΟ SALT & PEPPER
directory3 = r'' # ΔΗΛΩΣΗ ΤΟΥ 3ου ΜΟΝΟΠΑΤΙΟΥ ΠΟΥ ΘΑ ΑΠΟΘΗKEΥΤΟΥΝ ΟΙ ΕΙΚΟΝΕΣ ΠΟΥ ΑΦΟΡΟΥΝ ΤΟ ΘΟΡΥΒΟ POISSON

kernel = np.ones((5,5),np.float32)/25 # ΔΗΜΙΟΥΡΓΙΑ ΤΟΥ KERNEL 5Χ5 ΓΙΑ ΤΟ ΦΙΛΤΡΟ AVERAGE

img = cv2.imread('wolf.jpg') # ΕΙΣΟΔΟΣ ΤΗΣ ΕΙΚΟΝΑΣ WOLF ΣΤΗ ΜΕΤΑΒΛΗΤΗ IMG
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # ΜΕΤΑΤΡΟΠΗ ΤΗΣ ΕΙΚΟΝΑΣ ΣΕ GRAYSCALE

img_sp = sp_noise(img_gray,0.1) # ΚΑΛΕΣΜΑ ΤΗΣ ΣΥΝΑΡΤΗΣΗΣ sp_noise ΓΙΑ ΝΑ ΠΡΟΣΘΕΣΟΥΜΕ SALT & PEPPER ΘΟΡΥΒΟ ΣΤΗ ΓΚΡΙ ΕΙΚΟΝΑ
imgsp_median = cv2.medianBlur(img_sp, 5) # ΧΡΗΣΗ ΤΗΣ OPENCV ΓΙΑ ΝΑ ΕΦΑΡΜΟΣΟΥΜΕ ΤΟ MEDIAN ΦΙΛΤΡΟ ΣΤΗ ΓΚΡΙ ΕΙΚΟΝΑ ΜΕ SALT & PEPPER
imgsp_gauss = cv2.GaussianBlur(img_sp, (5,5), 0) # ΧΡΗΣΗ ΤΗΣ OPENCV ΓΙΑ ΝΑ ΕΦΑΡΜΟΣΟΥΜΕ ΤΟ GAUSS ΦΙΛΤΡΟ ΣΤΗ ΓΚΡΙ ΕΙΚΟΝΑ ΜΕ SALT & PEPPER
imgsp_averaging = cv2.filter2D(img_sp, -1, kernel) # ΧΡΗΣΗ ΤΗΣ OPENCV ΓΙΑ ΝΑ ΕΦΑΡΜΟΣΟΥΜΕ ΤΟ AVERAGING ΦΙΛΤΡΟ ΣΤΗ ΓΚΡΙ ΕΙΚΟΝΑ ΜΕ SALT & PEPPER

img_poisson = np.random.poisson(img_gray / 255.0 * 0.4) / 0.4 * 255 # ΕΙΣΑΓΩΓΗ POISSON ΘΟΡΥΒΟΥ ΣΤΗ ΓΚΡΙ ΕΙΚΟΝΑ ΜΕ ΤΗ ΒΟΗΘΕΙΑ ΤΗΣ np
img_poisson = np.floor(np.abs(img_poisson)).astype('uint8') # ΜΕΤΑΤΡΟΠΗ ΤΗΣ ΓΚΡΙ ΕΙΚΟΝΑΣ ΣΕ INT8
img_poisson_median = cv2.medianBlur(img_poisson, 3) # ΧΡΗΣΗ ΤΗΣ OPENCV ΓΙΑ ΝΑ ΕΦΑΡΜΟΣΟΥΜΕ ΤΟ MEDIAN ΦΙΛΤΡΟ ΣΤΗ ΓΚΡΙ ΕΙΚΟΝΑ ΜΕ POISSON
img_poisson_gauss = cv2.GaussianBlur(img_poisson, (5,5), 0) # ΧΡΗΣΗ ΤΗΣ OPENCV ΓΙΑ ΝΑ ΕΦΑΡΜΟΣΟΥΜΕ ΤΟ GAUSS ΦΙΛΤΡΟ ΣΤΗ ΓΚΡΙ ΕΙΚΟΝΑ ΜΕ POISSON
img_poisson_averaging = cv2.filter2D(img_poisson, -1, kernel) # ΧΡΗΣΗ ΤΗΣ OPENCV ΓΙΑ ΝΑ ΕΦΑΡΜΟΣΟΥΜΕ ΤΟ AVERAGING ΦΙΛΤΡΟ ΣΤΗ ΓΚΡΙ ΕΙΚΟΝΑ ΜΕ POISSON

os.chdir(directory1) # ΕΠΙΛΟΓΗ ΤΟΥ 1ου ΜΟΝΟΠΑΤΙΟΥ 

cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL) # ΕΝΤΟΛΗ ΠΟΥ ΕΠΙΤΡΕΠΕΙ ΤΟ RESIZE ΣΤΟ ΠΑΡΑΘΥΡΟ
cv2.imshow("Original Image", img) # ΕΝΤΟΛΗ ΓΙΑ ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
cv2.resizeWindow("Original Image", 500, 375) # ΕΝΤΟΛΗ ΓΙΑ ΠΡΟΚΑΘΟΡΙΣΜΟ ΤΟΥ ΜΕΓΕΘΟΥΣ ΤΟΥ ΠΑΡΑΘΥΡΟΥ

cv2.namedWindow("Grayscale Image", cv2.WINDOW_NORMAL) # ΕΝΤΟΛΗ ΠΟΥ ΕΠΙΤΡΕΠΕΙ ΤΟ RESIZE ΣΤΟ ΠΑΡΑΘΥΡΟ
cv2.imshow("Grayscale Image", img_gray) # ΕΝΤΟΛΗ ΓΙΑ ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
cv2.resizeWindow("Grayscale Image", 500, 375) # ΕΝΤΟΛΗ ΓΙΑ ΠΡΟΚΑΘΟΡΙΣΜΟ ΤΟΥ ΜΕΓΕΘΟΥΣ ΤΟΥ ΠΑΡΑΘΥΡΟΥ
cv2.imwrite("wolfGray.jpg", img_gray) # ΕΝΤΟΛΗ ΓΙΑ ΑΠΟΘΗΚΕΥΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
 
cv2.waitKey() # ΕΝΤΟΛΗ ΓΙΑ ΑΝΑΜΟΝΗ ΤΩΝ ΠΑΡΑΘΥΡΩΝ 
cv2.destroyWindow("Original Image") # ΕΝΤΟΛΗ ΓΙΑ ΚΛΕΙΣΙΜΟ ΤΟΥ ΠΑΡΑΘΥΡΟΥ "Original Image"
cv2.destroyWindow("Grayscale Image") # ΕΝΤΟΛΗ ΓΙΑ ΚΛΕΙΣΙΜΟ ΤΟΥ ΠΑΡΑΘΥΡΟΥ "Grayscale Image"

os.chdir(directory2) # ΕΠΙΛΟΓΗ ΤΟΥ 2ου ΜΟΝΟΠΑΤΙΟΥ 

cv2.namedWindow("Salt & Pepper Image", cv2.WINDOW_NORMAL) # ΕΝΤΟΛΗ ΠΟΥ ΕΠΙΤΡΕΠΕΙ ΤΟ RESIZE ΣΤΟ ΠΑΡΑΘΥΡΟ
cv2.imshow("Salt & Pepper Image", img_sp) # ΕΝΤΟΛΗ ΓΙΑ ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
cv2.resizeWindow("Salt & Pepper Image", 500, 375) # ΕΝΤΟΛΗ ΓΙΑ ΠΡΟΚΑΘΟΡΙΣΜΟ ΤΟΥ ΜΕΓΕΘΟΥΣ ΤΟΥ ΠΑΡΑΘΥΡΟΥ
cv2.imwrite("wolf_sp.jpg", img_sp) # ΕΝΤΟΛΗ ΓΙΑ ΑΠΟΘΗΚΕΥΣΗ ΤΗΣ ΕΙΚΟΝΑΣ

cv2.namedWindow("Median Filter On Salt & Pepper Image", cv2.WINDOW_NORMAL) # ΕΝΤΟΛΗ ΠΟΥ ΕΠΙΤΡΕΠΕΙ ΤΟ RESIZE ΣΤΟ ΠΑΡΑΘΥΡΟ
cv2.imshow("Median Filter On Salt & Pepper Image", imgsp_median) # ΕΝΤΟΛΗ ΓΙΑ ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
cv2.resizeWindow("Median Filter On Salt & Pepper Image", 500, 375) # ΕΝΤΟΛΗ ΓΙΑ ΠΡΟΚΑΘΟΡΙΣΜΟ ΤΟΥ ΜΕΓΕΘΟΥΣ ΤΟΥ ΠΑΡΑΘΥΡΟΥ
cv2.imwrite("wolf_sp_median_filter.jpg", imgsp_median) # ΕΝΤΟΛΗ ΓΙΑ ΑΠΟΘΗΚΕΥΣΗ ΤΗΣ ΕΙΚΟΝΑΣ

cv2.namedWindow("Gauss Filter On Salt & Pepper Image", cv2.WINDOW_NORMAL) # ΕΝΤΟΛΗ ΠΟΥ ΕΠΙΤΡΕΠΕΙ ΤΟ RESIZE ΣΤΟ ΠΑΡΑΘΥΡΟ
cv2.imshow("Gauss Filter On Salt & Pepper Image", imgsp_gauss) # ΕΝΤΟΛΗ ΓΙΑ ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
cv2.resizeWindow("Gauss Filter On Salt & Pepper Image", 500, 375) # ΕΝΤΟΛΗ ΓΙΑ ΠΡΟΚΑΘΟΡΙΣΜΟ ΤΟΥ ΜΕΓΕΘΟΥΣ ΤΟΥ ΠΑΡΑΘΥΡΟΥ
cv2.imwrite("wolf_sp_gauss_filter.jpg", imgsp_gauss) # ΕΝΤΟΛΗ ΓΙΑ ΑΠΟΘΗΚΕΥΣΗ ΤΗΣ ΕΙΚΟΝΑΣ

cv2.namedWindow("Averaging Filter On Salt & Pepper Image", cv2.WINDOW_NORMAL) # ΕΝΤΟΛΗ ΠΟΥ ΕΠΙΤΡΕΠΕΙ ΤΟ RESIZE ΣΤΟ ΠΑΡΑΘΥΡΟ
cv2.imshow("Averaging Filter On Salt & Pepper Image", imgsp_averaging) # ΕΝΤΟΛΗ ΓΙΑ ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
cv2.resizeWindow("Averaging Filter On Salt & Pepper Image", 500, 375) # ΕΝΤΟΛΗ ΓΙΑ ΠΡΟΚΑΘΟΡΙΣΜΟ ΤΟΥ ΜΕΓΕΘΟΥΣ ΤΟΥ ΠΑΡΑΘΥΡΟΥ
cv2.imwrite("wolf_sp_averaging_filter.jpg", imgsp_averaging) # ΕΝΤΟΛΗ ΓΙΑ ΑΠΟΘΗΚΕΥΣΗ ΤΗΣ ΕΙΚΟΝΑΣ

cv2.waitKey() # ΕΝΤΟΛΗ ΓΙΑ ΑΝΑΜΟΝΗ ΤΩΝ ΠΑΡΑΘΥΡΩΝ 
cv2.destroyWindow("Salt & Pepper Image") # ΕΝΤΟΛΗ ΓΙΑ ΚΛΕΙΣΙΜΟ ΤΟΥ ΠΑΡΑΘΥΡΟΥ "Salt & Pepper Image"
cv2.destroyWindow("Median Filter On Salt & Pepper Image") # ΕΝΤΟΛΗ ΓΙΑ ΚΛΕΙΣΙΜΟ ΤΟΥ ΠΑΡΑΘΥΡΟΥ "Median Filter On Salt & Pepper Image"
cv2.destroyWindow("Gauss Filter On Salt & Pepper Image") # ΕΝΤΟΛΗ ΓΙΑ ΚΛΕΙΣΙΜΟ ΤΟΥ ΠΑΡΑΘΥΡΟΥ "Gauss Filter On Salt & Pepper Image"
cv2.destroyWindow("Averaging Filter On Salt & Pepper Image") # ΕΝΤΟΛΗ ΓΙΑ ΚΛΕΙΣΙΜΟ ΤΟΥ ΠΑΡΑΘΥΡΟΥ "Averaging Filter On Salt & Pepper Image"


os.chdir(directory3) # ΕΠΙΛΟΓΗ ΤΟΥ 3ου ΜΟΝΟΠΑΤΙΟΥ

cv2.namedWindow("Poisson Image", cv2.WINDOW_NORMAL) # ΕΝΤΟΛΗ ΠΟΥ ΕΠΙΤΡΕΠΕΙ ΤΟ RESIZE ΣΤΟ ΠΑΡΑΘΥΡΟ
cv2.imshow("Poisson Image", img_poisson) # ΕΝΤΟΛΗ ΓΙΑ ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
cv2.resizeWindow("Poisson Image", 500, 375) # ΕΝΤΟΛΗ ΓΙΑ ΠΡΟΚΑΘΟΡΙΣΜΟ ΤΟΥ ΜΕΓΕΘΟΥΣ ΤΟΥ ΠΑΡΑΘΥΡΟΥ
cv2.imwrite("wolf_poisson.jpg", img_poisson) # ΕΝΤΟΛΗ ΓΙΑ ΑΠΟΘΗΚΕΥΣΗ ΤΗΣ ΕΙΚΟΝΑΣ

cv2.namedWindow("Median Filter On Poisson Image", cv2.WINDOW_NORMAL) # ΕΝΤΟΛΗ ΠΟΥ ΕΠΙΤΡΕΠΕΙ ΤΟ RESIZE ΣΤΟ ΠΑΡΑΘΥΡΟ
cv2.imshow("Median Filter On Poisson Image", img_poisson_median) # ΕΝΤΟΛΗ ΓΙΑ ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
cv2.resizeWindow("Median Filter On Poisson Image", 500, 375) # ΕΝΤΟΛΗ ΓΙΑ ΠΡΟΚΑΘΟΡΙΣΜΟ ΤΟΥ ΜΕΓΕΘΟΥΣ ΤΟΥ ΠΑΡΑΘΥΡΟΥ
cv2.imwrite("wolf_poisson_median_filter.jpg", img_poisson_median) # ΕΝΤΟΛΗ ΓΙΑ ΑΠΟΘΗΚΕΥΣΗ ΤΗΣ ΕΙΚΟΝΑΣ

cv2.namedWindow("Gauss Filter On Poisson Image", cv2.WINDOW_NORMAL) # ΕΝΤΟΛΗ ΠΟΥ ΕΠΙΤΡΕΠΕΙ ΤΟ RESIZE ΣΤΟ ΠΑΡΑΘΥΡΟ
cv2.imshow("Gauss Filter On Poisson Image", img_poisson_gauss) # ΕΝΤΟΛΗ ΓΙΑ ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
cv2.resizeWindow("Gauss Filter On Poisson Image", 500, 375) # ΕΝΤΟΛΗ ΓΙΑ ΠΡΟΚΑΘΟΡΙΣΜΟ ΤΟΥ ΜΕΓΕΘΟΥΣ ΤΟΥ ΠΑΡΑΘΥΡΟΥ
cv2.imwrite("wolf_poisson_averaging_filter.jpg", img_poisson_gauss) # ΕΝΤΟΛΗ ΓΙΑ ΑΠΟΘΗΚΕΥΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
 
cv2.namedWindow("Averaging Filter On Poisson Image", cv2.WINDOW_NORMAL) # ΕΝΤΟΛΗ ΠΟΥ ΕΠΙΤΡΕΠΕΙ ΤΟ RESIZE ΣΤΟ ΠΑΡΑΘΥΡΟ
cv2.imshow("Averaging Filter On Poisson Image", img_poisson_averaging) # ΕΝΤΟΛΗ ΓΙΑ ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
cv2.resizeWindow("Averaging Filter On Poisson Image", 500, 375) # ΕΝΤΟΛΗ ΓΙΑ ΠΡΟΚΑΘΟΡΙΣΜΟ ΤΟΥ ΜΕΓΕΘΟΥΣ ΤΟΥ ΠΑΡΑΘΥΡΟΥ
cv2.imwrite("wolf_poisson_gauss_filter.jpg", img_poisson_averaging) # ΕΝΤΟΛΗ ΓΙΑ ΑΠΟΘΗΚΕΥΣΗ ΤΗΣ ΕΙΚΟΝΑΣ

cv2.waitKey() # ΕΝΤΟΛΗ ΓΙΑ ΑΝΑΜΟΝΗ ΤΩΝ ΠΑΡΑΘΥΡΩΝ 
cv2.destroyWindow("Poisson Image") # ΕΝΤΟΛΗ ΓΙΑ ΚΛΕΙΣΙΜΟ ΤΟΥ ΠΑΡΑΘΥΡΟΥ "Poisson Image"
cv2.destroyWindow("Median Filter On Poisson Image") # ΕΝΤΟΛΗ ΓΙΑ ΚΛΕΙΣΙΜΟ ΤΟΥ ΠΑΡΑΘΥΡΟΥ "Median Filter On Poisson Image"
cv2.destroyWindow("Gauss Filter On Poisson Image") # ΕΝΤΟΛΗ ΓΙΑ ΚΛΕΙΣΙΜΟ ΤΟΥ ΠΑΡΑΘΥΡΟΥ "Gauss Filter On Poisson Image"
cv2.destroyWindow("Averaging Filter On Poisson Image") # ΕΝΤΟΛΗ ΓΙΑ ΚΛΕΙΣΙΜΟ ΤΟΥ ΠΑΡΑΘΥΡΟΥ "Averaging Filter On Poisson Image"

print("\n") # NEWLINE

print("Structural Similarity Index (SSIM) Method") # ΤΥΠΩΣΗ ΤΗΣ ΜΕΘΟΔΟΥ

print("\n") # NEWLINE

print("Similarity results for salt & pepper noise image:")
print("\n") # NEWLINE
(score, _) = skimage.measure.compare_ssim(img_gray, imgsp_median, full=True) # ΥΠΟΛΟΓΙΣΜΟΣ ΤΟΥ SCORE ΜΕΤΑΞΥ ΤΩΝ ΔΥΟ ΕΙΚΟΝΩΝ
print( " .. SSIM score between original wolf and denoised with median filter on salt & pepper noise: {:.4f}".format(score)) # ΤΥΠΩΣΗ ΤΟΥ SCORE
print("\n")
(score, _) = skimage.measure.compare_ssim(img_gray, imgsp_gauss, full=True) # ΥΠΟΛΟΓΙΣΜΟΣ ΤΟΥ SCORE ΜΕΤΑΞΥ ΤΩΝ ΔΥΟ ΕΙΚΟΝΩΝ
print( " .. SSIM score between original wolf and denoised with gauss filter on salt & pepper noise: {:.4f}".format(score)) # ΤΥΠΩΣΗ ΤΟΥ SCORE
print("\n") # NEWLINE
(score, _) = skimage.measure.compare_ssim(img_gray, imgsp_averaging, full=True) # ΥΠΟΛΟΓΙΣΜΟΣ ΤΟΥ SCORE ΜΕΤΑΞΥ ΤΩΝ ΔΥΟ ΕΙΚΟΝΩΝ
print( " .. SSIM score between original wolf and denoised with averaging filter on salt & pepper noise: {:.4f}".format(score)) # ΤΥΠΩΣΗ ΤΟΥ SCORE

print("\n") # NEWLINE

print("Similarity results for poisson noise image:")
print("\n") # NEWLINE
(score, _) = skimage.measure.compare_ssim(img_gray, img_poisson_median, full=True) # ΥΠΟΛΟΓΙΣΜΟΣ ΤΟΥ SCORE ΜΕΤΑΞΥ ΤΩΝ ΔΥΟ ΕΙΚΟΝΩΝ
print( " .. SSIM score between original wolf and denoised with median filter on poisson noise: {:.4f}".format(score)) # ΤΥΠΩΣΗ ΤΟΥ SCORE
print("\n") # NEWLINE
(score, _) = skimage.measure.compare_ssim(img_gray, img_poisson_gauss, full=True) # ΥΠΟΛΟΓΙΣΜΟΣ ΤΟΥ SCORE ΜΕΤΑΞΥ ΤΩΝ ΔΥΟ ΕΙΚΟΝΩΝ
print( " .. SSIM score between original wolf and denoised with gauss filter on poisson noise: {:.4f}".format(score)) # ΤΥΠΩΣΗ ΤΟΥ SCORE
print("\n") # NEWLINE
(score, _) = skimage.measure.compare_ssim(img_gray, img_poisson_averaging, full=True) # ΥΠΟΛΟΓΙΣΜΟΣ ΤΟΥ SCORE ΜΕΤΑΞΥ ΤΩΝ ΔΥΟ ΕΙΚΟΝΩΝ
print( " .. SSIM score between original wolf and denoised with averaging filter on poisson noise: {:.4f}".format(score)) # ΤΥΠΩΣΗ ΤΟΥ SCORE

print("\n\n") # DOUBLE NEWLINE

print("Mean Square Error (MSE) Method") # ΤΥΠΩΣΗ ΤΗΣ ΜΕΘΟΔΟΥ

print("\n") # NEWLINE

print("Similarity results for salt & pepper noise image:")
print("\n") # NEWLINE
score = np.square(np.subtract(img_gray,imgsp_median)).mean() # ΥΠΟΛΟΓΙΣΜΟΣ ΤΟΥ SCORE ΜΕΤΑΞΥ ΤΩΝ ΔΥΟ ΕΙΚΟΝΩΝ
print( " .. MSE score between original wolf and denoised with median filter on salt & pepper noise: {:.4f}".format(score)) # ΤΥΠΩΣΗ ΤΟΥ SCORE
print("\n") # NEWLINE
score = np.square(np.subtract(img_gray,imgsp_gauss)).mean() # ΥΠΟΛΟΓΙΣΜΟΣ ΤΟΥ SCORE ΜΕΤΑΞΥ ΤΩΝ ΔΥΟ ΕΙΚΟΝΩΝ
print( " .. MSE score between original wolf and denoised with gauss filter on salt & pepper noise: {:.4f}".format(score)) # ΤΥΠΩΣΗ ΤΟΥ SCORE
print("\n") # NEWLINE
score = np.square(np.subtract(img_gray,imgsp_averaging)).mean() # ΥΠΟΛΟΓΙΣΜΟΣ ΤΟΥ SCORE ΜΕΤΑΞΥ ΤΩΝ ΔΥΟ ΕΙΚΟΝΩΝ
print( " .. MSE score between original wolf and denoised with averaging filter on salt & pepper noise {:.4f}".format(score)) # ΤΥΠΩΣΗ ΤΟΥ SCORE

print("\n") # NEWLINE

print("Similarity results for poisson noise image:")
print("\n") # NEWLINE
score = np.square(np.subtract(img_gray,img_poisson_median)).mean() # ΥΠΟΛΟΓΙΣΜΟΣ ΤΟΥ SCORE ΜΕΤΑΞΥ ΤΩΝ ΔΥΟ ΕΙΚΟΝΩΝ
print( " .. MSE score between original wolf and denoised with median filter on poisson noise: {:.4f}".format(score)) # ΤΥΠΩΣΗ ΤΟΥ SCORE
print("\n") # NEWLINE
score = np.square(np.subtract(img_gray,img_poisson_gauss)).mean() # ΥΠΟΛΟΓΙΣΜΟΣ ΤΟΥ SCORE ΜΕΤΑΞΥ ΤΩΝ ΔΥΟ ΕΙΚΟΝΩΝ
print( " .. MSE score between original wolf and denoised with gauss filter on poisson noise: {:.4f}".format(score)) # ΤΥΠΩΣΗ ΤΟΥ SCORE
print("\n") # NEWLINE
score = np.square(np.subtract(img_gray,img_poisson_averaging)).mean() # ΥΠΟΛΟΓΙΣΜΟΣ ΤΟΥ SCORE ΜΕΤΑΞΥ ΤΩΝ ΔΥΟ ΕΙΚΟΝΩΝ
print( " .. MSE score between original wolf and denoised with averaging filter on poisson noise: {:.4f}".format(score)) # ΤΥΠΩΣΗ ΤΟΥ SCORE