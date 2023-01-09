# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:50:08 2020

@author: Teonix
"""
#Spyder 4.1.5

# ΕΙΣΟΔΟΣ ΒΙΒΛΙΟΘΗΚΩΝ
import cv2
import os
import numpy as np

# ΚΑΤΑΣΚΕΥΗ ΣΥΝΑΡΤΗΣΗΣ SOLARIZE
def solarize (originalImage, thresValue):
    output = np.zeros(originalImage.shape,np.uint8) # ΓΕΜΙΣΜΑ ΜΕΤΑΒΛΗΤΗΣ ΤΟΥ ΑΠΟΤΕΛΕΣΜΑΤΟΣ ΜΕ 0
    for i in range(originalImage.shape[0]): # ΛΟΥΠΑ ΣΤΗΝ 1 ΔΙΑΣΤΑΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
        for j in range(originalImage.shape[1]): # ΛΟΥΠΑ ΣΤΗΝ 2 ΔΙΑΣΤΑΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
            if originalImage[i][j] < thresValue: # ΑΝ ΤΟ PIXEL ΕΧΕΙ ΜΙΚΡΟΤΕΡΗ ΤΙΜΗ ΑΠΟ ΤΟ THRESVALUE
                output[i][j] = 255 - originalImage[i][j] # TO PIXEL ΑΛΛΑΖΕΙ ΜΕ ΤΟ ΣΥΜΠΛΗΡΩΜΑ ΤΟΥ
            else:
                output[i][j] = originalImage[i][j] # ΑΛΛΙΩΣ TΟ PIXEL ΠΑΡΑΜΕΝΕΙ ΙΔΙΟ
    return output # ΕΠΙΣΤΡΟΦΗ ΤΟΥ ΑΠΟΤΕΛΕΣΜΑΤΟΣ

directory = r'' # ΔΗΛΩΣΗ ΤΟΥ ΜΟΝΟΠΑΤΙΟΥ ΠΟΥ ΘΑ ΑΠΟΘΗKEΥΤΟΥΝ ΟΙ ΕΙΚΟΝΕΣ

img = cv2.imread('lion.jpg') # ΕΙΣΟΔΟΣ ΤΗΣ ΕΙΚΟΝΑΣ LION ΣΤΗ ΜΕΤΑΒΛΗΤΗ IMG
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # ΜΕΤΑΤΡΟΠΗ ΤΗΣ ΕΙΚΟΝΑΣ ΣΕ GRAYSCALE
img_sol64 = solarize(img_gray,64) # ΚΑΛΕΣΜΑ ΤΗΣ ΣΥΝΑΡΤΗΣΗΣ solarize ΓΙΑ ΜΕΤΑΤΡΟΠΗ ΤΗΣ ΓΚΡΙ ΕΙΚΟΝΑΣ ΜΕ ΤΙΜΗ ΚΑΤΩΦΛΙΟΥ 64
img_sol128 = solarize(img_gray,128) # ΚΑΛΕΣΜΑ ΤΗΣ ΣΥΝΑΡΤΗΣΗΣ solarize ΓΙΑ ΜΕΤΑΤΡΟΠΗ ΤΗΣ ΓΚΡΙ ΕΙΚΟΝΑΣ ΜΕ ΤΙΜΗ ΚΑΤΩΦΛΙΟΥ 128
img_sol192 = solarize(img_gray,192) # ΚΑΛΕΣΜΑ ΤΗΣ ΣΥΝΑΡΤΗΣΗΣ solarize ΓΙΑ ΜΕΤΑΤΡΟΠΗ ΤΗΣ ΓΚΡΙ ΕΙΚΟΝΑΣ ΜΕ ΤΙΜΗ ΚΑΤΩΦΛΙΟΥ 192

os.chdir(directory) # ΕΠΙΛΟΓΗ ΤΟΥ ΜΟΝΟΠΑΤΙΟΥ ΠΟΥ ΦΤΙΑΞΑΜΕ ΠΙΟ ΠΑΝΩ

cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL) # ΕΝΤΟΛΗ ΠΟΥ ΕΠΙΤΡΕΠΕΙ ΤΟ RESIZE ΣΤΟ ΠΑΡΑΘΥΡΟ
cv2.imshow("Original Image", img) # ΕΝΤΟΛΗ ΓΙΑ ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
cv2.resizeWindow("Original Image", 450, 450) # ΕΝΤΟΛΗ ΓΙΑ ΠΡΟΚΑΘΟΡΙΣΜΟ ΤΟΥ ΜΕΓΕΘΟΥΣ ΤΟΥ ΠΑΡΑΘΥΡΟΥ

cv2.namedWindow("Grayscale Image", cv2.WINDOW_NORMAL) # ΕΝΤΟΛΗ ΠΟΥ ΕΠΙΤΡΕΠΕΙ ΤΟ RESIZE ΣΤΟ ΠΑΡΑΘΥΡΟ
cv2.imshow("Grayscale Image", img_gray) # ΕΝΤΟΛΗ ΓΙΑ ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
cv2.resizeWindow("Grayscale Image", 450, 450) # ΕΝΤΟΛΗ ΓΙΑ ΠΡΟΚΑΘΟΡΙΣΜΟ ΤΟΥ ΜΕΓΕΘΟΥΣ ΤΟΥ ΠΑΡΑΘΥΡΟΥ
cv2.imwrite("lionGray.jpg", img_gray) # ΕΝΤΟΛΗ ΓΙΑ ΑΠΟΘΗΚΕΥΣΗ ΤΗΣ ΕΙΚΟΝΑΣ

cv2.namedWindow("Solarized Image with 64 thresValue", cv2.WINDOW_NORMAL) # ΕΝΤΟΛΗ ΠΟΥ ΕΠΙΤΡΕΠΕΙ ΤΟ RESIZE ΣΤΟ ΠΑΡΑΘΥΡΟ
cv2.imshow("Solarized Image with 64 thresValue", img_sol64) # ΕΝΤΟΛΗ ΓΙΑ ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
cv2.resizeWindow("Solarized Image with 64 thresValue", 450, 450) # ΕΝΤΟΛΗ ΓΙΑ ΠΡΟΚΑΘΟΡΙΣΜΟ ΤΟΥ ΜΕΓΕΘΟΥΣ ΤΟΥ ΠΑΡΑΘΥΡΟΥ
cv2.imwrite("lionSol64.jpg", img_sol64) # ΕΝΤΟΛΗ ΓΙΑ ΑΠΟΘΗΚΕΥΣΗ ΤΗΣ ΕΙΚΟΝΑΣ

cv2.namedWindow("Solarized Image with 128 thresValue", cv2.WINDOW_NORMAL) # ΕΝΤΟΛΗ ΠΟΥ ΕΠΙΤΡΕΠΕΙ ΤΟ RESIZE ΣΤΟ ΠΑΡΑΘΥΡΟ
cv2.imshow("Solarized Image with 128 thresValue", img_sol128) # ΕΝΤΟΛΗ ΓΙΑ ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
cv2.resizeWindow("Solarized Image with 128 thresValue", 450, 450) # ΕΝΤΟΛΗ ΓΙΑ ΠΡΟΚΑΘΟΡΙΣΜΟ ΤΟΥ ΜΕΓΕΘΟΥΣ ΤΟΥ ΠΑΡΑΘΥΡΟΥ
cv2.imwrite("lionSol128.jpg", img_sol128) # ΕΝΤΟΛΗ ΓΙΑ ΑΠΟΘΗΚΕΥΣΗ ΤΗΣ ΕΙΚΟΝΑΣ

cv2.namedWindow("Solarized Image with 192 thresValue", cv2.WINDOW_NORMAL) # ΕΝΤΟΛΗ ΠΟΥ ΕΠΙΤΡΕΠΕΙ ΤΟ RESIZE ΣΤΟ ΠΑΡΑΘΥΡΟ
cv2.imshow("Solarized Image with 192 thresValue", img_sol192) # ΕΝΤΟΛΗ ΓΙΑ ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
cv2.resizeWindow("Solarized Image with 192 thresValue", 450, 450) # ΕΝΤΟΛΗ ΓΙΑ ΠΡΟΚΑΘΟΡΙΣΜΟ ΤΟΥ ΜΕΓΕΘΟΥΣ ΤΟΥ ΠΑΡΑΘΥΡΟΥ
cv2.imwrite("lionSol192.jpg", img_sol192) # ΕΝΤΟΛΗ ΓΙΑ ΑΠΟΘΗΚΕΥΣΗ ΤΗΣ ΕΙΚΟΝΑΣ

cv2.waitKey() # ΕΝΤΟΛΗ ΓΙΑ ΑΝΑΜΟΝΗ ΤΩΝ ΠΑΡΑΘΥΡΩΝ 
cv2.destroyWindow("Original Image") # ΕΝΤΟΛΗ ΓΙΑ ΚΛΕΙΣΙΜΟ ΤΟΥ ΠΑΡΑΘΥΡΟΥ "Original Image"
cv2.destroyWindow("Grayscale Image") # ΕΝΤΟΛΗ ΓΙΑ ΚΛΕΙΣΙΜΟ ΤΟΥ ΠΑΡΑΘΥΡΟΥ "Grayscale Image"
cv2.destroyWindow("Solarized Image with 64 thresValue") # ΕΝΤΟΛΗ ΓΙΑ ΚΛΕΙΣΙΜΟ ΤΟΥ ΠΑΡΑΘΥΡΟΥ "Solarized Image with 64 thresValue"
cv2.destroyWindow("Solarized Image with 128 thresValue") # ΕΝΤΟΛΗ ΓΙΑ ΚΛΕΙΣΙΜΟ ΤΟΥ ΠΑΡΑΘΥΡΟΥ "Solarized Image with 128 thresValue"
cv2.destroyWindow("Solarized Image with 192 thresValue") # ΕΝΤΟΛΗ ΓΙΑ ΚΛΕΙΣΙΜΟ ΤΟΥ ΠΑΡΑΘΥΡΟΥ "Solarized Image with 192 thresValue"
