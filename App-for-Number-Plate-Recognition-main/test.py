import streamlit as st
import numpy as np
import cv2
import  imutils
import pytesseract
import pandas as pd


st.write("Number Plate Reader")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:

        image = cv2.imread(uploaded_file)

        image = imutils.resize(image, width=500)

        cv2.imshow("Original Image", image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("1 - Grayscale Conversion", gray)

        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        #cv2.imshow("2 - Bilateral Filter", gray)

        edged = cv2.Canny(gray, 170, 200)
        #cv2.imshow("4 - Canny Edges", edged)

        (new, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] 
        NumberPlateCnt = None 

        count = 0
        for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:  
                        NumberPlateCnt = approx 
                        break

        # Masking the part other than the number plate
        mask = np.zeros(gray.shape,np.uint8)
        new_image = cv2.drawContours(mask,[NumberPlateCnt],0,255,-1)
        new_image = cv2.bitwise_and(image,image,mask=mask)
        st.image("Final_image",new_image)

        # Configuration for tesseract
        config = ('-l eng --oem 1 --psm 3')

        # Run tesseract OCR on image
        text = pytesseract.image_to_string(new_image, config=config)

        st.write(text)
        
