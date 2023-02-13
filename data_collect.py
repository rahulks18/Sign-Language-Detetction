import cv2
import os
import numpy as np
chars = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','NULL','O','P','Q','R','S','SPACE','T','U','V','W','X','Y','Z']

# Create the directory structure
if not os.path.exists("data"):
    os.makedirs("data")
    os.makedirs("data/train")
    os.makedirs("data/test")
    for c in chars:
        os.makedirs("data/train/"+c)
        os.makedirs("data/test/"+c)

mode = 'test'
directory = 'data/'+mode+'/'
count = 0
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    # Coordinates of the ROI
    window_size = 250
    x1 = 360
    y1 = 20
    x2 = x1+ window_size
    y2 = y1 + window_size

    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (0,0,255) ,1)
    img = frame[y1:y2, x1:x2]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV) 

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # img = cv2.GaussianBlur(img,(9,9),sigmaX=0)

    crop_img = cv2.resize(img,(64,64))
    #cv2.putText(frame, str(80), (300, 150),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2) 
    blur = cv2.GaussianBlur(img,(5,5),2)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    resized=cv2.resize(res,(64,64))
    normalized=resized/255.0
    reshaped=np.reshape(normalized,(1,64,64,1))

    total = 0
    # Getting count of existing images
    count = {}
    for c in chars:
        count[c] = len(os.listdir(directory+c))
        total = total + count[c]  
    # Printing the count in each set to the screen
    cv2.putText(frame, "MODE : "+mode, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "TOTAL IMAGE COUNT : "+str(total), (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    y = 80
    for c in chars:
        cv2.putText(frame, c+" : "+str(count[c]), (10, y), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        y = y + 15

    cv2.imshow("frame", frame)
    cv2.imshow("img",res)

    interrupt = cv2.waitKey(10)
    key = interrupt & 0xFF
    if key == 27: # esc key
        break
    if key == 32: #space
       cv2.imwrite(directory+'SPACE/'+str(count['SPACE'])+'.jpg', resized) 
    
    if key == 13: #enter
      cv2.imwrite(directory+'NULL/'+str(count['NULL'])+'.jpg', resized) 
    if key >= ord('a') and key <= ord('z'):
       cv2.imwrite(directory+(chr(key).upper())+'/'+str(count[chr(key).upper()])+'.jpg', resized)
cap.release()
cv2.destroyAllWindows()