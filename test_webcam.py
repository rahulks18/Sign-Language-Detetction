import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import operator
import tensorflow as tf

chars = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','NULL','O','P','Q','R','S','SPACE','T','U','V','W','X','Y','Z']
#classifier = load_model('Trained_model.h5')

# Loading the model
json_file = open("Trained_model.json", "r")
model_json = json_file.read()
json_file.close()
classifier = model_from_json(model_json)
# load weights into new model
classifier.load_weights("Trained_model.h5")
print("Loaded model from disk")

def predict():
    
    test_image = tf.keras.utils.load_img('1.png', target_size=(64, 64))
    test_image = tf.keras.utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    prediction = {}
    i = 0
    for c in chars:
        prediction[c] = result[0][i]
        i = i+1
    # Sorting based on top prediction
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    return prediction[0]

cap = cv2.VideoCapture(0)

text = ''
word = ''
c = 0
last_char = 'NULL'

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
    #_, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV) 

    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
    #img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    #img = cv2.GaussianBlur(img,(9,9),sigmaX=0)

 
    blur = cv2.GaussianBlur(img,(5,5),2)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    resized=cv2.resize(res,(64,64))
    normalized=resized/255.0
    reshaped=np.reshape(normalized,(1,64,64,1))

    img_name = "1.png"
    save_img = cv2.resize(resized, (64,64))
    cv2.imwrite(img_name, save_img)
    result = predict()
    
    if c == 0:
        last = result[0]
    if(last == result[0]):
        c = c + 1
    else : 
        c = 0

    if c >= 10:
        cv2.putText(frame, 'Predicted : '+ str(result[0]) , (x1, y2+20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))
  
    if c >= 30 and result[0] != 'NULL':
        if result[0] == 'SPACE':
            text += ' _ '
		
            word = ''
    
        else:
            word += result[0]
            text += result[0]
        c = 0
    last = result[0]
    
    cv2.putText(frame, text, (20, 400), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 0))
    
    cv2.imshow("frame", frame)
    cv2.imshow("img",cv2.resize(res,(128,128)))

    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()
