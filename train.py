from calendar import c
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
import tensorflow

# Initialing the CNN
classifier = Sequential()

# Step 1 - Convolutio Layer 
classifier.add(Convolution2D(32, 3,  3, input_shape = (64, 64, 3), activation = 'relu'))

#step 2 - Pooling
classifier.add(MaxPooling2D(pool_size =(2,2)))

# Adding second convolution layer
classifier.add(Convolution2D(32, (3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))

# Adding Third convolution layer
classifier.add(Convolution2D(64, (3,3), activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(MaxPooling2D(pool_size =(2,2)))


#Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Full Connection
classifier.add(Dense(6122, activation = 'relu'))
classifier.add(Dense(28, activation = 'softmax'))

#Compiling The CNN
classifier.compile(
              optimizer = tensorflow.keras.optimizers.SGD(lr = 0.01),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

#Part 2 Fittting the CNN to the image
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'data/train',
        target_size=(64, 64),
        batch_size=5,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'data/test',
        target_size=(64, 64),
        batch_size=5,
        class_mode='categorical')

model = classifier.fit_generator(
        training_set,
        steps_per_epoch= 6002//25,
        epochs=25,
        validation_data = test_set,
        validation_steps =6002 
      )

#Saving the model
#classifier.save('Trained_model.h5')

model_json = classifier.to_json()
with open("Trained_model.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights('Trained_model.h5')






