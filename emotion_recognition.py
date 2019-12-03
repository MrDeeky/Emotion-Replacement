import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, SeparableConv2D, MaxPooling2D, Add, GlobalAveragePooling2D, Input
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import cv2
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split
import random


def create_dataset(dataset, expression):
    # convert pixels to original image size (48x48) and resize to (128, 128)
    pixels = dataset['pixels'].tolist()
    faces = []
    for sequence in pixels:
        face = []
        values = sequence.split(' ')
    for value in values:
        face.append(value)
    face = np.array(face).reshape(48, 48)
    face = cv2.resize(face.astype('uint8'), (128, 128))
    faces.append(face.astype('float32'))
    faces = np.expand_dims(faces, -1)

    # convert emotion value to 1D vector
    emotion = dataset['emotion'].tolist()
    emotions = []
    for value in emotion:
        emotions.append(expression[value])
    emotions = np.array(emotions)
    return faces, emotions

def XCeption(input):
    out = input
    for i in range(2):
        out = Conv2D(filters=8, kernel_size=3, padding='same')(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

    for filter in [16, 32, 64, 128]:
        old = Conv2D(filters=filter, kernel_size=1, strides=2, padding='same')(out)
        old = BatchNormalization()(old)

        for i in range(2):
            out = SeparableConv2D(filters=filter, kernel_size=3, padding='same')(out)
            out = BatchNormalization()(out)
            if i == 0:
                out = Activation('relu')(out)

        out = MaxPooling2D(pool_size=3,  strides=2, padding='same')(out)
        out = Add()([out, old])

    out = Conv2D(filters=7, kernel_size=3, padding='same')(out)
    out = GlobalAveragePooling2D()(out)
    out = Activation('softmax')(out)

    return Model(input, out)


def build_cnn_project():
    return XCeption(Input((128, 128, 1)))
    
    
def swap_emotion(prediction):
    # 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
    expression = ['anger', 'disgust',
                  'fear', 'happy',
                  'sad', 'surprise',
                  'neutral']
    index = np.argmax(prediction)

    possible_gan_emotions = [0,1,2,3,4,5,6]
    print('detected ', expression[index])
    if expression[index] == 'anger':
        possible_gan_emotions.remove(2)
    elif expression[index] == 'disgust':
        possible_gan_emotions.remove(3)
    elif expression[index] == 'fear':
        possible_gan_emotions.remove(1)
    elif expression[index] == 'happy':
        possible_gan_emotions.remove(4)
    elif expression[index] == 'sad':
        possible_gan_emotions.remove(5)
    elif expression[index] == 'surprise':
        possible_gan_emotions.remove(6)
    elif expression[index] == 'neutral':
        possible_gan_emotions.remove(0)

    gan_emotion = random.choice(possible_gan_emotions)
    gan_emotion_vector = np.zeros((1,7))
    gan_emotion_vector[0][gan_emotion] = 1
    print(gan_emotion_vector)
    return gan_emotion_vector
    
    
 
if __name__ == "__main__":
    # 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
    expression = [np.array([1, 0, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0, 0]),
                    np.array([0, 0, 1, 0, 0, 0, 0]), np.array([0, 0, 0, 1, 0, 0, 0]),
                    np.array([0 ,0 ,0 ,0, 1, 0 ,0]), np.array([0, 0, 0, 0, 0, 1, 0]),
                    np.array([0, 0, 0, 0, 0, 0, 1])]
    dataset = pd.read_csv('/content/drive/My Drive/fer2013.csv')
    faces, emotions = create_dataset(dataset, expression)
    
    # split dataset in a randomize manner so there's variation between trainings
    x_train, x_test, y_train, y_test = train_test_split(faces, emotions, shuffle=True)

    input = Input((128, 128, 1))
    model = XCeption(input)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # cnn.summary()
    model.load_weights('/content/drive/My Drive/emotion_recognition.h5')

    #mc = tf.keras.callbacks.ModelCheckpoint('/content/drive/My Drive/project_weights3/weight{epoch:04d}.h5', save_weights_only=True)

    generator = ImageDataGenerator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
    #cnn.fit_generator(generator.flow(x_train, y_train), steps_per_epoch=(len(x_train)/32), epochs=200, callbacks=[mc], validation_data=(x_test, y_test))
    
    # INSTRUCTIONS TO PREDICT:
    # -pass in image in dimensions (128, 128, 1)
    # -expand the dimensions
    # -predict
    # -get first index which is the 1D vector
    # -get index of expression
    test_img = faces[12345]
    test_img = np.expand_dims(img, axis=0)
    result = model.predict(img)[0]
    predicted_emotion = expression[np.argmax(result)]