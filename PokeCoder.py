from keras.models import Sequential,load_model
from keras.layers import Dense,\
    Conv2D,\
    Conv2DTranspose,\
    UpSampling2D,\
    MaxPool2D,\
    Activation,\
    BatchNormalization,\
    Reshape,\
    Convolution2D,\
    Flatten

activation = 'Adam'

import numpy as np

encoded_image_dims = (28,28)
decoded_image_dims = (64,64)

def make_encoder():
    encoder = Sequential()
    encoder.add(Convolution2D(64,5,strides=(2,2),padding='same',input_shape=encoded_image_dims+(3,)))
    encoder.add(Activation('tanh'))
    encoder.add(Convolution2D(64,5,strides=(2,2),padding='same'))
    encoder.add(Activation('tanh'))
    encoder.add(MaxPool2D((2,2)))
    # encoder.add(Convolution2D(64,5,strides=(2,2),padding='same'))
    # encoder.add(Convolution2D(64,5,strides=(2,2),padding='same'))
    # encoder.add(MaxPool2D((2,2)))
    encoder.add(Flatten())
    encoder.add(Dense(256))
    encoder.add(Activation('tanh'))


    encoder.compile(optimizer=activation,loss='binary_crossentropy')
    encoder.summary()

    return encoder

def make_decoder():
    decoder = Sequential()
    decoder.add(Dense(16 * 2 * 2, input_dim=256))
    decoder.add(Activation('tanh'))

    decoder.add(Dense(256 * 8 * 8))

    decoder.add(BatchNormalization())
    decoder.add(Activation('tanh'))

    print(decoder.output_shape)
    decoder.add(Reshape((8, 8, 256), input_shape=(128 * 8 * 8,)))

    decoder.add(UpSampling2D(size=(2, 2)))
    decoder.add(Conv2D(128, (5, 5), padding='same'))
    decoder.add(Activation('tanh'))

    decoder.add(UpSampling2D(size=(2, 2)))
    decoder.add(Conv2D(64, (5, 5), padding='same'))
    decoder.add(Activation('tanh'))

    decoder.add(UpSampling2D(size=(2, 2)))
    decoder.add(Conv2D(3, (5, 5), padding='same'))
    decoder.add(Activation('tanh'))

    decoder.compile(optimizer=activation,loss='binary_crossentropy')
    decoder.summary()

    return decoder

auto_encoder = Sequential()
auto_encoder.add(make_encoder())
auto_encoder.add(make_decoder())
auto_encoder.compile(optimizer=activation,loss='binary_crossentropy')

from ProjectUtils import load_and_scale_images,converter


X = np.array(load_and_scale_images('pokemon/',encoded_image_dims))
y = np.array(load_and_scale_images('pokemon/',decoded_image_dims))

# X = converter(X)

X = (X+1)/2
y = (y+1)/2

np.save('autoencoderX',X)
np.save('autoencodery',y)









def test_auto_encoder():
    import cv2

    for i in range(10):
        test_img =X[i]
        cv2.imshow('Autoencoder input',test_img)
        prediction = auto_encoder.predict(np.array([test_img]))[0]
        cv2.imshow('Autoencoder result',prediction)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()


# auto_encoder.fit(X,y,epochs=1)
# auto_encoder.save_weights('models/pokecoder.h5')
test_auto_encoder()

for i in range(5):
    auto_encoder.load_weights('models/pokecoder.h5')
    auto_encoder.fit(X,y,epochs=5)
    auto_encoder.save_weights('models/pokecoder.h5')
    test_auto_encoder()

