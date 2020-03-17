import cv2,os
from ProjectUtils import ensure_folder
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, \
    Convolution2D,\
    MaxPooling2D,\
    Conv2D,\
    BatchNormalization,\
    Dropout,\
    Conv2DTranspose,\
    UpSampling2D,\
    Reshape,\
    Flatten,\
    Activation,\
    ZeroPadding2D

from keras.layers import LeakyReLU
from keras.initializers import RandomNormal
from keras.optimizers import  Adam, SGD

"""

N.B.: keep colab open by typing the following js code into the console 

function ConnectButton(){
    console.log("Connect pushed"); 
    document.querySelector("#connect").click() 
}
setInterval(ConnectButton,60000);

N.B.: running well on Colab with versions
tf == 2.1.0
keras == 2.3.1


"""


""" Project Setup"""

img_dim = 64
img_dims = (img_dim,img_dim)
base_path = ''
ensure_folder(base_path+'GAN_images/')
ensure_folder(base_path+'models/')
ensure_folder(base_path+'pokemon/')



""" Class Setup """

class Generator:
    def __init__(self):
        self.model = Sequential()
        self.lr = 0.0002
        self.make_model_v4()


    def make_model_v4(self):

        self.model.add(Dense(64 * 2 * 2,input_dim=100))
        self.model.add(Activation('tanh'))

        self.model.add(Dense(256 * 8 * 8))

        self.model.add(BatchNormalization())
        self.model.add(Activation('tanh'))
        self.model.add(Reshape((8, 8, 256), input_shape=(128 * 8 * 8,)))


        self.model.add(UpSampling2D(size=(2, 2)))
        self.model.add(Conv2D(128, (5, 5), padding='same'))
        self.model.add(Activation('tanh'))

        self.model.add(UpSampling2D(size=(2, 2)))
        self.model.add(Conv2D(64, (5, 5), padding='same'))
        self.model.add(Activation('tanh'))

        self.model.add(UpSampling2D(size=(2, 2)))
        self.model.add(Conv2D(3, (5, 5), padding='same'))
        self.model.add(Activation('tanh'))


    def make_model_v3(self):

        # N.B.: this is for a 28x28x3 image

        self.model.add(Dense(128 * 7 * 7,input_dim=100))
        self.model.add(Activation('tanh'))
        self.model.add(Dense(128 * 7 * 7))

        self.model.add(BatchNormalization())
        self.model.add(Activation('tanh'))
        self.model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))


        self.model.add(UpSampling2D(size=(2, 2)))
        self.model.add(Conv2D(64, (5, 5), padding='same'))
        self.model.add(Activation('tanh'))

        self.model.add(UpSampling2D(size=(2, 2)))
        self.model.add(Conv2D(3, (5, 5), padding='same'))
        self.model.add(Activation('tanh'))

        # g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        # self.model.compile(loss='binary_crossentropy', optimizer="SGD")


    def make_model_v2(self):
        transforms = 5
        channels = 3
        start_dim = 4

        self.model.add(Dense(start_dim * start_dim * (channels**transforms), input_dim=100))
        self.model.add(BatchNormalization(momentum=0.9))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(Reshape((start_dim, start_dim, (channels**transforms))))
        self.model.add(Dropout(0.4))

        self.model.add(UpSampling2D())
        self.model.add(Conv2DTranspose(128, 5, padding='same'))
        self.model.add(BatchNormalization(momentum=0.9))
        self.model.add(LeakyReLU(alpha=0.3))

        self.model.add(UpSampling2D())
        self.model.add(Conv2DTranspose(128, 5, padding='same'))
        self.model.add(BatchNormalization(momentum=0.9))
        self.model.add(LeakyReLU(alpha=0.3))

        self.model.add(UpSampling2D())
        self.model.add(Conv2DTranspose(128, 5, padding='same'))
        self.model.add(BatchNormalization(momentum=0.9))
        self.model.add(LeakyReLU(alpha=0.3))

        self.model.add(UpSampling2D())
        self.model.add(Conv2DTranspose(3, 5, padding='same'))
        self.model.add(Activation('tanh'))

        self.model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.lr),metrics=['accuracy'])
        self.model.summary()

    def make_model(self):
        transforms = 5
        channels = 3
        start_dim = 4

        # Starting size
        self.model.add(Dense(start_dim * start_dim * 256, kernel_initializer=RandomNormal(0, 0.02), input_dim=100))
        self.model.add(LeakyReLU(0.2))
        # 4x4x256
        self.model.add(Reshape((start_dim, start_dim, 256)))

        # 8x8x128
        self.model.add(Conv2DTranspose(128, (4, 4), strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02)))
        self.model.add(LeakyReLU(0.2))

        # 16x16*128
        self.model.add(Conv2DTranspose(128, (4, 4), strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02)))
        self.model.add(LeakyReLU(0.2))

        # 32x32x128
        self.model.add(Conv2DTranspose(128, (4, 4), strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02)))
        self.model.add(LeakyReLU(0.2))

        # 64x64x128
        self.model.add(Conv2DTranspose(128, (4, 4), strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02)))
        self.model.add(LeakyReLU(0.2))

        # 64x64x3
        self.model.add(Conv2D(channels, (3, 3), padding='same', activation='sigmoid', kernel_initializer=RandomNormal(0, 0.02)))

        self.model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.lr,),metrics=['accuracy'])

        print('Generator Model')
        self.model.summary()



class Discriminator:
    def __init__(self):
        self.model = Sequential()
        self.img_rows= img_dim
        self.img_cols= img_dim
        self.channel= 3
        self.lr = 0.0003
        self.make_model_v3()

    def make_model_v3(self):
        self.model.add(
            Conv2D(64, (5, 5),
                   padding='same',
                   input_shape=(img_dim, img_dim, 3))
        )
        self.model.add(Activation('tanh'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (5, 5)))
        self.model.add(Activation('tanh'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (5, 5)))
        self.model.add(Activation('tanh'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(256))
        self.model.add(Activation('tanh'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

    def make_model_v2(self):
        # self.D = Sequential()
        depth = 64
        dropout = 0.4
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64
        input_shape = (img_dim,img_dim,3)
        self.model.add(Conv2D(depth * 1, 5, strides=2, input_shape=input_shape,padding='same', ))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(Dropout(dropout))

        self.model.add(Conv2D(depth * 2, 5, strides=2, padding='same'))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(Dropout(dropout))

        self.model.add(Conv2D(depth * 4, 5, strides=2, padding='same',))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(Dropout(dropout))

        self.model.add(Conv2D(depth * 8, 5, strides=1, padding='same',))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(Dropout(dropout))

        self.model.add(Flatten())
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.lr,beta_1=0.5),metrics=['accuracy'])

        self.model.summary()

    def make_model(self):

        img_shape = (img_dim,img_dim,3)
        self.kernel_size = 5

        self.model.add(Conv2D(32, kernel_size=self.kernel_size, strides=2, input_shape=img_shape, padding="same"))  # 192x256 -> 96x128
        self.model.add(LeakyReLU(alpha=0.2))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, kernel_size=self.kernel_size, strides=2, padding="same"))  # 96x128 -> 48x64
        self.model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        self.model.add(LeakyReLU(alpha=0.2))
        self.model.add(Dropout(0.25))
        self.model.add(BatchNormalization(momentum=0.8))

        self.model.add(Conv2D(128, kernel_size=self.kernel_size, strides=2, padding="same"))  # 48x64 -> 24x32
        self.model.add(LeakyReLU(alpha=0.2))
        self.model.add(Dropout(0.25))
        self.model.add(BatchNormalization(momentum=0.8))

        self.model.add(Conv2D(256, kernel_size=self.kernel_size, strides=1, padding="same"))  # 24x32 -> 12x16
        self.model.add(LeakyReLU(alpha=0.2))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(512, kernel_size=self.kernel_size, strides=1, padding="same"))  # 12x16 -> 6x8
        self.model.add(LeakyReLU(alpha=0.2))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(1, activation='sigmoid'))



class AdversarialModel:
    def __init__(self):

        self.save_version = 0


        self.model = Sequential()
        self.test_noise = np.random.normal(0, 1, (16, 100))
        self.discriminator = Discriminator()
        self.generator = Generator()

        d_optimizer = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        g_optimizer = SGD(lr=0.0005, momentum=0.9, nesterov=True)


        self.make_model(self.generator.model,self.discriminator.model)
        self.model.compile(loss='binary_crossentropy', optimizer=g_optimizer)

        self.generator.model.compile(loss='binary_crossentropy',optimizer="SGD")

        self.discriminator.model.trainable = True
        self.discriminator.model.compile(loss='binary_crossentropy',optimizer=d_optimizer)

        self.generator.model.summary()
        self.discriminator.model.summary()



    def make_model(self,generator,discriminator):
        self.model = Sequential()
        self.model.add(generator)
        discriminator.trainable = False
        self.model.add(discriminator)


    @staticmethod
    def assemble_images(images):
        img = images
        top_imgs = np.array(img[0])
        for i in range(1, 4):
            top_imgs = np.concatenate((top_imgs, img[i]), axis=1)

        bottom_imgs = np.array(img[4])
        for i in range(5, 8):
            bottom_imgs = np.concatenate((bottom_imgs, img[i]), axis=1)

        top_rows = np.concatenate((top_imgs, bottom_imgs), axis=0)

        top_imgs = np.array(img[0])
        for i in range(9, 12):
            top_imgs = np.concatenate((top_imgs, img[i]), axis=1)

        bottom_imgs = np.array(img[4])
        for i in range(13, 16):
            bottom_imgs = np.concatenate((bottom_imgs, img[i]), axis=1)

        bottom_rows = np.concatenate((top_imgs, bottom_imgs), axis=0)

        final_img = np.concatenate((top_rows, bottom_rows), axis=0)

        return final_img

    def train(self,images, epochs, batch_size=128, sample_interval=50,save_interval =100):
        X_train = np.array(images)
        valid = np.zeros((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            noise = np.random.normal(0, 1, (batch_size, 100))
            gen_imgs = self.generator.model.predict(noise)
            d_loss_real = self.discriminator.model.train_on_batch(imgs, valid) # real pokemon images
            d_loss_fake = self.discriminator.model.train_on_batch(gen_imgs, fake) # "fake" images
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, 100))
            g_loss = self.model.train_on_batch(noise, valid)

            text = "%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss[0])
            print(text)


            if epoch % sample_interval == 0:

                img = self.generator.model.predict(self.test_noise)


                final_img = self.assemble_images(img)
                final_img = (final_img +1)/2 # move range to [0,1]

                n_images = len(os.listdir('drive/My Drive/Colab Notebooks/GAN_images'))

                display_scale =3
                final_img = cv2.resize(final_img,(final_img.shape[1]*display_scale,final_img.shape[0]*display_scale))




                final_img = final_img*255 # move range to [0,255]
                cv2.imwrite('drive/My Drive/Colab Notebooks/GAN_images/'+str(n_images)+'.png',final_img)


            if epoch % save_interval == 0:
                print('Save interval reached. Saving models...')
                self.save()

    def train_v2(self,images, epochs, batch_size=128, sample_interval=50,save_interval =100):
        X_train = np.array(images) # convert list of images to numpy arr


        # start tracking previous inputs
        previous_inputs = np.array(self.generator.model.predict(np.random.normal(-1, 1, (batch_size, 100))))
        n_previous_inputs = 500
        previous_inputs_sample_size = 10

        print('Training model...')

        for epoch in range(epochs):


            index = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[index]
            noise = np.random.normal(-1, 1, (batch_size, 100))


            gen_imgs = self.generator.model.predict(noise)


            previous_inputs = np.concatenate((previous_inputs,gen_imgs[:previous_inputs_sample_size])) # take a sample from this epoch
            if previous_inputs.shape[0]>=n_previous_inputs: # if we're getting too many previous examples, then remove some
                previous_inputs = previous_inputs[previous_inputs_sample_size:]

            # prepare inputs for D
            X = np.concatenate((imgs, gen_imgs))
            y = [1] * batch_size + [0] * batch_size # N.B.: 1 is for valid, 0 is for fake

            # Train D
            d_loss = self.discriminator.model.train_on_batch(X,y)
            self.discriminator.model.train_on_batch(previous_inputs,[0]*previous_inputs.shape[0]) # train D on previous inputs


            # Train G on D
            self.discriminator.model.trainable = False
            noise = np.random.normal(-1, 1, (batch_size*2, 100))
            g_loss = self.model.train_on_batch(noise, [1]*batch_size*2)
            self.discriminator.model.trainable = True



            text = "%d [D loss: %f] [G loss: %f]" % (epoch, d_loss, g_loss)
            print(text)


            if epoch % sample_interval == 0:

                img = self.generator.model.predict(self.test_noise)

                final_img = self.assemble_images(img)
                final_img = (final_img +1)*(255/2) # move range to [0,1] from [-1,1]
                n_images = len(os.listdir(base_path+'GAN_images/'))

                display_scale =1
                final_img = cv2.resize(final_img,(final_img.shape[1]*display_scale,final_img.shape[0]*display_scale))


                if 'Colab' not in base_path: # do not do this if we're in Colab
                    cv2.imshow(text,final_img/255) # imshow uses range [0,1] for floating point numbers
                    cv2.waitKey(500)
                    cv2.destroyAllWindows()

                cv2.imwrite(base_path+'GAN_images/'+str(n_images)+'.png',final_img)


            if epoch % save_interval == 0:



                print('Save interval reached. Saving models...')
                self.save()


    def save(self):
        folder=base_path+'models/'

        self.save_version+=1

        if self.save_version %3==0:
            self.save_version = 1

        version = self.save_version
        self.discriminator.model.save_weights(folder+'discriminator'+str(version)+'.h5')
        self.generator.model.save_weights(folder+'generator'+str(version)+'.h5')
        self.model.save_weights(folder+'adversarial'+str(version)+'.h5')
        np.save(folder+'noise',self.test_noise)
        print('Model saved...')


    def load(self,version=0):
        folder=base_path+'models/'

        if version != 0:
            version = version
        else:
            version = self.save_version

        self.discriminator.model.load_weights(folder+'discriminator'+str(version)+'.h5')
        self.generator.model.load_weights(folder+'generator'+str(version)+'.h5')
        self.model.load_weights(folder+'adversarial'+str(version)+'.h5')
        self.test_noise = np.load(folder+'noise.npy')
        print('Model loaded...')




if "Colab" not in base_path:
    from ProjectUtils import load_and_scale_images
    images = load_and_scale_images(base_path+'pokemon/',size=img_dims)
    save_images = np.save('images.npy',images)
else:
    images = np.load(base_path+'images.npy')

AM = AdversarialModel()
AM.load(version=1)
AM.model.save('model.h5')

AM.train_v2(images,
         epochs=100000,
         batch_size=32,
         sample_interval=10,
         save_interval=50)