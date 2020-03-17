import os

def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_GAN_gif():
    import imageio, os

    images = []
    img_folder = 'GAN_images/'
    save_folder = 'media/'

    filenames = os.listdir(img_folder)

    for i,file in enumerate(filenames,1):
        try:
            images.append(imageio.imread(img_folder+str(i)+'.png'))

            print('making movie... {}% done'.format(round((i/len(filenames))*100,2)))
        except FileNotFoundError:
            pass
        imageio.mimsave(save_folder+'movie.gif', images)


# make_GAN_gif()


def make_GAN_movie():
    import os
    import moviepy.editor as me

    song = 'Pokémon Theme Song.mp3'
    song2 = 'Battle Music.mp3'
    song3 = 'A Rival Appears! [Pokémon Red & Blue].mp3'
    song4 = 'Pokemon Omega RubyAlpha Sapphire - Battle! Rival Music (HQ).mp3'

    folder = 'GAN_images/'
    save_folder = 'media/'

    files = os.listdir(folder)
    files = [x.split('.')[0] for x in files]
    files = [int(x) for x in files]
    files.sort()

    file_names = [folder+str(x)+'.png' for x in files]
    duration = len(file_names)/24

    audio = me.AudioFileClip(save_folder+song4).set_duration(duration)

    clip = me.ImageSequenceClip(file_names,fps=24).set_duration(duration)

    final_clip = clip.set_audio(audio)

    final_clip.write_videofile(save_folder+"movie.mp4",
                               temp_audiofile="temp-audio.m4a",
                               remove_temp=True,
                               codec="libx264",
                               audio_codec="aac")

def converter(x):
    #x has shape (batch, width, height, channels)
    return (0.21 * x[:,:,:,:1]) + (0.72 * x[:,:,:,1:2]) + (0.07 * x[:,:,:,-1:])


def load_and_scale_images(path,size=(64,64)):
    import cv2,os
    from tqdm import tqdm
    import numpy as np
    img_dims = size

    imgs = os.listdir(path)
    imgs = [img for img in imgs if imgs != '.DS_Store']

    img_data = []

    print('Loading images...')
    for i,img in enumerate(tqdm(imgs)):


        img = cv2.imread(path+img)

        if not isinstance(img,type(None)):
            corner_color = img[0][0]


            fill_val = [0,0,0]
            img[np.where((img == corner_color).all(axis=2))] = fill_val


            x = img.shape[0]
            y = img.shape[1]
            if x <= 64 and y<=64:
                if x > y:
                    img = cv2.copyMakeBorder(
                        img,
                        top=0,
                        bottom=x-y,
                        left=0,
                        right=0,
                        borderType=cv2.BORDER_CONSTANT,
                        value=fill_val)
                elif y>x:
                    img = cv2.copyMakeBorder(
                        img,
                        top=0,
                        bottom=0,
                        left=0,
                        right=y-x,
                        borderType=cv2.BORDER_CONSTANT,
                        value=fill_val)

                img = cv2.resize(img,img_dims)
                img = (img/(255/2))-1 # rescale from [0,255] to [-1,1]

                img_data.append(img)

    print('{} images have been loaded...'.format(len(img_data)))

    return img_data


# make_GAN_movie()