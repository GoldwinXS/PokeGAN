


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

make_GAN_movie()