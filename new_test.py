import os
import shutil

audio_dir='../dataset/audio/'
moved='/media/intern0/D560-049C/DCASE_data/audio'
goto ='/media/intern0/3461-93FE/dcase_audio'
audio_list=os.listdir(audio_dir)
moved=os.listdir(moved)

for aj in audio_list:
    if aj not in moved:
        shutil.copy(audio_dir+aj, goto)
        # exit()
print(moved)
print(audio_list)