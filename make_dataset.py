import cv2 as cv
import os
from tqdm import tqdm

def create_data(tracks, frames_folder):#tracks è il file di testo mot, frames folder che contiene i frame

    #apertura file txt con tracks
    file = open(tracks)
    lines = file.readlines()

    with tqdm(total=len(lines), desc='Progresso', unit='righe') as progress_bar:
        #ciclo su righe del file
        for box_info in lines:
            #per training e val split(', ') e varibile z, per test split(',') senza z
            frame_idx, id, left, top, w, h, conf, x,y = box_info.split(',')
            frame_idx, id, left, top, w, h = int(float(frame_idx)), int(float(id)), int(float(left)), int(float(top)), int(float(w)), int(float(h))

            #crea directory se non esiste già
            dir_path = os.path.join('test_set', f'{id:04d}')
            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)

            #path del crop
            path = os.path.join(dir_path, f'{id:04d}_{frame_idx}.png')

            #legge frame e fa il crop
            crop = cv.imread(f'{frames_folder}/{frame_idx-1:04d}.png')[top:top+h,left:left+w]

            #non considera immagini troppo piccole
            if crop.shape[0] > 20 and crop.shape[1] > 20:
                cv.imwrite(f"{path}", crop)

            progress_bar.update(1)

def obtain_tracks():
    print("")

tracks = ""#only the bbox obtained with yolov5 in the mot format
frames_folder = "data/Grapes_001"#all the frames w/o bbox

create_data(tracks, frames_folder)#this function should create a folder for each different identificated grape,
                                  #in this folder there will be the cropped image of that grape, for each frame