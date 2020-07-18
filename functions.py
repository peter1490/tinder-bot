try:
    import os
    os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))
    import face_recognition
except Exception:
    print("Could not import Packages !")
    exit()

 
import ctypes
import sys, os
from PIL import ImageGrab
import time
import pickle
import numpy as np
from tqdm import tqdm
import cv2


KNOWN_FACES_DIR = 'facedir/known_faces'
UNKNOWN_FACES_DIR = 'facedir/unknown_faces'
DATA_FILE = "dataset_faces.dat"
TOLERANCE = 0.52
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'cnn'  # 'hog' or 'cnn' - CUDA accelerated (if available) deep-learning pretrained model

user32 = ctypes.windll.user32
user32.SetProcessDPIAware()

screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)

def loadKnownFaces():
    print('Loading known faces...')
    all_face_encodings = {}
    known_faces = []
    known_names = []
    for name in os.listdir(KNOWN_FACES_DIR):

        # Next we load every file of faces of known person
        print(f"loading {name} faces...\n")
        filenames = os.listdir(f'{KNOWN_FACES_DIR}/{name}')
        for i in tqdm(range(len(filenames))):

            # Load an image
            image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filenames[i]}')

            # Get 128-dimension face encoding
            # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
            faces = face_recognition.face_encodings(image)
            if len(faces) > 0:
                encoding = faces[0]

                # Append encodings and name
                known_faces.append(encoding)
                known_names.append(name)
        all_face_encodings[0] = known_faces
        all_face_encodings[1] = known_names    
        with open(DATA_FILE, 'wb') as f:
            pickle.dump(all_face_encodings, f)

def hasMatch(picture_path, all_face_encodings=None):
    if all_face_encodings == None:
        with open(DATA_FILE, 'rb') as f:
            all_face_encodings = pickle.load(f)
    known_faces = np.array(list(all_face_encodings[0]))
    known_names = np.array(list(all_face_encodings[1]))

    image = face_recognition.load_image_file(picture_path)
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_encoding in encodings:
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            return True
    return False


def capture_picture(picture_width, picture_height, picture_folder, picture_name):
    im2 = ImageGrab.grab(bbox =(screen_width/2 - picture_width/2 - 714, screen_height/2 - picture_height/2 - 70, screen_width/2 + picture_width/2 - 714,  screen_height/2 + picture_height/2 - 70)) 
    im2.save(picture_folder +"/"+ picture_name)


def define_swipe(accepted_folder, denied_folder, all_face_encodings=None):
    test_folder = "img/models_test"
    test_picture = "test_pic.jpg"
    test_picture_extension = ".jpg"
    capture_picture(470, 700, test_folder, test_picture)
    if hasMatch(test_folder+"/"+test_picture, all_face_encodings):
        os.rename(test_folder+'/'+test_picture, accepted_folder+'/'+str(time.time())+test_picture_extension)
        return True
    else:
        os.rename(test_folder+'/'+test_picture, denied_folder+'/'+str(time.time())+test_picture_extension)
        return False