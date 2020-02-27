from api import * 
import ctypes
import sys, os
from PIL import ImageGrab

user32 = ctypes.windll.user32
user32.SetProcessDPIAware()

screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)

def capture_picture(picture_width, picture_height, picture_name):
    im2 = ImageGrab.grab(bbox =(screen_width/2 - picture_width/2, screen_height/2 - picture_height/2, screen_width/2 + picture_width/2,  screen_height/2 + picture_height/2)) 
    im2.save("img/training_models/" + picture_name + ".jpg")

def train_ai(pictures_folder, group_id):
    pictures_added = 0
    picturesInFolder  = os.listdir(pictures_folder)
    response = create_person_group(group_id)
    if response.status_code == 200 or response.json()["error"]["code"] == "PersonGroupExists":
        for picture in picturesInFolder:
            picture_name_extension = picture.split('.')
            picture_path = pictures_folder +'/'+ picture_name_extension[0]+'.'+picture_name_extension[1]
            response = get_person(group_id, picture_name_extension[0])
            if response.status_code == 404:
                pictures_added += 1
                response = create_person(group_id, picture_name_extension[0])
                if response.status_code == 200:
                    person_id = response.json()["personId"]
                    new_picture_path = pictures_folder +'/'+ person_id+'.'+picture_name_extension[1]
                    os.rename(picture_path,new_picture_path)
                    response = add_person_face(group_id, person_id, new_picture_path)
                    if response.status_code == 200:
                        print("Picture added: " + new_picture_path)
                    else:
                        os.rename(new_picture_path, picture_path)
                        delete_person(group_id, person_id)
                        print("Could not add Face :" + picture_name_extension[0])
                        print("Http response: " + response.text)
                else:
                    print("Could not create person :" + picture_name_extension[0])
                    print("Http response: " + response.text)
            else:
                print("Picture aldready exist: " + picture_path)
        if pictures_added > 0:
            response = train_person_group(group_id)
            print("Train api response Code :" + str(response.status_code))
        else:
            print("Model ready !")
    else:
        print("Could not create groupId : " + group_id)
        print("Http response: " + response.text)

train_ai("img/model_pics", "main_model")
