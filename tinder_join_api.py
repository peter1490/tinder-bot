from api import * 
import ctypes
import sys, os
from PIL import ImageGrab

user32 = ctypes.windll.user32
user32.SetProcessDPIAware()

screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)

def capture_picture(picture_width, picture_height, picture_folder, picture_name):
    im2 = ImageGrab.grab(bbox =(screen_width/2 - picture_width/2+230, screen_height/2 - picture_height/2 - 50, screen_width/2 + picture_width/2+230,  screen_height/2 + picture_height/2 - 50)) 
    im2.save(picture_folder +"/"+ picture_name +".jpg")

def train_ai(pictures_folder, group_id):
    pictures_added = 0
    pictures_in_folder  = os.listdir(pictures_folder)
    response = create_person_group(group_id)
    if response.status_code == 200 or response.json()["error"]["code"] == "PersonGroupExists":
        for picture in pictures_in_folder:
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

def add_new_pictures_to_model(pictures_to_add, pictures_folder, group_id):
    response = list_persons(group_id)
    if response.status_code == 200:
        persons = response.json()
        for person in persons:
            ref_num = int(person["name"][4:])
            for person_2 in persons:
                ref_person_2 = int(person_2["name"][4:])
                if ref_num < ref_person_2:
                    ref_num = ref_person_2
        
        pictures_in_folder_to_add  = os.listdir(pictures_to_add)
        for picture_to_add in pictures_in_folder_to_add:
            ref_num += 1
            picture_to_add_name_extension = picture_to_add.split(".")
            os.rename(pictures_to_add+"/"+picture_to_add_name_extension[0]+"."+picture_to_add_name_extension[1], pictures_folder+"/ref_"+str(ref_num)+"."+picture_to_add_name_extension[1])
            print(pictures_folder+"/ref_"+str(ref_num)+"."+picture_to_add_name_extension[1]+" Added !")
    else:
        print("Could not load groupId: " + group_id)
        print("Http response Code: " + response.text)

    

#train_ai("img/model_pics", "main_model")

#add_new_pictures_to_model("img/pre_model", "img/model_pics", "main_model")

#capture_picture(450, 600, "img/models_test", "test")
