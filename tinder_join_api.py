import api
from PIL import ImageGrab
import ctypes

user32 = ctypes.windll.user32
user32.SetProcessDPIAware()

screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)

def capture_picture(picture_width, picture_height, picture_name):
    im2 = ImageGrab.grab(bbox =(screen_width/2 - picture_width/2, screen_height/2 - picture_height/2, screen_width/2 + picture_width/2,  screen_height/2 + picture_height/2)) 
    im2.save("img/training_models/" + picture_name + ".jpg")

