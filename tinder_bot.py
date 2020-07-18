from selenium import webdriver
import time
import datetime
import random

from secrets import username, password
from functions import *

class TinderBot():
    def __init__(self):
        self.driver = webdriver.Chrome()

    def login(self):
        self.driver.get('https://tinder.com')

        time.sleep(4)

        fb_btn = self.driver.find_element_by_xpath('//*[@id="modal-manager"]/div/div/div/div/div[3]/span/div[2]/button')
        fb_btn.click()


        # switch to login popup
        base_window = self.driver.window_handles[0]
        self.driver.switch_to_window(self.driver.window_handles[1])

        email_in = self.driver.find_element_by_xpath('//*[@id="email"]')
        email_in.send_keys(username)

        pw_in = self.driver.find_element_by_xpath('//*[@id="pass"]')
        pw_in.send_keys(password)

        login_btn = self.driver.find_element_by_xpath('//*[@id="u_0_0"]')
        login_btn.click()

        self.driver.switch_to_window(base_window)
        time.sleep(7)

        popup_1 = self.driver.find_element_by_xpath('//*[@id="modal-manager"]/div/div/div/div/div[3]/button[1]')
        popup_1.click()

        popup_2 = self.driver.find_element_by_xpath('//*[@id="modal-manager"]/div/div/div/div/div[3]/button[1]')
        popup_2.click()

    def like(self):
        like_btn = self.driver.find_element_by_xpath('//*[@id="content"]/div/div[1]/div/main/div[1]/div/div/div[1]/div/div[2]/div[4]/button')
        like_btn.click()

    def dislike(self):
        dislike_btn = self.driver.find_element_by_xpath('//*[@id="content"]/div/div[1]/div/main/div[1]/div/div/div[1]/div/div[2]/div[2]/button')
        dislike_btn.click()

    def auto_swipe(self):
        match = 0
        un_match = 0
        with open(DATA_FILE, 'rb') as f:
            all_face_encodings = pickle.load(f)
        while True:
            wait_time = datetime.timedelta(0,random.randint(900, 1800),0)
            self.swipe_number = random.randint(250, 350)

            for x in range(self.swipe_number):
                confidence = define_swipe("img/accepted", "img/denied", all_face_encodings)
                if confidence:
                    time.sleep(random.uniform(0.5,1))
                    try:
                        self.like()
                        match += 1
                    except Exception:
                        try:
                            self.close_popup()
                        except Exception:
                            try:
                                self.close_match()
                            except Exception:
                                print("No more likes, exiting...")
                                exit()
                else:
                    try:
                        self.dislike()
                        un_match += 1
                    except Exception:
                        try:
                            self.close_popup()
                        except Exception:
                            try:
                                popup_buy = self.driver.find_element_by_xpath('//*[@id="modal-manager"]/div/div/div[3]/button[2]')
                                popup_buy.click()
                            except Exception:
                                print("oops !")
                    
                time.sleep(2)
            
            print("Like number: ", match)
            print("Dislike number: ", un_match)


    def close_popup(self):
        popup_3 = self.driver.find_element_by_xpath('//*[@id="modal-manager"]/div/div/div[2]/button[2]')
        popup_3.click()

    def close_match(self):
        match_popup = self.driver.find_element_by_xpath('//*[@id="modal-manager-canvas"]/div/div/div[1]/div/div[3]/a')
        match_popup.click()


bot = TinderBot()
bot.login()
time.sleep(10)
bot.auto_swipe()