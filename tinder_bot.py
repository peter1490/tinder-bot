from selenium import webdriver
import time
import datetime
import random

from secrets import username, password

class TinderBot():
    def __init__(self):
        self.driver = webdriver.Chrome()
    swipe_number = 0
    def login(self):
        self.driver.get('https://tinder.com')

        time.sleep(3)

        fb_btn = self.driver.find_element_by_xpath('//*[@id="modal-manager"]/div/div/div/div/div[3]/div[2]/button')
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
        time.sleep(1)

        popup_1 = self.driver.find_element_by_xpath('//*[@id="modal-manager"]/div/div/div/div/div[3]/button[1]')
        popup_1.click()

        popup_2 = self.driver.find_element_by_xpath('//*[@id="modal-manager"]/div/div/div/div/div[3]/button[1]')
        popup_2.click()

    def like(self):
        like_btn = self.driver.find_element_by_xpath('//*[@id="content"]/div/div[1]/div/main/div[1]/div/div/div[1]/div/div[2]/button[3]')
        like_btn.click()

    def dislike(self):
        dislike_btn = self.driver.find_element_by_xpath('//*[@id="content"]/div/div[1]/div/main/div[1]/div/div/div[1]/div/div[2]/button[1]')
        dislike_btn.click()

    def auto_swipe(self):
        match = 0
        un_match = 0
        while True:
            wait_time = datetime.timedelta(0,random.randint(2700, 3600),0)
            self.swipe_number = random.randint(250, 350)

            for x in range(self.swipe_number):
                if random.randint(0,10) > 1:
                    time.sleep(random.uniform(0.5,1))
                    try:
                        self.like()
                        match += 1
                        print("like number: ",match)
                    except Exception:
                        try:
                            self.close_popup()
                        except Exception:
                            self.close_match()
                else:
                    try:
                        self.dislike()
                        un_match += 1
                        print("dislike number: ",un_match)
                    except Exception:
                        print('oops')
            countdown_timer(int(wait_time.total_seconds()))


    def close_popup(self):
        popup_3 = self.driver.find_element_by_xpath('//*[@id="modal-manager"]/div/div/div[2]/button[2]')
        popup_3.click()

    def close_match(self):
        match_popup = self.driver.find_element_by_xpath('//*[@id="modal-manager-canvas"]/div/div/div[1]/div/div[3]/a')
        match_popup.click()

def countdown_timer(x, now=datetime.datetime.now):
    target = now()
    one_second_later = datetime.timedelta(seconds=1)
    for remaining in range(x, 0, -1):
        target += one_second_later
        print(datetime.timedelta(seconds=remaining), 'remaining', end='\r')
        time.sleep((target - now()).total_seconds())
    print('\nTIMER ended')

bot = TinderBot()
bot.login()
#bot.auto_swipe()