from config import *
from menus import *

class CarDetailsMenu(Menu):
    def __init__(self, dimensions, pos, color):
        super().__init__(dimensions, pos, color)
        
        self.car_acceleration = CAR_ACCELERATION
        
        plus_button = Button((30, 30), self, (dimensions[0] - 70, 10))
        plus_button.set_text("+")
        plus_button.set_font_size(32)
        plus_button.set_backcolor(RGBA_LIGHT_RED)
        plus_button.on_press(lambda:plus_button.set_backcolor(RGBA_DARK_RED))
        plus_button.on_release(lambda:plus_button.set_backcolor(RGBA_LIGHT_RED))
        plus_button.on_click(lambda:self.add_to_car_acceleration(0.0005))
        
        self.add_text_field("AccSet", TextField((5, 20), "Car Acceleration : {:.4f}".format(self.car_acceleration), 14))
        
        minus_button = Button((30, 30), self, (dimensions[0] - 35, 10))
        minus_button.set_text("-")
        minus_button.set_font_size(32)
        minus_button.set_backcolor(RGBA_LIGHT_BLUE)
        minus_button.on_press(lambda:minus_button.set_backcolor(RGBA_DARK_BLUE))
        minus_button.on_release(lambda:minus_button.set_backcolor(RGBA_LIGHT_BLUE))
        minus_button.on_click(lambda:self.add_to_car_acceleration(-0.0005))
        
    def add_to_car_acceleration(self, delta):
        self.car_acceleration = self.car_acceleration + delta
        self.set_text_field("AccSet", "Car Acceleration : {:.4f}".format(self.car_acceleration))
        
    def get_car_acceleration(self):
        return self.car_acceleration