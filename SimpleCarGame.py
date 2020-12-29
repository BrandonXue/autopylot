from config import *
from rgb_colors import *
import sys, pygame
import tensorflow as tf
from tensorflow import keras
import numpy as np
from math import sqrt, radians, sin, cos, exp, fabs
from random import randint

from Menus import *


##TEST
class Car_Details_Menu(Menu):
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

#############################################################


    
def rotate_around_player2(player_pos, obj_pos, rotation_matrix: np.array):
    temp_vec = obj_pos - player_pos
    return rotation_matrix.dot(temp_vec)

def rotate_around_player(player_pos, obj_pos, rotation)->np.array:
    temp_vec = obj_pos - player_pos
    rotated_position = rotation_matrix.dot(temp_vec) + player_pos
    return rotated_position
    
def lin_seg_intersection(p1, p2, p3, p4): #p is a point either tuple or list in format x, y. p1 and p2 together form a line and p3 and p4 together form a line
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p4[0] - p3[0], p4[1] - p3[1])
    odd1 = p3[0] - p1[0]
    odd2 = p1[1] - p3[1]

    numt = v2[1]*odd1 + v2[0]*odd2
    nums = odd2*v1[0] + odd1*v1[1]
    
    den = v1[0]*v2[1] - v1[1]*v2[0]
    
    try:
        t = numt / den
        s = nums / den
        if t >= 0 and t <= 1 and s >= 0 and s <= 1:
            return True
    except ArithmeticError:
        return False
    return False
    
def check_collision(ob_coords1, ob_coords2):
    for i in range(0, 4):
        for j in range(0, 4):
            if lin_seg_intersection(ob_coords1[i-1], ob_coords1[i], ob_coords2[j-1], ob_coords2[j]): return True
    return False
    
def clamp(value, min, max):
    if value > max:
        value = max
    elif value < min:
        value = min
    return value

class rectangle:
    def __init__(self, x=0, y=0, width=10, height=10):
        self.coordinates_ = np.array([[x, y],[x+width, y], [x+width, y+height], [x, y+height]], dtype=float)
        self.center_ = np.array([x+width/2,y+height/2], dtype=float)
        self.front_center_ = np.array([x+width/2, y], dtype=float)
        self.max_dim = width if width > height else height

    def get_coords2(self, point_of_rotation, rotation_matrix, view_adjust):
        new_coords = []
        for point in self.coordinates_:
            new_coords.append(rotation_matrix.dot(point - point_of_rotation) + view_adjust)
        return new_coords
        
    #for the main player
    def get_rotated(self, point_of_rotation, rotation_matrix):
        return np.array([rotate_around_player(point_of_rotation, [x, y], rotation_matrix) for [x,y] in self.coordinates_])
        
    def move(self, move_values:np.array(2) = np.array([0.,0.])):
        self.coordinates_ += move_values
        self.center_ += move_values
        self.front_center_ += move_values
        
    def get_center(self):
        return self.center_
        
    def get_front_center(self):
        return self.front_center_
       
    #doesn't work properly
    def check_collision(self, coordinates)->bool:
        for [x,y] in coordinates:
            if x >= self.coordinates_[0][0] and x <= self.coordinates_[2][0] and y >= self.coordinates_[0][1]  and y <= self.coordinates_[2][1]:
                return True
        return False

    def get_max_dim(self):
        return self.max_dim

def main():
    pygame.init()
    keys = np.array([False]*KEY_ARRAY_SIZE)
    viewport = pygame.display.set_mode(VIEWPORT_SIZE)
    half_viewport = np.array(VIEWPORT_SIZE)/2
    viewport_diag = sqrt(half_viewport[0]**2 + half_viewport[1]**2)
    screen_rect = viewport.get_rect()
    
    font = pygame.freetype.Font(None, 32)
    
    temp_menu = Car_Details_Menu((300, 500), (5, 5), (150, 150, 150, 200))
    
    rotation = 0.0
    angular_velocity = 0.0 # Can think of this as steering wheel position
    velocity = 0.0
    pressed = False
    collision = False
    
    collision = False
    
    #object position    x, y
    abs_pos = np.array([0, 0])
    
    player_test = rectangle(0, 0, CAR_WIDTH, CAR_HEIGHT)
    #object_test = rectangle(-100, -100, 50, 50)
    objects = list()
    
    for i in range(0, 50):
        objects.append(
            rectangle(
                randint(-MAX_X_LOC_BOX, MAX_X_LOC_BOX), randint(-MAX_Y_LOC_BOX, MAX_Y_LOC_BOX), # Location coords
                randint(MIN_BOX_WIDTH, MAX_BOX_WIDTH), randint(MIN_BOX_HEIGHT, MAX_BOX_HEIGHT) # width height
            )
        )
    
    #viewport position needs to center on object
    view_pos = np.array([0., 0.])
    player_rotation_matrix = np.array([[1.0, 0.0], [0.0, 1.0]]) # I2

    max_collision_check_dist = (
        sqrt((CAR_WIDTH/2)**2 + (CAR_HEIGHT/2)**2) # player/car box hypoetenuse
        + sqrt((MAX_BOX_WIDTH/2)**2 + (MAX_BOX_HEIGHT/2)**2) # terrain object max hypotenuse
    )
    while True:
        #update events
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
            
            #MENU FUNCTIONS
            if event.type == pygame.MOUSEMOTION:
                temp_menu.set_mouse_pos(pygame.mouse.get_pos())
            if event.type == pygame.MOUSEBUTTONDOWN:
                temp_menu.set_state(PRESS)
                
            if event.type == pygame.MOUSEBUTTONUP:
                temp_menu.set_state(RELEASE)
                temp_menu.set_state(CLICK)
            #END MENU FUNCTIONS
            
            #VEHICLE FUNCTIONS
            elif event.type == pygame.KEYDOWN and event.key < KEY_ARRAY_SIZE:
                keys[event.key] = True
            elif event.type == pygame.KEYUP and event.key < KEY_ARRAY_SIZE:
                keys[event.key] = False
            
        #game logic
        move = np.array([0., 0.])
        rads = radians(rotation)
        
        if keys[pygame.K_w]:
            velocity -= temp_menu.get_car_acceleration()#CAR_ACCELERATION
            pressed=True
            
        if keys[pygame.K_s]:
            velocity += temp_menu.get_car_acceleration()#CAR_ACCELERATION 
            pressed=True
            
        # Turning left only
        if keys[pygame.K_a] and not keys[pygame.K_d]:
            angular_velocity = clamp(
                angular_velocity+CAR_ANGULAR_ACCELERATION,
                -CAR_MAX_ANGULAR_VELOCITY, CAR_MAX_ANGULAR_VELOCITY
            )

        # Turning right only 
        elif keys[pygame.K_d] and not keys[pygame.K_a]:
            angular_velocity = clamp(
                angular_velocity-CAR_ANGULAR_ACCELERATION,
                -CAR_MAX_ANGULAR_VELOCITY, CAR_MAX_ANGULAR_VELOCITY
            )

        # Else either both pressed or neither pressed. Move wheel towards center.
        else:
            if angular_velocity < CAR_ANGULAR_ZERO_THRESHOLD and angular_velocity > -CAR_ANGULAR_ZERO_THRESHOLD:
                angular_velocity = 0.0
            if angular_velocity != 0.0:
                angular_velocity *= CAR_ANGULAR_VELOCITY_DECAY
            
        # Only turn if car is in motion
        if velocity != 0:
            # Make turn speed dependent on velocity
            rotation -= angular_velocity * velocity

        velocity = clamp(velocity, -CAR_MAX_VELOCITY, CAR_MAX_VELOCITY)
        move[0] += sin(rads) * velocity
        move[1] += cos(rads) * velocity
        
        if not pressed:
            if velocity < CAR_VELOCITY_ZERO_THRESHOLD and velocity > -CAR_VELOCITY_ZERO_THRESHOLD:
                velocity = 0.0
            if velocity != 0:
                velocity *= CAR_VELOCITY_DECAY
                
        pressed = False
        
        player_test.move(move)
        view_pos += move
        
        #screen reset
        viewport.fill(RGB_BLACK)
        
        radian_val = radians(rotation)
        rotation_matrix = np.array([[cos(radian_val), -sin(radian_val)], [sin(radian_val), cos(radian_val)]])
        
        collision_points = player_test.get_rotated(player_test.get_center(), rotation_matrix)
        
        #render
        rotation_matrix = np.array([[cos(rads), -sin(rads)], [sin(rads), cos(rads)]])

        player_view_adjust = half_viewport + player_test.get_front_center() - view_pos
        view_ajdust = half_viewport + player_test.get_center() - view_pos
        
        player_coords = player_test.get_coords2(player_test.get_front_center(), player_rotation_matrix, player_view_adjust)

        pygame.draw.polygon(viewport, RGB_BLUE, player_coords)

        player_pos = player_test.get_center()
        player_max_dim = player_test.get_max_dim()
        for obj in objects:
            obj_center = obj.get_center()
            max_viewable_dist = viewport_diag + obj.get_max_dim()
            if (fabs(player_pos[0] - obj_center[0]) < max_viewable_dist
                and fabs(player_pos[1] - obj_center[1]) < max_viewable_dist):

                object_coords = obj.get_coords2(player_pos, rotation_matrix, view_ajdust)
                
                if (not collision 
                    and fabs(player_pos[0] - obj_center[0]) < max_collision_check_dist
                    and fabs(player_pos[1] - obj_center[1]) < max_collision_check_dist):
                    if check_collision(player_coords, object_coords): collision = True
                
                pygame.draw.polygon(viewport, RGB_GREEN, object_coords)
            
        
        
        #render menuds
        temp_menu.draw(viewport)
        col_text = "Collisions: " + ("True" if collision else "False")
        font.render_to(viewport, (5, 100), col_text, RGBA_LIGHT_RED)
        collision = False
        
        #swap buffers
        pygame.display.flip()
        
if __name__ == '__main__':
    main()