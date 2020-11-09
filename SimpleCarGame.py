import sys, pygame
import numpy as np
from math import sqrt, radians, sin, cos
from random import randint

def get_relative_coordinates(abs_pos, view_pos, viewport_size)->np.array:
    return abs_pos - view_pos + np.array(viewport_size)/2
    
def adjust_viewport_position(map_size, viewport_size, view_pos)->np.array:
    #check x
    if view_pos[0] < -map_size[0]/2 + viewport_size[0]/2: view_pos[0] = -map_size[0]/2 + viewport_size[0]/2
    elif view_pos[0] > map_size[0]/2 + viewport_size[0]/2: view_pos[0] = map_size[0]/2 + viewport_size[0]/2
    #check y
    if view_pos[1] < -map_size[1]/2 + viewport_size[1]/2: view_pos[1] = -map_size[1]/2 + viewport_size[1]/2
    elif view_pos[1] > map_size[1]/2 + viewport_size[1]/2: view_pos[1] = map_size[1]/2 + viewport_size[1]/2
    
    return view_pos
    
def rotate_around_player(player_pos, obj_pos, rotation)->np.array:
    temp_vec = obj_pos - player_pos
    radian_val = radians(rotation)
    rotation_matrix = np.array([[cos(radian_val), -sin(radian_val)], [sin(radian_val), cos(radian_val)]])
    rotated_position = rotation_matrix.dot(temp_vec) + player_pos
    return rotated_position
    
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
        
    def get_coords(self, point_of_rotation, view_position, viewport_size, rotation):
        new_coords = np.array([rotate_around_player(point_of_rotation, [x, y], rotation) for [x,y] in self.coordinates_])
        new_coords = np.array([get_relative_coordinates([x, y], view_position, viewport_size) for [x,y] in new_coords])
        return new_coords.tolist()
        
    def move(self, move_values:np.array(2) = np.array([0.,0.])):
        self.coordinates_ += move_values
        self.center_ += move_values
        self.front_center_ += move_values
        
    def get_center(self):
        return self.center_
        
    def get_front_center(self):
        return self.front_center_

def main():
    pygame.init()
    keys = np.array([False]*1024)
    viewport_size = width, height = 1080, 720
    viewport = pygame.display.set_mode(viewport_size)
    screen_rect = viewport.get_rect()
    
    map_size = width, height = 2000, 2000
    
    #some colors
    black = 0, 0, 0
    blue = 0, 0, 255
    green = 0, 255, 0
    red = 255, 0, 0
    
    rotation = 0.0
    velocity = 0.0
    acceleration = 0.005
    pressed = False
    
    #object position    x, y
    abs_pos = np.array([0, 0])
    
    player_test = rectangle(0, 0, 20, 40)
    #object_test = rectangle(-100, -100, 50, 50)
    objects = list()
    
    for i in range(0, 10):
        objects.append(rectangle(randint(-750,750), randint(-750, 750), randint(25, 75), randint(25, 75)))
    
    #viewport position needs to center on object
    view_pos = np.array([0., 0.])

    while True:
        #update events
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
            
            if event.type == pygame.KEYDOWN:
                keys[event.key] = True
            
            if event.type == pygame.KEYUP:
                keys[event.key] = False
            
        #game logic
        move = np.array([0., 0.])
        rads = radians(rotation)
        
        if keys[pygame.K_w]:
            velocity -= acceleration
            pressed=True
            
        if keys[pygame.K_s]:
            velocity += acceleration 
            pressed=True
            
        if keys[pygame.K_a]:
            rotation+=0.1
            
        if keys[pygame.K_d]:
            rotation-=0.1
            
        velocity = clamp(velocity, -2., 2.)
        move[0] += sin(rads) * velocity
        move[1] += cos(rads) * velocity
        
        if not pressed:
            if velocity > 0.0: velocity -= acceleration
            elif velocity < 0.0: velocity += acceleration
            if velocity < 0.002 and velocity > -0.002: velocity = 0.0
                
        pressed = False
        
        player_test.move(move)
        view_pos += move
        
        #screen reset
        viewport.fill(black)
        
        #render
        pygame.draw.polygon(viewport, blue, player_test.get_coords(player_test.get_front_center(), view_pos, viewport_size, 0))
        for object in objects:
            pygame.draw.polygon(viewport, green, object.get_coords(player_test.get_center(), view_pos, viewport_size, rotation))
        
        #swap buffers
        pygame.display.flip()
        
if __name__ == '__main__':
    main()