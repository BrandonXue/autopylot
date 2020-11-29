from config import *
from rgb_colors import *
from car_details_menu import *
from rectangle import *
import numpy as np
import pygame

from math import sqrt, radians, sin, cos, fabs
from random import randint
from skimage import color, transform, exposure
    
def clamp(value, min, max):
    if value > max:
        value = max
    elif value < min:
        value = min
    return value

class CarGame:
    def __init__(self, pipe_conn, exit_keyword):
        # We may want to init() selective modules (to look into if we have time)
        pygame.init()

        # Set inter-process communication properties
        self.pipe_conn = pipe_conn
        self.exit_keyword = exit_keyword

        # Keyboard related
        self.key_list = [False] * KEY_ARRAY_SIZE # Lists are faster for simple random access

        # Display related
        self.display = pygame.display.set_mode(
            VIEWPORT_SIZE,
            pygame.DOUBLEBUF | pygame.HWSURFACE
        )
        self.display.set_alpha(None)
       

        self.cover_display = pygame.Surface((80, 80))

        self.font = pygame.freetype.Font(None, 32)
        self.car_details_menu = CarDetailsMenu((300, 500), (5, 5), (150, 150, 150, 200))
    
        # Initial car state
        self.rotation = 0.0
        self.angular_velocity = 0.0 # Can think of this as steering wheel position
        self.velocity = 0.0
        self.collision = False

        # View transform related 
        # self.view_pos = np.array([0., 0.]) # viewport position needs to center on object

        # Pre-calculated game constants 
        self.half_viewport = pygame.math.Vector2(VIEWPORT_SIZE)/2
        self.display_offset = self.half_viewport
        self.viewport_diag = sqrt(self.half_viewport[0]**2 + self.half_viewport[1]**2)

        self.collision_check_dist = (
            sqrt((CAR_WIDTH/2)**2 + (CAR_HEIGHT/2)**2) # player/car box hypoetenuse
            + sqrt((MAX_BOX_WIDTH/2)**2 + (MAX_BOX_HEIGHT/2)**2) # terrain object max hypotenuse
            + 20 # Arbitrary number to ensure we don't filter out anything that looks like it should be colliding
        )

        # Frame counter for throttling certain events
        self.frame_count = 0


    def create_player_rect(self):
        self.player_rect = PlayerRectangle(0, 0, CAR_WIDTH, CAR_HEIGHT)


    def create_random_rects(self):
        self.rects = []
        for i in range(0, 50):
            self.rects.append(
                Rectangle(
                    randint(-MAX_X_LOC_BOX, MAX_X_LOC_BOX), randint(-MAX_Y_LOC_BOX, MAX_Y_LOC_BOX), # Location coords
                    randint(MIN_BOX_WIDTH, MAX_BOX_WIDTH), randint(MIN_BOX_HEIGHT, MAX_BOX_HEIGHT) # width height
                )
            )

    def store_input_keys(self):
        for event in pygame.event.get():
            # Treat quit event as escape button pressed
            if event.type == pygame.QUIT:
                self.key_list[pygame.K_ESCAPE] = True
            
            #MENU FUNCTIONS
            if event.type == pygame.MOUSEMOTION:
                self.car_details_menu.set_mouse_pos(pygame.mouse.get_pos())
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.car_details_menu.set_state(PRESS)
            if event.type == pygame.MOUSEBUTTONUP:
                self.car_details_menu.set_state(RELEASE)
                self.car_details_menu.set_state(CLICK)
            
            #VEHICLE FUNCTIONS
            elif event.type == pygame.KEYDOWN and event.key < KEY_ARRAY_SIZE:
                self.key_list[event.key] = True
            elif event.type == pygame.KEYUP and event.key < KEY_ARRAY_SIZE:
                self.key_list[event.key] = False

    def update_positions(self):
        move_vec = np.array([0., 0.])
        self.rads = radians(self.rotation)

        pressed = False

        if self.key_list[pygame.K_w]:
            self.velocity -= self.car_details_menu.get_car_acceleration()#CAR_ACCELERATION
            pressed=True
            
        if self.key_list[pygame.K_s]:
            self.velocity += self.car_details_menu.get_car_acceleration()#CAR_ACCELERATION 
            pressed=True
            
        # Turning left only
        if self.key_list[pygame.K_a] and not self.key_list[pygame.K_d]:
            self.angular_velocity = clamp(
                self.angular_velocity + CAR_ANGULAR_ACCELERATION, # new value
                -CAR_MAX_ANGULAR_VELOCITY, CAR_MAX_ANGULAR_VELOCITY # clamp bounds
            )

        # Turning right only 
        elif self.key_list[pygame.K_d] and not self.key_list[pygame.K_a]:
            self.angular_velocity = clamp(
                self.angular_velocity - CAR_ANGULAR_ACCELERATION, # new value
                -CAR_MAX_ANGULAR_VELOCITY, CAR_MAX_ANGULAR_VELOCITY # clamp bounds
            )

        # Else either both pressed or neither pressed. Move wheel towards center.
        else:
            if (self.angular_velocity < CAR_ANGULAR_ZERO_THRESHOLD 
                and self.angular_velocity > -CAR_ANGULAR_ZERO_THRESHOLD):
                self.angular_velocity = 0.0
            if self.angular_velocity != 0.0:
                self.angular_velocity *= CAR_ANGULAR_VELOCITY_DECAY
            
        # Only turn if car is in motion
        if self.velocity != 0:
            # Make turn speed dependent on velocity
            self.rotation -= self.angular_velocity * self.velocity

        self.velocity = clamp(self.velocity, -CAR_MAX_VELOCITY, CAR_MAX_VELOCITY)
        move_vec[0] += sin(self.rads) * self.velocity
        move_vec[1] += cos(self.rads) * self.velocity
        
        if not pressed:
            if self.velocity < CAR_VELOCITY_ZERO_THRESHOLD and self.velocity > -CAR_VELOCITY_ZERO_THRESHOLD:
                self.velocity = 0.0
            if self.velocity != 0:
                self.velocity *= CAR_VELOCITY_DECAY
                
        pressed = False

        # Player position and view position
        self.player_rect.move(move_vec)
        # self.view_pos += move_vec

    def render_updates(self):
        # Fill background color
        self.display.fill(RGB_WHITE)
        
        # These locations are used as references
        player_center = self.player_rect.get_center()
        rear_axle = self.player_rect.get_rear_center()

        # Render the player
        player_coords = self.player_rect.pivot_and_offset(rear_axle, 0, self.display_offset)
        pygame.draw.polygon(self.display, RGB_BLUE, player_coords)

        collision = False
        
        for obj in self.rects:

            obj_center = obj.get_center()
            max_viewable_dist = self.viewport_diag + obj.get_max_dim()

            if (fabs(player_center[0] - obj_center[0]) < max_viewable_dist
                and fabs(player_center[1] - obj_center[1]) < max_viewable_dist):

                object_coords = obj.pivot_and_offset(rear_axle, self.rotation, self.display_offset)
                
                if (not collision 
                    and fabs(player_center[0] - obj_center[0]) < self.collision_check_dist
                    and fabs(player_center[1] - obj_center[1]) < self.collision_check_dist):
                    if check_collision(player_coords, object_coords): collision = True
                
                pygame.draw.polygon(self.display, RGB_GREEN, object_coords)
        
        # ============== After scene is drawn, but before overhead display ===============
        self.frame_count = (self.frame_count + 1) % 30
        if self.key_list[pygame.K_p] and self.frame_count == 0:
            self.cover_display.fill(RGB_WHITE)
            pixels = pygame.surfarray.pixels3d(self.display)

            self.pipe_conn.send(pixels)
            
            pixels = color.rgb2gray(pixels)
            pixels = transform.resize(pixels,(80,80))
            pixels = exposure.rescale_intensity(pixels, out_range=(0, 255))
            
            pygame.surfarray.blit_array(self.cover_display, pixels)
            
            self.display.blit(self.cover_display, (VIEWPORT_SIZE[0]-180,VIEWPORT_SIZE[1]-180))
        
        # ============= After input buffer is used for neural net draw HUD ===============
        # temp_menu.draw(viewport)
        col_text = "Collisions: " + ("True" if collision else "False")
        self.font.render_to(self.display, (5, 5), col_text, RGBA_LIGHT_RED)

        # ========================= Swap the buffer to display ===========================
        #swap buffers
        pygame.display.flip()

    def start(self):
        self.create_player_rect()
        self.create_random_rects()
        while True:
            self.store_input_keys()

            # If escape is pressed, break main event loop
            if self.key_list[pygame.K_ESCAPE]:
                self.pipe_conn.send(self.exit_keyword)
                break

            self.update_positions()
            self.render_updates()


# def start_game(pipe_connection, exit_keyword):
#     pygame.init()
#     keys = np.array([False]*KEY_ARRAY_SIZE)
#     viewport = pygame.display.set_mode(
#         VIEWPORT_SIZE, 
#         pygame.DOUBLEBUF | pygame.HWSURFACE,
#         depth=0 # let pygame choose depth
#     )
#     viewport.set_alpha(None)

#     half_viewport = np.array(VIEWPORT_SIZE)/2
#     viewport_diag = sqrt(half_viewport[0]**2 + half_viewport[1]**2)
#     screen_rect = viewport.get_rect()

#     frame_count = 0
    
#     cover_viewport = pygame.Surface((80,80))
#     cover_viewport.fill(RGB_WHITE)
    
#     font = pygame.freetype.Font(None, 32)
    
#     temp_menu = CarDetailsMenu((300, 500), (5, 5), (150, 150, 150, 200))
    
#     rotation = 0.0
#     angular_velocity = 0.0 # Can think of this as steering wheel position
#     velocity = 0.0
#     pressed = False
    
#     collision = False
    
#     #object position    x, y
#     abs_pos = np.array([0, 0])
    
#     player_test = Rectangle(0, 0, CAR_WIDTH, CAR_HEIGHT)
#     objects = list()
    
#     for i in range(0, 50):
#         objects.append(
#             Rectangle(
#                 randint(-MAX_X_LOC_BOX, MAX_X_LOC_BOX), randint(-MAX_Y_LOC_BOX, MAX_Y_LOC_BOX), # Location coords
#                 randint(MIN_BOX_WIDTH, MAX_BOX_WIDTH), randint(MIN_BOX_HEIGHT, MAX_BOX_HEIGHT) # width height
#             )
#         )
    
#     #viewport position needs to center on object
#     view_pos = np.array([0., 0.])
#     player_rotation_matrix = np.array([[1.0, 0.0], [0.0, 1.0]]) # I2

#     # Only used for broad phase collision detection
#     max_collision_check_dist = (
#         sqrt((CAR_WIDTH/2)**2 + (CAR_HEIGHT/2)**2) # player/car box hypoetenuse
#         + sqrt((MAX_BOX_WIDTH/2)**2 + (MAX_BOX_HEIGHT/2)**2) # terrain object max hypotenuse
#         + 50 # Arbitrary number to ensure we don't filter out anything that looks like it should be colliding
#     )
#     while True:
#         #update events
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT: sys.exit()
            
#             #MENU FUNCTIONS
#             if event.type == pygame.MOUSEMOTION:
#                 temp_menu.set_mouse_pos(pygame.mouse.get_pos())
#             if event.type == pygame.MOUSEBUTTONDOWN:
#                 temp_menu.set_state(PRESS)
                
#             if event.type == pygame.MOUSEBUTTONUP:
#                 temp_menu.set_state(RELEASE)
#                 temp_menu.set_state(CLICK)
#             #END MENU FUNCTIONS
            
#             #VEHICLE FUNCTIONS
#             elif event.type == pygame.KEYDOWN and event.key < KEY_ARRAY_SIZE:
#                 keys[event.key] = True
#             elif event.type == pygame.KEYUP and event.key < KEY_ARRAY_SIZE:
#                 keys[event.key] = False

#         # Game logic
#         move = np.array([0., 0.])
#         rads = radians(rotation)
        
#         if keys[pygame.K_w]:
#             velocity -= temp_menu.get_car_acceleration()#CAR_ACCELERATION
#             pressed=True
            
#         if keys[pygame.K_s]:
#             velocity += temp_menu.get_car_acceleration()#CAR_ACCELERATION 
#             pressed=True
            
#         if keys[pygame.K_ESCAPE]:
#             pipe_connection.send(exit_keyword)
#             break
       
            
            
#         # Turning left only
#         if keys[pygame.K_a] and not keys[pygame.K_d]:
#             angular_velocity = clamp(
#                 angular_velocity+CAR_ANGULAR_ACCELERATION,
#                 -CAR_MAX_ANGULAR_VELOCITY, CAR_MAX_ANGULAR_VELOCITY
#             )

#         # Turning right only 
#         elif keys[pygame.K_d] and not keys[pygame.K_a]:
#             angular_velocity = clamp(
#                 angular_velocity-CAR_ANGULAR_ACCELERATION,
#                 -CAR_MAX_ANGULAR_VELOCITY, CAR_MAX_ANGULAR_VELOCITY
#             )

#         # Else either both pressed or neither pressed. Move wheel towards center.
#         else:
#             if angular_velocity < CAR_ANGULAR_ZERO_THRESHOLD and angular_velocity > -CAR_ANGULAR_ZERO_THRESHOLD:
#                 angular_velocity = 0.0
#             if angular_velocity != 0.0:
#                 angular_velocity *= CAR_ANGULAR_VELOCITY_DECAY
            
#         # Only turn if car is in motion
#         if velocity != 0:
#             # Make turn speed dependent on velocity
#             rotation -= angular_velocity * velocity

#         velocity = clamp(velocity, -CAR_MAX_VELOCITY, CAR_MAX_VELOCITY)
#         move[0] += sin(rads) * velocity
#         move[1] += cos(rads) * velocity
        
#         if not pressed:
#             if velocity < CAR_VELOCITY_ZERO_THRESHOLD and velocity > -CAR_VELOCITY_ZERO_THRESHOLD:
#                 velocity = 0.0
#             if velocity != 0:
#                 velocity *= CAR_VELOCITY_DECAY
                
#         pressed = False
        
#         player_test.move(move)
#         view_pos += move
        
#         #screen reset
#         viewport.fill(RGB_WHITE)
        
#         #render
#         rotation_matrix = np.array([[cos(rads), -sin(rads)], [sin(rads), cos(rads)]])

#         player_view_adjust = half_viewport + player_test.get_front_center() - view_pos
#         view_ajdust = half_viewport + player_test.get_center() - view_pos
        
#         player_coords = player_test.get_coords(player_test.get_front_center(), player_rotation_matrix, player_view_adjust)

#         pygame.draw.polygon(viewport, RGB_BLUE, player_coords)

#         player_pos = player_test.get_center()

#         collision = False
        
#         for obj in objects:
#             obj_center = obj.get_center()
#             max_viewable_dist = viewport_diag + obj.get_max_dim()
#             if (fabs(player_pos[0] - obj_center[0]) < max_viewable_dist
#                 and fabs(player_pos[1] - obj_center[1]) < max_viewable_dist):

#                 object_coords = obj.get_coords(player_pos, rotation_matrix, view_ajdust)
                
#                 if (not collision 
#                     and fabs(player_pos[0] - obj_center[0]) < max_collision_check_dist
#                     and fabs(player_pos[1] - obj_center[1]) < max_collision_check_dist):
#                     if check_collision(player_coords, object_coords): collision = True
                
#                 pygame.draw.polygon(viewport, RGB_GREEN, object_coords)
        
#         # ============== After scene is drawn, but before overhead display ===============
#         frame_count = (frame_count + 1) % 30
#         if keys[pygame.K_p] and frame_count == 0:
#             cover_viewport.fill(RGB_WHITE)
#             pixels = pygame.surfarray.pixels3d(viewport)

#             pipe_connection.send(pixels)
            
#             pixels = color.rgb2gray(pixels)
#             pixels = transform.resize(pixels,(80,80))
#             pixels = exposure.rescale_intensity(pixels, out_range=(0, 255))
            
#             pygame.surfarray.blit_array(cover_viewport, pixels)
            
#             viewport.blit(cover_viewport, (VIEWPORT_SIZE[0]-180,VIEWPORT_SIZE[1]-180))
        
#         # ============= After input buffer is used for neural net draw HUD ===============
#         # temp_menu.draw(viewport)
#         col_text = "Collisions: " + ("True" if collision else "False")
#         font.render_to(viewport, (5, 5), col_text, RGBA_LIGHT_RED)

#         # ========================= Swap the buffer to display ===========================
#         #swap buffers
#         pygame.display.flip()