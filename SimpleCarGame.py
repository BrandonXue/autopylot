import sys, pygame
import numpy as np

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
    
    #object position    x, y
    abs_pos = np.array([0, 0])
    
    #viewport position needs to center on object
    view_pos = np.array([0, 0])
    viewport_sizenp = np.array([1080, 720], dtype=int)

    while True:
        #update events
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
            
            if event.type == pygame.KEYDOWN:
                keys[event.key] = True
            
            if event.type == pygame.KEYUP:
                keys[event.key] = False
            
        #game logic
        if keys[pygame.K_w]:    
            if abs_pos[1] == view_pos[1]: view_pos[1] -= 1
            abs_pos[1] -= 1
            
        if keys[pygame.K_s]:
            if abs_pos[1] == view_pos[1]: view_pos[1] += 1
            abs_pos[1] += 1
            
        if keys[pygame.K_a]:
            if abs_pos[0] == view_pos[0]: view_pos[0] -= 1
            abs_pos[0] -= 1
            
        if keys[pygame.K_d]:
            if abs_pos[0] == view_pos[0]: view_pos[0] += 1
            abs_pos[0] += 1
           
        view_pos = adjust_viewport_position(map_size, viewport_size, view_pos)
        rel_pos = get_relative_coordinates(abs_pos, view_pos, viewport_size)
        
        #prepare position of main object
        test_rect = pygame.Rect(rel_pos[0],rel_pos[1], 10, 10)
        
        #test_rect = test_rect.clamp(screen_rect)
        
        #screen reset
        viewport.fill(black)
        
        #render
        pygame.draw.rect(viewport, blue, test_rect)
        
        #swap buffers
        pygame.display.flip()
        
if __name__ == '__main__':
    main()