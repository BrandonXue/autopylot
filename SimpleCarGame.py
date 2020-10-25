import sys, pygame
pygame.init()

size = width, height = 320, 240
speed = [2,2]
black = 0, 0, 0
blue = 0, 0, 255

screen = pygame.display.set_mode(size)

testRect = pygame.Rect(0, 0, 10, 10)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()
        
        #game logic
        
        #screen reset
        screen.fill(black)
        
        #render
        pygame.draw.rect(screen, blue, testRect)
        
        #swap buffers
        pygame.display.flip()