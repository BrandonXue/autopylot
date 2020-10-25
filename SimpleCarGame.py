import sys, pygame
pygame.init()

size = width, height = 1080, 720
speed = [2,2]
black = 0, 0, 0
blue = 0, 0, 255

keys = [False]*1024

vel = 1, 1

screen = pygame.display.set_mode(size)

testRect = pygame.Rect(0, 0, 10, 10)

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
        testRect = testRect.move(0, -1)
    if keys[pygame.K_s]:
        testRect = testRect.move(0, 1)
    if keys[pygame.K_a]:
        testRect = testRect.move(-1, 0)
    if keys[pygame.K_d]:
        testRect = testRect.move(1, 0)
        
    
    #screen reset
    screen.fill(black)
    
    #render
    pygame.draw.rect(screen, blue, testRect)
    
    #swap buffers
    pygame.display.flip()