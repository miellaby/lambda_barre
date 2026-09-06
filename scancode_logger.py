"""Log scancodes for key presses. Press keys to see their scancode.
ESC or Ctrl-C to quit."""
import pygame

pygame.init()
screen = pygame.display.set_mode((480, 200))
pygame.display.set_caption("Scancode logger — ESC to quit")
font = pygame.font.SysFont("monospace", 18)
clock = pygame.time.Clock()

log = []
running = True
while running:
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False
        elif ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                running = False
            else:
                log.append(f"key={ev.key:3d} scancode={ev.scancode:3d} name={pygame.key.name(ev.key)}")
                log = log[-8:]

    screen.fill((24, 26, 34))
    for i, line in enumerate(log):
        s = font.render(line, True, (200, 206, 220))
        screen.blit(s, (12, 12 + i * 22))
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
