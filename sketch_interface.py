import pygame
import sys


class Interface():
    def __init__(self):
        pass

    def draw(self):
        # Initialize PyGame
        pygame.init()

        # Constants
        WIDTH, HEIGHT = 1111, 1111
        BG_COLOR = (255, 255, 255)  # White background
        DRAW_COLOR = (0, 0, 0)      # Black drawing color
        BRUSH_SIZE = 5              # Brush thickness

        # Create display
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Sketch Pad")

        # Initialize canvas
        canvas = pygame.Surface((WIDTH, HEIGHT))
        canvas.fill(BG_COLOR)

        # Main loop
        drawing = False  # Tracks if the user is drawing
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.image.save(canvas, "user_sketch.png")  # Save the drawing on exit
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    drawing = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    drawing = False
                elif event.type == pygame.MOUSEMOTION and drawing:
                    pygame.draw.circle(canvas, DRAW_COLOR, event.pos, BRUSH_SIZE)

            # Display canvas
            screen.blit(canvas, (0, 0))
            pygame.display.flip()



        # To do:
        # 1. format the png in form of CNN input
        # 2. CLEAR feature

interface = Interface()
interface.draw()