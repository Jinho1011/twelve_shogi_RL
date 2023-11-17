import pygame as pg
import numpy as np

# Constants for rendering
grid_size = 100
board_color = (255, 204, 153)
line_color = (0, 0, 0)
white_stone = (255, 255, 255)
black_stone = (0, 0, 0)
radius = grid_size // 2 - 2
font_size = 36


class ShogiRenderer:
    def __init__(self, env, row_size, col_size):
        self.env = env
        self.row_size = row_size
        self.col_size = col_size
        pg.init()
        self.screen = pg.display.set_mode(
            (col_size * grid_size, row_size * grid_size))
        pg.display.set_caption('Connect6 Game')
        self.font = pg.font.SysFont('AppleSDGothicNeo', font_size)

    def render(self):
        self.screen.fill(board_color)

        # Draw the grid
        for i in range(self.row_size):
            for j in range(self.col_size):
                rect = pg.Rect(j * grid_size, i * grid_size,
                               grid_size, grid_size)
                pg.draw.rect(self.screen, line_color, rect, 1)

        # Draw the stones and text
        agent1 = self.env.get_state(0)
        agent2 = self.env.get_state(1)

        for i in range(self.row_size):
            for j in range(self.col_size):
                center = (j * grid_size + grid_size // 2,
                          i * grid_size + grid_size // 2)

                if agent1[i][j] == 1:
                    pg.draw.circle(self.screen, black_stone, center, radius)
                    text_surface = self.font.render('장', True, white_stone)
                    text_rect = text_surface.get_rect(center=center)
                    self.screen.blit(text_surface, text_rect)

                if agent2[i][j] == 1:
                    pg.draw.circle(self.screen, white_stone, center, radius)
                    text_surface = self.font.render('장', True, black_stone)
                    text_rect = text_surface.get_rect(center=center)
                    self.screen.blit(text_surface, text_rect)

        pg.display.flip()

    def close(self):
        pg.quit()
