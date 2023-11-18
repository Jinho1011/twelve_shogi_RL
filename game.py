import pygame as pg
import numpy as np

from env import TwelveShogi

# Constants for rendering
grid_size = 100
board_color = (255, 204, 153)
line_color = (0, 0, 0)
white = (255, 255, 255)
black = (0, 0, 0)
radius = grid_size // 2 - 2
font_size = 36


class ShogiRenderer:
    def __init__(self, env: TwelveShogi, row_size, col_size):
        self.env = env
        self.row_size = row_size
        self.col_size = col_size
        pg.init()
        self.screen = pg.display.set_mode(
            (col_size * grid_size, row_size * grid_size))
        pg.display.set_caption('Shogi Game')
        self.font = pg.font.SysFont('AppleSDGothicNeo', font_size)

    def draw_piece(self, label, bg_color, text_color, center):
        pg.draw.circle(self.screen, bg_color, center, radius)
        text_surface = self.font.render(label, True, text_color)
        text_rect = text_surface.get_rect(center=center)
        self.screen.blit(text_surface, text_rect)

    def render(self):
        self.screen.fill(board_color)

        # 상단 플레이어 1의 영역 색상 정의
        player1_area_color = (220, 220, 220)  # 연한 회색
        # 하단 플레이어 2의 영역 색상 정의
        player2_area_color = (255, 200, 200)  # 연한 붉은색

        # 플레이어 1의 영역 채우기
        for i in range(self.row_size):
            rect = pg.Rect(0, i*grid_size, grid_size, grid_size)
            pg.draw.rect(self.screen, player1_area_color, rect)

        # 플레이어 2의 영역 채우기
        for i in range(self.row_size):
            rect = pg.Rect((self.col_size-1) * grid_size, i *
                           grid_size, grid_size, grid_size)
            pg.draw.rect(self.screen, player2_area_color, rect)

        # Draw the grid
        for i in range(self.row_size):
            for j in range(self.col_size):
                rect = pg.Rect(j * grid_size, i * grid_size,
                               grid_size, grid_size)
                pg.draw.rect(self.screen, line_color, rect, 1)

        for i in range(self.row_size):
            for j in range(self.col_size):
                center = (j * grid_size + grid_size // 2,
                          i * grid_size + grid_size // 2)

                if self.env.state[i][j] == -5:
                    self.draw_piece('후', black, white, center)
                elif self.env.state[i][j] == -4:
                    self.draw_piece('자', black, white, center)
                elif self.env.state[i][j] == -3:
                    self.draw_piece('왕', black, white, center)
                elif self.env.state[i][j] == -2:
                    self.draw_piece('상', black, white, center)
                elif self.env.state[i][j] == -1:
                    self.draw_piece('장', black, white, center)
                elif self.env.state[i][j] == 1:
                    self.draw_piece('장', white, black, center)
                elif self.env.state[i][j] == 2:
                    self.draw_piece('상', white, black, center)
                elif self.env.state[i][j] == 3:
                    self.draw_piece('왕', white, black, center)
                elif self.env.state[i][j] == 4:
                    self.draw_piece('자', white, black, center)
                elif self.env.state[i][j] == 5:
                    self.draw_piece('후', white, black, center)

        pg.display.flip()

    def close(self):
        pg.quit()
