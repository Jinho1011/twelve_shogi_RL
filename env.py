# -*- coding:utf-8 -*-

import pygame as pg
import numpy as np

row_size = 3
col_size = 4
grid_size = 100


board_color = (255, 204, 153)  # A light brown color
line_color = (0, 0, 0)  # Black lines for the grid
white_stone = (255, 255, 255)
black_stone = (0, 0, 0)


class TwelveShogi():
    def __init__(self) -> None:
        self.state = {0: np.zeros((row_size, col_size), dtype=np.int32), 1: np.zeros(
            (row_size, col_size), dtype=np.int32)}
        self.reward_dict = {"victory": 10.0, "defeat": -
                            10.0, "step": 0.0001, "overlap": -0.1}
        self.screen = None

    def reset(self):
        self.state = {0: np.zeros((row_size, col_size), dtype=np.int32), 1: np.zeros(
            (row_size, col_size), dtype=np.int32)}
        obs = self.state[0]
        return obs

    def finish_check(self) -> bool:
        for i in range(0, row_size):
            for j in range(0, col_size):
                for k in [1, -1]:
                    try:
                        if (k == self.state[0][i, j] == self.state[0][i + 1, j] == self.state[0][i + 2, j] == self.state[0][i + 3, j] == self.state[0][i + 4, j] == self.state[0][i + 5, j]):
                            return k
                    except:
                        pass

        for j in range(0, row_size):
            for i in range(0, col_size):
                for k in [1, -1]:
                    try:
                        if (k == self.state[0][i, j] == self.state[0][i, j + 1] == self.state[0][i, j + 2] == self.state[0][i, j + 3] == self.state[0][i, j + 4] == self.state[0][i, j + 5]):
                            return k
                    except:
                        pass

        for i in range(0, row_size):
            for j in range(0, col_size):
                for k in [1, -1]:
                    try:
                        if (k == self.state[0][i, j] == self.state[0][i + 1, j + 1] == self.state[0][i + 2, j + 2] == self.state[0][i + 3, j + 3] == self.state[0][i + 4, j + 4] == self.state[0][i + 5, j + 5]):
                            return k
                    except:
                        pass

        for i in range(0, row_size):
            for j in range(0, col_size):
                for k in [1, -1]:
                    try:
                        if (k == self.state[0][i, j] == self.state[0][i - 1, j + 1] == self.state[0][i - 2, j + 2] == self.stat[0][i - 3, j + 3] == self.state[0][i - 4, j + 4] == self.state[0][i - 5, j + 5]):
                            return k
                    except:
                        pass

        return 0

    def heuristic(self, turn: int, layer: int = 0):
        """
        returns heuristic information about states
        turn : turn of agnet (0 or 1)
        layer : positive layer only (1), negative layer only (-1), both (0)
        """
        hlayer1 = np.zeros((row_size, col_size), dtype=np.int32)
        hlayer2 = np.zeros((row_size, col_size), dtype=np.int32)

        for i in range(0, row_size):
            for j in range(0, col_size):
                # init
                hlayer1[i, j] = 0
                hlayer2[i, j] = 0

                # left to right
                count1 = 0
                count2 = 0
                for k in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
                    try:
                        if (1 == self.state[turn][i + k, j]):
                            count1 += 1
                    except:
                        pass
                    try:
                        if (-1 == self.state[turn][i + k, j]):
                            count2 += 1
                    except:
                        pass

                hlayer1[i, j] += count1
                hlayer2[i, j] += count2

                # up to down
                count1 = 0
                count2 = 0
                for k in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
                    try:
                        if (1 == self.state[turn][i, j + k]):
                            count1 += 1
                    except:
                        pass
                    try:
                        if (-1 == self.state[turn][i, j + k]):
                            count2 += 1
                    except:
                        pass

                hlayer1[i, j] += count1
                hlayer2[i, j] += count2

                # diag upper left to bottom right \
                count1 = 0
                count2 = 0
                for k in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
                    try:
                        if (1 == self.state[turn][i + k, j - k]):
                            count1 += 1
                    except:
                        pass
                    try:
                        if (-1 == self.state[turn][i + k, j - k]):
                            count2 += 1
                    except:
                        pass

                hlayer1[i, j] += count1
                hlayer2[i, j] += count2

                # diag upper right to bottom left /
                count1 = 0
                count2 = 0
                for k in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
                    try:
                        if (1 == self.state[turn][i - k, j + k]):
                            count1 += 1
                    except:
                        pass
                    try:
                        if (-1 == self.state[turn][i - k, j + k]):
                            count2 += 1
                    except:
                        pass

                hlayer1[i, j] += count1
                hlayer2[i, j] += count2

        if layer == 1:
            return hlayer1
        elif layer == -1:
            return hlayer2
        else:
            return hlayer1, hlayer2

    def get_state(self, turn: int):
        return self.state[turn]

    def put_check(self, action: int, turn: int):
        idx = (action // col_size, action % row_size)

        if self.state[turn][idx] != 0.0:
            return False

        return True

    def step(self, action: int, turn: int):
        idx = (action // col_size, action % row_size)

        if (turn == 0):
            if self.state[0][idx] == 0.0:
                self.state[0][idx] = 1.0
                self.state[1][idx] = -1.0
                reward = self.reward_dict["step"]
                info = {"pass": True}
            else:
                reward = self.reward_dict["overlap"]
                info = {"pass": False}

        else:
            if self.state[1][idx] == 0.0:
                self.state[1][idx] = 1.0
                self.state[0][idx] = -1.0
                reward = self.reward_dict["step"]
                info = {"pass": True}
            else:
                reward = self.reward_dict["overlap"]
                info = {"pass": False}

        next_obs = self.get_state(turn)

        w = self.finish_check()

        if w != 0 and (not info["pass"]):
            done = True
            if (w == 1 and turn == 0) or (w == -1 and turn == 1):
                reward += self.reward_dict["victory"]
            else:
                reward += self.reward_dict["defeat"]
        else:
            done = False

        return next_obs, reward, done, info

    def seton(self, action, turn):
        idx = (action // col_size, action % row_size)
        self.state[turn][idx] = 1.0
        self.state[0 if turn == 1 else 1][idx] = -1.0

    def update(self):
        for k in self.state:
            print(k)

    def render(self):
        radius = grid_size // 2 - 2

        if not self.screen:
            pg.init()
            self.screen = pg.display.set_mode(
                (row_size * grid_size, col_size * grid_size))
            pg.display.set_caption('Shogi Game')
            self.font = pg.font.SysFont("AppleSDGothicNeo", 40, True, False)

        self.screen.fill(board_color)

        # Draw the grid
        for i in range(row_size):
            for j in range(col_size):
                rect = pg.Rect(i * grid_size, j * grid_size,
                               grid_size, grid_size)
                pg.draw.rect(self.screen, line_color, rect, 1)

        # Draw the stones
        for i in range(row_size):
            for j in range(col_size):
                center = (j * grid_size + grid_size // 2,
                          i * grid_size + grid_size // 2)

                if self.state[0][i][j] == 1:  # Player 1's 장
                    pg.draw.circle(self.screen, black_stone, center, radius)
                    text_surface = self.font.render(
                        '장', True, white_stone)  # White text on black stone
                    text_rect = text_surface.get_rect(center=center)
                    self.screen.blit(text_surface, text_rect)
                elif self.state[1][i][j] == 1:  # Player 2 stone
                    pg.draw.circle(self.screen, white_stone, center, radius)
                    text_surface = self.font.render(
                        '장', True, black_stone)  # Black text on white stone
                    text_rect = text_surface.get_rect(center=center)
                    self.screen.blit(text_surface, text_rect)

        # Update the display
        pg.display.flip()
