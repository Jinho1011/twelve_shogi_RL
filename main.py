# -*- coding:utf-8 -*-

from env import TwelveShogi
from game import ShogiRenderer
import pygame as pg
import random
import time


def main():
    row_size = 3
    col_size = 4

    env = TwelveShogi(row_size, col_size)
    renderer = ShogiRenderer(env, row_size, col_size)

    obs = env.reset()
    done = False
    target = 0

    step = 0
    cum_reward = 0.0

    while not done:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                done = True

        action = random.randint(0, row_size * col_size - 1)
        next_obs, reward, done, info = env.step(action, target)

        cum_reward += reward
        print(f"Step: {step}, Reward: {cum_reward}")

        renderer.render()
        target ^= 1

        time.sleep(1)

    # renderer.close()


if __name__ == "__main__":
    main()
