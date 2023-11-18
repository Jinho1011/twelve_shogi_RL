# -*- coding:utf-8 -*-

from agent import Agent
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

    env.reset()
    done = False
    turn = 0
    agent1 = Agent(env, 0)
    agent2 = Agent(env, 1)

    cum_reward = 0.0

    while not done:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                done = True

        if turn == 0:
            action = agent1.select_action()
        else:
            action = agent2.select_action()

        next_state, reward, done = env.step(action, turn)

        cum_reward += reward
        print(f"Step: {next_state}, Reward: {cum_reward}")

        renderer.render()
        turn ^= 1

        time.sleep(0.1)

    # renderer.close()


if __name__ == "__main__":
    main()
