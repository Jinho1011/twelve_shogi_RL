# -*- coding:utf-8 -*-
from env import TwelveShogi
from game import ShogiRenderer
from dqn import DQNAgent
import pygame as pg
import random
import time
import numpy as np


def main():
    row_size = 3
    col_size = 4

    env = TwelveShogi(row_size, col_size)
    agent1 = DQNAgent(0, epsilon=0.01, load_model=True)
    agent2 = DQNAgent(1, epsilon=0.01, load_model=True)
    renderer = ShogiRenderer(env, row_size, col_size)

    env.reset()
    done = False
    turn = 0

    cum_reward = {
        0: 0,
        1: 1
    }

    renderer.render()

    while not done:
        # 스페이스 바를 누를 때까지 기다림
        wait_for_space = True
        while wait_for_space:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    done = True
                    wait_for_space = False
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_SPACE:
                        wait_for_space = False

        poro0 = np.array(env.poros[0])
        poro1 = np.array(env.poros[1])

        poro0_padded = np.pad(
            poro0, (0, max(0, 6 - len(poro0))), constant_values=0)
        poro1_padded = np.pad(
            poro1, (0, max(0, 6 - len(poro1))), constant_values=0)

        # Concatenate along the first axis
        state = np.concatenate(
            (env.state, poro0_padded, poro1_padded), axis=None)
        state = np.append(state, turn)  # 0 대신에 turn append 하도록 수정
        state = np.array(state).reshape(5, 5)
        if turn == 0:
            action = agent1.get_action(state, turn, env)
        else:
            action = agent2.get_action(state, turn, env)

        next_state, reward, done = env.step(action, turn)

        renderer.render()

        cum_reward[turn] += reward
        print(f"Step: {next_state}, turn: {turn}, Reward: {cum_reward[turn]}")

        turn ^= 1

        time.sleep(0.1)

    # renderer.close()


if __name__ == "__main__":
    main()
