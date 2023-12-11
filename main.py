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

    env.reset()
    done = False
    turn = 0

    cum_reward = 0.0

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

        actions = env.get_all_possible_actions(turn)
        action = random.choice(actions)

        next_state, reward, done = env.step(action, turn)

        renderer.render()

        cum_reward += reward
        print(f"Step: {next_state}, Reward: {cum_reward}")

        turn ^= 1

        time.sleep(0.1)

    # renderer.close()


if __name__ == "__main__":
    main()
