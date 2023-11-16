from env import Connect6EnvAdversarial
import pygame as pg
import random
import time

env = Connect6EnvAdversarial()

obs = env.reset()
done = False

step = 0
cum_reward = 0.0

pg.init()
pg.display.set_caption('Connect6 Game')
clock = pg.time.Clock()
target = 0
    

while not done:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            done = True
    
    action = random.randint(0, 3 * 4 - 1)
    next_obs, reward, done, info = env.step(action, target)

    cum_reward += reward
    print(f"Step: {step}, Reward: {cum_reward}")

    # Render the game state
    env.render()

    step += 1

    # Cap the frame rate
    clock.tick(60)
    
    if target == 0: target = 1
    else: target = 0

done = False
while not done:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            done = True