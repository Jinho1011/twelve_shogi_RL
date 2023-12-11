from env import TwelveShogi
from game import ShogiRenderer
import copy
from mcts import mcts_go
from collections import deque
import datetime
import random
import time
import numpy as np
import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_v2_behavior()


row_size, col_size = 3, 4
state_size = 24
action_size = 156

load_model = False
train_mode = True

batch_size = 64
mem_maxlen = 50000
discount_factor = 1.0
learning_rate = 0.0002

run_episode = 30000
test_episode = 100

max_step = 226

start_train_episode = 1000

target_update_step = 25
print_interval = 1
save_interval = 500

epsilon_init = 0.95
epsilon_min = 0.05

date_time = datetime.datetime.now().strftime("%d-%H-%M")

save_path = "./saved_models/" + date_time + "_DQN_MCTS"
load_path = "./saved_models/"


class Model():
    def __init__(self, model_name):
        self.input = tf.placeholder(
            shape=[None, 1, state_size, state_size], dtype=tf.float32)

        with tf.variable_scope(name_or_scope=model_name):
            self.conv1 = tf.layers.conv2d(
                self.input, 32, [3, 3], padding='SAME', activation=tf.nn.relu)
            self.pool1 = tf.layers.max_pooling2d(
                self.conv1, [2, 2], [1, 1], padding='SAME')
            self.conv2 = tf.layers.conv2d(
                self.pool1, 32, [3, 3], padding='SAME', activation=tf.nn.relu)
            self.pool2 = tf.layers.max_pooling2d(
                self.conv2, [2, 2], [1, 1], padding='SAME')

            self.flat = tf.layers.flatten(self.pool2)

            self.fc1 = tf.layers.dense(self.flat, 128, activation=tf.nn.relu)
            self.Q_Out = tf.layers.dense(
                self.fc1, action_size, activation=tf.nn.softmax)

        self.predict = tf.argmax(self.Q_Out, 1)

        self.target_Q = tf.placeholder(
            shape=[None, action_size], dtype=tf.float32)

        self.loss = tf.losses.huber_loss(self.target_Q, self.Q_Out)
        self.UpdateModel = tf.train.AdamOptimizer(
            learning_rate).minimize(self.loss)
        self.trainable_var = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, model_name)


class DQNAgent():
    def __init__(self):
        self.model1 = Model("Q1")
        self.target_model1 = Model("target1")
        self.model2 = Model("Q2")
        self.target_model2 = Model("target2")

        self.memory1 = deque(maxlen=mem_maxlen)
        self.memory2 = deque(maxlen=mem_maxlen)

        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self.epsilon = epsilon_init

        self.Saver = tf.train.Saver()
        self.Summary, self.Merge = self.Make_Summary()

        if load_model == True:
            self.Saver.restore(self.sess, load_path)

        self.game = TwelveShogi(3, 4)
        self.renderer = ShogiRenderer(self.game, row_size, col_size)

    def reset_mcts(self):
        self.game = TwelveShogi(3, 4)

    def print_mcts(self):
        self.renderer.render()

    def get_action(self, state, turn: int):
        if self.epsilon > np.random.rand():
            turn = -1 if turn == 0 else 1
            move = mcts_go(current_game=copy.deepcopy(
                self.game), turn=turn, stats=True)
            action = move[0] * state_size[0] + move[1]
            return action
        else:
            if turn == 0:
                predict1 = self.sess.run(self.model1.predict, feed_dict={
                                         self.model1.input: [[state]]})
                return np.isscalar(predict1)
            else:
                predict2 = self.sess.run(self.model2.predict, feed_dict={
                                         self.model2.input: [[state]]})
                return np.isscalar(predict2)

    def append_sample(self, data, turn: int):
        if turn == 0:
            self.memory1.append(
                ([data[0]], data[1], data[2], [data[3]], data[4]))
        else:
            self.memory2.append(
                ([data[0]], data[1], data[2], [data[3]], data[4]))

    def save_model(self):
        self.Saver.save(self.sess, save_path + "/model/model")

    def train_model(self, model, target_model, memory, done):
        if done:
            if self.epsilon > epsilon_min:
                self.epsilon -= (0.5 / (run_episode -
                                 start_train_episode)) * 15

        mini_batch = random.sample(memory, batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for i in range(batch_size):
            states.append(mini_batch[i][0])
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states.append(mini_batch[i][3])
            dones.append(mini_batch[i][4])

        target = self.sess.run(model.Q_Out, feed_dict={model.input: states})
        target_val = self.sess.run(target_model.Q_Out,
                                   feed_dict={target_model.input: next_states})

        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + \
                    discount_factor * np.amax(target_val[i])

        _, loss = self.sess.run([model.UpdateModel, model.loss],
                                feed_dict={model.input: states,
                                           model.target_Q: target})

        return loss

    def update_target(self, model, target_model):
        for i in range(len(model.trainable_var)):
            self.sess.run(target_model.trainable_var[i].assign(
                model.trainable_var[i]))

    def Make_Summary(self):
        self.summary_end_step = tf.placeholder(dtype=tf.float32)
        self.summary_mean_rewards = tf.placeholder(dtype=tf.float32)
        self.summary_max_rewards = tf.placeholder(dtype=tf.float32)
        self.summary_loss1 = tf.placeholder(dtype=tf.float32)
        self.summary_reward1 = tf.placeholder(dtype=tf.float32)
        self.summary_loss2 = tf.placeholder(dtype=tf.float32)
        self.summary_reward2 = tf.placeholder(dtype=tf.float32)

        tf.summary.scalar("end_step", self.summary_end_step)
        tf.summary.scalar("mean_rewards", self.summary_mean_rewards)
        tf.summary.scalar("max_rewards", self.summary_max_rewards)
        tf.summary.scalar("loss1", self.summary_loss1)
        tf.summary.scalar("reward1", self.summary_reward1)
        tf.summary.scalar("loss2", self.summary_loss2)
        tf.summary.scalar("reward2", self.summary_reward2)

        Summary = tf.summary.FileWriter(
            logdir=save_path, graph=self.sess.graph)
        Merge = tf.summary.merge_all()

        return Summary, Merge

    def Write_Summray(self, end_step, rewards, max_reward, reward1, loss1, reward2, loss2, episode):
        self.Summary.add_summary(
            self.sess.run(self.Merge, feed_dict={self.summary_end_step: end_step,
                                                 self.summary_mean_rewards: rewards,
                                                 self.summary_max_rewards: max_reward,
                                                 self.summary_loss1: loss1,
                                                 self.summary_reward1: reward1,
                                                 self.summary_loss2: loss2,
                                                 self.summary_reward2: reward2}), episode)


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

        if turn == 0:
            action = agent1.select_action()
        else:
            action = agent2.select_action()

        next_state, reward, done = env.step(action, turn)

        renderer.render()

        cum_reward += reward
        print(f"Step: {next_state}, Reward: {cum_reward}")

        turn ^= 1

        time.sleep(0.1)
