import numpy as np
import time
import random
import datetime
from collections import deque
import copy
from env import TwelveShogi
import os
import tensorflow._api.v2.compat.v1 as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# from mcts import mcts_go

tf.disable_v2_behavior()
tf.disable_eager_execution()

directions = [
    [-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]
]

max_a = 12
max_b = 5
max_c = 9

row_size, col_size = 3, 4
state_size = 24  # 3 * 4 + 12
action_size = 540

# load_model = False
train_mode = True

batch_size = 64
mem_maxlen = 50000
discount_factor = 1.0
learning_rate = 0.0002

run_episode = 50000
test_episode = 100

max_step = 226

start_train_episode = 10000
start_predict_episode = 10000

target_update_step = 25
print_interval = 1
save_interval = 500

epsilon_init = 0.8
epsilon_min = 0.05

date_time = datetime.datetime.now().strftime("%d-%H-%M")

agent_0_save_path = "./saved_models0/"
agent_1_save_path = "./saved_models1/"


def num_to_abc(num):
    a = num // (max_b * max_c)
    num = num % (max_b * max_c)
    b = num // max_c
    c = num % max_c
    return [a, b, c]


def abc_to_num(abc):
    a, b, c = abc
    num = (a * max_b + b) * max_c + c
    return num


class Model():
    def __init__(self, model_name):
        self.input = tf.placeholder(
            shape=[None, 1, 5, 5], dtype=tf.float32)  # 6x4인데 정사각형 모양을 맞추기 위해 5x5으로 만듦

        with tf.variable_scope(name_or_scope=model_name, reuse=tf.AUTO_REUSE):
            self.initializer = tf.initializers.zeros()

            self.conv1 = tf.layers.conv2d(  # type: ignore
                self.input, 16, [3, 3], padding='SAME', activation=tf.nn.relu)
            self.pool1 = tf.layers.max_pooling2d(  # type: ignore
                self.conv1, [2, 2], [1, 1], padding='SAME')
            self.conv2 = tf.layers.conv2d(  # type: ignore
                self.pool1, 16, [3, 3], padding='SAME', activation=tf.nn.relu)
            self.pool2 = tf.layers.max_pooling2d(  # type: ignore
                self.conv2, [2, 2], [1, 1], padding='SAME')
            self.flat = tf.layers.flatten(self.pool2)  # type: ignore

            self.fc1 = tf.layers.dense(  # type: ignore
                self.flat, 128, activation=tf.nn.relu)
            self.Q_Out = tf.layers.dense(  # type: ignore
                self.fc1, action_size, activation=tf.nn.softmax, kernel_initializer=self.initializer)
            self.predict = tf.argmax(self.Q_Out, 1)

            self.target_Q = tf.placeholder(
                shape=[None, action_size], dtype=tf.float32)

            self.loss = tf.losses.huber_loss(self.target_Q, self.Q_Out)
            self.UpdateModel = tf.train.AdamOptimizer(
                learning_rate).minimize(self.loss)
            self.trainable_var = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, model_name)


class DQNAgent():
    def __init__(self, turn, epsilon=0.95, load_model=False):
        self.turn = turn
        self.model = Model("Q")
        self.target_model = Model("targetQ")

        self.memory = deque(maxlen=mem_maxlen)

        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self.epsilon = epsilon

        self.Saver = tf.train.Saver()
        self.Summary, self.Merge = self.Make_Summary()

        if load_model == True:
            load_path = agent_0_save_path if self.turn == 0 else agent_1_save_path
            # self.Saver.restore(self.sess, load_path)
            self.Saver.restore(
                self.sess, f"./saved_models{turn}/")

        self.game = TwelveShogi(row_size, col_size)

    def reset_mcts(self):
        self.game = TwelveShogi(row_size, col_size)

    def set_mcts(self, action, turn):
        self.game.step(action, turn)

    def get_action(self, state, turn: int, env):
        while True:
            if self.epsilon > np.random.rand():
                # 탐험
                actions = env.get_all_possible_actions(turn)
                action = random.choice(actions)
                return action
            else:
                predict1 = self.sess.run(self.model.predict, feed_dict={
                    self.model.input: [[state]]})
                action = num_to_abc(predict1[0])  # type: ignore

                action_0 = (action[0]) // col_size
                action_1 = (action[0]) % col_size
                type = action[1]+1
                if turn != 0:
                    type *= -1
                direction = directions[action[2]]

                action = ((action_0, action_1), type, direction)

                # if action is not valid, then train model that action you predicted for current state is wrong, so Q value of that action should be 0
                if not env.validate_action(action, turn):
                    # target = self.sess.run(self.model.Q_Out, feed_dict={
                    #     self.model.input: [[state]]})
                    # target[0][predict1[0]] = 0.0
                    # self.sess.run(self.model.UpdateModel, feed_dict={
                    #     self.model.input: [[state]], self.model.target_Q: target})
                    continue

                state = np.array(state)

                # Flatten the array and take the first 12 elements
                state_flattened = state.flatten()[:12]

                # Reshape the array to 3x4
                state_reshaped = state_flattened.reshape((3, 4))

                # Print the reshaped array
                print(state_reshaped)
                print(((action_0, action_1), type,
                      (direction[0], direction[1])))

                return ((action_0, action_1), type, (direction[0], direction[1]))

    def append_sample(self, data):
        self.memory.append(
            ([data[0]], data[1], data[2], [data[3]], data[4]))

    def save_model(self):
        save_path = agent_0_save_path if self.turn == 0 else agent_1_save_path
        self.Saver.save(self.sess, save_path)

    def train_model(self, model: Model, target_model: Model, memory, done, episode):
        if done:
            if self.epsilon > epsilon_min and start_predict_episode < episode:
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
            coord = actions[i][0]*col_size + actions[i][1]
            type = abs(actions[i][2])-1
            direction_index = directions.index(
                [actions[i][3], actions[i][4]])
            num = abc_to_num([coord, type, direction_index])

            future_reward = 0 if dones[i] else discount_factor * \
                np.amax(target_val[i])  # type: ignore
            target[i][num] = rewards[i] + future_reward  # type: ignore

        _, loss = self.sess.run([model.UpdateModel, model.loss],
                                feed_dict={model.input: states,
                                           model.target_Q: target})  # type: ignore

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

        save_path = agent_0_save_path if self.turn == 0 else agent_1_save_path
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


if __name__ == '__main__':
    env = TwelveShogi(row_size, col_size)
    agent1 = DQNAgent(0)
    agent2 = DQNAgent(1)

    rewards = {0: [], 1: []}
    losses = {0: [], 1: []}

    end_step = []

    for episode in range(run_episode + test_episode):
        if episode == run_episode:
            train_mode = False

        init_state = env.reset()
        agent1.reset_mcts()
        agent2.reset_mcts()
        done = False
        turn = 1

        episode_rewards = {0: 0.0, 1: 0.0}

        step = 0
        print(f"episode : {episode}")
        while not done:
            turn ^= 1

            agent = agent1 if turn == 0 else agent2
            opponent_agent = agent2 if turn == 0 else agent1

            poro0 = np.array(env.poros[0])
            poro1 = np.array(env.poros[1])

            poro0_padded = np.pad(
                poro0, (0, max(0, 6 - len(poro0))), constant_values=0)
            poro1_padded = np.pad(
                poro1, (0, max(0, 6 - len(poro1))), constant_values=0)

            # Concatenate along the first axis
            state = np.concatenate(
                (env.state, poro0_padded, poro1_padded), axis=None)
            state = np.append(state, 0)  # 0 대신에 turn append 하도록 수정
            state = np.array(state).reshape(5, 5)
            action = agent.get_action(state, turn, env)
            # [i, j, type, x, y]
            next_state, reward, done = env.step(action, turn)
            # agent.set_mcts(action, turn)

            episode_rewards[turn] += reward
            done = done

            if train_mode:
                action = [action[0][0], action[0][1],
                          action[1], action[2][0], action[2][1]]
                data = [state, action, reward, next_state, done]
                agent.append_sample(data)
            else:
                agent.epsilon = 0.0

            if episode > start_train_episode and train_mode:
                # train behavior networks
                loss1 = agent.train_model(
                    agent.model, agent.target_model, agent.memory, done, episode)
                losses[turn].append(loss1)

                # update target networks
                # 25번째 step마다 target network를 behavior network로 업데이트
                # 느릴 수도 있으니까 나중에 20번 마다 업데이트하도록 바꿔야함
                if step % target_update_step == 0:
                    agent.update_target(
                        agent.model, agent.target_model)
            step += 1

        end_step.append(step)

        rewards[0].append(episode_rewards[0])
        rewards[1].append(episode_rewards[1])

        # 각 에피소드 별 step 수
        # 선공 후공 승률
        # 선공 후공 평균 보상
        # 선공 후공 로스 평균

        # if episode % print_interval == 0 and episode != 0:
        # print("step: {} / episode: {} / epsilon: {:.3f}".format(step,  # type: ignore
        #         episode, agent.epsilon))
        # print("reward: {:.2f} / reward1: {:.2f} / loss1: {:.4f} / reward2: {:.2f} / loss2: {:.4f}".format(
        #       np.mean(rewards[0]) + np.mean(rewards[1]), np.mean(rewards[0]), np.mean(losses[0]), np.mean(rewards[1]), np.mean(losses[1])))
        # print('------------------------------------------------------------')

        # agent1.Write_Summray(np.mean(end_step), np.mean(rewards[0]) + np.mean(rewards[1]), max(max(rewards[0]), max(rewards[1])), np.mean(rewards[0]), np.mean(losses[0]),  # type: ignore
        #                      np.mean(rewards[1]), np.mean(losses[1]), episode)
        # agent2.Write_Summray(np.mean(end_step), np.mean(rewards[0]) + np.mean(rewards[1]), max(max(rewards[0]), max(rewards[1])), np.mean(rewards[0]), np.mean(losses[0]),  # type: ignore
        #                      np.mean(rewards[1]), np.mean(losses[1]), episode)

        rewards = {0: [], 1: []}
        losses = {0: [], 1: []}

        if episode % save_interval == 0 and episode != 0:
            agent1.save_model()  # 선공 모델
            agent2.save_model()  # 후공 모델
            print("Save Model {}".format(episode))
