# -*- coding:utf-8 -*-
import numpy as np

PIECES = {
    '장': 1,
    '상': 2,
    '왕': 3,
    '자': 4,
    '후': 5,
}

"""
게임이 시작되면 선 플레이어부터 말 1개를 1칸 이동시킬 수 있다. 
말을 이동시켜 상대방의 말을 잡은 경우, 해당 말을 포로로 잡게 되며 포로로 잡은 말은 다음 턴부터 자신의 말로 사용할 수 있다.

게임 판에 포로로 잡은 말을 내려놓는 행동도 턴을 소모하는 것이며 이미 말이 놓여진 곳이나 상대의 진영에는 말을 내려놓을 수 없다.

상대방의 후(侯)를 잡아 자신의 말로 사용할 경우에는 자(子)로 뒤집어서 사용해야 한다.

게임은 한 플레이어가 상대방의 왕(王)을 잡으면 해당 플레이어의 승리로 종료된다.

만약 자신의 왕(王)이 상대방의 진영에 들어가 자신의 턴이 다시 돌아올 때까지 한 턴을 버틸 경우 해당 플레이어의 승리로 게임이 종료된다.

1: 장(將). 자신의 진영 오른쪽에 놓이는 말로 앞, 뒤와 좌, 우로 이동이 가능하다.
2: 상(相). 자신의 진영 왼쪽에 놓이며 대각선 4방향으로 이동할 수 있다.
3: 왕(王). 자신의 진영 중앙에 위치하며 앞, 뒤, 좌, 우, 대각선 방향까지 모든 방향으로 이동이 가능하다.
4: 자(子). 왕의 앞에 놓이며 오로지 앞으로만 이동할 수 있다.
5: 하지만, 자(子)는 상대 진영에 들어가면 뒤집어서 후(侯)로 사용된다. 후(侯)는 대각선 뒤쪽 방향을 제외한 전 방향으로 이동할 수 있다.

초기 상태는 아래와 같음

2 0 0 0
3 4 0 0
1 0 0 0

0 0 0 1
0 0 4 3
0 0 0 2

step 함수 의사코드
1. 먼저 piece를 랜덤하게 선택한다
2. 선택한 piece에서 취할 수 있는 액션 목록 중에서 랜덤하게 하나를 선택한다
 - 만약 취할 수 있는 액션이 없다면, 다시 1번으로 돌아가 랜덤하게 piece를 선택하고 2를 반복한다
3. 만약 어떠한 액션이라도 수행하고 나면, finish_check로 종료 여부를 확인한다

"""


class TwelveShogi():
    def __init__(self, row_size, col_size) -> None:
        self.row_size = row_size
        self.col_size = col_size
        self.state = {0: np.zeros((self.row_size, self.col_size), dtype=np.int32), 1: np.zeros(
            (self.row_size, self.col_size), dtype=np.int32)}
        self.reward_dict = {"victory": 10.0, "defeat": -
                            10.0, "step": 0.0001, "overlap": -0.1}
        self.screen = None

    def reset(self):
        self.state = {0: np.zeros((self.row_size, self.col_size), dtype=np.int32), 1: np.zeros(
            (self.row_size, self.col_size), dtype=np.int32)}
        obs = self.state[0]
        return obs

    # TODO: finish_check 함수 수정
    def finish_check(self) -> bool:
        for i in range(0, self.row_size):
            for j in range(0, self.col_size):
                for k in [1, -1]:
                    try:
                        if (k == self.state[0][i, j] == self.state[0][i + 1, j] == self.state[0][i + 2, j] == self.state[0][i + 3, j] == self.state[0][i + 4, j] == self.state[0][i + 5, j]):
                            return k
                    except:
                        pass

        for j in range(0, self.row_size):
            for i in range(0, self.col_size):
                for k in [1, -1]:
                    try:
                        if (k == self.state[0][i, j] == self.state[0][i, j + 1] == self.state[0][i, j + 2] == self.state[0][i, j + 3] == self.state[0][i, j + 4] == self.state[0][i, j + 5]):
                            return k
                    except:
                        pass

        for i in range(0, self.row_size):
            for j in range(0, self.col_size):
                for k in [1, -1]:
                    try:
                        if (k == self.state[0][i, j] == self.state[0][i + 1, j + 1] == self.state[0][i + 2, j + 2] == self.state[0][i + 3, j + 3] == self.state[0][i + 4, j + 4] == self.state[0][i + 5, j + 5]):
                            return k
                    except:
                        pass

        for i in range(0, self.row_size):
            for j in range(0, self.col_size):
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
        hlayer1 = np.zeros((self.row_size, self.col_size), dtype=np.int32)
        hlayer2 = np.zeros((self.row_size, self.col_size), dtype=np.int32)

        for i in range(0, self.row_size):
            for j in range(0, self.col_size):
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

    # TODO: put_check 함수 수정
    def put_check(self, action: int, turn: int):
        idx = (action // self.col_size, action % self.row_size)

        if self.state[turn][idx] != 0.0:
            return False

        return True

    # TODO: step 함수 수정
    def step(self, action: int, turn: int):
        idx = (action // self.col_size, action % self.row_size)

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
        idx = (action // self.col_size, action % self.row_size)
        self.state[turn][idx] = 1.0
        self.state[0 if turn == 1 else 1][idx] = -1.0

    def update(self):
        for k in self.state:
            print(k)
