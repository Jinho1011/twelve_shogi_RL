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


2 0 -4 -1
3 5 4 -3
1 0 3 -2
- 왼쪽 양수가 0
- 오른쪽 음수가 1



step 함수 의사코드
1. 먼저 piece를 랜덤하게 선택한다
2. 선택한 piece에서 취할 수 있는 액션 목록 중에서 랜덤하게 하나를 선택한다
 - 만약 취할 수 있는 액션이 없다면, 다시 1번으로 돌아가 랜덤하게 piece를 선택하고 2를 반복한다
3. 만약 어떠한 액션이라도 수행하고 나면, finish_check로 종료 여부를 확인한다


def 말의액션가져오기(type, turn):
    장_action = [(1, 0), (0, 1), (0, -1), (-1, 0)]
    장_action = [(1, 1), (0, 1), (0, -1), (-1, 0)]
    장_action = [(1, 0), (0, 1), (0, -1), (-1, 0)]
    자_action = {
        0: [(1, 0)],
        1: [(-1, 0)]
    }
    
"""


class TwelveShogi():
    def __init__(self, row_size, col_size) -> None:
        self.row_size = row_size
        self.col_size = col_size
        self.poros = {0: [], 1: []}
        self.state = [
            [2, 0, 0, -1],
            [3, 4, -4, -3],
            [1, 0, 0, -2]
        ]
        self.reward_dict = {"victory": 10.0, "defeat": -
                            10.0, "step": 0.0001, "catch": 0.01}
        self.screen = None

    def reset(self):
        self.state = [
            [2, 0, 0, -1],
            [3, 4, -4, -3],
            [1, 0, 0, -2]
        ]

    # 내 영역인지 아닌지 확인하는 함수, 내 영역이면 True
    def area_check(self, turn, i, j) -> bool:
        if turn == 0:
            return j == 0
        else:
            return j == 3

    def get_action(self, type, turn):
        장_action = [(1, 0), (0, 1), (0, -1), (-1, 0)]
        상_action = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        왕_action = [(1, 0), (0, 1), (0, -1), (-1, 0),
                    (1, 1), (1, -1), (-1, 1), (-1, -1)]
        자_action = {
            0: [(0, 1)],
            1: [(0, -1)]
        }
        후_action = {
            0: [(1, 0), (0, 1), (0, -1), (-1, 0),
                (1, 1), (-1, 1)],
            1: [(1, 0), (0, 1), (0, -1), (-1, 0),
                (1, -1),  (-1, -1)]
        }
        match abs(type):
            case 1:
                return 장_action
            case 2:
                return 상_action
            case 3:
                return 왕_action
            case 4:
                return 자_action[turn]
            case 5:
                return 후_action[turn]

    # TODO: finish_check 함수 수정
    def finish_check(self, stage, turn) -> bool:
        if stage == 0:
            king = 3 if turn == 0 else -3
            if turn == 0:
                for i in range(0, self.row_size):
                    if self.state[i][3] == king:
                        return True
            else:
                for i in range(0, self.row_size):
                    if self.state[i][0] == king:
                        return True
            return False

        else:
            if turn == 0:
                for row in self.state:
                    for item in row:
                        if item == -3:
                            return False
                return True
            
            else:
                for row in self.state:
                    for item in row:
                        if item == 3:
                            return False
                return True

    def get_type(self, i, j):
        return abs(self.state[i, j])

    def move_piece(self, i, j, x, y):
        temp = self.state[i][j]
        self.state[i+x][j+y] = temp
        self.state[i][j] = 0

    def step(self, action: ((int, int), int, (int, int)), turn: int):
        coord, type, direction = action
        i, j = coord
        done = self.finish_check(0, turn)
        reward = 0

        if direction:
            # 기존 말을 이동시키는 경우
            x, y = direction
            target_x, target_y = i + x, j + y

            if self.state[target_x][target_y] != 0:
                # 상대 말을 잡은 경우
                poro = self.state[target_x][target_y]
                if abs(poro) == 5:
                    poro = 4 if turn == 0 else -4
                else:
                    poro = poro * -1
                self.poros[turn].append(poro)
                self.move_piece(i, j, x, y)
                reward += self.reward_dict["catch"]  # 상대 말을 잡으면 추가 보상
            else:
                # 빈 칸으로 이동
                self.move_piece(i, j, x, y)
                reward += self.reward_dict["step"]
        else:
            # 포로에서 말을 꺼내서 두는 경우
            if self.state[i][j] == 0:
                self.state[i][j] = type
                self.poros[turn].remove(type)
                reward += self.reward_dict["step"]

        done = self.finish_check(1, turn)
        if done:
            reward += self.reward_dict["victory"] if done == 1 else self.reward_dict["defeat"]

        next_state = self.state
        return next_state, reward, done
