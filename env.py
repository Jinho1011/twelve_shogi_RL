# -*- coding:utf-8 -*-
import copy
import random
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
        self.poros = {0: [],
                      1: []}
        self.state = [
            [2, 0, 0, -1],
            [3, 4, -4, -3],
            [1, 0, 0, -2]
        ]
        self.reward_dict = {"victory": 10.0, "defeat": -
                            100.0, "step": 0.0000, "catch": 0.0001}
        self.move_history = []
        self.moves = 0

    def reset(self):
        self.state = [
            [2, 0, 0, -1],
            [3, 4, -4, -3],
            [1, 0, 0, -2]
        ]
        self.poros[0] = []
        self.poros[1] = []

    def area_check(self, turn, i, j) -> bool:
        if turn == 0:
            return j == 0
        else:
            return j == 3

    # get action of each pieces

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
            case _:
                return []
        return []

    def finish_region_check(self, turn) -> bool:
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

    def finish_catch_check(self, turn) -> bool:
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
        return abs(self.state[i][j])

    def move_piece(self, i, j, x, y):
        temp = self.state[i][j]
        self.state[i+x][j+y] = temp
        self.state[i][j] = 0

    def turn_hu(self, type, i, j):
        if type == 4:
            if self.area_check(1, i, j):
                self.state[i][j] = 5
        elif type == -4:
            if self.area_check(0, i, j):
                self.state[i][j] = -5

    def step(self, action, turn: int):
        coord, type, direction = action
        i, j = coord
        done_region = self.finish_region_check(turn)
        reward = 0

        if direction != (0, 0):
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
                if abs(type) == 3:
                    reward -= self.reward_dict["step"] / 10
                else:
                    reward += self.reward_dict["step"]
            self.turn_hu(type, target_x, target_y)
        else:
            # 포로에서 말을 꺼내서 두는 경우
            if self.state[i][j] == 0:
                self.state[i][j] = type
                self.poros[turn].remove(type)
                # self.poros[turn] = np.delete(self.poros[turn], index)

                reward += self.reward_dict["step"]

        done_catch = self.finish_catch_check(turn)
        if done_region or done_catch:
            reward += self.reward_dict["victory"]

        poro0 = np.array(self.poros[0])
        poro1 = np.array(self.poros[1])

        poro0_padded = np.pad(
            poro0, (0, max(0, 6 - len(poro0))), constant_values=0)
        poro1_padded = np.pad(
            poro1, (0, max(0, 6 - len(poro1))), constant_values=0)
        next_state = np.concatenate(
            (self.state, poro0_padded, poro1_padded), axis=None)
        next_state = np.append(next_state, turn)
        next_state = np.array(next_state).reshape(5, 5)

        if self.is_check_mate(self.state, turn):
            reward -= 20

        return next_state, reward, done_region or done_catch

    def is_check_mate(self, state, turn):
        # 내 왕이 잡히는 경우 체크메이트
        # 내 왕이 상대편 말들에서 나올 수 있는 위치들에 있으면 체크메이트
        # 상대편 말들에서 나올 수 있는 위치들을 구한다
        opponents = []
        king_position = None

        if turn == 0:
            for i in range(self.row_size):
                for j in range(self.col_size):
                    if state[i][j] == 3:
                        king_position = (i, j)
                        break
        else:
            for i in range(self.row_size):
                for j in range(self.col_size):
                    if state[i][j] == -3:
                        king_position = (i, j)
                        break

        if turn == 0:
            for i in range(self.row_size):
                for j in range(self.col_size):
                    if state[i][j] < 0:
                        opponents.append((i, j))
        else:
            for i in range(self.row_size):
                for j in range(self.col_size):
                    if state[i][j] > 0:
                        opponents.append((i, j))

        positions = []

        for opponent in opponents:
            opponent_type = state[opponent[0]][opponent[1]]
            opponent_actions = self.get_action(opponent_type, turn)
            for opponent_action in opponent_actions:
                opponent_x, opponent_y = opponent_action
                opponent_target_x, opponent_target_y = opponent[0] + \
                    opponent_x, opponent[1] + opponent_y
                positions.append((opponent_target_x, opponent_target_y))

        return king_position in positions

    def get_obvious_moves(self, turn):
        # 이번 턴에 취할 수 있는 모든 액션을 가져옴
        # coord, type, direction
        # ((1, 1), 4, (0, 1)) 1,1에 있는 type 4의 말을 (0,1) 방향으로 이동: 기존 말을 이동시키는 경우
        # ((1, 1), 4, None) 1,1에 있는 type 4의 말을 꺼내서 놓기: 포로에서 꺼내서 새로 두는 경우

        # MISSION: 상대 왕의 위치를 얻고, 그 방향으로 이동하는 액션만 남기기
        # ex. 상대 왕이 (1, 3)에 있고

        actions = self.get_all_possible_actions(turn)
        # 상대 왕의 위치 찾기
        opponent_king = -3 if turn == 0 else 3
        king_position = None
        for i in range(self.row_size):
            for j in range(self.col_size):
                if self.state[i][j] == opponent_king:
                    king_position = (i, j)
                    break
            if king_position:
                break

        if king_position is None:
            return actions

        # 상대 왕의 위치로의 방향으로 이동할 수 있는 액션만 필터링
        filtered_actions = []
        for coord, type, action in actions:
            if action == (0, 0):
                continue
            x, y = action

            direction_to_king = (
                king_position[0] - coord[0], king_position[1] - coord[1])

            # 상대 왕을 향하는 방향으로 이동하는 액션만 필터링
            if (direction_to_king[0] >= 0 and x >= 0) or (direction_to_king[0] < 0 and x < 0):
                if (direction_to_king[1] >= 0 and y >= 0) or (direction_to_king[1] < 0 and y < 0):
                    filtered_actions.append((coord, type, action))

        for coord, type, action in filtered_actions:
            temp = copy.deepcopy(self.state)
            i, j = coord
            x, y = action
            target_x, target_y = i + x, j + y
            temp[target_x][target_y] = temp[i][j]
            temp[i][j] = 0
            if self.is_check_mate(temp, turn):
                filtered_actions.remove((coord, type, action))

        if len(filtered_actions) == 0:
            return actions
        else:
            return filtered_actions

    def get_all_possible_actions(self, turn: int):
        actions = []
        pieces = self.get_pieces(turn)
        actions.extend(self.get_actions_for_pieces(pieces, turn))
        actions.extend(self.get_poro_actions(turn))
        return list(set(actions))

    def get_pieces(self, turn: int):
        return [(i, j, self.state[i][j]) for i in range(self.row_size) for j in range(self.col_size) if (turn == 0 and self.state[i][j] > 0) or (turn == 1 and self.state[i][j] < 0)]

    def get_actions_for_pieces(self, pieces, turn):
        actions = []
        for piece in pieces:
            i, j, type = piece
            actions_for_piece = [action for action in self.get_action(
                type, turn) if self.is_valid_action(i, j, action, turn)]
            actions.extend([((i, j), type, action)
                            for action in actions_for_piece])
        return actions

    def is_valid_action(self, i, j, action, turn):
        x, y = action
        if (x, y) != (0, 0):
            if not (0 <= i + x <= 2) or not (0 <= j + y <= 3):
                return False
            if (turn == 0 and self.state[i + x][j + y] > 0) or (turn == 1 and self.state[i + x][j + y] < 0):
                return False
        return True

    def get_poro_actions(self, turn):
        locations = [(i, j) for i in range(self.row_size) for j in range(self.col_size) if (
            turn == 0 and j != 3 and self.state[i][j] == 0) or (turn == 1 and j != 0 and self.state[i][j] == 0)]
        return [(location, poro, (0, 0)) for location in locations for poro in self.poros[turn] if poro != 0]

    def undo(self):
        "remove the last placed piece"
        if self.move_history:  # is not empty
            # ((1, 1), 4, (0, 1)) 1,1에 있는 type 4의 말을 (0,1) 방향으로 이동
            last_action = self.move_history[-1]
            self.state[last_action[0][0]][last_action[0][1]] = 0
            self.move_history.pop()
            self.moves -= 1
        else:
            raise IndexError("No moves have been played.")

    def validate_action(self, action, turn):
        try:
            area_check = self.is_valid_action(
                action[0][0], action[0][1], action[2], turn)

            type = action[1]
            direction = action[2]

            valid_action = self.get_action(type, turn)

            if direction == (0, 0):
                locations = []
                for i in range(0, self.row_size):
                    for j in range(0, self.col_size):
                        if turn == 0:
                            if j != 3 and self.state[i][j] == 0:
                                locations.append((i, j))
                        else:
                            if j != 0 and self.state[i][j] == 0:
                                locations.append((i, j))
                direction_check = action[0] in locations
            else:
                direction_check = (direction[0], direction[1]) in valid_action and self.state[action[0]
                                                                                              [0]][action[0][1]] == type
            return area_check and direction_check
        except:
            print("exception occured")
            return False
