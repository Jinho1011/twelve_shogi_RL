import random
from env import TwelveShogi


class Agent():
    def __init__(self, env: TwelveShogi, turn) -> None:
        self.env = env
        self.turn = turn

        self.pieces = []
        # [ (i, j, type), ... ]

    def get_empty_location(self):
        locations = []
        for i in range(0, self.env.row_size):
            for j in range(0, self.env.col_size):
                if self.turn == 0:
                    if j != 3 and self.env.state[i][j] == 0:
                        locations.append((i, j))
                else:
                    if j != 0 and self.env.state[i][j] == 0:
                        locations.append((i, j))

        return locations

    def get_actions(self, piece: (int, int, int)):
        i, j, type = piece

        actions = self.env.get_action(type, self.turn)
        black_list = []

        for action in actions:
            x, y = action
            if (i + x) < 0 or (i + x) > 2:
                black_list.append(action)
                continue
            if (j + y) < 0 or (j + y) > 3:
                black_list.append(action)
                continue
            if self.turn == 0 and self.env.state[i + x][j + y] > 0:
                black_list.append(action)
                continue
            if self.turn == 1 and self.env.state[i + x][j + y] < 0:
                black_list.append(action)
                continue

        # actions 에서 black_list 제외
        result = [item for item in actions if item not in black_list]
        # 반환형식 [(1,0),(1,1),(0,1)...] direction만 포함
        return result

    def get_pieces(self, turn):
        for i in range(0, self.env.row_size):
            for j in range(0, self.env.col_size):
                if turn == 0:
                    if self.env.state[i][j] > 0:
                        self.pieces.append((i, j, self.env.state[i][j]))
                else:
                    if self.env.state[i][j] < 0:
                        self.pieces.append((i, j, self.env.state[i][j]))

    def select_action(self):
        """
         0. 말을 선택할지 포로를 선택할지 정한다.
            - 포로가 존재하는지 검사한다
            - random하게 0에서 1 사이의 숫자를 생성하고 0.5 이상이면 말을 선택하고 0.5 미만이면 포로를 선택한다
         1. 말을 선택하거나 포로를 선택한다
            - 만약 turn이 0이라면 self.env.state에서 양수인 것들을, 1이라면 음수인 것들을 가져온다
         A. 말을 선택한 경우
         a. 랜덤으로 말 선택
         b. 선택한 말이 움직일 수 있는 곳 체크
         c. 움직일 수 있는 곳 중에서 랜덤으로 선택 -> 선택한 말, 위치 반환 -> 만약 갈 수 없는 곳(자신의 말이 이미 있는 경우)
                                                                               0으로 가기
         B. 포로를 선택한 경우
         a. 포로중 랜덤으로 말 선택
         b. 선택한 말을 놓을 수 있는 곳을 랜덤하게 고른다(내 영역 중에서 0으로 된 곳 선택) -> 내 영역 중 0으로 된 곳이 없으면 0으로 감
         c. 놓을 수 있는 곳 중에서 랜덤으로 선택 -> 선택한 말, 위치 반환
        """
        done = False

        while not done:
            if len(self.env.poros[self.turn]) == 0:
                # self.pieces 중에서 랜덤하게 선택 -> 가능한 액션 받아오고
                self.get_pieces(self.turn)
                piece = random.choice(self.pieces)
                i, j, type = piece

                actions = self.get_actions(piece)
                # 액션 랜덤 선택
                if len(actions) == 0:
                    continue
                action = random.choice(actions)
                # 반환형식: action = ((1, 2), 2, (1,1)))
                return ((i, j), type, action)
            else:
                if len(self.pieces) / (len(self.pieces) + len(self.env.poros)) <= random.random():
                    self.get_pieces(self.turn)
                    piece = random.choice(self.pieces)
                    i, j, type = piece

                    actions = self.get_actions(piece)
                    # 액션 랜덤 선택
                    if len(actions) == 0:
                        continue
                    action = random.choice(actions)
                    # 반환형식: action = ((1, 2), 2, (1,1)))
                    return ((i, j), type, action)
                else:
                    # self.env.poros 중에서 랜덤하게 선택
                    piece = random.choice(self.env.poros[self.turn])
                    # 비어있는 위치 배열 받아오기
                    locations = self.get_empty_location()
                    # 빈 배열이면 되돌아가기
                    if len(locations) == 0:
                        continue

                    # 비어있는 위치 중 랜덤 선택
                    location = random.choice(locations)

                    return (location, piece, None)
