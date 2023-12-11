"stock mcts implementation"
import sys
import copy
import random
import math
import copy

from env import TwelveShogi

state_size = 24
action_size = 156

MCTS_ITERATIONS = 1000


def mcts_go(current_game: TwelveShogi, turn, iterations=MCTS_ITERATIONS, stats=False):
    "MCTS"
    # Initialize the tree with possible moves and current position
    tree = [Node()]  # for general tracking and debugging
    for action in current_game.get_obvious_moves(turn):
        new_node = Node(parent=tree[0], move_to=action)
        tree[0].children.append(new_node)
        tree.append(new_node)

    for _ in range(iterations):
        # iterations
        current_node = tree[0]  # origin node, current board.
        while not current_node.is_leaf():
            children_scores = tuple(
                map(lambda x: x.ucb1(), current_node.children))
            current_node = current_node.children[children_scores.index(
                max(children_scores))]

        board_updates = 0
        for action in current_node.moves_to:
            current_game.step(action, turn)
            board_updates += 1

        # quickly check if the game if is in a terminal state
        do_rollout = True
        rollout_res = current_game.finish_catch_check(turn)
        if rollout_res:
            # the game is already terminal, look no further.
            do_rollout = False

        if not current_node.visits and do_rollout:  # ==0
            # rollout
            rollout_res = rollout(copy.deepcopy(current_game), turn)
        elif current_node.visits and do_rollout:
            # let's go deeper!!!!!!111!!!
            for move in current_game.get_obvious_moves(turn):
                new_node = Node(parent=current_node, move_to=list(move))
                current_node.children.append(new_node)
                tree.append(new_node)
            if not current_node.children:
                rollout_res = 0
            else:
                current_node = current_node.children[0]
                # update board again
                board_updates += 1
                current_game.step(
                    current_node.moves_to[-1], turn)
                # rollout
                rollout_res = rollout(copy.deepcopy(current_game), turn)

        # revert board
        for _ in range(board_updates):
            current_game.undo()

        # backpropogate the rollout
        while current_node.parent:  # not None. only the top node has None as a parent
            current_node.visits += 1
            current_node.score += rollout_res
            current_node = current_node.parent
        current_node.visits += 1  # for the mother node

    # pick the move with the most visits
    if stats:
        print('Stats for nerds\n' f'Search tree size: {len(tree)}')
    current_node = tree[0]
    visit_map = tuple(map(lambda x: x.visits, current_node.children))
    best_move = visit_map.index(max(visit_map))
    return current_game.get_obvious_moves(turn)[best_move]


class Node:
    def __init__(self, parent: 'Node' | None = None, move_to=None):
        self.parent = parent
        if parent and not move_to:
            raise TypeError("A parent is provided with no move_to paramenter.")
        elif parent:
            self.moves_to = copy.deepcopy(self.parent.moves_to)  # type: ignore
            self.moves_to.append(move_to)
        else:
            self.moves_to = []
        self.score = 0
        self.visits = 0
        self.children = []

    def is_leaf(self):
        return not bool(self.children)

    def ucb1(self):
        try:
            return self.score / self.visits + 2 * math.sqrt(math.log(self.parent.visits)  # type: ignore
                                                            / self.visits)
        except ZeroDivisionError:
            # equivalent to infinity
            # assuming log(parent visits) / visits will not exceed 100
            return 10000


def rollout(game: TwelveShogi, turn):
    "Rollout a game"
    while True:
        check_win = game.finish_region_check(turn)

        if check_win:
            return (turn + 1) // 2

        action = random.choice(game.get_all_possible_actions(turn))
        game.step(action, turn)
        if game.finish_catch_check(turn):
            return (turn + 1) // 2
