from env import TwelveShogi


env = TwelveShogi(3, 4)


state = [[0, 3, 0, -4],
         [2,-2,-1,-3],
         [0, 4, 0, 0]]

print(env.is_check_mate(state, 0))
