max_a = 12
max_b = 5
max_c = 9


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


abc = [9, 1, 3]
num = abc_to_num(abc)
print(num)
print(num_to_abc(num))
