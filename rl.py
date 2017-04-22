from model import MountainCar
from tile_coding import TileCoding
from math import inf
from random import choice

# ALPHA = 0.1
GAMMA = 0.999
REPS = 5e6
EPISODE_LIMIT = 2000
# EPS = 0.05
INTERACTIVE = False


def episodes(n):
    tiles = TileCoding([[-1.2, 0.5], [-0.07, 0.07]])
    for i in range(n):
        model = MountainCar(rand_assign = True)
        rew = 0
        while not model.terminated():
            x = model.state()
            best_act = None
            best_value = -inf
            for action in (-1, 0, 1):
                xp, next_reward = model.check_action(action)
                if xp is None:
                    val = next_reward
                else:
                    val = next_reward + GAMMA * tiles.value(xp)
                if val > best_value:
                    best_act = [action]
                    best_value = val
                elif val == best_value:
                    best_act += [action]
            tiles.update(x, best_value)
            rew += model.take_action(choice(best_act))
        yield rew, i


def value_function_estimation(file, tiles = None):
    i = 0
    if INTERACTIVE:
        old_eye = 0
    if tiles is None:
        tiles = TileCoding([[-1.2, 0.5], [-0.07, 0.07]])
        mode = 'w'
    else:
        mode = 'a'
    # eps_rewards = []
    # eps_endings = []
    with open(file, mode) as fout:
        while i < REPS:
            model = MountainCar(rand_assign = True)
            x = model.state()
            if INTERACTIVE:
                print(i - old_eye)
                if input() == 'STOP':
                    break
                old_eye = i
            ep_rew = 0
            ep_i = i
            while not model.terminated():
                if INTERACTIVE:
                    print(x)
                i += 1
                if i % 1e4 == 0:
                    print(i)
                if i - ep_i == EPISODE_LIMIT:
                    break
                best_act = None
                best_value = -inf
                for action in (-1, 0, 1):
                    xp, next_reward = model.check_action(action)
                    if xp is None:
                        val = next_reward
                    else:
                        val = next_reward + GAMMA * tiles.value(xp)
                    if val > best_value:
                        best_act = [action]
                        best_value = val
                    elif val == best_value:
                        best_act += [action]
                tiles.update(x, best_value)
                next_reward = model.take_action(choice(best_act))
                ep_rew += next_reward
                x = model.state()
            fout.write(str(ep_rew) + '\t' + str(i) + '\n')
            # eps_rewards += [ep_rew]
            # eps_endings += [i]
    # return eps_rewards, eps_endings
    return tiles


if __name__ == '__main__':
    value_function_estimation('../test.txt')
