import rl
import time
from math import fsum
from tile_coding import read_tiles


def compute(n, folder = None):
    if folder is None:
        folder = '../run/'
    for i in range(n):
        file = folder + 'rewards' + str(i) + '.txt'
        rl.value_function_estimation(file).print(folder + 'tiles' + str(i) + '.txt')


def interactive_test():
    tiles = None
    i = 0
    while True:
        reps = int(input('REPS = '))
        rl.REPS = reps
        tiles = rl.value_function_estimation('../test.txt', tiles = tiles)
        with open('../test.txt') as fin:
            lines = [int(line[0:line.find('\t')]) for line in fin]
            print((fsum(lines[0:500]) / 500, fsum(lines[-500:]) / 500))
        if input('Show view? ') == 'yes':
            win = tiles.view()
            input('Close view? ')
            win.close()
        if reps == 0:
            break
    tiles.print('../tiles_test.txt')


def naive_test():
    from model import MountainCar
    m = MountainCar()
    start = s = m.state()
    action = 1
    i = 0
    while not m.terminated():
        # print(s)
        i += 1
        if s[1] * action <= 0:
            action *= -1
        m.take_action(action)
        s = m.state()
    print((i, start))


def old_times():
    import tile_coding
    tile_coding.OLD_TILES = True


def read(file):
    tiles = read_tiles(file, ((-1.2, 0.5), (-0.07, 0.07)))
    win = tiles.view()
    win.getMouse()
    win.close()


if __name__ == '__main__':
    print(time.asctime())
