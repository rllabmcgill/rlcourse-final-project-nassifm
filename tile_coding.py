from math import inf
from random import choice
from ast import literal_eval

try:
    from graphics import *
except:
    print('cannot import graphics')

UPDATE_LIMIT = 50
ALPHA = 0.1
OLD_TILES = False


class ParseError(Exception):
    pass


def read_tiles(file, limits):
    with open(file) as fin:
        tiles = TileCoding(limits)
        known_nodes = {}
        for line in fin:
            try:
                n_id, p, n_type, extra = line.split('\t', 3)
            except ValueError as e:
                print(line)
                raise e
            if p == 'root':
                parent = None
            elif p[-1] == 'l':
                parent = known_nodes[p[0:-1]]
            elif p[-1] == 'r':
                parent = known_nodes[p[0:-1]]
            else:
                raise ParseError('Invalid parent code')
            if n_type == 'c':
                ft, thres = extra.split('\t')
                ft = int(ft)
                thres = float(thres)
                node = Condition(thres, ft, parent)
            elif n_type == 't':
                node = NATile(parent, len(limits), value = extra)
            elif n_type == 'ot':
                val, lims = extra.split('\t', 1)
                val = float(val)
                lims = literal_eval(lims)
                node = ATile(parent, lims, val)
            else:
                raise ParseError('Unknown node type')
            known_nodes[n_id] = node
            if p == 'root':
                tiles._root = node
            elif p[-1] == 'l':
                parent.set_children(left = node)
            elif p[-1] == 'r':
                parent.set_children(right = node)
                del known_nodes[p[0:-1]]
    return tiles


class TileCoding:
    def __init__(self, limits, split: list = 0.5):
        dims = len(limits)
        if dims <= 0:
            raise ValueError('Limits must be of the same length as the number of dimensions.')
        try:
            if len(split) != dims:
                raise ValueError('Length of splits does not match length of limits.')
        except TypeError:
            split = [split] * dims
        prevs = []
        for i in range(dims):
            s = split[i]
            l = limits[i]
            t = (1 - s) * l[0] + s * l[1]
            if len(prevs) == 0:
                self._root = Condition(t, i, None)
                prevs = [self._root]
            else:
                n_prevs = []
                for prev in prevs:
                    copy1 = Condition(t, i, prev)
                    copy2 = Condition(t, i, prev)
                    prev.set_children(copy1, copy2)
                    n_prevs += [copy1, copy2]
                prevs = n_prevs
        if OLD_TILES:
            self._init_old(self._root, limits)
        else:
            for prev in prevs:
                prev.set_children(NATile(prev, dims), NATile(prev, dims))

    def _init_old(self, cond, lims):
        left, right = cond.get_children()
        t = cond.threshold()
        i = cond.feature()
        ll = [lim for lim in lims]
        ll[i] = (ll[i][0], t)
        lr = [lim for lim in lims]
        lr[i] = (t, lr[i][1])
        if left is None:
            cond.set_children(left = ATile(cond, ll))
        else:
            self._init_old(left, ll)
        if right is None:
            cond.set_children(right = ATile(cond, lr))
        else:
            self._init_old(right, lr)

    def _get_tile(self, x):
        c = self._root
        while type(c) is Condition:
            if x[c.feature()] < c.threshold():
                c = c.get_children()[0]
            else:
                c = c.get_children()[1]
        return c

    def value(self, x):
        return self._get_tile(x).value()

    def update(self, x, expected_reward):
        tile = self._get_tile(x)
        tile.update(x, expected_reward)

    def _draw_condition(self, cond, win, lims = None, w = 1000, h = 1000):
        if lims is None:
            lims = ((0, w), (0, h))
        t = cond.threshold()
        if cond.feature() == 0:
            t += 1.2
            t *= w / 1.7
            t = int(t)
        else:
            t += 0.07
            t *= h / 0.14
            t = int(t)
        left, right = cond.get_children()
        if cond.feature() == 0:
            line = Line(Point(t, lims[1][0]), Point(t, lims[1][1]))
            line.draw(win)
            if type(left) is Condition:
                self._draw_condition(left, win, ((lims[0][0], t), lims[1]))
            if type(right) is Condition:
                self._draw_condition(right, win, ((t, lims[0][1]), lims[1]))
        else:
            line = Line(Point(lims[0][0], t), Point(lims[0][1], t))
            line.draw(win)
            if type(left) is Condition:
                self._draw_condition(left, win, (lims[0], (lims[1][0], t)))
            if type(right) is Condition:
                self._draw_condition(right, win, (lims[0], (t, lims[1][1])))

    def view(self):
        w = 1000
        h = 1000
        win = GraphWin('Tiles', w, h)
        self._draw_condition(self._root, win, lims = None, w = w, h = h)
        return win

    def _print_condition(self, cond, fout, i, parent):
        next_i = i + 1
        if type(cond) is Condition:
            fout.write('\t'.join([str(i), str(parent), 'c', str(cond.feature()), str(cond.threshold())]) + '\n')
            left, right = cond.get_children()
            next_i = self._print_condition(left, fout, next_i, str(i) + 'l')
            next_i = self._print_condition(right, fout, next_i, str(i) + 'r')
        elif type(cond) is NATile:
            fout.write('\t'.join([str(i), str(parent), 't', str(cond.value())]) + '\n')
        elif type(cond) is ATile:
            fout.write('\t'.join([str(i), str(parent), 'ot', str(cond.value()), str(cond._lims)]) + '\n')
        return next_i

    def print(self, file):
        with open(file, 'w') as fout:
            self._print_condition(self._root, fout, 0, 'root')


class Condition:
    def __init__(self, threshold, feature, parent):
        self._parent = parent
        self._threshold = threshold
        self._feature = feature
        self._left = None
        self._right = None

    def feature(self):
        return self._feature

    def threshold(self):
        return self._threshold

    def get_children(self):
        return self._left, self._right

    def set_children(self, left = None, right = None):
        if left is not None:
            self._left = left
        if right is not None:
            self._right = right

    # REMOVE FOR EVIL RECURSION LIMIT
    # def get_tile(self, x):
    #     if x[self._feature] < self._threshold:
    #         return self._left.get_tile(x)
    #     else:
    #         return self._right.get_tile(x)

    def parent(self):
        return self._parent


class ATile:
    def __init__(self, parent, lims, value = 0):
        self._parent = parent
        self._dims = len(lims)
        self._value = value
        self._lims = lims
        self._mid = [(lim[0] + lim[1]) / 2 for lim in lims]
        self._lvalues = self._rvalues = [value] * self._dims
        self._nb_updates = 0
        self._min_delta = inf

    def value(self):
        return self._value

    def parent(self):
        return self._parent

    def dims(self):
        return self._dims

    def _split(self):
        best_split = None
        delta = -inf
        for i in range(self._dims):
            d = abs(self._lvalues[i] - self._rvalues[i])
            if d > delta:
                best_split = [i]
            elif d == delta:
                best_split += [i]
        split = choice(best_split)
        c = Condition(self._mid[split], split, self._parent)
        l1 = [l for l in self._lims]
        l1[split] = (self._lims[split][0], self._mid[split])
        t1 = ATile(c, l1, self._lvalues[split])
        l2 = [l for l in self._lims]
        l2[split] = (self._mid[split], self._lims[split][1])
        t2 = ATile(c, l2, self._rvalues[split])
        c.set_children(t1, t2)
        if self is self._parent.get_children()[0]:
            self._parent.set_children(left = c)
        else:
            self._parent.set_children(right = c)

    def update(self, x, reward):
        # Update value
        delta = reward - self._value
        self._value += ALPHA * delta
        # Update subs
        for i in range(self._dims):
            if x[i] < self._mid[i]:
                self._lvalues[i] += ALPHA * (reward - self._lvalues[i])
            else:
                self._rvalues[i] += ALPHA * (reward - self._rvalues[i])
        # Check min delta
        if delta < self._min_delta:
            self._min_delta = delta
            self._nb_updates = 0
        else:
            self._nb_updates += 1
            if self._nb_updates >= UPDATE_LIMIT:
                self._split()


class NATile:
    def __init__(self, parent, dims, value = 0):
        self._dims = dims
        self._parent = parent
        self._value = value
        self._nb_updates = 0
        self._min_delta = inf
        self._min_rew = inf
        self._max_rew = -inf
        # (pos, weight)
        self._min = None  # [[0, 0]] * dims
        self._max = None  # [[0, 0]] * dims
        self._var = [0] * dims

    def value(self):
        return self._value

    def parent(self):
        return self._parent

    def dims(self):
        return self._dims

    # def get_tile(self, x):
    #     return self

    def _best_split(self):
        best_var = inf
        best_ft = None
        for i in range(self._dims):
            if self._var[i] < best_var:
                best_ft = [i]
                best_var = self._var[i]
            elif self._var[i] == best_var:
                best_ft += [i]
        best_ft = choice(best_ft)
        x = self._min[best_ft]
        y = self._max[best_ft]
        thres = (x[0] * x[1] + y[0] * y[1]) / (x[1] + y[1])
        return best_ft, thres

    def _split(self):
        ft, t = self._best_split()
        c = Condition(t, ft, self._parent)
        t1 = NATile(c, self._dims, self._value)
        t2 = NATile(c, self._dims, self._value)
        c.set_children(t1, t2)
        if self is self._parent.get_children()[0]:
            self._parent.set_children(left = c)
        else:
            self._parent.set_children(right = c)

    def _update_mid(self, x, reward):
        if self._max_rew == self._min_rew:
            return
        for i in range(self._dims):
            mins = self._min[i]
            maxs = self._max[i]
            ny = (reward - self._min_rew) / (self._max_rew - self._min_rew)
            if maxs[0] == mins[0]:
                if ny < 0.5:
                    r = 1
                elif ny == 0.5:
                    r = 0.5
                else:
                    r = 0
            else:
                nx = (x[i] - mins[0]) / (maxs[0] - mins[0])
                if ny > 0.5:
                    r = 0.5 * nx / ny
                else:
                    r = (nx - 1) / (ny - 1) * -0.5 + 1
            self._var[i] += (r - mins[1] / (mins[1] + maxs[1])) ** 2
            mins[1] += (1 - r)
            maxs[1] += r

    def update(self, x, reward):
        # Update split
        if self._max is None:
            self._max = self._min = [[pos, 0.5] for pos in x]
            self._min_rew = self._max_rew = reward
        elif reward > self._max_rew:
            omaxr = self._max_rew
            nmaxr = reward
            minr = self._min_rew
            for i in range(self._dims):
                old_max, self._max[i][0] = self._max[i][0], x[i]
                if self._min[i][0] < old_max < self._max[i][0] or self._min[i][0] > old_max > self._max[i][0]:
                    oweight = self._min[i][1] + self._max[i][1]
                    if nmaxr - minr > 2 * (omaxr - minr):
                        nx = (old_max - self._min[i][0]) / (self._max[i][0] - self._min[i][0])
                        ny = (omaxr - minr) / (nmaxr - minr)
                        r = (nx - 1) / (ny - 1) * -0.5 + 1
                        self._min[i][1] = (1 - r) * oweight
                        self._max[i][1] = r * oweight
                    else:
                        nx1 = (old_max - self._min[i][0]) / (self._max[i][0] - self._min[i][0])
                        # noinspection PyTypeChecker
                        nx0 = (self._min[i][0] * self._min[i][1] + old_max * self._max[i][1]) / oweight
                        ny1 = (omaxr - minr) / (nmaxr - minr)
                        ny0 = ny1 / 2
                        ry = (0.5 - ny0) / (ny1 - ny0)
                        rx = nx0 + ry * (nx1 - nx0)
                        self._min[i][1] = (1 - rx) * oweight
                        self._max[i][1] = rx * oweight
                else:
                    self._min[i][1] = self._max[i][1] = 0.5
                    self._var[i] = 0
        elif reward < self._min_rew:
            maxr = self._max_rew
            nminr = reward
            ominr = self._min_rew
            for i in range(self._dims):
                old_min, self._min[i][0] = self._min[i][0], x[i]
                if self._min[i][0] < old_min < self._max[i][0] or self._min[i][0] > old_min > self._max[i][0]:
                    oweight = self._min[i][1] + self._max[i][1]
                    if maxr - nminr > 2 * (maxr - ominr):
                        nx = (old_min - self._min[i][0]) / (self._max[i][0] - self._min[i][0])
                        ny = (ominr - nminr) / (maxr - nminr)
                        r = nx / ny * 0.5
                        self._min[i][1] = (1 - r) * oweight
                        self._max[i][1] = r * oweight
                    else:
                        nx0 = (old_min - self._min[i][0]) / (self._max[i][0] - self._min[i][0])
                        # noinspection PyTypeChecker
                        nx1 = (old_min * self._min[i][1] + self._max[i][0] * self._max[i][1]) / oweight
                        ny0 = (ominr - nminr) / (maxr - nminr)
                        ny1 = (ny0 + 1) / 2
                        ry = (0.5 - ny0) / (ny1 - ny0)
                        rx = nx0 + ry * (nx1 - nx0)
                        self._min[i][1] = (1 - rx) * oweight
                        self._max[i][1] = rx * oweight
                else:
                    self._min[i][1] = self._max[i][1] = 0.5
                    self._var[i] = 0
        else:
            self._update_mid(x, reward)
        # Update value
        delta = reward - self._value
        self._value += ALPHA * delta
        # Check min delta
        if delta < self._min_delta:
            self._min_delta = delta
            self._nb_updates = 0
        else:
            self._nb_updates += 1
            if self._nb_updates >= UPDATE_LIMIT:
                self._split()
