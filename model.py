from random import random
from math import cos

_limits = (-1.2, 0.5)
_speed_limit = 0.07
_time_step = 0.001
_gravity = 0.0025
_compression = 3


class MountainCar:
    def __init__(self, rand_assign = True, pos = -0.5, vel = 0):
        if rand_assign:
            self._position = random() * (_limits[1] - _limits[0]) + _limits[0]
            self._velocity = random() * _speed_limit * 2 - _speed_limit
        else:
            self._position = pos
            self._velocity = vel

    # noinspection PyMethodMayBeStatic
    def _reward(self):
        return -1

    def take_action(self, action):
        if action not in (-1, 0, 1):
            raise ValueError('Invalid action')
        self._velocity += action * _time_step - _gravity * cos(_compression * self._position)
        if self._velocity < -_speed_limit:
            self._velocity = -_speed_limit
        elif self._velocity > _speed_limit:
            self._velocity = _speed_limit

        self._position += self._velocity
        if self._position < _limits[0]:
            self._position = _limits[0]
        elif self._position > _limits[1]:
            self._position = _limits[1]

        if self._position == _limits[0] and self._velocity < 0:
            self._velocity = 0
        return self._reward()

    def check_action(self, action):
        if action not in (-1, 0, 1):
            raise ValueError('Invalid action')
        old_pos, old_vel = self._position, self._velocity
        rew = self.take_action(action)
        end = self.terminated()
        pos, vel, self._position, self._velocity = self._position, self._velocity, old_pos, old_vel
        if end:
            return None, rew
        else:
            return (pos, vel), rew

    def state(self):
        return self._position, self._velocity

    def terminated(self):
        return self._position >= _limits[1]
