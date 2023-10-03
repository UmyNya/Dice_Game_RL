from abc import ABC, abstractmethod


class DiceGameAgent(ABC):
    def __init__(self, game):
        self.game = game

    @abstractmethod
    def play(self, state):
        pass


def get_next_states_cached(game, cache, action, state):
    """
    缓存 (action, state)，加速 get_next_states
    :param game:
    :param cache:
    :param action:
    :param state:
    :return:
    """
    if (action, state) not in cache:
        cache[(action, state)] = game.get_next_states(action, state)
    return cache[(action, state)]
