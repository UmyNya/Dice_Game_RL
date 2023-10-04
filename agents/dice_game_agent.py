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


def play_game_with_agent(agent, game, verbose=False):
    """
    :param agent:
    :param game:
    :param verbose: bool 是否输出详细信息（冗余信息）
    :return:
    """
    state = game.reset()

    if verbose:
        print(f"Testing agent: \n\t{type(agent).__name__}")
    if verbose:
        print(f"Starting dice: \n\t{state}\n")

    game_over = False

    # agent 行动的计数器
    actions = 0

    while not game_over:
        action = agent.play(state)
        actions += 1

        if verbose:
            print(f"Action {actions}: \t{action}")
        _, state, game_over = game.roll(action)
        if verbose and not game_over:
            print(f"Dice: \t\t{state}")

    if verbose:
        print(f"\nFinal dice: {state}, score: {game.score}")

    return game.score
