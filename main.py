import numpy as np
import matplotlib.pyplot as plt
import time

from dice_game import DiceGame
from abc import ABC, abstractmethod

# 是否 Debug，是就输出调试信息
DBG = True


def log(msg):
    if DBG:
        print(msg)


class DiceGameAgent(ABC):
    def __init__(self, game):
        self.game = game

    @abstractmethod
    def play(self, state):
        pass


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


class MyAgent(DiceGameAgent):
    """
    使用最佳价值迭代实现的 agent
    """

    def __init__(self, game, gamma=0.96, theta=0.1):
        """Initializes the agent by performing a value iteration

        After the value iteration is run an optimal policy is returned. This
        policy instructs agent on what action to take in any possible state.
        """
        # this calls the superclass constructor (does self.game = game)
        super().__init__(game)

        # value iteration 最优价值迭代
        local_cache = {}
        v_arr = {}
        policy = {}
        for state in game.states:
            v_arr[state] = 0
            policy[state] = ()

        delta_max = theta + 1  # initialize to be over theta treshold
        while delta_max >= theta:
            delta_max = 0
            for state in game.states:
                s_val = v_arr[state]
                max_action = 0
                for action in game.actions:
                    s1_sum = 0
                    # states, game_over, reward, probabilities = game.get_next_states(action, state)
                    states, game_over, reward, probabilities = get_next_states_cached(game, local_cache, action, state)
                    for s1, p1 in zip(states, probabilities):
                        if not game_over:
                            s1_sum += p1 * (reward + gamma * v_arr[s1])
                        else:
                            s1_sum += p1 * (reward + gamma * game.final_score(state))
                    if s1_sum > max_action:
                        max_action = s1_sum
                        policy[state] = action
                v_arr[state] = max_action
                delta_max = max(delta_max, abs(s_val - v_arr[state]))

        self._policy = policy

    def play(self, state):
        """
        given a state, return the chosen action for this state
        at minimum you must support the basic rules: three six-sided fair dice

        if you want to support more rules, use the values inside self.game, e.g.
            the input state will be one of self.game.states
            you must return one of self.game.actions

        read the code in dicegame.py to learn more
        """
        # YOUR CODE HERE

        return self._policy[state]


def stats(basic=True, extended=True):
    """
    统计
    :param basic:
    :param extended:
    :return:
    """
    if basic:
        print("Testing basic rules.")
        print()

        scores = []
        times = []
        for _ in range(10):
            total_score = 0
            total_time = 0
            n = 1000

            np.random.seed()
            game = DiceGame()

            start_time = time.process_time()
            test_agent = MyAgent(game)
            total_time += time.process_time() - start_time

            for i in range(n):
                start_time = time.process_time()
                score = play_game_with_agent(test_agent, game)
                total_time += time.process_time() - start_time

                # print(f"Game {i} score: {score}")
                total_score += score

            scores.append(total_score / n)
            times.append(total_time)
            # print(f"Average score: {total_score / n}")
            # print(f"Total time: {total_time:.4f} seconds")
        print("10x1000 - Overall AVG score {} and AVG time {}".format(np.mean(scores), np.mean(times)))
        for s, t in zip(scores, times):
            print(round(s, 2), round(t, 2))

    if extended:

        print("Testing extended rules – two three-sided dice.")
        print()

        scores = []
        times = []
        for _ in range(10):
            total_score = 0
            total_time = 0
            n = 1000
            np.random.seed()
            game = DiceGame(dice=2, sides=3)

            start_time = time.process_time()
            test_agent = MyAgent(game)
            total_time += time.process_time() - start_time

            for i in range(n):
                start_time = time.process_time()
                score = play_game_with_agent(test_agent, game)
                total_time += time.process_time() - start_time
                total_score += score

            scores.append(total_score / n)
            times.append(total_time)
            # print(f"Average score: {total_score / n}")
            # print(f"Average time: {total_time / n:.5f} seconds")
        print("10x1000 - Overall AVG score {} and AVG time {}".format(np.mean(scores), np.mean(times)))
        for s, t in zip(scores, times):
            print(round(s, 2), round(t, 2))

        print("Testing extended rules – six six-sided dice.")
        print()

        scores = []
        times = []
        for _ in range(1):
            total_score = 0
            total_time = 0
            n = 1000
            np.random.seed()
            game = DiceGame(dice=6, sides=6)

            start_time = time.process_time()
            test_agent = MyAgent(game)
            total_time += time.process_time() - start_time

            for i in range(n):
                start_time = time.process_time()
                score = play_game_with_agent(test_agent, game)
                total_time += time.process_time() - start_time
                total_score += score

            scores.append(total_score / n)
            times.append(total_time)
            # print(f"Average score: {total_score / n}")
            # print(f"Average time: {total_time / n:.5f} seconds")
        print("10x1000 - Overall AVG score {} and AVG time {}".format(np.mean(scores), np.mean(times)))
        for s, t in zip(scores, times):
            print(round(s, 2), round(t, 2))


def hyper_tuning():
    """
    超参数调试。绘制超参数相关图表。
    :return:
    """
    print("Tuning parameters gamma and theta")
    print()

    # test_gamma = np.arange(0.88, 0.999, 0.002)
    test_gamma = [0.95]
    test_theta = np.arange(1, 50, 1)

    scores = []
    times = []
    for g in test_gamma:
        for t in test_theta:

            total_score = 0
            total_time = 0
            n = 1000

            np.random.seed()
            game = DiceGame()

            start_time = time.process_time()
            test_agent = MyAgent(game, gamma=g, theta=t)
            total_time += time.process_time() - start_time

            for i in range(n):
                start_time = time.process_time()
                score = play_game_with_agent(test_agent, game)
                total_time += time.process_time() - start_time
                total_score += score

            scores.append(total_score / n)
            times.append(total_time)

    # for g,s,t in zip(test_gamma,scores,times):
    for g, s, t in zip(test_theta, scores, times):
        print(round(g, 3), round(s, 3), round(t, 3))

    print("10x20 -[gamma = {:0.3f}, theta = {:0.3f}] Overall AVG score {:0.3f} and AVG time {:0.3f}".format(g, t,
                                                                                                            np.mean(
                                                                                                                scores),
                                                                                                            np.mean(
                                                                                                                times)))
    # plt.plot(test_gamma, scores)
    # plt.xlabel('gamma')
    plt.plot(test_theta, scores)
    plt.xlabel('theta')
    plt.ylabel('game score')
    plt.title('Average game score of 1000 games')
    plt.grid(True)
    # plt.savefig("gamma_tunning.png")
    plt.savefig("theta_tunning.png")
    plt.show()

    # plt.plot(test_gamma, times)
    # plt.xlabel('gamma')
    plt.plot(test_theta, times)
    plt.xlabel('theta')
    plt.ylabel('time')
    plt.title('Average time to converge')
    plt.grid(True)
    plt.savefig("gamma_tunning_times.png")
    plt.show()


if __name__ == "__main__":
    stats(basic=True, extended=True)
    # hyper_tuning()
