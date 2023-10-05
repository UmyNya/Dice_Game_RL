"""
最优价值迭代
"""
import time

import numpy as np
from matplotlib import pyplot as plt

from agents.dice_game_agent import DiceGameAgent, get_next_states_cached, play_game_with_agent
from dice_game import DiceGame


class MyAgent(DiceGameAgent):
    """
    使用最优价值迭代实现的 agent
    """

    def __init__(self, game, gamma=0.96, theta=0.1):
        """
        最优价值迭代
        用最优贝尔曼方程迭代近似状态价值函数。
        动作策略是选择一个使得价值最大的状态（贪心策略）。
        """
        super().__init__(game)

        local_cache = {}
        v_arr = {}
        policy = {}

        # 初始化状态
        for state in game.states:
            v_arr[state] = 0
            policy[state] = ()

        # 学习最优价值（价值迭代）
        delta_max = theta + 1  # 初始化 delta 使得比阈值 theta 大
        while delta_max >= theta:
            delta_max = 0
            for state in game.states:
                s_val = v_arr[state]
                max_action = 0
                for action in game.actions:
                    # 评估 （state，action）的未来期望收益 s1_sum （价值评估）
                    s1_sum = 0
                    states, game_over, reward, probabilities = get_next_states_cached(game, local_cache, action, state)
                    for s1, p1 in zip(states, probabilities):
                        if not game_over:
                            s1_sum += p1 * (reward + gamma * v_arr[s1])
                        else:
                            s1_sum += p1 * (reward + gamma * game.final_score(state))

                    # 选取未来期望收益最大的动作作为最优动作（策略改进）
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

    print(
        "10x20 -[gamma = {:0.3f}, theta = {:0.3f}] Overall AVG score {:0.3f} and AVG time {:0.3f}".format(g, t, np.mean(
            scores), np.mean(times)))
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
    hyper_tuning()
