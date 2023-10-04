import time

import matplotlib.pyplot as plt
import numpy as np

from agents.dice_game_agent import play_game_with_agent
from dice_game import DiceGame

# 通过切换导入的 MyAgent 来完成不同策略的测试
# from agents.best_value_iteration import MyAgent
from agents.epsilon_mc import MyAgent

# 是否 Debug，是就输出调试信息
DBG = True


def log(msg):
    if DBG:
        print(msg)


def stats(basic=True, extended=False):
    """
    统计
    :param basic: bool 是否测试 3d6 场景
    :param extended: bool 是否测试其他场景（2d3,6d6)
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

        print("Testing extended rules - two three-sided dice.")
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

        print("Testing extended rules - six six-sided dice.")
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
    stats(basic=True, extended=False)
    # hyper_tuning()
