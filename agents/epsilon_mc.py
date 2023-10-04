"""
基于 ε 策略的的首次访问型 MC 控制算法
"""
import numpy as np
import random

from agents.dice_game_agent import DiceGameAgent, get_next_states_cached
from dice_game import DiceGame


class MyAgent(DiceGameAgent):
    """
    使用ε-贪心策略实现的 agent
    """

    def __init__(self, game, gamma=0.96, theta=0.1):
        """
        ε-Q 学习策略
        """
        # this calls the superclass constructor (does self.game = game)
        super().__init__(game)

        # ======== 初始化参数=========

        # ε , 试探的概率
        self._epsilon = 0.5  # TODO：可以优化为缓慢下降/波动

        # 动作的数量
        self._action_num = len(game.actions)
        # 动作的index
        self._action_index = {action: index for action, index in zip(game.actions, range(0, self._action_num))}

        # self._best_action[state] = 当前 state 下的最佳 action
        self._best_action = {}
        # 动作价值函数
        # q_arr[state][0][self._action_index[action]] = 动作的平均价值
        # q_arr[state][1][self._action_index[action]] = 动作的累计次数
        q_arr = {}

        for state in game.states:
            self._best_action[state] = (0, 1, 2)  # TODO 可以优化
            q_arr[state] = [np.zeros(len(game.actions), dtype=float),
                            np.zeros(len(game.actions), dtype=int)]  # TODO 可以优化

        # ================== 训练模型 ========================= #

        local_cache = {}

        delta_max = theta + 1  # initialize to be over theta treshold
        while delta_max >= theta:
            print(self._best_action)
            # 幕（状态、动作列表）
            state_list = []
            action_list = []
            # 初始化游戏状态
            state = game.reset()
            game_over = False
            # 根据策略生成一个幕
            while not game_over and len(action_list) <= 10:
                action = self.play(state)
                # TODO: 参数化12
                # len(action_list) < 12 是剪枝，因为重置 12 次以上基本没啥好分数了。12 是个超参数
                if len(action_list) > 12:
                    action = (0, 1, 2)

                _, state, game_over = self.game.roll(action)
                # 记录幕
                action_list.append(action)
                state_list.append(state)

            # 该幕的得分
            score = self.game.score

            # 策略评估
            # 未来期望收益
            reward = score
            # 逆序遍历幕
            for step in range(len(state_list) - 1, -1, -1):
                # TODO: 只更新首次访问的
                state = state_list[step]
                action = action_list[step]
                # 动作的累计次数
                q_arr[state][1][self._action_index[action]] += 1
                num_a = q_arr[state][1][self._action_index[action]]
                # 动作的平均价值
                # TODO: 可以用 TD 算法优化 https://www.cnblogs.com/xiaohuiduan/p/12977830.html
                pre_a = q_arr[state][0][self._action_index[action]]
                q_arr[state][0][self._action_index[action]] += (reward - q_arr[state][0][
                    self._action_index[action]]) / num_a

                # 更新下一个状态的 g
                # TODO：把 -1 变成 - self.game._penalty
                reward = gamma * reward - 1

                # 策略改进: 如果 action 的平均值比 best action 好，就作为新的 best action
                if q_arr[state][0][self._action_index[action]] > q_arr[state][0][
                    self._action_index[self._best_action[state]]]:
                    self._best_action[state] = action

                # TODO：怎么终止循环呢
                # 一次幕中如果状态-动作的平均价值的最大变化小于theta，就终止循环
                delta_max = max(delta_max, abs(pre_a - q_arr[state][0][self._action_index[action]]))

    def play(self, state):
        """
        动作策略函数
        :param state: 当前状态
        :return: agent的动作
        """
        # 对频率进行加权
        # 最佳动作的概率
        p_best = 1 - self._epsilon * (1 - 1 / self._action_num)
        # 其他动作的概率
        p_others = self._epsilon / self._action_num

        # TODO: 做缓存
        # 所有动作的概率数组
        p_array = np.linspace(p_others, p_others, self._action_num, endpoint=True, dtype=float)
        best_index = self.game.actions.index(self._best_action[state])
        p_array[best_index] = p_best

        # 按照 epsilon 贪心策略选择一个动作
        return random.choices(self.game.actions, p_array, k=1)[0]


# 用来测试
if __name__ == "__main__":
    game = DiceGame()
    agent = MyAgent(game)
    state = game.reset()
    action = agent.play(state)
