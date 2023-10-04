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

    def __init__(self, game, gamma=0.99, theta=0.5):
        """
        ε-Q 学习策略
        """
        # this calls the superclass constructor (does self.game = game)
        super().__init__(game)

        # ======== 初始化参数=========

        # 动作的数量
        self._action_num = len(game.actions)
        # 动作的index
        self._action_index = {action: index for action, index in zip(game.actions, range(0, self._action_num))}

        # 计算动作概率的缓存
        self._p_best = None
        self._init_p_array = None
        # ε , 试探的概率
        self._epsilon = 0  # TODO：可以优化为缓慢下降/波动
        self.update_epsilon(0.1)

        # 当前 state 下的最佳 action = self._best_action[state]
        self._best_action = {}
        # 动作价值函数
        # 动作的平均价值 = q_arr[state][0][self._action_index[action]]
        # 动作的累计次数 = q_arr[state][1][self._action_index[action]]
        q_arr = {}

        # TODO: 优化初始值
        for state in game.states:
            # 计算当前 state 的临时分数
            score = self.game.get_state_score(state)
            # 初始化最优动作 TODO
            self._best_action = {(1, 1, 1): (0, 1, 2), (1, 1, 2): (0, 1), (1, 1, 3): (0, 1), (1, 1, 4): (0, 1, 2),
                                 (1, 1, 5): (0, 1, 2), (1, 1, 6): (0, 1, 2), (1, 2, 2): (1, 2), (1, 2, 3): (0,),
                                 (1, 2, 4): (0,), (1, 2, 5): (0,), (1, 2, 6): (0,), (1, 3, 3): (0,), (1, 3, 4): (0,),
                                 (1, 3, 5): (0,), (1, 3, 6): (0,), (1, 4, 4): (0,), (1, 4, 5): (0,), (1, 4, 6): (0,),
                                 (1, 5, 5): (0,), (1, 5, 6): (0,), (1, 6, 6): (0,), (2, 2, 2): (0, 1, 2),
                                 (2, 2, 3): (0, 1), (2, 2, 4): (0, 1, 2), (2, 2, 5): (0, 1, 2), (2, 2, 6): (0, 1, 2),
                                 (2, 3, 3): (0,), (2, 3, 4): (0,), (2, 3, 5): (0,), (2, 3, 6): (0,), (2, 4, 4): (0,),
                                 (2, 4, 5): (0,), (2, 4, 6): (0,), (2, 5, 5): (0,), (2, 5, 6): (0, 1, 2),
                                 (2, 6, 6): (0,), (3, 3, 3): (), (3, 3, 4): (), (3, 3, 5): (0, 1, 2),
                                 (3, 3, 6): (0, 1, 2), (3, 4, 4): (), (3, 4, 5): (), (3, 4, 6): (0, 1, 2),
                                 (3, 5, 5): (), (3, 5, 6): (0, 1, 2), (3, 6, 6): (), (4, 4, 4): (), (4, 4, 5): (),
                                 (4, 4, 6): (), (4, 5, 5): (), (4, 5, 6): (0, 1, 2), (4, 6, 6): (), (5, 5, 5): (),
                                 (5, 5, 6): (0, 2), (5, 6, 6): (0, 1), (6, 6, 6): ()}
            # if score > 9:
            #     self._best_action[state] = (0, 1, 2)
            # else:
            #     self._best_action[state] = ()
            # 动作的平均价值 TODO
            actions_val = np.zeros(len(game.actions), dtype=float)
            # actions_val[7] = score
            # 动作的累计次数
            actions_num = np.zeros(len(game.actions), dtype=int)
            # actions_num[7] = 1
            q_arr[state] = [actions_val, actions_num]

        # ================== 训练模型 ========================= #

        delta_max = theta + 1  # initialize to be over theta treshold
        for i in range(10000):

            # 幕（状态、动作列表）
            state_list = []
            action_list = []
            # 初始化游戏状态
            state = game.reset()
            game_over = False
            # 记录最后一幕的reward。当然也可以弄个 reward list，但是由于中间步骤的 reward 都是 -1 ，所以只用记录最后一个
            reward = 0
            # 根据策略生成一个幕
            while not game_over:
                action = self.play(state)
                # TODO: 参数化12
                # 剪枝，因为重置 8 次以上基本没啥好分数了。8 是个超参数
                if len(action_list) > 8:
                    # 强制结束游戏
                    action = (0, 1, 2)

                reward, state, game_over = self.game.roll(action)
                # 记录幕
                state_list.append(state)
                action_list.append(action)

            # 策略评估
            # 访问记录。只更新该幕中首次访问的状态-动作
            visit_history = []
            # 逆序遍历幕 TODO:debug
            for step in range(len(state_list) - 1, -1, -1):
                state = state_list[step]
                action = action_list[step]

                # 只更新首次访问
                if (state, action) in visit_history:
                    reward = gamma * reward - 1
                    continue
                visit_history.append((state, action))

                # 动作的累计次数
                q_arr[state][1][self._action_index[action]] += 1
                num_a = q_arr[state][1][self._action_index[action]]
                # 动作的平均价值
                pre_a = q_arr[state][0][self._action_index[action]]
                q_arr[state][0][self._action_index[action]] += (reward - q_arr[state][0][
                    self._action_index[action]]) / num_a

                # 更新下一个状态的未来期望收益 G_t
                # TODO：把 -1 变成 - self.game._penalty
                reward = gamma * reward - 1

                # 策略改进:取平均值最大的 action 作为 best action
                max_index = np.argmax(q_arr[state][0])
                self._best_action[state] = self.game.actions[max_index]

                # TODO：怎么终止循环呢
                # 一次幕中如果状态-动作的平均价值的最大变化小于theta，就终止循环
                delta_max = max(delta_max, abs(pre_a - q_arr[state][0][self._action_index[action]]))

        # print('finish')
        # # 打开文件
        # f = open("best_action.txt", "w")
        # # 写入列表中的每一个元素
        # f.write(str(self._best_action))
        # # 关闭文件
        # f.close()
        # # 打开文件
        # f = open("q_arr.txt", "w")
        # # 写入列表中的每一个元素
        # f.write(str(q_arr))
        # # 关闭文件
        # f.close()

    def play(self, state):
        """
        动作策略函数
        :param state: 当前状态
        :return: agent的动作
        """
        best_index = self._action_index[self._best_action[state]]
        p_array = self._init_p_array.copy()
        p_array[best_index] = self._p_best

        # 按照 epsilon 贪心策略选择一个动作
        return random.choices(self.game.actions, p_array, k=1)[0]

    def update_epsilon(self, e):
        # 更新 epsilon 的值
        self._epsilon = e

        # 重新计算概率
        # 最佳动作的概率
        p_best = 1 - self._epsilon * (1 - 1 / self._action_num)
        # 其他动作的概率
        p_others = self._epsilon / self._action_num
        # 初始化动作的概率数组
        init_p_array = np.linspace(p_others, p_others, self._action_num, endpoint=True, dtype=float)
        # 更新缓存
        self._p_best = p_best
        self._init_p_array = init_p_array


# 用来测试
if __name__ == "__main__":
    game = DiceGame()
    agent = MyAgent(game)
    # print('训练完成')
    # state = game.reset()
    # action = agent.play(state)
