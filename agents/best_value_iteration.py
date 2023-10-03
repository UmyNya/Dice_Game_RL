"""
最优价值迭代
"""
from agents.dice_game_agent import DiceGameAgent, get_next_states_cached


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
