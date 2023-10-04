"""
���� �� ���Եĵ��״η����� MC �����㷨
"""
import numpy as np
import random

from agents.dice_game_agent import DiceGameAgent, get_next_states_cached
from dice_game import DiceGame


class MyAgent(DiceGameAgent):
    """
    ʹ�æ�-̰�Ĳ���ʵ�ֵ� agent
    """

    def __init__(self, game, gamma=0.96, theta=0.1):
        """
        ��-Q ѧϰ����
        """
        # this calls the superclass constructor (does self.game = game)
        super().__init__(game)

        # ======== ��ʼ������=========

        # �� , ��̽�ĸ���
        self._epsilon = 0.5  # TODO�������Ż�Ϊ�����½�/����

        # ����������
        self._action_num = len(game.actions)
        # ������index
        self._action_index = {action: index for action, index in zip(game.actions, range(0, self._action_num))}

        # self._best_action[state] = ��ǰ state �µ���� action
        self._best_action = {}
        # ������ֵ����
        # q_arr[state][0][self._action_index[action]] = ������ƽ����ֵ
        # q_arr[state][1][self._action_index[action]] = �������ۼƴ���
        q_arr = {}

        for state in game.states:
            self._best_action[state] = (0, 1, 2)  # TODO �����Ż�
            q_arr[state] = [np.zeros(len(game.actions), dtype=float),
                            np.zeros(len(game.actions), dtype=int)]  # TODO �����Ż�

        # ================== ѵ��ģ�� ========================= #

        local_cache = {}

        delta_max = theta + 1  # initialize to be over theta treshold
        while delta_max >= theta:
            print(self._best_action)
            # Ļ��״̬�������б�
            state_list = []
            action_list = []
            # ��ʼ����Ϸ״̬
            state = game.reset()
            game_over = False
            # ���ݲ�������һ��Ļ
            while not game_over and len(action_list) <= 10:
                action = self.play(state)
                # TODO: ������12
                # len(action_list) < 12 �Ǽ�֦����Ϊ���� 12 �����ϻ���ûɶ�÷����ˡ�12 �Ǹ�������
                if len(action_list) > 12:
                    action = (0, 1, 2)

                _, state, game_over = self.game.roll(action)
                # ��¼Ļ
                action_list.append(action)
                state_list.append(state)

            # ��Ļ�ĵ÷�
            score = self.game.score

            # ��������
            # δ����������
            reward = score
            # �������Ļ
            for step in range(len(state_list) - 1, -1, -1):
                # TODO: ֻ�����״η��ʵ�
                state = state_list[step]
                action = action_list[step]
                # �������ۼƴ���
                q_arr[state][1][self._action_index[action]] += 1
                num_a = q_arr[state][1][self._action_index[action]]
                # ������ƽ����ֵ
                # TODO: ������ TD �㷨�Ż� https://www.cnblogs.com/xiaohuiduan/p/12977830.html
                pre_a = q_arr[state][0][self._action_index[action]]
                q_arr[state][0][self._action_index[action]] += (reward - q_arr[state][0][
                    self._action_index[action]]) / num_a

                # ������һ��״̬�� g
                # TODO���� -1 ��� - self.game._penalty
                reward = gamma * reward - 1

                # ���ԸĽ�: ��� action ��ƽ��ֵ�� best action �ã�����Ϊ�µ� best action
                if q_arr[state][0][self._action_index[action]] > q_arr[state][0][
                    self._action_index[self._best_action[state]]]:
                    self._best_action[state] = action

                # TODO����ô��ֹѭ����
                # һ��Ļ�����״̬-������ƽ����ֵ�����仯С��theta������ֹѭ��
                delta_max = max(delta_max, abs(pre_a - q_arr[state][0][self._action_index[action]]))

    def play(self, state):
        """
        �������Ժ���
        :param state: ��ǰ״̬
        :return: agent�Ķ���
        """
        # ��Ƶ�ʽ��м�Ȩ
        # ��Ѷ����ĸ���
        p_best = 1 - self._epsilon * (1 - 1 / self._action_num)
        # ���������ĸ���
        p_others = self._epsilon / self._action_num

        # TODO: ������
        # ���ж����ĸ�������
        p_array = np.linspace(p_others, p_others, self._action_num, endpoint=True, dtype=float)
        best_index = self.game.actions.index(self._best_action[state])
        p_array[best_index] = p_best

        # ���� epsilon ̰�Ĳ���ѡ��һ������
        return random.choices(self.game.actions, p_array, k=1)[0]


# ��������
if __name__ == "__main__":
    game = DiceGame()
    agent = MyAgent(game)
    state = game.reset()
    action = agent.play(state)
