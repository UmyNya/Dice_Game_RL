"""
骰子游戏
参考了 https://github.com/andrejlukic/dicegame-solver，更新了版本不兼容的代码和增加了注释
"""

from functools import partial
from scipy.stats import multinomial

import numpy as np
import itertools


class DiceGame:
    def __init__(self, dice=3, sides=6, *, values=None, bias=None, penalty=1):
        # 骰子数
        self._dice = dice
        # 每个骰子的面数
        self._sides = sides
        # 重新投掷骰子的惩罚（扣 penalty 分/点数）
        self._penalty = penalty
        # 骰子的每个面数上值。默认初始化为[1,sides+1)。或者接受一个长度和 sides 相同的数组
        if values is None:
            self._values = np.arange(1, self._sides + 1)
        else:
            if len(values) != sides:
                raise ValueError("Length of values must equal sides")
            self._values = np.array(values)
        # 骰子各个面的概率，默认平均
        if bias is None:
            self._bias = np.ones(self._sides) / self._sides
        else:
            # 错误检查，至少要保证bias长度等于sides（即values)
            if len(self._values) != len(self._bias):
                raise ValueError("Dice values and biases must be equal length")
            self._bias = np.array(bias)

        # 骰子点数的翻转数值。用于在计算 final_score 时翻转相同的点数。
        # 初始化为 self._flip = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}，[::-1]是切片操作符，表示逆序访问。
        self._flip = {a: b for a, b in zip(self._values, self._values[::-1])}

        # 投掷骰子时选择要保留的点数.()表示不保留点数。(0，2) 表示保留序号为 0 和 2 的骰子的点数
        # 3d6 的话，self.actions = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
        self.actions = []
        for i in range(0, self._dice + 1):
            self.actions.extend(itertools.combinations(range(0, self._dice), i))

        # 骰子的所有可能的组合。3个6面骰子就是从(1,1,1)到(6,6,6）
        # combinations_with_replacement 第一个参数是值域，第二个参数是元组长度。产生有序，元素可重复的元组。
        # 详见 https://docs.python.org/zh-cn/3/library/itertools.html
        self.states = [a for a in itertools.combinations_with_replacement(self._values, self._dice)]

        # 记录每个状态的最终得分
        self.final_scores = {state: self.final_score(state) for state in self.states}

        # 重置骰子
        # self.reset()
        self._current_dice = np.zeros(self._dice, dtype=np.int_)
        self.score = 0
        self._game_over = False

    def reset(self):
        """
        重置骰子
        :return:
        """
        self._game_over = False
        self.score = self._penalty
        self._current_dice = np.zeros(self._dice, dtype=int)
        _, dice, _ = self.roll()
        return dice

    def final_score(self, dice):
        """
        计算最终得分。
        :param dice: tuple 投出的点数的元组(即一个状态 state）。
        :return: int 返回一个得分
        """
        # 统计有多少点数相同的骰子。根据游戏规则，点数相同的骰子要翻转点数。eg. (2,2,1) -> (5,5,1)
        uniques, counts = np.unique(dice, return_counts=True)
        uniques[counts > 1] = np.array([self._flip[x] for x in uniques[counts > 1]])
        return np.sum(uniques[counts == 1]) + np.sum(uniques[counts > 1] * counts[counts > 1])

    def flip_duplicates(self):
        """
        将 self._current_dice 中重复的点数翻转，并重新排序
        :return:
        """
        uniques, counts = np.unique(self._current_dice, return_counts=True)
        if np.any(counts > 1):
            # \ 表示换行。这里是用 self._flip 中对应值（翻转的点数）替换重复的点数。
            self._current_dice[np.isin(self._current_dice, uniques[counts > 1])] = \
                [self._flip[x] for x in self._current_dice[np.isin(self._current_dice, uniques[counts > 1])]]
        self._current_dice.sort()

    def roll(self, hold=()):
        """
        投掷骰子
        :param hold: tuple 选择要保留的点数。在 self.actions 中选择
        :return: int, tuple, bool 本轮得分，投出的点数（状态），是否 game over。 本轮得分如果重投扣惩罚分数，如果保留所有骰子则游戏结束获得所有点数之和。
        """
        if hold not in self.actions:
            raise ValueError("hold must be a valid tuple of dice indices")

        if self._game_over:
            return 0, (), True

        count = len(hold)
        if count == self._dice:
            # 保留所有骰子的点数，游戏结束
            self.flip_duplicates()
            self.score += np.sum(self._current_dice)
            return np.sum(self._current_dice), self.get_dice_state(), True
        else:
            # 只保留一部分骰子的点数，剩下骰子重新投掷
            mask = np.ones(self._dice, dtype=bool)
            hold = np.array(hold, dtype=int)
            mask[hold] = False
            # 对不保留的点数重新投掷
            self._current_dice[mask] = np.random.choice(self._values, self._dice - count,
                                                        p=self._bias, replace=True)
            # 对新的结果进行排序
            self._current_dice.sort()
            # 得分减去重新投掷的惩罚
            self.score -= self._penalty
            return -1 * self._penalty, self.get_dice_state(), False

    def get_dice_state(self):
        return tuple(self._current_dice)

    def get_next_states(self, action, dice_state):
        """
        获取当前 dice_state 下做出 action 可能的所有 next_state
        Get all possible results of taking an action from a given state.

        :param action: the action taken
        :param dice_state: the current dice
        :return: state, game_over, reward, probabilities
                 state:
                    a list containing each possible resulting state as a tuple,
                    or a list containing None if it is game_over, to indicate
                    the terminal state
                 game_over:
                    a Boolean indicating if all dice were held
                 reward:
                    the reward for this action, equal to the final value of the
                    dice if game_over, otherwise equal to -1 * penalty
                 probabilities:
                    a list of size equal to state containing the probability of
                    each state occurring from this action
        """
        if action not in self.actions:
            raise ValueError("action must be a valid tuple of dice indices")
        if dice_state not in self.states:
            raise ValueError("state must be a valid tuple of dice values")

        count = len(action)
        if count == self._dice:
            # 保留所有点数，游戏结束
            return [None], True, self.final_score(dice_state), np.array([1])
        else:
            # first, build a mask (array of True/False) to indicate which values are held
            mask = np.zeros(self._dice, dtype=bool)
            hold = np.array(action, dtype=int)
            mask[hold] = True

            # get all possible combinations of values for the non-held dice
            other_vals = np.array(list(itertools.combinations_with_replacement(self._values,
                                                                               self._dice - count)),
                                  dtype=int)

            # in v1, dice only went from 1 to n
            # now dice can have any values, but values don't matter for probability, so get same data with 0 to n-1
            other_index = np.array(list(itertools.combinations_with_replacement(range(self._sides),
                                                                                self._dice - count)),
                                   dtype=int)

            # other_index will look like this, a numpy array of combinations
            #   [[0, 0], [0, 1], ..., [5, 5]]
            # need to calculate the probability of each one, so will query a multinomial distribution
            # if dice show (1, 3) then the correct query format is index based: [1, 0, 1, 0, 0, 0]
            queries = np.apply_along_axis(partial(np.bincount, minlength=self._sides), 1, other_index)
            probabilities = multinomial.pmf(queries, self._dice - count, self._bias)

            other_vals = np.insert(other_vals, np.zeros(count, dtype=int),
                                   np.asarray(dice_state, dtype=int)[mask], axis=1)

            other_vals.sort(axis=1)

            other_vals = [tuple(x) for x in other_vals]

            return other_vals, False, -1 * self._penalty, probabilities


def main():
    print("Let's play the game!")
    game = DiceGame()
    while True:
        dice = game.reset()
        print(f"Your dices are {dice}")
        print(f"Your score is {game.score}")
        while True:
            try:
                print("Type which dice you want to hold separated by spaces indexed from 0, blank to reroll all")
                print("Hold all dice to stick and get your final score (input \"0 1 2\")")
                holds = input(">")
                if holds == "":
                    holds = tuple()
                else:
                    holds = tuple(map(int, holds.split(" ")))
                reward, dice, game_over = game.roll(holds)
                if game_over:
                    print(f"Your final dices are {dice} (after flipping)")
                    print(f"Your final score is {game.score}")
                    break
                else:
                    print(f"Your dices are {dice}")
                    print(f"Your score is {game.score}")
            except KeyboardInterrupt:
                return
            # TODO(zzj)：异常太广泛，有空修修
            except:
                continue

        print("Play again? y/n")
        again = input(">")
        if again != "y":
            break


if __name__ == "__main__":
    main()
