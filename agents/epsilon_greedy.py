"""
ε-贪心策略
"""
from agents.dice_game_agent import DiceGameAgent, get_next_states_cached


class MyAgent(DiceGameAgent):
    """
    使用ε-贪心策略实现的 agent
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
