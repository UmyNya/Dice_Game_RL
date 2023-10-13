import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd  # 导入pandas库

# 状态空间和行动空间的大小
n_states = 19  # 总得分可以从3到18
n_actions = 2  # 可以选择重新投骰子（行动0）或者保留当前分数（行动1）

# 初始化Q值表
Q = np.zeros((n_states, n_actions))

# 设置学习率和折扣因子，这里可调超参数
alpha = 0.5
gamma = 0.98

# 设置环境的奖励函数
R = np.zeros((n_states, n_actions))
R[:, 0] = -1  # 重新投骰子的奖励为-1

# 创建一个离散的概率分布来模拟每次投掷3个骰子的结果
dice_distribution = np.zeros(n_states)
for i in range(1, 7):
    for j in range(1, 7):
        for k in range(1, 7):
            dice_distribution[i + j + k] += 1
dice_distribution /= np.sum(dice_distribution)  # 归一化以得到概率分布

# 初始化总回报和游戏回合数
total_return = 0
n_episodes = 50000

epsilon = 0.1  # 设置ϵ的值

# 初始化存储平均总回报和所花费时间的列表
average_returns = []
elapsed_times = []

# Q-learning算法
start_time = time.time()
for episode in range(n_episodes):
    s = np.random.choice(n_states, p=dice_distribution)  # 根据概率分布选择初始状态
    done = False

    while not done:
        if np.random.rand() < epsilon:  # 以ϵ的概率进行随机探索
            a = np.random.randint(0, n_actions)
        else:  # 以1-ϵ的概率选择当前认为最好的行动
            a = np.argmax(Q[s, :])

        if a == 0:  # 如果选择重新投骰子
            s_next = np.random.choice(n_states, p=dice_distribution)  # 根据概率分布选择新的状态
        else:  # 如果选择保留当前分数
            R[s, a] = s  # 将当前总分作为奖励
            s_next = s
            done = True

        # 更新Q值
        Q[s, a] = (1 - alpha) * Q[s, a] + alpha * (R[s, a] + gamma * np.max(Q[s_next, :]))

        s = s_next

    total_return += R[s, a]  # 更新所有游戏回合的总回报

    if (episode + 1) % 1000 == 0:  # 每1000局结束时计算并打印所花费的时间
        average_return = total_return / (episode + 1)
        elapsed_time = time.time() - start_time
        print(f"{episode + 1} 次游戏后的平均总回报: {average_return}")
        print(f"前一个1000局游戏所花费的时间: {elapsed_time} 秒")

        # 将平均总回报和所花费的时间添加到对应的列表中
        average_returns.append(average_return)
        elapsed_times.append(elapsed_time)
        start_time = time.time()  # 更新start_time为当前时间

# 自动导入Excel文件
# df = pd.DataFrame({'alpha': alpha, 'gamma': gamma, '分数': average_returns, '时间（s）': elapsed_times})
# df.to_excel(f'alpha={alpha}_gamma={gamma}.xlsx', index=False)

# 所有游戏回合结束后，绘制图形
plt.rcParams['font.sans-serif'] = ['SimHei']   # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False   # 显示负号

plt.figure(1)   # 创建第一个图形窗口
plt.plot(range(1000, n_episodes + 1, 1000), average_returns)
plt.xlabel('n')
plt.ylabel('game score')
plt.title(f'Average game score of 1000 games(alpha={alpha},gamma={gamma})')
# plt.savefig(f'Qlearning_game_score(alpha={alpha},gamma={gamma}).png')

plt.figure(2)   # 创建第二个图形窗口
plt.plot(range(1000, n_episodes + 1, 1000), elapsed_times)
plt.xlabel('n')
plt.ylabel('time')
plt.title(f'Average time to converge(alpha={alpha},gamma={gamma})')
# plt.savefig(f'Qlearning_time(alpha={alpha},gamma={gamma}).png')

plt.show()
