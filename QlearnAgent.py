import numpy as np
import pickle
from Env import Env


class QlearnAgent:
    def __init__(self, env):
        # 学习率
        self.alpha = 0.1
        # 奖励衰减系数
        self.gamma = 0.9
        # epsilon贪心
        self.epsilon = 0.05
        # 动作维度
        self.action_dim = env.action_dim
        # Q表
        self.Q = np.zeros((2, 2, 2, 2, 2, 2, 2, 10, 10, 4))

    def take_action(self, state, evaluate = False):
        # 评测时就不使用epsilon贪心策略了
        if evaluate:
            return self.Q[state].argmax()
        t = np.random.random()
        if t < self.epsilon:
            return np.random.choice(np.arange(self.action_dim))
        else:
            return self.Q[state].argmax()

    def learn(self, s0, a0, s1, r):
        indx0 = tuple(list(s0) + [a0])
        self.Q[indx0] = self.Q[indx0] + self.alpha * (r + self.gamma * self.Q[s1].max() - self.Q[indx0])

    def save(self, path):
        # 保存参数
        pickle.dump(self.Q, open(path, 'wb'))

    def load(self, path):
        # 读取参数
        self.Q = pickle.load(open(path, 'rb'))

if __name__ == '__main__':
    episodes = 60000

    env = Env()
    agent = QlearnAgent(env)
    Q = agent.Q.copy()
    reward_list = []
    for episode in range(1, episodes + 1):
        state0 = env.reset()
        totalReward = 0
        while True:
            action = agent.take_action(state0)
            state1, reward, done = env.step(action)
            totalReward += reward
            agent.learn(state0, action, state1, reward)
            if done:
                break
            state0 = state1

        if episode % 100 == 0:
            print("episode %d, reward %d" % (episode, totalReward))
            reward_list.append(totalReward)

            # Q表误差0.1之内
            if abs(Q - agent.Q).sum() < 0.1:
                agent.save('Qlearn.pkl')
                break

            Q = agent.Q.copy()

    # 由于Qleanring是异策略算法， 因此单独设置模块测试学习到的最优路径
    state0 = env.reset()
    while True:
        action = agent.take_action(state0, evaluate=True)
        state1, reward, done = env.step(action)
        if done:
            break
        state0 = state1
    env.printPath()

    import matplotlib.pyplot as plt
    plt.plot([i for i in range(len(reward_list))], reward_list)
    plt.xlabel("Iterations (x100)")
    plt.ylabel("Reward")
    plt.savefig('Qlearn收敛曲线.png')
    plt.show()