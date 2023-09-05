import numpy as np
import pickle
from Env import Env

class PolicyGradientAgent:
    def __init__(self, env):
        # 学习率
        self.alpha = 0.001
        # 奖励衰减系数
        self.gamma = 0.9

        # 动作维度
        self.action_dim = env.action_dim

        # 策略参数
        self.w = np.random.rand(env.state_dim, env.action_dim)
        self.w /= self.w.sum(axis=1).reshape(-1, 1)

    def softmax(self,a):
        c = np.max(a)
        exp_a = np.exp(a - c)  # 溢出对策
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

    def cal_policy(self, state):
        # 将9维状态转化为1维的数
        i1 = int("".join(list(map(str, state[:7]))), 2)
        i2 = state[-2] * 10 + state[-1]
        probs = self.w[i1 * 100 + i2]
        # 防止上溢
        z = np.max(probs)
        probs = np.exp(probs - z)
        probs /= probs.sum()
        return probs

    def take_action(self, state):
        probs = self.cal_policy(state)
        action = np.random.choice(np.arange(self.action_dim), p=probs)
        return action

    def learn(self, states, actions, rewards):
        G = 0
        for i in reversed(range(len(rewards))):
            reward = rewards[i]
            state = states[i]
            action = actions[i]
            G = self.gamma * G + reward
            probs = self.cal_policy(state)
            i1 = int("".join(list(map(str, state[:7]))), 2)
            i2 = state[-2] * 10 + state[-1]
            self.w[i1 * 100 + i2][action] += self.alpha * G * (1 - probs[action])

    def save(self, path):
        # 保存参数
        pickle.dump(self.w, open(path, 'wb'))

    def load(self, path):
        # 读取参数
        self.w = pickle.load(open(path, 'rb'))

if __name__ == '__main__':
    episodes = 3000000

    env = Env()
    agent = PolicyGradientAgent(env)
    W = agent.w.copy()
    reward_list = []
    for episode in range(1, episodes + 1):
        state0 = env.reset()
        totalReward = 0
        states = [state0]
        actions = []
        rewards = []
        while True:
            action = agent.take_action(state0)
            actions.append(action)
            state1, reward, done = env.step(action)
            states.append(state1)
            rewards.append(reward)
            totalReward += reward
            if done:
                break
            state0 = state1

        # totalReward += reward
        agent.learn(states, actions, rewards)
        if episode % 1000 == 0:
            print(actions)
            print("episode %d, reward %d" % (episode, totalReward))
            reward_list.append(totalReward)

            sameNum = 0
            for r in reversed(reward_list):
                if r == totalReward:
                    sameNum += 1
                else:
                    break
            # 连续50次回报相等 就结束
            if sameNum == 50:
                agent.save('PG.pkl')
                env.printPath()
                break

    import matplotlib.pyplot as plt
    plt.plot([i for i in range(len(reward_list))], reward_list)
    plt.xlabel("Iterations (x1000)")
    plt.ylabel("Reward")
    plt.savefig('PG收敛曲线.png')
    plt.show()