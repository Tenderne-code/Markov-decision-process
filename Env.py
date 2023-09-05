import numpy as np


class Env:
    def __init__(self):
        # 网格为10x10
        self.lengh = 10
        self.board = [[0 for j in range(self.lengh)] for i in range(self.lengh)]
        # 城市数量
        self.cityNum = 7
        # 定义城市
        self.board[2][1] = 1
        self.board[6][2] = 1
        self.board[2][3] = 1
        self.board[0][4] = 1
        self.board[3][5] = 1
        self.board[9][5] = 1
        self.board[7][7] = 1
        self.citys = [(2, 1), (6, 2),
                      (2, 3), (0, 4), (3, 5), (9, 5), (7, 7)]
        # 定义城市是否访问过
        self.cityState = dict()
        # 旅行商初始所在位置
        self.init_pos = (0, 0)
        # 旅行商的行走路径
        self.path = []
        # 状态维度
        self.state_dim = 2 ** self.cityNum * self.lengh * self.lengh
        # 动作维度
        self.action_dim = 4

    def action_space_sample(self):
        return np.random.choice(np.arange(self.action_dim))

    def getState(self):
        state_list = []
        for key in self.citys:
            state_list.append(self.cityState[key])
        state_list.append(self.pos[0])
        state_list.append(self.pos[1])
        return tuple(state_list)

    def reset(self):
        # 初始化位置(0,0), 路径为空
        self.pos = self.init_pos
        self.path = []
        # 所有城市设定为未访问过
        for key in self.citys:
            self.cityState[key] = 0
        self.cityState[(0, 0)] = 1
        return self.getState()

    def step(self, action):
        """
        :param action: 0,1,2,3 分别表示上下左右4个方向
        :return: state, reward, done 返回状态，奖励，是否游戏结束
        """
        dir = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dx, dy = dir[action]
        x, y = self.pos[0] + dx, self.pos[1] + dy
        if 0 <= x < self.lengh and 0 <= y < self.lengh:
            nx, ny = x, y
        else:
            # 出界了
            done = True
            reward = -1
            state = self.getState()
            return (state, reward, done)
        self.path.append((nx, ny))
        self.pos = (nx, ny)

        if self.board[self.pos[0]][self.pos[1]] == 1:
            if self.cityState[self.pos] == 1:
                done = True
                reward = -1
                state = self.getState()
                return (state, reward, done)
            else:
                self.cityState[self.pos] = 1

        # 检测是否访问全部城市并回到起点
        passNum = 0
        for key in self.citys:
            passNum += self.cityState[key]

        if passNum == self.cityNum and self.pos == self.init_pos:
            done = True
            reward = 10000
            state = self.getState()
            return (state, reward, done)

        state = self.getState()
        reward = -1
        if self.board[self.pos[0]][self.pos[1]] == 1:
            reward = 25
        done = False
        return (state, reward, done)

    def printPath(self):
        print("路径长度",len(self.path))
        print("路径","->".join(['(0, 0)']+list(map(str,self.path))))