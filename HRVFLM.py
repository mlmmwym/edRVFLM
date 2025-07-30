import json

import numpy as np
import torch
from torch import nn
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score

def standardization(arr):
    mean = arr.mean()
    std = arr.std()
    arr = (arr - mean) / (std + 0.000001)
    return arr


def to_prob(x):
    return torch.exp(x) / sum(torch.exp(x))


# 多个矩阵顺序相乘
def mul_matmul(matList):
    M1 = matList[0]
    for i in range(1, len(matList)):
        M1 = torch.mm(M1, matList[i])
    return M1


def save_data(data, file):
    with open(file, 'w', encoding='utf8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)



class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hiddens = num_hiddens

        self.lin1 = nn.Linear(num_inputs, num_hiddens, bias=True)
        self.lin2 = nn.Linear(num_inputs + num_hiddens, num_hiddens, bias=True)
        self.lin3 = nn.Linear(num_inputs + num_hiddens, num_hiddens, bias=True)
        self.lin4 = nn.Linear(num_inputs + num_hiddens, num_hiddens, bias=True)
        self.lin5 = nn.Linear(num_inputs + num_hiddens, num_hiddens, bias=True)
        self.lin6 = nn.Linear(num_inputs + num_hiddens, num_hiddens, bias=True)
        self.lin7 = nn.Linear(num_inputs + num_hiddens, num_hiddens, bias=True)

        self.out0 = nn.Linear(num_inputs, num_outputs, bias=False)
        self.out0.weight = nn.Parameter(self.out0.weight * 0)

        self.out1 = nn.Linear(num_inputs + num_hiddens, num_outputs, bias=False)
        self.out1.weight = nn.Parameter(self.out1.weight * 0)

        self.out2 = nn.Linear(num_inputs + num_hiddens, num_outputs, bias=False)
        self.out2.weight = nn.Parameter(self.out2.weight * 0)

        self.out3 = nn.Linear(num_inputs + num_hiddens, num_outputs, bias=False)
        self.out3.weight = nn.Parameter(self.out3.weight * 0)

        self.out4 = nn.Linear(num_inputs + num_hiddens, num_outputs, bias=False)
        self.out4.weight = nn.Parameter(self.out4.weight * 0)

        self.out5 = nn.Linear(num_inputs + num_hiddens, num_outputs, bias=False)
        self.out5.weight = nn.Parameter(self.out5.weight * 0)

        self.out6 = nn.Linear(num_inputs + num_hiddens, num_outputs, bias=False)
        self.out6.weight = nn.Parameter(self.out6.weight * 0)

        self.out7 = nn.Linear(num_inputs + num_hiddens, num_outputs, bias=False)
        self.out7.weight = nn.Parameter(self.out7.weight * 0)

        self.activation = nn.ReLU()

        self.c = 1
        self.M0 = torch.eye(num_inputs) * self.c
        self.M1 = torch.eye(num_inputs + num_hiddens) * self.c
        self.M2 = torch.eye(num_inputs + num_hiddens) * self.c
        self.M3 = torch.eye(num_inputs + num_hiddens) * self.c
        self.M4 = torch.eye(num_inputs + num_hiddens) * self.c
        self.M5 = torch.eye(num_inputs + num_hiddens) * self.c
        self.M6 = torch.eye(num_inputs + num_hiddens) * self.c
        self.M7 = torch.eye(num_inputs + num_hiddens) * self.c

        self.out_list = [self.out0, self.out1, self.out2, self.out3, self.out4, self.out5, self.out6, self.out7]
        self.M_list = [self.M0, self.M1, self.M2, self.M3, self.M4, self.M5, self.M6, self.M7]

        self.r1 = 0.99
        self.r2 = 0.9995

        self.lmd_list = [self.r2] * 8

        self.weight_list = [1 / len(self.out_list)] * len(self.out_list)

        self.bate = 0.8

        self.smooth = 0.5

        self.loss = torch.nn.MSELoss(reduction='sum')

    def hidden_forward(self, X):
        with torch.no_grad():
            H1 = self.activation(self.lin1(X))
            H1 = standardization(H1)
            X = standardization(X)
            H1 = torch.cat([X, H1], dim=0)

            H2 = self.activation(self.lin2(H1))
            H2 = standardization(H2)
            H2 = torch.cat([X, H2], dim=0)

            H3 = self.activation(self.lin3(H2))
            H3 = standardization(H3)
            H3 = torch.cat([X, H3], dim=0)

            H4 = self.activation(self.lin4(H3))
            H4 = standardization(H4)
            H4 = torch.cat([X, H4], dim=0)

            H5 = self.activation(self.lin5(H4))
            H5 = standardization(H5)
            H5 = torch.cat([X, H5], dim=0)

            H6 = self.activation(self.lin6(H5))
            H6 = standardization(H6)
            H6 = torch.cat([X, H6], dim=0)

            H7 = self.activation(self.lin7(H6))
            H7 = standardization(H7)
            H7 = torch.cat([X, H7], dim=0)

            return [X, H1, H2, H3, H4, H5, H6, H7]

    def update_classify_weight(self, loss_list):
        for i in range(len(loss_list)):
            self.weight_list[i] = self.weight_list[i] * (self.bate ** loss_list[i])
            if self.weight_list[i] < self.smooth * (1 / len(loss_list)):
                self.weight_list[i] = self.smooth * (1 / len(loss_list))
        self.weight_list = [w / sum(self.weight_list) for w in self.weight_list]

    def update_forgetting_factor(self):
        alpha_min = min(self.weight_list)
        alpha_max = max(self.weight_list)

        for i in range(len(self.weight_list)):
            self.lmd_list[i] = self.r1 + (self.weight_list[i] - alpha_min + 0.000001) / (
                        alpha_max - alpha_min + 0.000001) * (self.r2 - self.r1)

    def online_training(self, X, y):
        with torch.no_grad():
            predict_list = []
            loss_list = []
            H_list = self.hidden_forward(X)

            for i in range(len(H_list)):
                y_hat = self.out_list[i](H_list[i])
                predict_list.append(y_hat)
                loss_list.append(self.loss(y_hat, y))
                H = H_list[i]
                H = H.reshape(-1, 1)
                self.M_list[i] = self.lmd_list[i] * self.M_list[i] + mul_matmul([H, H.T]) + torch.eye(H.shape[0]) * (1 - self.lmd_list[i])
                P = torch.linalg.inv(self.M_list[i])
                k = mul_matmul([P, H])
                bias = (y - y_hat).reshape(1, -1)
                new_weight = self.out_list[i].weight + mul_matmul([k, bias]).T
                self.out_list[i].weight = nn.Parameter(new_weight)
            total_y_hat = sum([to_prob(predict_list[i]) * self.weight_list[i] for i in range(len(predict_list))])
            self.update_classify_weight(loss_list)
            self.update_forgetting_factor()
            return total_y_hat


def classify(y_hat, y):
    if y_hat.type(y.dtype) == y:
        return 1
    else:
        return 0


def train(net, X_train, y_train):
    resultList = []
    dataBlock = 100
    i = 0
    pred = []
    true_label = []
    AccDict = {}
    while i < len(X_train):
        X = X_train[i]
        y = y_train[i]
        y_hat = net.online_training(X, y)
        classifyRusult = classify(torch.tensor(y.argmax(axis=0)), y_hat.argmax(axis=0))
        pred.append(int(y_hat.argmax(axis = 0)))
        true_label.append(int(y.argmax(axis = 0)))
        resultList.append(classifyRusult)
        if len(resultList) % dataBlock == 0:
            AccDict[len(resultList)] = {"blockRate":sum(resultList[-dataBlock:]) * 1.0 / dataBlock,"totalRate":sum(resultList) * 1.0 / len(resultList)}
        i += 1
    return AccDict


def OnlineLearning(fileName):
    data = np.load(fileName)
    X_train = torch.tensor(torch.from_numpy(data["x_train"]), dtype=torch.float32)
    y_train = torch.tensor(torch.from_numpy(data["y_train"]), dtype=torch.float32)
    num_inputs = X_train[0].shape[0]
    num_outputs = y_train[0].shape[0]
    num_hiddens = 256

    net = Net(num_inputs, num_outputs, num_hiddens)
    AccDict = train(net, X_train, y_train)
    save_data(AccDict, "result.json")



OnlineLearning("Electricity.npz")
