import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(777)

# =====================================================================
# Question 1:

x1_train = [[10], [9], [3], [2]]
y1_train = [[90], [80], [50], [30]]

X_1 = Variable(torch.Tensor(x1_train))
Y_1 = Variable(torch.Tensor(y1_train))
# xW + b
model_1 = torch.nn.Linear(1, 1, bias = True)
# cost criterion
criterion_1 = torch.nn.MSELoss()
# Minimize
optimizer_1 = torch.optim.SGD(model_1.parameters(), lr = 0.01)

for step in range(10000):
    optimizer_1.zero_grad()
    hypothesis_1 = model_1(X_1)
    cost_1 = criterion_1(hypothesis_1, Y_1)
    cost_1.backward()
    optimizer_1.step()

    if step % 20 == 0:
        print(step, cost_1.data.numpy(), model_1.weight.data.numpy(), model_1.bias.data.numpy())

# Testing
predicted_1 = model_1(Variable(torch.Tensor(([3.5], [4.5]))))
print("[5], predicted.data.numpy(): \n", predicted_1.data.numpy())

# =====================================================================
# Question 2:
x2_train = [[3], [4.5], [5.5], [6.5], [7.5], [8.5], [8], [9], [9.5], [10]]
y2_train = [[8.49], [11.93], [16.18], [18.08], [21.45], [24.35], [21.24], [24.84], [25.94], [26.02]]

X_2 = Variable(torch.Tensor(x2_train))
Y_2 = Variable(torch.Tensor(y2_train))

# 주어진 학습률로 모델을 학습하는 함수 정의
def train_model(x, y, learning_rate, epochs=10000):
    model = torch.nn.Linear(1, 1, bias=True)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for step in range(epochs):
        optimizer.zero_grad()
        hypothesis = model(x)
        loss = criterion(hypothesis, y)
        loss.backward()
        optimizer.step()

    # 학습된 weight와 bias를 스칼라로 반환
    return model.weight.data.numpy(), model.bias.data.numpy()


# 서로 다른 학습률 설정
lr_small = 0.00005  # 작은 학습률
lr_middle = 0.001  # 기존 학습률
lr_default = 0.01  # 기존 학습률
lr_large = 0.016  # 큰 학습률

# 각 학습률별로 모델 학습
w_small, b_small = train_model(X_2, Y_2, lr_small)
w_middle, b_middle = train_model(X_2, Y_2, lr_middle)
w_default, b_default = train_model(X_2, Y_2, lr_default)
w_large, b_large = train_model(X_2, Y_2, lr_large)

print("Parameters with lr=0.00005: w = {}, b = {}".format(w_small, b_small))
print("Parameters with lr=0.001:  w = {}, b = {}".format(w_middle, b_middle))
print("Parameters with lr=0.01:  w = {}, b = {}".format(w_default, b_default))
print("Parameters with lr=0.016:   w = {}, b = {}".format(w_large, b_large))

# numpy 배열로 변환 (데이터 포인트 플롯용)
x2_array = np.array(x2_train).flatten()
y2_array = np.array(y2_train).flatten()

# x 값 범위 생성 (예측 선 그리기)
x_range = np.linspace(x2_array.min(), x2_array.max(), 100)
# 각 모델의 직선 예측값 계산
y_small = w_small.item() * x_range + b_small
y_middle = w_middle.item() * x_range + b_middle
y_default = w_default.item() * x_range + b_default
y_large = w_large.item() * x_range + b_large

# 플롯 생성: 데이터 포인트와 3개의 직선 (각기 다른 색상)
plt.figure(figsize=(8, 8))
plt.scatter(x2_array, y2_array, color='black', label='Data Points')
plt.plot(x_range, y_small, ':', color='red', label='LR = 0.00005')
#plt.plot(x_range, y_middle, ':', color='olive', label='LR = 0.001')
plt.plot(x_range, y_default, '-', color='limegreen', label='LR = 0.01')
#plt.plot(x_range, y_large, ':', color='blue', label='LR = 0.016')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Linear Regression Fit on Question 2 with Different Learning Rates")
plt.legend()
plt.show()
