# Hand-written

# Q値をエージェントが行動するたびに更新する手法
# 時刻 tで
#　状態state s_t
#　行動state a_t
#　s_tのもとでa_tをおこすことによって　報酬reward r_tが得られる

# Q(s_t, a_t)はs_tにおいてある行動a_tを取った時の価値

# Q(s_t, a_t)　← Q(s_t, a_t) + \alpha(r_(t+1)) + \gamma max_{a_(t+1)} Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t))

# gamma : 0 ~ 1  : 将来の価値をどれほど考慮するか
# alpha : 0 ~ 1 : 学習の大きさを制御する。

# Q(s_t, a_t) : 現在のの価値
# alpha(r_(t+1)-Q(s_t, a_t)) :　未来の報酬と現在の価値の差
# alphaが大きすぎると「偶然に得た」報酬に過剰依存してしまう。
# alpha * gamma * max_{a_(t+1)} Q(s_{t+1}, a_{t+1}) : 未来の価値

# A : 0~2000 random
# B : 0~1000 random

# より大きな数字をえることができる可能性が高い方を選択できるようなモデルを作成する。

# Q_t(A), Q_t(B)
# ここで、tは時刻ではなく更新回数
# initialize : 0
# 
# Softmax function (like Boltzmann Distribution), 規格化の役割を果たす
# P_{t+1} (A) = exp{\beta * Q_t(A)}/ exp{\beta * Q_t(A)} + exp{\beta * Q_t(B)}
# beta = 1/t t : entropy
# βが小さい時、より活発に探索しようとする。
# A , B という「系」には　時間発展がないため

# Q_t+1(X) = Q_t(X) + \alpha(r_t - Q_t(X))

import numpy as np
import matplotlib.pyplot as plt

# seed 固定
# コードを実行したとき、再現性を確保するために
np.random.seed(39)

#　引いた総数
N = 2000

# カード山A : 0~2000
pileA = np.arange(2001)
# カード山B : 0~1000
pileB = np.arange(1001)

#parameters
alpha = 0.02
beta = 0.007

"""
Q_A : (時間ごとの)カード山Aの価値
Q_B : (時間ごとの)カード山Bの価値
P_A : (時間ごとの)カード山Aのを引く確率
P_B : (時間ごとの)カード山Bのを引く確率
select_pile : (時間ごと)選択したカード山(A = 1 B = 0)
"""

Q_A = [0] #Q_A[i] = Q_(t+1)(A)
Q_B = [0] #Q_B[i] = Q_(t+1)(B)
P_A = [] #P_A[i] = P_(t+1)(A)
P_B = [] #P_B[i] = P_(t+1)(B)
select_pile = []


# Q_t+1(X) = Q_t(X) + \alpha(r_t - Q_t(X))
for i in range(N):
    deno = (np.exp(beta * Q_A[i]) + np.exp(beta * Q_B[i]))
    P_A.append(np.exp(beta * Q_A[i]) / deno)
    P_B.append(np.exp(beta * Q_B[i]) / deno)
    if P_A[i] >= np.random.rand():
        select_pile.append(1)
        #r_t = 山から出たカードが報酬
        Q_A.append(Q_A[i] + alpha * (np.random.choice(pileA) - Q_A[i]))
        Q_B.append(Q_B[i])
    else:
        select_pile.append(0)
        #r_t = 山から出たカードが報酬
        Q_A.append(Q_A[i])
        Q_B.append(Q_B[i] + alpha * (np.random.choice(pileB) - Q_B[i]))

#　時間
t = np.arange(N)

# 1. AとB　とっちを選んだか
plt.scatter(t, select_pile, c="black", marker=".")
plt.title("Choice(A=1, B=0)")
plt.show()

# 1-1. Cumulative Optimal Action Rate
# あるステップまでどのような割合で選択したか
cumsum_A_ratio = np.cumsum(np.array(select_pile) == 1) / np.arange(1, N+1)

# print("np.cumsum(np.array(select_pile) == 1)", np.cumsum(np.array(select_pile) == 1))
# print("np.arrange(1, N+1)",np.arange(1, N+1))

plt.plot(t, cumsum_A_ratio, color="b", linewidth = 2)
plt.title("Cumulative Ratio of Choosing Pile A")
plt.xlabel("Step (t)")
plt.ylabel("Ratio(0.0 ~ 1.0)")
plt.axhline(y=1.0, color = "b", linestyle = '--')
plt.show()

# 2.AとBの　価値
plt.plot(t, Q_A[1:], color="red", linewidth = 2, label="A")
plt.plot(t, Q_B[1:], color="orange", linewidth = 2, label="B")
plt.xlabel("Step (t)")
plt.ylabel("Quality(=Action Value) Q")
plt.legend(loc = "lower right")
plt.show()

# 3.AとBを引く　確率
plt.plot(t, P_A, color="red", linewidth = 2, label="A")
plt.plot(t, P_B, color="orange", linewidth = 2, label="B")
plt.xlabel("Step (t)")
plt.ylabel("Probability P")
plt.legend(loc = "lower right")
plt.show()