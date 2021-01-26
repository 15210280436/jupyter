import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

# 梯度下降推导

# 设置学习率（步长）
rate = 0.001

# 训练集
x_train = np.array([[2],[3],[4],[5],[6],[7],[8],[9],[10],[11]])
y_train = np.array([2,3.5,3.5,5,6,6.5,8.5,9,10.5,11])

# 初始化W和B
w=np.random.normal()
b=np.random.normal()

# 定义回归方程
def h(x):
    return w*x[0]+b
W=[]
B=[]
L=[]
# 梯度下降 
for i in range(7000):
    sum_w=0
    sum_b=0
    lose=0
    for x, y in zip(x_train, y_train):
        sum_w = sum_w - rate*(y-h(x))*x[0]
        sum_b = sum_b - rate*(y-h(x))
        lose=lose+(y-h(x))*(y-h(x))
    w = w - sum_w
    b = b - sum_b
    W.append([w])
    B.append([b])
    L.append([lose])

df=pd.concat([pd.DataFrame(list(W)),pd.DataFrame(list(B)),pd.DataFrame(list(L))],axis=1)
df.columns=['W','B','L']


# In[51]:


print('W=%f,B=%f,L=%f' % (w,b,lose))


# In[46]:


mpl.rcParams['legend.fontsize'] = 10
 
fig = plt.figure(figsize=(20,8))
ax = fig.gca(projection='3d')
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
ax.plot(df['W'], df['B'], df['L'], label='Lose')
ax.legend()
ax.set_xlabel('W')
ax.set_ylabel('B')
ax.set_zlabel('L')
plt.show()




