# 损失函数（对数似然损失）cost(hθ(x),y)=∑(i=1 to m)-yilog(hθ(x))-(1-yi)log(1-hθ(x)) 推导
import numpy as np

# 准备数据集
eps=1e-15
y_true = [0, 0, 1, 1]
y_pred = [0.1, 0.2, 0.7, 0.99]
y_true = np.array(y_true)
y_pred = np.array(y_pred)
assert (len(y_true) and len(y_true) == len(y_pred))

# 把概率规范到0-1之间
p = np.clip(y_pred, eps, 1-eps)
loss = np.sum(- y_true * np.log(p) - (1 - y_true) * np.log(1-p))

loss_self=loss / len(y_true)
print ("自定义对数似然函数:{}".format(loss_self))

from sklearn.metrics import log_loss
print ("skearn对数似然函数:{} ".format(log_loss(y_true, y_pred)))