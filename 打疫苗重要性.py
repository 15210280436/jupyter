#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 给大家科普一下，为什么我们国家要求80%的打新冠疫苗
# R0表示如果不对病毒进行控制，每个感染者会将病毒传染给R0个人
# 科学家估计新冠病毒传染R0≈3
# 世代间隔表示病毒的快慢，科学家估计新冠的世代间隔≈4天


# In[46]:


import pandas as pd
import numpy as np
import math
my_font=font_manager.FontProperties(fname='/System/Library/Fonts/PingFang.ttc')


# In[68]:


# 假设需要N天全世界70亿人全部被感染
P=7500000000
N=1000
R0=3
interval=4
p=0
m=0
days=[]
people=[]
for i in np.arange(1,N,interval):
    if i==1:
        p=1
    else:
        p+=R0**(m+1)
        m+=1
#     print(i,p)

    days.append(i)
    people.append(p)
    if p>=P:
        break
np.set_printoptions(suppress=True)
df_day=pd.DataFrame(days,columns=['days'])
df_people=pd.DataFrame(people,columns=['people'])
plt.figure(figsize=(20,8))
plt.plot(df_day,df_people)
y_ticks=[]
plt.xlabel('天数',fontproperties=my_font,size=15)
plt.ylabel('感染人数',fontproperties=my_font,size=15)
plt.title('全球感染人数',fontproperties=my_font,size=25)
plt.legend(prop=my_font,fontsize=6,loc=2)
plt.show()
    


# In[79]:


# 假设人群中有r的比例注射疫苗具有了免疫力
# 那么1-r不具有免疫力
# 此时一个人就不能感染R0个人，而是R0*(1-r)个人
# 传染病就会消失
P=7500000000
N=1000
r=0
R0=3
interval=4
p=0
m=0
days=[]
people=[]
for r in np.arange(0,1,0.01):
    for i in np.arange(1,N,interval):
        if i==1:
            p=1
        else:
            p+=(R0*(1-r))**(m+1)
            m+=1
    #     print(i,p)
        days.append(i)
        people.append(p)
    df_people=pd.DataFrame(people,columns=['people'])
    if (df_people.iloc[-1]['people']-df_people.iloc[0]['people'])/N<=1:
        break
r


# In[80]:


r/0.8


# In[ ]:




