#!/usr/bin/env python
# coding: utf-8

#决策树
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# 加载数据
engine = create_engine("mysql+pymysql://root:123456@127.0.0.1:3306/temp?charset=utf8")
sql="select * from information_entropy"
data=pd.read_sql_query(sql, engine)

# g(D,A)=H(D)-H(D|A) H(D)=-∑x∈X P(D)logP(D)
# H(D)
d_1=data.groupby('是否借款')['是否借款'].agg(['count']).reset_index()['count'][0]
d_2=data.groupby('是否借款')['是否借款'].agg(['count']).reset_index()['count'][1]
d_sum=data['是否借款'].count()
h_d=-(d_1/d_sum*np.log2(d_1/d_sum)+d_2/d_sum*np.log2(d_2/d_sum))

# H(D|年龄)
# H(青年)
age_young_d_1=data[data['年龄']=='青年'].groupby('是否借款')['是否借款'].agg(['count']).reset_index()['count'][0]
age_young_d_2=data[data['年龄']=='青年'].groupby('是否借款')['是否借款'].agg(['count']).reset_index()['count'][1]
age_young_sum=data[data['年龄']=='青年']['是否借款'].count()
age_young_h_d=-(age_young_d_1/age_young_sum*np.log2(age_young_d_1/age_young_sum)+age_young_d_2/age_young_sum*np.log2(age_young_d_2/age_young_sum))
# H(中年)
age_mid_d_1=data[data['年龄']=='中年'].groupby('是否借款')['是否借款'].agg(['count']).reset_index()['count'][0]
age_mid_d_2=data[data['年龄']=='中年'].groupby('是否借款')['是否借款'].agg(['count']).reset_index()['count'][1]
age_mid_sum=data[data['年龄']=='中年']['是否借款'].count()
age_mid_h_d=-(age_mid_d_1/age_mid_sum*np.log2(age_mid_d_1/age_mid_sum)+age_mid_d_2/age_mid_sum*np.log2(age_mid_d_2/age_mid_sum))
# H(老年)
age_old_d_1=data[data['年龄']=='老年'].groupby('是否借款')['是否借款'].agg(['count']).reset_index()['count'][0]
age_old_d_2=data[data['年龄']=='老年'].groupby('是否借款')['是否借款'].agg(['count']).reset_index()['count'][1]
age_old_sum=data[data['年龄']=='老年']['是否借款'].count()
age_old_h_d=-(age_old_d_1/age_old_sum*np.log2(age_old_d_1/age_old_sum)+age_old_d_2/age_old_sum*np.log2(age_old_d_2/age_old_sum))
# H(D|年龄)
h_d_age=(5/15*age_young_h_d+5/15*age_mid_h_d+5/15*age_old_h_d)
g_d_age=h_d-h_d_age

# H(D|有工作)
# H(是)
work_1_d_1=data[data['有工作']=='是'].groupby('是否借款')['是否借款'].agg(['count']).reset_index()['count'][0]
work_1_d_2=data[data['有工作']=='是'].groupby('是否借款')['是否借款'].agg(['count']).reset_index()['count'][0]
work_1_sum=data[data['有工作']=='是']['是否借款'].count()
work_1_h_d=-(work_1_d_1/work_1_sum*np.log2(work_1_d_1/work_1_sum)+work_1_d_2/work_1_sum*np.log2(work_1_d_2/work_1_sum))
# H(否)
work_0_d_1=data[data['有工作']=='否'].groupby('是否借款')['是否借款'].agg(['count']).reset_index()['count'][0]
work_0_d_2=data[data['有工作']=='否'].groupby('是否借款')['是否借款'].agg(['count']).reset_index()['count'][1]
work_0_sum=data[data['有工作']=='否']['是否借款'].count()
work_0_h_d=-(work_0_d_1/work_0_sum*np.log2(work_0_d_1/work_0_sum)+work_0_d_2/work_0_sum*np.log2(work_0_d_2/work_0_sum))
# H(D|有工作)
h_d_work=(5/15*work_1_h_d+10/15*work_0_h_d)
g_d_work=h_d-h_d_work

# H(D|有房子)
# H(是)
home_1_d_1=data[data['有房子']=='是'].groupby('是否借款')['是否借款'].agg(['count']).reset_index()['count'][0]
home_1_d_2=data[data['有房子']=='是'].groupby('是否借款')['是否借款'].agg(['count']).reset_index()['count'][0]
home_1_sum=data[data['有房子']=='是']['是否借款'].count()
home_1_h_d=-(home_1_d_1/home_1_sum*np.log2(home_1_d_1/home_1_sum)+home_1_d_2/home_1_sum*np.log2(home_1_d_2/home_1_sum))
# H(否)
home_0_d_1=data[data['有房子']=='否'].groupby('是否借款')['是否借款'].agg(['count']).reset_index()['count'][0]
home_0_d_2=data[data['有房子']=='否'].groupby('是否借款')['是否借款'].agg(['count']).reset_index()['count'][1]
home_0_sum=data[data['有房子']=='否']['是否借款'].count()
home_0_h_d=-(home_0_d_1/home_0_sum*np.log2(home_0_d_1/home_0_sum)+home_0_d_2/home_0_sum*np.log2(home_0_d_2/home_0_sum))
# H(D|有房子)
h_d_home=(6/15*home_1_h_d+9/15*home_0_h_d)
g_d_home=h_d-h_d_home
g_d_home

# H(D|信贷情况)
# H(一般)
credit_1_d_1=data[data['信贷情况']=='一般'].groupby('是否借款')['是否借款'].agg(['count']).reset_index()['count'][0]
credit_1_d_2=data[data['信贷情况']=='一般'].groupby('是否借款')['是否借款'].agg(['count']).reset_index()['count'][1]
credit_1_sum=data[data['信贷情况']=='一般']['是否借款'].count()
credit_1_h_d=-(credit_1_d_1/credit_1_sum*np.log2(credit_1_d_1/credit_1_sum)+credit_1_d_2/credit_1_sum*np.log2(credit_1_d_2/credit_1_sum))
# H(好)
credit_2_d_1=data[data['信贷情况']=='好'].groupby('是否借款')['是否借款'].agg(['count']).reset_index()['count'][0]
credit_2_d_2=data[data['信贷情况']=='好'].groupby('是否借款')['是否借款'].agg(['count']).reset_index()['count'][1]
credit_2_sum=data[data['信贷情况']=='好']['是否借款'].count()
credit_2_h_d=-(credit_2_d_1/credit_2_sum*np.log2(credit_2_d_1/credit_2_sum)+credit_2_d_2/credit_2_sum*np.log2(credit_2_d_2/credit_2_sum))
# H(非常好)
credit_3_d_1=data[data['信贷情况']=='非常好'].groupby('是否借款')['是否借款'].agg(['count']).reset_index()['count'][0]
credit_3_d_2=data[data['信贷情况']=='非常好'].groupby('是否借款')['是否借款'].agg(['count']).reset_index()['count'][0]
credit_3_sum=data[data['信贷情况']=='非常好']['是否借款'].count()
credit_3_h_d=-(credit_3_d_1/credit_3_sum*np.log2(credit_3_d_1/credit_3_sum)+credit_3_d_2/credit_3_sum*np.log2(credit_3_d_2/credit_3_sum))
# H(D|信贷情况)
h_d_credit=(5/15*credit_1_h_d+6/15*credit_2_h_d+4/15*credit_3_h_d)
g_d_credit=h_d-h_d_credit
g_d_all=[]
g_d_all.append([g_d_age,g_d_work,g_d_home,g_d_credit])
g_d_all_df=pd.DataFrame(list(g_d_all),columns=['年龄','有工作','有房子','信贷情况'])

print(g_d_all_df)




