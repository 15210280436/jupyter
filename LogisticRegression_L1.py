# 正则化力度对权重W的影响
# 把高次项的W逐渐调整趋向0
# 定义数据
df_wine=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)
df_wine.columns=['Class label','Alcohol',
                 'Malic acid','Ash',
                 'Alcalinity of ash','Magnesium',
                 'Total phenols','Flavanoids',
                 'Nonflavanoid phenols',
                 'Proanthocyanins',
                 'Color intensity','Hue',
                 'OD280/OD315 of diluted wines',
                 'Proline'
                ]

print("Class labels:",np.unique(df_wine['Class label']))
# 分割数据集和训练集
X,y=df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

# 进行标准化处理
# 特征值标准化
std_x=StandardScaler()
x_train_std=std_x.fit_transform(x_train)
x_test_std=std_x.transform(x_test)
# 目标值标准化 由于sklearn1.9以上，StandardScaler入参是2维数组,y_train.reshape(-1,1)
std_y=StandardScaler()
y_train_std=std_y.fit_transform(y_train.reshape(-1,1))
y_test_std=std_y.transform(y_test.reshape(-1,1))

lr=LogisticRegression(penalty='l1',C=0.1)
lr.fit(x_train_std,y_train)

# print('Training accuracy:',lr.score(x_train_std,y_train))
# print('Test accuracy:',lr.score(x_test_std,y_test))
print('intercept_:',lr.intercept_)
print('coef_:',lr.coef_)

# 可视化C和W趋势图
fig=plt.figure(figsize=(20,8))
ax=plt.subplot(111)

colors=['blue','green','red','cyan','magenta','yellow','black','pink','lightgreen','lightblue','gray','indigo','orange']
weights,params=[],[]

for c in np.arange(-4,6,dtype=float):
    lr=LogisticRegression(penalty='l1',C=10**c,random_state=0)
    lr.fit(x_train_std,y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
    
weights=np.array(weights)

for column,color in zip(range(weights.shape[1]),colors):
    plt.plot(params,weights[:,column],label=df_wine.columns[column+1],color=color)
    
plt.axhline(0,color='black',linestyle='--',linewidth=3)
plt.xlim([10**(-5),10**5])
plt.ylabel('weight coefficient',size=20)
plt.xlabel('C',size=20)
plt.xscale('log')
plt.legend(loc='upper left')
plt.legend(loc='upper left',bbox_to_anchor=(1.38,1.03),ncol=1,fancybox=True)

plt.show()
