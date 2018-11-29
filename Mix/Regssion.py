import pandas as pd
from sklearn import model_selection
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble
from sklearn import neighbors
from sklearn import svm
from sklearn import naive_bayes
import xgboost
from sklearn.preprocessing import minmax_scale


def randomforest(X_train,X_test,Y_train,Y_test):
    rf = ensemble.RandomForestRegressor(random_state=100)
    rf.fit(X_train,Y_train)
    rf_predict = rf.predict(X_test)
    data_rf = pd.DataFrame({'predict':rf_predict,'real':Y_test})
    return data_rf

data = pd.read_csv('/home/hongqiaochen/Desktop/Link_predict/USAir/mix_4.csv',sep=',')
prediction = data.columns[2:]
X = data[prediction]
Y = data['connect']
# 变量的量纲不一样，需要对自变量进行归一化
X = minmax_scale(X)
X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.1,random_state=1234)

# 检测因变量是否存在失衡
counts = data.connect.value_counts()
print((counts[1]/(counts[0]+counts[1])))
# 经过检验严重失衡,用SMOTE平衡
over_sample = SMOTE(random_state=1234)
X_train,Y_train = over_sample.fit_sample(X_train,Y_train)


# auc_gnb,gnb = gnb(X_train,X_test,Y_train,Y_test)
# auc_rf,rf = randomforest(X_train,X_test,Y_train,Y_test)
# auc_svc,svc = svc_nonlinearity(X_train,X_test,Y_train,Y_test)
# auc_XGboost,XGboost = XGboost(X_train,X_test,Y_train,Y_test)
data = randomforest(X_train,X_test,Y_train,Y_test)
data.sort_values(by='real',ascending=False,inplace=True)
print(data)