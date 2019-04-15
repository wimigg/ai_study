#教程網址: https://blog.csdn.net/qilixuening/article/details/75151026
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy.stats import norm
from scipy import stats
%matplotlib inline

df_train = pd.read_csv(r'D:\自我學習\AI\Kaggle\Case3_House_price_regression_房價預測\Dataset\train.csv')  #利用pandas導入得到DataFrame數據

df_train.head() # 可以查看（默認）前5行數據信息
# df_train.tail() # 可以查看後10行數據信息

df_train.columns # 查看各個特徵的具體名稱

df_train.describe() # df_train['SalePrice'].describe()能獲得某一列的基本統計特徵

#我們也可以利用直方圖查看某一特徵數據的具體分佈情況：
sns.distplot(df_train['SalePrice']) # 圖中的藍色曲線是默認參數 kde=True 的擬合曲線特徵

#由上圖可見，房價的並不服從正態分佈，我們可以查看其斜度skewness和峭度kurtosis，這是很重要的兩個統計量
print('skewness_斜度: {0}, kurtosis_峭度: {1}'.format(df_train['SalePrice'].skew(), df_train['SalePrice'].kurt()))

#利用DataFrame的自身特性，我們可以很容易地做出反映變量關係的散點圖
output,var,var1,var2 = 'SalePrice', 'GrLivArea', 'TotalBsmtSF', 'OverallQual'
fig, axes = plt.subplots(nrows=1,ncols=3,figsize=(16,5))
df_train.plot.scatter(x=var,y=output,ylim=(0,800000),ax=axes[0])
df_train.plot.scatter(x=var1,y=output,ylim=(0,800000),ax=axes[1])
df_train.plot.scatter(x=var2,y=output,ylim=(0,800000),ax=axes[2])

#從上圖我們注意到，OverQual屬性雖然是數值型變量，但具有明顯的有序性，此時對於這樣的變量，採用箱形圖顯示效果更佳：
fig, ax = plt.subplots(figsize=(8,6))
sns.boxplot(x=var2,y=output,data=df_train)
ax.set_ylim(0,800000)
plt.show()

#對於下面YearBuilt(建立年份)這個特徵，用seaborn繪製出來的效果簡潔而美觀：
var3 = 'YearBuilt'
fig, ax = plt.subplots(figsize=(16,8))
sns.boxplot(x=var3,y=output,data=df_train)
ax.set_ylim(0,800000)
plt.xticks(rotation=90)
plt.show()


#除此之外，seaborn一個比較強大而方便的功能在於，可以對多個特徵的散點圖、直方圖信息進行整合，得到各個特徵兩兩組合形成的圖矩陣：
#既然有了這樣一個功能，那就不能不提seaborn下另外一個與其類似的操作，不過它更加自由與靈活。
var_set = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.set(font_scale=1.25) # 設置橫縱坐標軸的字體大小
sns.pairplot(df_train[var_set]) # 7*7圖矩陣
# 可在kind和diag_kind參數下設置不同的顯示類型，此處分別為散點圖和直方圖，還可以設置每個圖內的不同類型的顯示
plt.show()


#由於數據特徵較多，為了便於展示，我們先另外創建一些數據：
df_tr = pd.read_csv(r'D:\自我學習\AI\Kaggle\Case3_House_price_regression_房價預測\Dataset\train.csv').drop('Id',axis=1)
df_X = df_tr.drop('SalePrice',axis=1)
df_y = df_tr['SalePrice']
quantity = [attr for attr in df_X.columns if df_X.dtypes[attr] != 'object'] # 數值變量集合
quality = [attr for attr in df_X.columns if df_X.dtypes[attr] == 'object'] # 類型變量集合


#我們對數值型數據進行melt操作，使其具有兩列，分別為變量名、取值。這其實相當於將所有選定的特徵的數據df1,...dfn進行pd.concat([df1,...dfn],axis=0)操作
melt_X = pd.melt(df_X, value_vars=quantity)
melt_X.head() # 可以查看（默認）前5行數據信息
#melt_X.tail() # 可以查看後10行數據信息

#sns.FacetGrid()默認會根據melt_X['variable']內的取值做unique操作，得到最終子圖的數量，然後可以利用col_wrap設置每行顯示的子圖數量（不要求必須填滿最後一行），sharex、sharey設置是否共享坐標軸；

#g.map()其實就類似於函數式編程裡面的map()函數，第一個參數表示繪製圖的方法（此處為直方圖），後面的參數為此繪圖方法下的參數設置。
g = sns.FacetGrid(melt_X, col="variable", col_wrap=5, sharex=False, sharey=False)
g = g.map(sns.distplot, "value") # 以melt_X['value']作為數據

#上述操作主要是單個或兩個特徵的數據分佈進行分析，下面我們對各個特徵間的關係進行分析：
#最簡單地，直接獲取整個DataFrame數據的協方差矩陣並利用sns.heatmaP()進行可視化
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True, ax=ax) # square參數保證corrmat為非方陣時，圖形整體輸出仍為正方形
plt.show()

#然後，我們可以選取與output變量相關係數最高的10個特徵查看其相關情況，找出那些相互關聯性較強的特徵
k = 10
top10_attr = corrmat.nlargest(k, output).index
top10_mat = corrmat.loc[top10_attr, top10_attr]
fig,ax = plt.subplots(figsize=(8,6))
sns.set(font_scale=1.25)
sns.heatmap(top10_mat, annot=True, annot_kws={'size':12}, square=True)
# 設置annot使其在小格內顯示數字，annot_kws調整數字格式
plt.show()





