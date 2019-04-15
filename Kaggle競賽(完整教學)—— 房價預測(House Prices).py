Kaggle競賽(完整教學)—— 房價預測(House Prices)
https://blog.csdn.net/wydyttxs/article/details/79680814



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy.stats import norm
from scipy import stats
%matplotlib inline

df_train = pd.read_csv(r'D:\自我學習\AI\Kaggle\Case3_House_price_regression_房價預測\Dataset\train.csv')

df_train.head() # 可以查看（默認）前5行數據信息

df_train.describe() # df_train['SalePrice'].describe()能獲得某一列的基本統計特徵

#探索性可視化（Exploratory Visualization）
#由於原始特徵較多，這裡只選擇建造年份(YearBuilt) 來進行可視化：
plt.figure(figsize=(15,8))
sns.boxplot(df_train.YearBuilt, df_train.SalePrice)

#一般认为新房子比较贵，老房子比较便宜，从图上看大致也是这个趋势，由于建造年份 (YearBuilt) 这个特征存在较多的取值 (从1872年到2010年)，直接one hot encoding会造成过于稀疏的数据，因此在特征工程中会将其进行数字化编码 (LabelEncoder) 

#数据清洗 (Data Cleaning)
#这里主要的工作是处理缺失值，首先来看各特征的缺失值数量：
aa = df_train.isnull().sum()
aa[aa>0].sort_values(ascending=False)

#如果我們仔細觀察一下data_description裡面的內容的話，會發現很多缺失值都有跡可循，比如上表第一個PoolQC，表示的是游泳池的質量，其值缺失代表的是這個房子本身沒有游泳池，因此可以用“None” 來填補。 

#下面給出的這些特徵都可以用“None” 來填補：
cols1 = ["PoolQC" , "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish", "GarageYrBlt", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]
for col in cols1:
    df_train[col].fillna("None", inplace=True)

	
#下面的這些特徵多為表示XX面積，比如TotalBsmtSF 表示地下室的面積，如果一個房子本身沒有地下室，則缺失值就用0來填補。	
cols=["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]
for col in cols:
    df_train[col].fillna(0, inplace=True)

#LotFrontage這個特徵與LotArea(地塊面積)和Neighborhood(鄰居)有比較大的關係，所以這裡用這兩個特徵分組後的中位數進行插補。
df_train['LotFrontage']=df_train.groupby(['LotArea','Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))

#特徵工程(Feature Engineering)
#離散型變量的排序賦值
#對於離散型特徵，一般採用pandas中的get_dummies進行數值化，但在這個比賽中光這樣可能還不夠，所以下面我採用的方法是按特徵進行分組，計算該特徵每個取值下SalePrice的平均數和中位數，再以此為基準排序賦值，下面舉個例子：

#MSSubClass這個特徵表示房子的類型，將數據按其分組：
df_train.groupby(['MSSubClass'])[['SalePrice']].agg(['mean','median','count'])


