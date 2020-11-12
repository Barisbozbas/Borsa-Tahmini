import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import yfinance as yf
##XU100.IS
#SPY AAPL
plt.style.use('fivethirtyeight')
#data = yf.download("^GSPC", start="2019-08-01", end="2020-08-01",group_by="ticker")
##XU100.IS
##data=web.DataReader('GOOG',data_source='yahoo',start='2020-09-15',end='2020-10-15')
##df=web.data.get_data_yahoo('BIST100',start='2020-09-15',end='2020-10-15')
#goog=web.get_data_yahoo('GOOG',start='2020-01-01')
#goog.head()

#deneme=web.get_data_yahoo('TSKB.IS',start='2020-10-01')
#deneme.head()


#Verileri internetten çekme
data=web.DataReader('KERVN.IS',data_source='yahoo',start='2020-08-15',end='2020-10-15')
data.shape

#Verileri grafikte gösterme
plt.title('TSKAB')
plt.ylabel('Fiyatlar')
data['Close'].plot(figsize=(10,5))
plt.show()

#Hisselerin kapanış fiyatlarını verilere atatık bunun yerine herhangi bir değere de atama yapabilirdik
data=data[['Close']]
data.tail()

gelecekteki_gunler=5
data['Tahmin']=data[['Close']].shift(-gelecekteki_gunler)
data.tail()

X=np.array(data.drop(['Tahmin'],1))[:-gelecekteki_gunler]
print(X)

Y=np.array(data['Tahmin'])[:-gelecekteki_gunler]
print(Y)

#Regresyon Kısmı
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=25)                                         
tree=DecisionTreeRegressor().fit(x_train,y_train)
#lr=LinearRegression().fit(x_train,y_test)

x_gelecek=data.drop(['Tahmin'],1)[-gelecekteki_gunler:]
x_gelecek=x_gelecek.tail(gelecekteki_gunler)
x_gelecek=np.array(x_gelecek)
x_gelecek

tree_tahminleri=tree.predict(x_gelecek)
print(tree_tahminleri)

tahmin=tree_tahminleri
valid=data[X.shape[0]:]
valid['Tahminler']=tahmin
plt.figure(figsize=(10,6))
plt.title('Karar Modeli')
plt.xlabel('Günler')
plt.ylabel('Fiyat')
plt.plot(data['Close'])
plt.plot(valid[['Close','Tahminler']])
plt.legend(['Orjinal','Değer','Tahminler'])
plt.show()