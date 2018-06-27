
# －－－－主題 : 財報公布前預測企業價值,持有報酬高?－－－－－

## ．目錄:
一、 資料說明

二、 資料處理

三、 迴圈前測試

四、 迴圈---以code做groupby,做預測

## 一、【資料說明】
1. 資料樣本 : 台灣加權股市上市股票
2. 資料期間 : 2008~2017
3. 輸入變數 : 財報資訊  
（GPM、OPM、NIM、EBITDA、現金、應收帳款、存貨、固定資產、廠房與設備投資、流動資產、流動負債、leverage、revenue、市值、GOR）


```python
import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection, ensemble, preprocessing, metrics
```

．輸入資料　（pd.read_csv）


```python
data=pd.read_csv("python_data1.csv",engine='python')
```

．查看前2筆資料　（.head(2)）


```python
data.head(2)
```






<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>code</th>
      <th>date</th>
      <th>GPM</th>
      <th>OPM</th>
      <th>NIM</th>
      <th>EBITDA</th>
      <th>cash</th>
      <th>account_receviable</th>
      <th>inventory</th>
      <th>fixed_asset</th>
      <th>invest</th>
      <th>liq_asset</th>
      <th>liq_liability</th>
      <th>leverage</th>
      <th>revenue</th>
      <th>mkt_value</th>
      <th>GOR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1101</td>
      <td>200803</td>
      <td>13.91</td>
      <td>9.89</td>
      <td>9.53</td>
      <td>4041764</td>
      <td>12423117</td>
      <td>7631637</td>
      <td>7302870</td>
      <td>115287797</td>
      <td>-2650659</td>
      <td>57091512</td>
      <td>29783960</td>
      <td>46.86</td>
      <td>17906537</td>
      <td>191344000.0</td>
      <td>-0.1019</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1101</td>
      <td>200806</td>
      <td>12.17</td>
      <td>8.28</td>
      <td>11.23</td>
      <td>8364504</td>
      <td>11760556</td>
      <td>9501174</td>
      <td>6841077</td>
      <td>119771536</td>
      <td>-7462419</td>
      <td>54362283</td>
      <td>42792997</td>
      <td>51.40</td>
      <td>38460834</td>
      <td>133194000.0</td>
      <td>0.0683</td>
    </tr>
  </tbody>
</table>
</div>



．觀察資料缺值、型態　（.info()）


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 25960 entries, 0 to 25959
    Data columns (total 17 columns):
    code                  25960 non-null int64
    date                  25960 non-null int64
    GPM                   25960 non-null float64
    OPM                   25960 non-null float64
    NIM                   25960 non-null float64
    EBITDA                25960 non-null int64
    cash                  25960 non-null int64
    account_receviable    25960 non-null int64
    inventory             25960 non-null int64
    fixed_asset           25960 non-null int64
    invest                25960 non-null int64
    liq_asset             25960 non-null int64
    liq_liability         25960 non-null int64
    leverage              25960 non-null float64
    revenue               25960 non-null int64
    mkt_value             25960 non-null float64
    GOR                   25960 non-null float64
    dtypes: float64(6), int64(11)
    memory usage: 3.4 MB
    

筆記: 合併資料-->  類似R裡面的c-bind: pd.concat ( [A,B] ,axis=1 )    |  R裡面的r-bind: A.append(B)

## 二、【資料處理】

新增欄位:


```python
# 企業價值(EV)欄位
data['EV'] = data['mkt_value'] + data['liq_liability'] - data['cash']

# 計算EBITDA/EV，並將數值X1000 (為了使後續轉整數值時，仍可保持數值相對大小。如:將0.019轉成19)
data['EBITDA_EV'] = (data['EBITDA'] / data['EV'])*1000   # 0.1% 單位

# 將如上述相同的GOR數值X100
data['GOR'] = data['GOR']*100
```

將數值轉整數:　（.astype(int)）


```python
data = data.astype(int)
```

敘述統計:　　（.describe()）


```python
data.describe()
```






<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>code</th>
      <th>date</th>
      <th>GPM</th>
      <th>OPM</th>
      <th>NIM</th>
      <th>EBITDA</th>
      <th>cash</th>
      <th>account_receviable</th>
      <th>inventory</th>
      <th>fixed_asset</th>
      <th>invest</th>
      <th>liq_asset</th>
      <th>liq_liability</th>
      <th>leverage</th>
      <th>revenue</th>
      <th>mkt_value</th>
      <th>GOR</th>
      <th>EV</th>
      <th>EBITDA_EV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>25960.000000</td>
      <td>25960.000000</td>
      <td>25960.000000</td>
      <td>25960.000000</td>
      <td>25960.000000</td>
      <td>2.596000e+04</td>
      <td>2.596000e+04</td>
      <td>2.596000e+04</td>
      <td>2.596000e+04</td>
      <td>2.596000e+04</td>
      <td>2.596000e+04</td>
      <td>2.596000e+04</td>
      <td>2.596000e+04</td>
      <td>25960.000000</td>
      <td>2.596000e+04</td>
      <td>2.596000e+04</td>
      <td>25960.000000</td>
      <td>2.596000e+04</td>
      <td>25960.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3366.784284</td>
      <td>201257.500000</td>
      <td>19.193413</td>
      <td>-7.795185</td>
      <td>1.429545</td>
      <td>1.877395e+06</td>
      <td>4.359460e+06</td>
      <td>4.545351e+06</td>
      <td>4.301350e+06</td>
      <td>9.354451e+06</td>
      <td>-8.061903e+05</td>
      <td>1.553805e+07</td>
      <td>1.076097e+07</td>
      <td>42.571726</td>
      <td>1.636215e+07</td>
      <td>2.200790e+07</td>
      <td>52.925270</td>
      <td>2.784365e+07</td>
      <td>57.502465</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2351.025044</td>
      <td>287.253248</td>
      <td>18.725125</td>
      <td>401.746241</td>
      <td>678.636725</td>
      <td>8.046860e+06</td>
      <td>2.277929e+07</td>
      <td>2.318394e+07</td>
      <td>1.682119e+07</td>
      <td>3.217948e+07</td>
      <td>3.370938e+06</td>
      <td>6.527966e+07</td>
      <td>4.785069e+07</td>
      <td>17.557707</td>
      <td>7.519858e+07</td>
      <td>8.074474e+07</td>
      <td>2894.147994</td>
      <td>9.971105e+07</td>
      <td>100.636994</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1101.000000</td>
      <td>200803.000000</td>
      <td>-557.000000</td>
      <td>-32157.000000</td>
      <td>-22185.000000</td>
      <td>-2.146405e+07</td>
      <td>4.840000e+02</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>-9.596002e+07</td>
      <td>-1.548831e+09</td>
      <td>1.475000e+03</td>
      <td>0.000000</td>
      <td>-1.916478e+09</td>
      <td>8.500000e+04</td>
      <td>-99.000000</td>
      <td>-2.147484e+09</td>
      <td>-3196.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1733.000000</td>
      <td>201008.250000</td>
      <td>9.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.800350e+04</td>
      <td>3.387805e+05</td>
      <td>3.124900e+05</td>
      <td>3.721892e+05</td>
      <td>6.244542e+05</td>
      <td>-3.571130e+05</td>
      <td>1.846876e+06</td>
      <td>8.713890e+05</td>
      <td>30.000000</td>
      <td>1.101516e+06</td>
      <td>2.214750e+06</td>
      <td>-11.000000</td>
      <td>2.938812e+06</td>
      <td>21.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2464.000000</td>
      <td>201257.500000</td>
      <td>17.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>3.250075e+05</td>
      <td>9.045880e+05</td>
      <td>9.186610e+05</td>
      <td>9.416095e+05</td>
      <td>1.659830e+06</td>
      <td>-8.008600e+04</td>
      <td>4.094638e+06</td>
      <td>2.221060e+06</td>
      <td>43.000000</td>
      <td>3.048604e+06</td>
      <td>5.188500e+06</td>
      <td>0.000000</td>
      <td>6.818246e+06</td>
      <td>48.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3481.000000</td>
      <td>201506.750000</td>
      <td>27.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>1.009981e+06</td>
      <td>2.414978e+06</td>
      <td>2.547179e+06</td>
      <td>2.772695e+06</td>
      <td>4.976073e+06</td>
      <td>-1.530900e+04</td>
      <td>9.887622e+06</td>
      <td>6.163521e+06</td>
      <td>55.000000</td>
      <td>8.930965e+06</td>
      <td>1.243225e+07</td>
      <td>13.000000</td>
      <td>1.636509e+07</td>
      <td>88.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9958.000000</td>
      <td>201712.000000</td>
      <td>100.000000</td>
      <td>160.000000</td>
      <td>81135.000000</td>
      <td>2.891660e+08</td>
      <td>7.550959e+08</td>
      <td>1.229237e+09</td>
      <td>5.609549e+08</td>
      <td>4.663030e+08</td>
      <td>0.000000e+00</td>
      <td>2.045529e+09</td>
      <td>2.025502e+09</td>
      <td>102.000000</td>
      <td>2.045071e+09</td>
      <td>2.027462e+09</td>
      <td>417465.000000</td>
      <td>2.078569e+09</td>
      <td>5802.000000</td>
    </tr>
  </tbody>
</table>
</div>



偏態:  　　　（.skew(axis=0)）


```python
data.skew(axis=0)
```




    code                    1.575811
    date                    0.000000
    GPM                    -5.194147
    OPM                   -54.849269
    NIM                    77.035320
    EBITDA                 14.442787
    cash                   21.175347
    account_receviable     20.561184
    inventory              15.891328
    fixed_asset             7.945374
    invest                -11.958904
    liq_asset              16.998355
    liq_liability          18.068881
    leverage                0.072254
    revenue                 7.855768
    mkt_value               9.680757
    GOR                   120.939992
    EV                      7.746659
    EBITDA_EV              10.405664
    dtype: float64




```python
data.head()
```






<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>code</th>
      <th>date</th>
      <th>GPM</th>
      <th>OPM</th>
      <th>NIM</th>
      <th>EBITDA</th>
      <th>cash</th>
      <th>account_receviable</th>
      <th>inventory</th>
      <th>fixed_asset</th>
      <th>invest</th>
      <th>liq_asset</th>
      <th>liq_liability</th>
      <th>leverage</th>
      <th>revenue</th>
      <th>mkt_value</th>
      <th>GOR</th>
      <th>EV</th>
      <th>EBITDA_EV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1101</td>
      <td>200803</td>
      <td>13</td>
      <td>9</td>
      <td>9</td>
      <td>4041764</td>
      <td>12423117</td>
      <td>7631637</td>
      <td>7302870</td>
      <td>115287797</td>
      <td>-2650659</td>
      <td>57091512</td>
      <td>29783960</td>
      <td>46</td>
      <td>17906537</td>
      <td>191344000</td>
      <td>-10</td>
      <td>208704843</td>
      <td>19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1101</td>
      <td>200806</td>
      <td>12</td>
      <td>8</td>
      <td>11</td>
      <td>8364504</td>
      <td>11760556</td>
      <td>9501174</td>
      <td>6841077</td>
      <td>119771536</td>
      <td>-7462419</td>
      <td>54362283</td>
      <td>42792997</td>
      <td>51</td>
      <td>38460834</td>
      <td>133194000</td>
      <td>6</td>
      <td>164226441</td>
      <td>50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1101</td>
      <td>200809</td>
      <td>11</td>
      <td>7</td>
      <td>9</td>
      <td>11993727</td>
      <td>10615331</td>
      <td>9283486</td>
      <td>8550177</td>
      <td>122733210</td>
      <td>-10158525</td>
      <td>45692556</td>
      <td>39599056</td>
      <td>52</td>
      <td>60944205</td>
      <td>61399000</td>
      <td>10</td>
      <td>90382725</td>
      <td>132</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1101</td>
      <td>200812</td>
      <td>9</td>
      <td>6</td>
      <td>8</td>
      <td>15111414</td>
      <td>13033535</td>
      <td>8386187</td>
      <td>6908626</td>
      <td>125807120</td>
      <td>-14729131</td>
      <td>43762069</td>
      <td>39174857</td>
      <td>51</td>
      <td>78476558</td>
      <td>88724000</td>
      <td>2</td>
      <td>114865322</td>
      <td>131</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1101</td>
      <td>200903</td>
      <td>9</td>
      <td>5</td>
      <td>2</td>
      <td>2829888</td>
      <td>14200950</td>
      <td>9431949</td>
      <td>5539320</td>
      <td>127230295</td>
      <td>-1963180</td>
      <td>48506002</td>
      <td>38714520</td>
      <td>51</td>
      <td>18071221</td>
      <td>92016000</td>
      <td>0</td>
      <td>116529570</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>



## 三、【迴圈前測試】

建立 code 清單


```python
code_list=pd.DataFrame(data['code'].unique())
code_list.columns=['code']
```

建立空資料（DataFrame）以記錄迴圈後資料


```python
result = pd.DataFrame()    
```

－－－－－－－－－－－－－－－－－－－－－　開始試迴圈內的程式　－－－－－－－－－－－－－－－－－－－－－

### ．個股新資料-->pre_data　資料

選擇清單中第2列,第1欄資料


```python
code_list.iloc[1,0]
```




    1102



依照個股代號，從母體資料切割成新資料 (這裡命名為　pre_data　資料)


```python
# pre_data
pre_data = data[data['code']==code_list.iloc[1,0]]
```

將 EBITDA_EV 欄位做lag一期。  （.shift(+1)）

再將 EBITDA_EV 與前一季做相減，作為　change　欄位的資料。 

且將第一筆缺值去除。　（.dropna()）


```python
pre_data['EBITDA_EV_lag'] = pre_data['EBITDA_EV'].shift(+1)
pre_data['change'] = pre_data['EBITDA_EV']-pre_data['EBITDA_EV_lag']
pre_data = pre_data.dropna()
```

若 EBITDA/EV 欄位比上一季，變動>3%。

則 label 欄位為 1 ,否則為 0          （　np.where(data['A']>=30,1,0) 類似R的IF條件句　）


```python
pre_data['label'] = np.where(pre_data['change']>=30,1,0)
```

### ．train資料 、text資料

201512 前 為訓練集資料

201512 後 為測試集資料


```python
train = pre_data[pre_data['date']<=201512]
test = pre_data[pre_data['date']>201512]
#text = text[text['date']<=201612]
```

選擇訓練、測試欄位


```python
select_column = ['GPM','OPM','NIM','EBITDA','cash','account_receviable','inventory','fixed_asset','invest','liq_asset','liq_liability','leverage','revenue','mkt_value','GOR','EV','EBITDA_EV','change']

x_train = train[select_column]
y_train = train['label']
x_test = test[select_column]
y_test = test['label']
```

### ．使用隨機森林預測


```python
# 建立模組
rfc = RandomForestClassifier() 
# 訓練模組
rfc.fit(x_train, y_train)
# 預測
EBITDA_EV_predict = rfc.predict(x_test)
EBITDA_EV_predict
```




    array([0, 0, 0, 0, 0, 0, 0, 0])



### ．績效


```python
# 績效
accuracy = metrics.accuracy_score(y_test, EBITDA_EV_predict)
fpr, tpr, thresholds = metrics.roc_curve(y_test, EBITDA_EV_predict)
auc = metrics.auc(fpr, tpr)
#print(auc)
print('準確率: {}'.format(auc))
print('AUC值: {}'.format(accuracy))
```

    準確率: 0.5
    AUC值: 0.5
    

### ．紀錄結果


```python
# EBITDA_EV_predict
apple = pd.DataFrame(EBITDA_EV_predict)
```


```python
result1 = pd.DataFrame()  
result1 = result1.append(apple)
```


```python
# code、AUC
result1['code'] = code_list.iloc[1,0]
result1['AUC'] = accuracy
```

選擇特定列數 :  （.loc[[0,4],:]）

只選擇2016、2017第一季之預測資料


```python
result1 = result1.loc[[0,4],:]
```

將個股預測資訊，統整為新資料


```python
result = result.append(result1)
```


```python
result
```






<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>code</th>
      <th>AUC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1101</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1101</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1102</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1102</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
</div>



## 四、【迴圈 --- 以code做groupby,　做預測】

建立 code 清單


```python
code_list=pd.DataFrame(data['code'].unique())
code_list.columns=['code']
```

建立空資料（DataFrame）以記錄迴圈後資料


```python
result = pd.DataFrame()    
```


```python
for ix in range(0,code_list.size):
    #創立新的空資料、code清單。
    result1 = pd.DataFrame()  
    pre_data = data[data['code']==code_list.iloc[ix,0]]
    
    # 建立指標欄位
    pre_data['EBITDA_EV_lag'] = pre_data['EBITDA_EV'].shift(+1)
    pre_data['change'] = pre_data['EBITDA_EV']-pre_data['EBITDA_EV_lag']
    pre_data = pre_data.dropna()
    
    pre_data['label'] = np.where(pre_data['change']>=30,1,0)
    
    # train、test
    train = pre_data[pre_data['date']<=201512]
    test = pre_data[pre_data['date']>201512]
    
    select_column = ['GPM','OPM','NIM','EBITDA','cash','account_receviable','inventory','fixed_asset','invest','liq_asset','liq_liability','leverage','revenue','mkt_value','GOR','EV','EBITDA_EV','change']
    x_train = train[select_column]
    y_train = train['label']
    x_test = test[select_column]
    y_test = test['label']
    
    # 使用隨機森林預測
    rfc = RandomForestClassifier() 
    rfc.fit(x_train, y_train)
    EBITDA_EV_predict = rfc.predict(x_test)

    # 績效
    accuracy = metrics.accuracy_score(y_test, EBITDA_EV_predict)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, EBITDA_EV_predict)
    auc = metrics.auc(fpr, tpr)
    
    #紀錄結果
    # EBITDA_EV_predict
    apple = pd.DataFrame(EBITDA_EV_predict)
    result1 = result1.append(apple)
    # code、AUC
    result1['code'] = code_list.iloc[ix,0]
    result1['AUC'] = accuracy
    result1 = result1.loc[[0,4],:]
    
    #合併資料
    result = result.append(result1)
        
```

更改欄名


```python
result.columns = ['label','code','AUC']
```


```python
result.head(10)
```






<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>code</th>
      <th>AUC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1101</td>
      <td>0.875</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1101</td>
      <td>0.875</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1102</td>
      <td>0.875</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1102</td>
      <td>0.875</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1103</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1103</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1104</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1104</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1108</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1108</td>
      <td>1.000</td>
    </tr>
  </tbody>
</table>
</div>



以年份 (2016 、2017)、季拆表 


```python
# 選2016、2017第一季
pre_2016 = result.loc[[0],:]
pre_2017 = result.loc[[4],:]
```


```python
pre_2016.head()
```






<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>code</th>
      <th>AUC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1101</td>
      <td>0.875</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1102</td>
      <td>0.875</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1103</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1104</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1108</td>
      <td>1.000</td>
    </tr>
  </tbody>
</table>
</div>



依照 EVIT/EV 增加比例（label欄位）和 準確率（AUC），做排序。 

.sort_values(['要降冪排序的欄位'],ascending=False) 


```python
pre_2016 = pre_2016.sort_values(['label','AUC'],ascending=False)
pre_2017 = pre_2016.sort_values(['label','AUC'],ascending=False)
```


```python
pre_2016.head()
```






<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>code</th>
      <th>AUC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1432</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1472</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2009</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2364</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2405</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
pre_2016.to_csv('pre_2016.csv')
pre_2017.to_csv('pre_2017.csv')
```
