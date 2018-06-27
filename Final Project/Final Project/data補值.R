library(dplyr)

#--------------------------------- 公司治理排名資料 --------------------------------- 
data <- read.csv("C:/Users/User/Desktop/Master/Bigdata/Final Project/python_data.CSV")
#colnames(data) <- c('code','date','GPM','OPM','NIM','EBITDA','cash','account_receviable','inventory','fixed_asset','invest','liq_asset','liq_liability','leverage','revenue','GOR','mkt_value')

#發現缺值，以mice處理
library(mice)
md.pattern(data)                   #看缺失值狀況(圖、統計)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>［global_StakeholderRisk］
fit <- mice(data[,c(6,12,13,16)],     # 需要填充的數據集
            m = 5,                    # 多重填補法的'填補矩陣數'。默認為5
            maxit = 50,               # 迭代次數，默認50
            method = "cart",          # 填補用的方法。這邊用cart，進行遺漏值預測
            seed = 500)               # set.seed()，令抽樣每次都一樣)
summary(fit)

# 看看哪一個dataset比較好，挑一個表現最好(R squared最高的)的dataset。
formula = "GOR ~ liq_asset + liq_liability + EBITDA"
R2 = sapply(1:5, function(i)summary( lm(formula, complete(fit,i)) ) $ r.squared)

# 合併資料
data<- data.frame(data[,-c(16)],
                  complete(fit, which.max(R2)))
data <- data[,-c(17:19)]

md.pattern(data)                   #看缺失值狀況(圖、統計)

write.csv(data,file='C:/Users/User/Desktop/Master/Bigdata/Final Project/python_data1.CSV',row.names=FALSE)
