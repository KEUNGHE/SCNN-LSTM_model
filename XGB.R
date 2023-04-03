#A code file for using XGBoost model for prediction

library(xgboost)
library(mcga)

ori_data <- read.csv("Blel_2008_2009_2010_2011.csv")
ori_data <- ori_data[-1,-1]
ori_data <- ori_data[,-c(14,15)]
colnames(ori_data) <- c("WT.5","WT1","WT2","WT3","WT4",
                        "wT5","WT6","WT7","WT8","WT9","WT10",
                        "WT12","AT")

ori_data <- na.omit(ori_data)
for(i in 1:12){
  ori_data[,i] <- as.numeric(ori_data[,i])
}

lookback <- 144
step <- 6
delay <- 12
n_sample <- dim(ori_data)[1]
n_train <- round(n_sample*0.5)
n_val <- round(n_sample*0.25)
n_test <- n_sample-n_train-n_val-lookback-delay+1

mean <- apply(ori_data[1:n_train,],2,mean)
sd <- apply(ori_data[1:n_train,],2,sd)
to_train <- scale(ori_data,center = mean,scale = sd)

{
  #training
  train <- array(0,dim = c(n_train,lookback/step*13))
  twp5 <- array(0,dim = n_train)
  tw12 <- array(0,dim = n_train)
  for(i in 1:n_train){
    train[i,] <- c(to_train[seq(i,i+lookback-1,
                                length.out = lookback/step),])
    twp5[i] <- to_train[i+lookback-1+delay,1]
    tw12[i] <- to_train[i+lookback-1+delay,12]
  }
  for(j in 1:10){
    m_0 <- array(0,dim = n_train)
    for(i in 1:n_train){
      m_0[i] <- to_train[i+lookback-1+delay,j+1]
    }
    assign(paste0("tw",j),m_0)
  }
  #validation
  val <- array(0,dim = c(n_val,lookback/step*13))
  vwp5 <- array(0,dim = n_val)
  vw12 <- array(0,dim = n_val)
  for(i in 1:n_val){
    val[i,] <- c(to_train[seq(i+n_train,i+n_train+lookback-1,
                               length.out = lookback/step),])
    vwp5[i] <- to_train[i+n_train+lookback-1+delay,1]
    vw12[i] <- to_train[i+n_train+lookback-1+delay,12]
  }
  for(j in 1:10){
    m_0 <- array(0,dim = n_val)
    for(i in 1:n_val){
      m_0[i] <- to_train[i+n_train+lookback-1+delay,j+1]
    }
    assign(paste0("vw",j),m_0)
  }
  #test
  test <- array(0,dim = c(n_test,lookback/step*13))
  tewp5 <- array(0,dim = n_test)
  tew12 <- array(0,dim = n_test)
  for(i in 1:n_test){
    test[i,] <- c(to_train[seq(i+n_train+n_val,i+n_train+n_val+lookback-1,
                                length.out = lookback/step),])
    tewp5[i] <- to_train[i+n_train+n_val+lookback-1+delay,1]
    tew12[i] <- to_train[i+n_train+n_val+lookback-1+delay,12]
  }
  for(j in 1:10){
    m_0 <- array(0,dim = n_test)
    for(i in 1:n_test){
      m_0[i] <- to_train[i+n_train+n_val+lookback-1+delay,j+1]
    }
    assign(paste0("tew",j),m_0)
  }
  
}

#Fitness function of the GA
ajfun <- function(x){
  xgb <- xgboost(data = train, label = twp5, verbose = 0,
                 nrounds = ceiling(x[1]), max_depth = ceiling(x[2]))
  pre_val <- predict(xgb,val)
  return(-mean((pre_val - vwp5)^2))
}

#Using GA to find the best hyper-parameters
GA_XGB <- mcga2(fitness = ajfun,min = c(3,6),max = c(40,25),
               maxiter = 30,popSize = 20)

par_xgb <- list(nrounds = GA_XGB@solution[1,1], max_depth = GA_XGB@solution[1,2])

mark <- c("p5","1","2","3","4","5","6","7","8","9","10","12")
par_xgb <- par_xgb
indicator <- matrix(0, nrow = 12, ncol = 12)
rownames(indicator) <- paste0("D",mark)
colnames(indicator) <- c(paste0("Train",c("MAE","MAPE","RMSE","NSE")),
                         paste0("Val",c("MAE","MAPE","RMSE","NSE")),
                         paste0("Test",c("MAE","MAPE","RMSE","NSE")))
for(k in 1:12){
  label_train <- get(paste0("tw",mark[k]))
  label_val <- get(paste0("vw",mark[k]))
  label_test <- get(paste0("tew",mark[k]))
  xgb <- xgboost(data = train, label = label_train, 
                 nrounds = par_xgb[1], max_depth = par_xgb[2])
  label_train <- label_train*sd[k]+mean[k]
  label_val <- label_val*sd[k]+mean[k]
  label_test <- label_test*sd[k]+mean[k]
  pre_train <- predict(xgb, train)
  pre_train <- pre_train*sd[k]+mean[k]
  pre_val <- predict(xgb, val)
  pre_val <- pre_val*sd[k]+mean[k]
  pre_test <- predict(xgb, test)
  pre_test <- pre_test*sd[k]+mean[k]
  
  train_mae <- mean(abs(label_train-pre_train))
  train_mape <- mean(abs(label_train-pre_train)/label_train)
  train_rmse <- sqrt(mean((label_train-pre_train)^2))
  train_nse <- 1 - sum((label_train-pre_train)^2)/sum((label_train-mean(label_train))^2)
  
  val_mae <- mean(abs(label_val-pre_val))
  val_mape <- mean(abs(label_val-pre_val)/label_val)
  val_rmse <- sqrt(mean((label_val-pre_val)^2))
  val_nse <- 1 - sum((label_val-pre_val)^2)/sum((label_val-mean(label_val))^2)
  
  test_mae <- mean(abs(label_test-pre_test))
  test_mape <- mean(abs(label_test-pre_test)/label_test)
  test_rmse <- sqrt(mean((label_test-pre_test)^2))
  test_nse <- 1 - sum((label_test-pre_test)^2)/sum((label_test-mean(label_test))^2)
  
  indicator[k,] <- c(train_mae,train_mape,train_rmse,train_nse,
                     val_mae,val_mape,val_rmse,val_nse,
                     test_mae,test_mape,test_rmse,test_nse)
  
  save(xgb,file = paste0("XGB",mark[k],".RData"))
}

write.csv(indicator,"Indicator_xgb.csv")

