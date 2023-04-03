#A code file for implementing CCNN-LSTM model
library(keras)
#importing dataset
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
step <- 1
delay <- 12
n_sample <- dim(ori_data)[1]
n_train <- round(n_sample*0.5)
n_val <- round(n_sample*0.25)
n_test <- n_sample-n_train-n_val-lookback-delay+1

mean <- apply(ori_data[1:n_train,],2,mean)
sd <- apply(ori_data[1:n_train,],2,sd)
to_train <- scale(ori_data,center = mean,scale = sd)

#Dividing dataset
{
  #training
  train <- array(0,dim = c(n_train,lookback/step,13))
  tw.5 <- array(0,dim = n_train)
  tw12 <- array(0,dim = n_train)
  for(i in 1:n_train){
    train[i,,] <- to_train[seq(i,i+lookback-1,
                               length.out = lookback/step),]
    tw.5[i] <- to_train[i+lookback-1+delay,1]
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
  val <- array(0,dim = c(n_val,lookback/step,13))
  vw.5 <- array(0,dim = n_val)
  vw12 <- array(0,dim = n_val)
  for(i in 1:n_val){
    val[i,,] <- to_train[seq(i+n_train,i+n_train+lookback-1,
                             length.out = lookback/step),]
    vw.5[i] <- to_train[i+n_train+lookback-1+delay,1]
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
  test <- array(0,dim = c(n_test,lookback/step,13))
  tew.5 <- array(0,dim = n_test)
  tew12 <- array(0,dim = n_test)
  for(i in 1:n_test){
    test[i,,] <- to_train[seq(i+n_train+n_val,i+n_train+n_val+lookback-1,
                              length.out = lookback/step),]
    tew.5[i] <- to_train[i+n_train+n_val+lookback-1+delay,1]
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

mark <- c(".5",1,2,3,4,5,6,7,8,9,10,12)
for(kk in 1:12){
  matr <- matrix(0,ncol = 9,nrow = 10)
  colnames(matr) <- rep(c("MAE","RMSE","NSE"),3)
  for(i in 1:10){
    
    input <- layer_input(shape = c(lookback/step,13))
    output <- input %>% layer_conv_1d(filters = 64,kernel_size = 5,
                                      activation = "relu") %>% 
      layer_lstm(units = 32) %>% 
      layer_dense(units = 1)
    model_ccnnlstm <- keras_model(input,output)
    model_ccnnlstm %>% compile(optimizer="nadam",
                           loss="mae")
    callback <- list(
      callback_model_checkpoint(monitor = "val_loss",
                                save_best_only = T,
                                filepath = "watertemp.h5"),
      callback_reduce_lr_on_plateau(monitor = "val_loss",
                                    factor = 0.2,
                                    patience = 4)
    )
    #training model
    model_ccnnlstm %>% fit(train,get(paste0("tw",mark[kk])),
                       validation_data=list(val,get(paste0("vw",mark[kk]))),
                       epoch=15,
                       batch_size=32,
                       callbacks=callback,
                       verbose=2)
    
    model_ccnnlstm <- load_model_hdf5("watertemp.h5")
    save_model_hdf5(model_ccnnlstm,paste0("CCNN-LSTM_",mark[kk],"_",i))
    
    pre <- model_ccnnlstm %>% predict(test)*sd[kk]+mean[kk]
    ori <- get(paste0("tew",mark[kk]))*sd[kk]+mean[kk]
    test_mae <- mean(abs(pre[,1]-ori))
    test_rmse <- sqrt(mean((pre[,1]-ori)^2))
    test_nse <- 1-sum((pre[,1]-ori)^2)/sum((ori-mean(ori))^2)
    
    pre <- model_ccnnlstm %>% predict(val)*sd[kk]+mean[kk]
    ori <- get(paste0("vw",mark[kk]))*sd[kk]+mean[kk]
    val_mae <- mean(abs(pre[,1]-ori))
    val_rmse <- sqrt(mean((pre[,1]-ori)^2))
    val_nse <- 1-sum((pre[,1]-ori)^2)/sum((ori-mean(ori))^2)
    
    pre <- model_ccnnlstm %>% predict(train)*sd[kk]+mean[kk]
    ori <- get(paste0("tw",mark[kk]))*sd[kk]+mean[kk]
    train_mae <- mean(abs(pre[,1]-ori))
    train_rmse <- sqrt(mean((pre[,1]-ori)^2))
    train_nse <- 1-sum((pre[,1]-ori)^2)/sum((ori-mean(ori))^2)
    
    matr[i,] <- c(train_mae,train_rmse,train_nse,
                  val_mae,val_rmse,val_nse,
                  test_mae,test_rmse,test_nse)
    print(c(kk,i))
  }
  assign(paste0("CCNN_",mark[kk]),matr)
  write.csv(get(paste0("CCNN_",mark[kk])),paste0("CCNN_",mark[kk],".csv"))
}


