#A code file for prediction using air2water model
k <- 12
tyr <- 365*24
lower <- c(0,0,-0.12,0,-0.02,0,0,0)
upper <- c(0.33,1,0.28,0.01,0,15,15,0.5)

n_y1 <- dim(Y1)[1] - k
n_y2 <- dim(Y2)[1] - k
n_y3 <- dim(Y3)[1] - k
train_labels <- array(c(Y1[1:n_y1+k,1],Y2[1:n_y2+k,1],Y3[1:n_y3+k,1]),
                      dim = c(n_y1+n_y2+n_y3))

fitness_air2 <- function(x){
  train_results <- array(0, dim = c(n_y1+n_y2+n_y3))
  #Y1
  for(i in 1:n_y1){
    TW_i <- Y1[i,1] + dw_dt(Y1[i,1], 4, Y1[i,3], x, i)*k
    train_results[i] <- TW_i
  }
  #Y2
  for(i in 1:n_y2){
    TW_i <- Y2[i,1] + dw_dt(Y2[i,1], 4, Y2[i,3], x, i)*k
    train_results[i+n_y1] <- TW_i
  }
  #Y3
  for(i in 1:n_y3){
    TW_i <- Y3[i,1] + dw_dt(Y3[i,1], 4, Y3[i,3], x, i)*k
    train_results[i+n_y1+n_y2] <- TW_i
  }
  NSC <- 1 - sum((train_labels-train_results)^2)/
    sum((train_labels-mean(train_labels))^2)
  RMSE <- sqrt(mean((train_labels-train_results)^2))
  return(-RMSE)
}


air2_train <- function(x){
  n_y1 <- dim(Y1)[1] - k
  n_y2 <- dim(Y2)[1] - k
  n_y3 <- dim(Y3)[1] - k
  train_results <- array(0, dim = c(n_y1+n_y2+n_y3))
  train_labels <- array(c(Y1[1:n_y1+k,1],Y2[1:n_y2+k,1],Y3[1:n_y3+k,1]),
                        dim = c(n_y1+n_y2+n_y3))
  #Y1
  for(i in 1:n_y1){
    TW_i <- Y1[i,1] + dw_dt(Y1[i,1], 4, Y1[i,3], x, i)*k
    train_results[i] <- TW_i
  }
  #Y2
  for(i in 1:n_y2){
    TW_i <- Y2[i,1] + dw_dt(Y2[i,1], 4, Y2[i,3], x, i)*k
    train_results[i+n_y1] <- TW_i
  }
  #Y3
  for(i in 1:n_y3){
    TW_i <- Y3[i,1] + dw_dt(Y3[i,1], 4, Y3[i,3], x, i)*k
    train_results[i+n_y1+n_y2] <- TW_i
  }
  return(list(raw = train_labels, prediction = train_results))
}


air2_test <- function(x){
  n_y4 <- dim(Y4)[1] - k
  train_results <- array(0, dim = c(n_y4))
  train_labels <- array(c(Y4[1:n_y4+k,1]),dim = c(n_y4))
  #Y4
  for(i in 1:n_y4){
    TW_i <- Y4[i,1] + dw_dt(Y4[i,1], 4, Y4[i,3], x, i)*k
    train_results[i] <- TW_i
  }
  return(list(raw = train_labels, prediction = train_results))
}
