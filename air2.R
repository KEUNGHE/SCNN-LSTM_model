#A code file for importing data and building air2water model
ori_data <- read.csv("Blel_2008_2009_2010_2011.csv")
ori_data <- ori_data[-1,-1]
ori_data <- na.omit(ori_data)
ori_data <- ori_data[,c(1,12,13)]
#TW = DX0.5; TR = DX12; TA = AT
colnames(ori_data) <- c("TW","TR","TA")
ori_data <- apply(ori_data,2,as.numeric)
ori_data <- data.frame(ori_data)

Y1 <- ori_data[1:8784,]
Y2 <- ori_data[8785:17454,]
Y3 <- ori_data[17545:26304,]
Y4 <- ori_data[26305:35051,]

delta_1 <- function(TR, TW, p6){
  delta <- exp((TR-TW)/p6)
  return(delta)
}
delta_2 <- function(TR, TW, p7, p8){
  delta <- exp((TW-TR)/p7) + exp(-TW/p8)
  return(delta)
}

dw_dt <- function(TW, TR, TA, p_i, t){
  if(TW>=TR){
    delta <- delta_1(4, TW, p_i[6])
  }
  if(TW<TR){
    delta <- delta_2(4, TW, p_i[7], p_i[8])
  }
  dTW <- 1/delta*(p_i[1]*cos(2*pi*(t/tyr-p_i[2])) + p_i[3] + p_i[4]*(TA-TW) + p_i[5]*TW)
  return(dTW)
}

