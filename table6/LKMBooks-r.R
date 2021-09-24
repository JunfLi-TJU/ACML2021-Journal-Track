setwd("F:/experiment/ML 2021/code/table6")
rm(list = ls())

dpath          <- file.path("F:/experiment/ML 2021/dataset/regression/")  

d_index <- 1

Dataset        <- c("elevators","housing","ailerons","ailerons-v",
                    "TomsHardware","TomsHardware-v","Twitter","Twitter-v")                   

savepath1      <- paste0("F:/experiment/ML 2021/Result/",paste0("LKMBooks-r-",Dataset[d_index],".txt"))
savepath2      <- paste0("F:/experiment/ML 2021/Result/",paste0("LKMBooks-r-all-",Dataset[d_index],".txt"))

traindatapath <- file.path(dpath, paste0(Dataset[d_index], ".train"))

traindatamatrix <- as.matrix(read.table(traindatapath))
trdata     <- traindatamatrix[ ,-1] 
ylabel     <- traindatamatrix[ ,1]                                             

length_tr  <- nrow(trdata)                                               
feature_tr <- ncol(trdata)              

para1_setting <- list(
  B      = 220,
  Y      = 1
)

nu         <- 1/3

reptimes   <- 20

power      <- seq(-4,4,1)
sigma      <- 2^(power)
len_sigma  <- length(sigma)

eta_EA     <- sqrt(8*log(len_sigma))/sqrt(length_tr)
all_K      <- c(rep(0, reptimes))
all_p      <- matrix(0,nrow=reptimes,ncol=len_sigma)
lambda     <- sqrt((1+nu)*para1_setting$B)/(sqrt((1-nu))*length_tr)

runtime    <- c(rep(0, reptimes))
errorrate  <- c(rep(0, reptimes))
all_error  <- c(rep(0, reptimes))

for(re in 1:reptimes)
{
  order    <- sample(1:length_tr,length_tr,replace = F)
  k        <- 0      ## record the number of support vector
  p        <- c(rep(1/len_sigma,len_sigma))
  
  svpara   <- array(0,1)
  svmat    <- matrix(0,nrow = feature_tr,ncol=1)
  ct       <- c(rep(0, len_sigma))
  fx       <- c(rep(0, len_sigma))
  barpt    <- c(rep(0, len_sigma))
  g_max    <- 1
  err      <- 0
  
  t1       <- proc.time()                                          ######## proc.time()
  for (tau in 1:length_tr)
  {
    p_rho       <- para1_setting$B/((1-nu)*length_tr^(1-nu)*tau^(nu))
    rho         <- rbinom(1,1,p_rho)
    
    diff_S_i    <- svmat - trdata[order[tau], ]
    colsum_in_S <- colSums(diff_S_i*diff_S_i)
    sv_coef     <- svpara
    if(rho==1 && k<para1_setting$B)
    {
      ########### updating budget
      if(k==0)
      {
        svmat[,1]  <- trdata[order[tau],]
      }else{
        svmat      <- cbind(svmat,trdata[order[tau],])
      }
    }
    sum <- 0
    for(i in 1:len_sigma)
    {
      kvalue_S  <- exp(colsum_in_S/(-2*(sigma[i]^2)))
      fx[i]     <- svpara%*% kvalue_S
      sum       <- sum+p[i]*fx[i]
    }
    err         <- err+abs(sum-ylabel[order[tau]])
    nabla       <- sign(sum-ylabel[order[tau]])
    if(nabla >0)
    {
      for(i in 1:len_sigma)
      {
        ct[i]     <- nabla*(fx[i]-min(fx))/g_max
        barpt[i]  <- p[i]*exp(-eta_EA*ct[i])
      }
    }else
    {
      for(i in 1:len_sigma)
      {
        ct[i]     <- nabla*(fx[i]-max(fx))/g_max
        barpt[i]  <- p[i]*exp(-eta_EA*ct[i])
      }
    }
    if(rho==1 && k<para1_setting$B)
    {
      svpara[k+1]    <- -lambda*nabla/p_rho
      k            <- k+1
    }
    g_max <- max(g_max,max(fx)-min(fx))
    p     <- barpt/sum(barpt)
  }
  t2 <- proc.time()
  runtime[re] <- (t2 - t1)[3]
  all_error[re] <- err
  errorrate[re] <- err/length_tr
  all_K[re]     <- k
  all_p[re,]    <- p
}

save_result <- list(
  note     = c(" the next term are:alg_name--dataname--eta--beta--Norm_f--Norm_X--ave_run_time--all_MSE--sd_time--sd_MSE"),
  alg_name = c("LKMBooks-r-v2"),
  dataname = paste0(Dataset[d_index], ".train"),
  run_time = as.character(runtime),
  B        = para1_setting$B,
  ave_run_time = sum(runtime)/reptimes,
  ave_err_rate = sum(errorrate)/reptimes,
  sd_time  <- sd(runtime),
  sd_err    <-sd(errorrate)
)

save_result2 <- list(
  all_supp  <- all_K,
  all_p  <- all_p 
)
write.table(save_result,file=savepath1,row.names =TRUE, col.names =FALSE, quote = T) 
write.table(save_result2,file=savepath2,row.names =TRUE, col.names =FALSE, quote = T) 

sprintf("the kernel parameter is %f", para1_setting$sigma)
sprintf("the number of sample is %d", length_tr)
sprintf("the number of support vectors is %d", k)
sprintf("total running time is %.1f in dataset", sum(runtime))
sprintf("average running time is %.1f in dataset", sum(runtime)/reptimes)
sprintf("the AAL is %f", sum(errorrate)/reptimes)
sprintf("standard deviation of runing time is %.5f in dataset", sd(runtime))
sprintf("standard deviation of AAL is %.5f in dataset", sd(errorrate))
sprintf("the average per-round running time is %.5f in dataset", sum(runtime)/reptimes/length_tr*10^4)
sprintf("standard deviation of running time is %.5f in dataset", sd(runtime)/length_tr*10^4)
