setwd("F:/experiment/ML 2021/code/table5")
rm(list = ls())

d_index <- 1

dpath          <- file.path("F:/experiment/ML 2021/dataset/classification/") 

Dataset        <- c("magic04","a9a")

savepath1      <- paste0("F:/experiment/ML 2021/Result/",paste0("Adv-LKMBooks-c-",Dataset[d_index],".txt"))
savepath2      <- paste0("F:/experiment/ML 2021/Result/",paste0("Adv-LKMBooks-c-all-",Dataset[d_index],".txt"))

traindatapath <- file.path(dpath, paste0(Dataset[d_index], ".train"))

traindatamatrix <- as.matrix(read.table(traindatapath))
trdata     <- traindatamatrix[ ,-1] 
ylabel     <- traindatamatrix[ ,1]                                             

length_tr  <- nrow(trdata)                                               
feature_tr <- ncol(trdata)

trdata_copy <- trdata 

para1_setting <- list(
  B      = 200
)

nu         <- 1/3

reptimes   <- 20

power      <- seq(-4,4,1)
sigma      <- 2^(power)
len_sigma  <- length(sigma)

eta_EA     <- sqrt(8*log(len_sigma))/sqrt(length_tr)
all_K      <- c(rep(0, reptimes))
all_p      <- matrix(0,nrow=reptimes,ncol=len_sigma)
lambda     <- 5/sqrt(length_tr)
varespilon <- 1

runtime    <- c(rep(0, reptimes))
errorrate  <- c(rep(0, reptimes))
all_error  <- c(rep(0, reptimes))

for(re in 1:reptimes)
{
  order    <- sample(1:length_tr,length_tr,replace = F)
  k        <- 0      ## record the number of support vector
  p        <- c(rep(1/len_sigma,len_sigma))
  
  svpara   <- array(0,1)
  sv_index <- array(0,1)
  svmat    <- matrix(0,nrow = feature_tr,ncol=1)
  ct       <- c(rep(0, len_sigma))
  fx       <- c(rep(0, len_sigma))
  barpt    <- c(rep(0, len_sigma))
  g_max    <- 1
  ET       <- 0
  err      <- 0
  for(tau in ceiling(length_tr/20):length_tr)
    trdata[order[tau], ]  <- 2^{-3}*trdata[order[tau], ]
  t1       <- proc.time()                                          ######## proc.time()
  for(tau in 1:length_tr)
  {
    nabla       <- 0
    diff_S_i    <- svmat - trdata[order[tau], ]
    colsum_in_S <- colSums(diff_S_i*diff_S_i)
    for(i in 1:len_sigma)
    {
      kvalue_S  <- exp(colsum_in_S/(-2*(sigma[i]^2)))
      fx[i]     <- svpara%*% kvalue_S
    }
    sum       <- crossprod(p,fx)
    hatyt <- 1
    if(sum<0)
      hatyt <- -1
    if(hatyt != ylabel[order[tau]])
      err   <- err+1
    if(ylabel[order[tau]]*sum < varespilon)
    {
      nabla <- -ylabel[order[tau]]
      if(nabla >0)
      {
        for(i in 1:len_sigma)
        {
          ct[i]     <- nabla*(fx[i]-min(fx))/g_max
          barpt[i]  <- p[i]*exp(-eta_EA*ct[i])
        }
      }else{
        for(i in 1:len_sigma)
        {
          ct[i]     <- nabla*(fx[i]-max(fx))/g_max
          barpt[i]  <- p[i]*exp(-eta_EA*ct[i])
        }
      }
      p_rho       <- min(1,para1_setting$B/((1-nu)*length_tr^(1-nu)*(ET+1)^(nu)))
      rho         <- rbinom(1,1,p_rho)
      if(rho==1 && k<para1_setting$B)
      {
        ########### updating budget
        if(k==0)
        {
          svmat[,1]  <- trdata[order[tau],]
        }else{
          svmat      <- cbind(svmat,trdata[order[tau],])
        }
        k            <- k+1
        sv_index[k]  <- order[tau]
        svpara[k]    <- -lambda*nabla/p_rho
      }
      ET <- ET+1
    }else{
      barpt <- p
    }
    g_max <- max(g_max,max(fx)-min(fx))
    p     <- barpt/sum(barpt)
  }
  t2 <- proc.time()
  runtime[re]   <- (t2 - t1)[3]
  trdata        <- trdata_copy
  all_error[re] <- err
  errorrate[re] <- err/length_tr
  all_K[re]     <- k
  all_p[re,]    <- p
}

save_result <- list(
  note     = c(" the next term are:alg_name--dataname--eta--beta--Norm_f--Norm_X--ave_run_time--all_MSE--sd_time--sd_MSE"),
  alg_name = c("LKMBooks-c"),
  B        =para1_setting$B,
  dataname = paste0(Dataset[d_index], ".train"),
  run_time = as.character(runtime),
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
sprintf("the AMR is %f", sum(errorrate)/reptimes)
sprintf("standard deviation of runing time is %.5f in dataset", sd(runtime))
sprintf("standard deviation of AMR is %.5f in dataset", sd(errorrate))
