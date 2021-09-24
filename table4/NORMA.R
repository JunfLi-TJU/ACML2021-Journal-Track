setwd("F:/experiment/ML 2021/code/table4")
rm(list = ls())

dpath          <- file.path("F:/experiment/ML 2021/dataset/regression/")  

d_index <- 1

Dataset        <- c("housing", "elevators","Hardware","Twitter","slice") 

savepath1      <- paste0("F:/experiment/ML 2021/Result/",paste0("NORMA-",Dataset[d_index],".txt"))
savepath2      <- paste0("F:/experiment/ML 2021/Result/",paste0("NORMA-all-",Dataset[d_index],".txt"))

traindatapath    <- file.path(dpath, paste0(Dataset[d_index], ".train"))
traindatamatrix  <- as.matrix(read.table(traindatapath))
trdata           <- traindatamatrix[ ,-1]
ylabel           <- traindatamatrix[ ,1]                                             

length_tr  <- nrow(trdata)                                               
feature_tr <- ncol(trdata)

para1_setting <- list( 
  B          = 200,
  varepsilon = 0.001,
  nu         = 0.5
)

sigma     <- 2^(0)

eta       <- 5/sqrt(length_tr)
lambda    <- 0.1

reptimes  <- 20
runtime   <- c(rep(0, reptimes))
errorrate <- c(rep(0, reptimes))
error     <- c(rep(0, reptimes))


for(re in 1:reptimes)
{
  order     <- sample(1:length_tr,length_tr,replace = F)   
  svmat     <- matrix(0,nrow = feature_tr,ncol=1)
  sv_coeff  <- array(0,1)
  k         <- 0
  flag      <- 1
  varepsilon_t <- para1_setting$varepsilon
  
  t1    <- proc.time()  #proc.time()
  for (i in 1:length_tr)
  {
    diff_S_i <- svmat - trdata[order[i], ]
    tem      <- colSums(diff_S_i*diff_S_i)
    svpara   <- sv_coeff[1:ncol(svmat)]
    fx       <- svpara %*% exp(tem/(-2*sigma^2))
    err      <- abs(fx-ylabel[order[i]])
    if(err > varepsilon_t)
    {
      if(k<para1_setting$B)
      {
        if(k == 0)
        {
          svmat[,1] <- trdata[order[i], ]
        }else{
          svmat     <- cbind(svmat,trdata[order[i], ])
        }
        k <- k+1
        nabla         <- sign(fx-ylabel[order[i]])
        sv_coeff      <- (1-eta*lambda)*sv_coeff
        sv_coeff[k]   <- -eta*nabla
        varepsilon_t  <- varepsilon_t + (1-para1_setting$nu)*eta
      }else{
        ## remove the old support vector
        nabla         <- sign(fx-ylabel[order[i]])
        sv_coeff      <- (1-eta*lambda)*sv_coeff
        sv_coeff[flag]   <- -eta*nabla
        svmat[,flag]     <- trdata[order[i], ]
        flag <- flag+1
        if(flag == (para1_setting$B))
          flag <- 1
        varepsilon_t  <- varepsilon_t + (1-para1_setting$nu)*eta
      }
    }else{
      sv_coeff      <- (1-eta*lambda)*sv_coeff
      varepsilon_t  <- varepsilon_t -para1_setting$nu*eta
    }
    error[re]   <- error[re] + err
  }
  t2 <- proc.time()
  runtime[re]    <- (t2 - t1)[3]
  errorrate[re]  <- error[re]/length_tr
}

save_result <- list(
  note     = c("the next term are:alg_name--dataname--sv_num--run_time--err_num--tot_run_time--ave_run_time--ave_err_rate--sd_time--sd_error"),
  alg_name = c("NORMA"),
  dataname = paste0(Dataset[d_index], ".train"),
  varepsilon = para1_setting$varepsilon,
  sigma     = sigma,
  nu       = para1_setting$nu,
  lambda   = lambda,
  B        = para1_setting$B,
  run_time = as.character(runtime),
  err_num  = as.character(errorrate), 
  tot_run_time = sum(runtime),
  ave_run_time = sum(runtime)/reptimes,
  ave_err_rate = sum(errorrate)/reptimes,
  sd_time  <- sd(runtime),
  sd_err    <-sd(errorrate)
)

write.table(save_result,file=savepath1,row.names =TRUE, col.names =FALSE, quote = T)

sprintf("the candidate kernel parameter are :")
sprintf("%.5f", sigma)
sprintf("the number of sample is %d", length_tr)
sprintf("total training time is %.4f in dataset", sum(runtime))
sprintf("average training time is %.5f in dataset", sum(runtime)/reptimes)
sprintf("the AAL is %f", sum(errorrate)/reptimes)
sprintf("standard deviation of run_time is %.5f in dataset", sd(runtime))
sprintf("standard deviation of AAL is %.5f in dataset", sd(errorrate))
