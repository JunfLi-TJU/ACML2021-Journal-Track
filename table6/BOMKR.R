setwd("F:/experiment/ML 2021/code/table6")
rm(list = ls())

dpath          <- file.path("F:/experiment/ML 2021/dataset/regression/")  

d_index <- 1

Dataset        <- c("elevators","housing","ailerons","ailerons-v",
                    "TomsHardware","TomsHardware-v","Twitter","Twitter-v")                  

savepath1      <- paste0("F:/experiment/ML 2021/Result/",paste0("BOMKR-",Dataset[d_index],".txt"))
savepath2      <- paste0("F:/experiment/ML 2021/Result/",paste0("BOMKR-all-",Dataset[d_index],".txt"))

traindatapath    <- file.path(dpath, paste0(Dataset[d_index], ".train"))
traindatamatrix  <- as.matrix(read.table(traindatapath))
trdata           <- traindatamatrix[ ,-1]
ylabel           <- traindatamatrix[ ,1]                                             

length_tr  <- nrow(trdata)                                               
feature_tr <- ncol(trdata)

para1_setting <- list( 
  B          = 90,
  varepsilon = 0.001,
  nu         = 0.5
)
x         <- seq(-4,4,1)
sigma     <- 2^(x)
len_sigma <- length(sigma)
beta      <- sqrt(length_tr)/(sqrt(length_tr)+sqrt(len_sigma))
eta       <- 5/sqrt(length_tr)
lambda    <- 0.1

reptimes  <- 20
runtime   <- c(rep(0, reptimes))
errorrate <- c(rep(0, reptimes))
error     <- c(rep(0, reptimes))
all_p     <- matrix(0,nrow = reptimes, ncol = len_sigma)


all_infor <- matrix(0,nrow = reptimes, ncol = 3*len_sigma)

for(re in 1:reptimes)
{
  order     <- sample(1:length_tr,length_tr,replace = F)
  sum       <- c(rep(0, len_sigma))
  L         <- c(rep(1/len_sigma, len_sigma))
  k         <- c(rep(0, len_sigma))
  p         <- c(rep(1/len_sigma,len_sigma))
  flag      <- c(rep(1, len_sigma))
  varepsilon_t  <- c(rep(para1_setting$varepsilon, len_sigma))
  
  sv_coe_list <- list(
    svpara1   = array(0,1),
    svpara2   = array(0,1),
    svpara3   = array(0,1),
    svpara4   = array(0,1),
    svpara5   = array(0,1),
    svpara6   = array(0,1),
    svpara7   = array(0,1),
    svpara8   = array(0,1),
    svpara9   = array(0,1)
  )
  sv_max_list <- list(
    svmat1  = matrix(0,nrow = feature_tr,ncol=1),
    svmat2  = matrix(0,nrow = feature_tr,ncol=1),
    svmat3  = matrix(0,nrow = feature_tr,ncol=1),
    svmat4  = matrix(0,nrow = feature_tr,ncol=1),
    svmat5  = matrix(0,nrow = feature_tr,ncol=1),
    svmat6  = matrix(0,nrow = feature_tr,ncol=1),
    svmat7  = matrix(0,nrow = feature_tr,ncol=1),
    svmat8  = matrix(0,nrow = feature_tr,ncol=1),
    svmat9  = matrix(0,nrow = feature_tr,ncol=1)
  )
  
  t1    <- proc.time()  #proc.time()
  for(i in 1:length_tr)
  {
    for(r in 1:len_sigma)
    {
      diff_S_i <- sv_max_list[[r]] - trdata[order[i], ]
      tem      <- colSums(diff_S_i*diff_S_i)
      sum[r]   <- sv_coe_list[[r]] %*% exp(tem/(-2*(sigma[r])^2))
      err      <- abs(sum[r]-ylabel[order[i]])
      if(err > varepsilon_t[r])
      {
        if(k[r] < para1_setting$B)
        {
          if(k[r] == 0)
          {
            sv_max_list[[r]][,1] <- trdata[order[i], ]
          }else{
            sv_max_list[[r]]     <- cbind(sv_max_list[[r]],trdata[order[i], ])
          }
          k[r]   <-  k[r]+1
          sv_coe_list[[r]]           <- (1-eta*lambda)*sv_coe_list[[r]]
          sv_coe_list[[r]][k[r]]     <- -eta*sign(sum[r]-ylabel[order[i]])
          varepsilon_t[r]            <- varepsilon_t[r] + (1-para1_setting$nu)*eta
        }else{
          ## remove the oldest support vector 
          sv_max_list[[r]][,flag[r]]  <- trdata[order[i], ]
          sv_coe_list[[r]]            <- (1-eta*lambda)*sv_coe_list[[r]]
          sv_coe_list[[r]][flag[r]]   <-  -eta*sign(sum[r]-ylabel[order[i]])
          varepsilon_t[r]             <- varepsilon_t[r] + (1-para1_setting$nu)*eta
          flag[r]                     <- flag[r]+1
          if(flag[r] == (k[r]+1))
          {
            flag[r] <- 1
          }
        }
      }else{
        varepsilon_t[r]  <- varepsilon_t[r] -para1_setting$nu*eta
        sv_coe_list[[r]] <- (1-eta*lambda)*sv_coe_list[[r]]
      }
      L[r]       <- L[r]*beta^err
    }
    fx <- crossprod(p,sum)
    error[re]   <- error[re] + abs(fx-ylabel[order[i]])
    p = L/sum(L)
    L <- p
  }
  t2 <- proc.time()
  runtime[re]    <- (t2 - t1)[3]
  errorrate[re]  <- error[re]/length_tr
  all_p[re,]     <- p
}

save_result <- list(
  note     = c("the next term are:alg_name--dataname--sv_num--run_time--err_num--tot_run_time--ave_run_time--ave_err_rate--sd_time--sd_error"),
  alg_name = c("BOMKR"),
  dataname = paste0(Dataset[d_index], ".train"),
  B        = para1_setting$B,
  run_time = as.character(runtime),
  err_num  = as.character(errorrate), 
  tot_run_time = sum(runtime),
  ave_run_time = sum(runtime)/reptimes,
  ave_err_rate = sum(errorrate)/reptimes,
  sd_time  <- sd(runtime),
  sd_err    <-sd(errorrate)
)

all_infor[,(len_sigma+1):(2*len_sigma)] = all_p[,1:len_sigma]

write.table(save_result,file=savepath1,row.names =TRUE, col.names =FALSE, quote = T) 
write.table(all_infor,file=savepath2,row.names =TRUE, col.names =FALSE, quote = T) 

sprintf("the candidate kernel parameter are :")
sprintf("%.5f", sigma)
sprintf("the number of sample is %d", length_tr)
sprintf("total training time is %.4f in dataset", sum(runtime))
sprintf("average training time is %.5f in dataset", sum(runtime)/reptimes)
sprintf("the AAL is %f", sum(errorrate)/reptimes)
sprintf("standard deviation of tun_time is %.5f in dataset", sd(runtime))
sprintf("standard deviation of AAL is %.5f in dataset", sd(errorrate))