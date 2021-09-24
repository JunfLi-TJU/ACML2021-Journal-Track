setwd("F:/experiment/ML 2021/code/table5")
rm(list = ls())

d_index <- 1

dpath          <- file.path("F:/experiment/ML 2021/dataset/classification/") 

Dataset        <- c("mushrooms","magic04")

savepath1      <- paste0("F:/experiment/ML 2021/Result/",paste0("OMKC-DD-",Dataset[d_index],".txt"))
savepath2      <- paste0("F:/experiment/ML 2021/Result/",paste0("OMKC-DD-all-",Dataset[d_index],".txt"))

traindatapath    <- file.path(dpath, paste0(Dataset[d_index], ".train"))
traindatamatrix  <- as.matrix(read.table(traindatapath))
trdata           <- traindatamatrix[ ,-1]
ylabel           <- traindatamatrix[ ,1]                                             

length_tr  <- nrow(trdata)                                               
feature_tr <- ncol(trdata)

para1_setting <- list( 
  B       = 200
)
x         <- seq(-4,4,1)
sigma     <- 2^(x)
len_sigma <- length(sigma)
beta      <- 0.8

reptimes  <- 20
runtime   <- c(rep(0, reptimes))
errorrate <- c(rep(0, reptimes))
haty      <- c(rep(0, len_sigma))
error     <- c(rep(0, reptimes))
all_p     <- matrix(0,nrow = reptimes, ncol = len_sigma)

all_infor<- matrix(0,nrow = reptimes, ncol = 3*len_sigma)

for(re in 1:reptimes)
{
  order    <- sample(1:length_tr,length_tr,replace = F)   
  
  L        <- c(rep(1, len_sigma))
  k        <- c(rep(0, len_sigma))
  diff_vec <- 0
  p        <- c(rep(1/len_sigma,len_sigma))
  
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
  for (i in 1:length_tr)
  {
    flag_updating <- 0
    for(r in 1:len_sigma)
    {
      err <- 0
      tem_svmat    <- sv_max_list[[r]]
      tem_svpara   <- sv_coe_list[[r]]
      
      diff_S_i <- tem_svmat - trdata[order[i], ]
      tem      <- colSums(diff_S_i*diff_S_i)
      sum      <- tem_svpara %*% exp(tem/(-2*(sigma[r])^2))
      haty[r]  <- 1
      if(sum<0)
        haty[r] <- -1
      if(haty[r] !=ylabel[order[i]])
      {
        err <- 1
        flag_updating <- 1
        if(diff_vec<para1_setting$B)
        {
          k[r]    <- k[r]+1
          if(k[r] == 1)
          {
            tem_svmat[,1] <- trdata[order[i], ]
          }else{
            tem_svmat <- cbind(tem_svmat,trdata[order[i], ])
          }
          tem_svpara[k[r]]   <- ylabel[order[i]]
          sv_max_list[[r]]   <- tem_svmat
          sv_coe_list[[r]]   <- tem_svpara
        }
      }
      L[r]       <- L[r]*beta^(err)
    }
    if(flag_updating)
      diff_vec <- diff_vec+1
    fx <- crossprod(p,haty)
    hatyt <-1
    if(fx < 0)
      hatyt <- -1
    if(hatyt !=ylabel[order[i]])
      error[re]   <- error[re] + 1
    p = L/sum(L)
  }
  t2 <- proc.time()
  runtime[re]    <- (t2 - t1)[3]
  errorrate[re]  <- error[re]/length_tr
  all_p[re,]     <- p
}

save_result <- list(
  note     = c("the next term are:alg_name--dataname--sv_num--run_time--err_num--tot_run_time--ave_run_time--ave_err_rate--sd_time--sd_error"),
  alg_name = c("OMKC-DD"),
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
sprintf("the AMR is %f", sum(errorrate)/reptimes)
sprintf("standard deviation of tun_time is %.5f in dataset", sd(runtime))
sprintf("standard deviation of AMR is %.5f in dataset", sd(errorrate))