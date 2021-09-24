setwd("F:/experiment/ML 2021/code/table7")
rm(list = ls())

d_index <- 1

dpath          <- file.path("F:/experiment/ML 2021/dataset/classification/") 

Dataset        <- c("mushrooms","cod-rna")

savepath1      <- paste0("F:/experiment/ML 2021/Result/",paste0("BATBooks-",Dataset[d_index],".txt"))
savepath2      <- paste0("F:/experiment/ML 2021/Result/",paste0("BATBooks-all-",Dataset[d_index],".txt"))

traindatapath  <- file.path(dpath, paste0(Dataset[d_index], ".train"))

traindatamatrix <- as.matrix(read.table(traindatapath))
trdata     <- traindatamatrix[ ,-1] 
ylabel     <- traindatamatrix[ ,1]                                             

length_tr  <- nrow(trdata)                                               
feature_tr <- ncol(trdata)

para1_setting <- list(
  B      = 2900,
  Y      = 1
)
nu         <- 1/8
U          <- para1_setting$B^(1/3)
reptimes   <- 20

power      <- seq(-4,4,1)
sigma      <- 2^(power)
len_sigma  <- length(sigma)

max_loss   <- 1

lambda     <- U*sqrt((1+nu)*para1_setting$B)/(sqrt(2*(1-nu))*length_tr)
all_K      <- matrix(0,nrow=reptimes,ncol=len_sigma)
all_p      <- matrix(0,nrow=reptimes,ncol=len_sigma)
p_ini      <- c(rep(1/len_sigma,len_sigma))

runtime    <- c(rep(0, reptimes))
errorrate  <- c(rep(0, reptimes))
all_error  <- c(rep(0, reptimes))

for(re in 1:reptimes)
{
  order       <- sample(1:length_tr,length_tr,replace = F)
  p           <- p_ini
  q           <- p_ini
  tilde_L     <- c(rep(0,len_sigma))
  min_tilde_L <- 4
  eta_EA      <- 4*sqrt(log(len_sigma))/sqrt(len_sigma*min_tilde_L)
  
  svpara   <- matrix(0,nrow = len_sigma,ncol=1)
  k        <- c(rep(0, len_sigma))      ## record the number of support vector
  ct       <- c(rep(0, len_sigma))
  
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
  Norm_overlin_f_t   <- c(rep(0, len_sigma))                       ######## record the norm of \vertlone{f}_t
  err                <- 0
  
  t1       <- proc.time()                                          ######## proc.time()
  for (tau in 1:length_tr)
  {
    class_err   <- c(rep(0, len_sigma))
    It          <- sample(1:len_sigma, 1, replace = T, prob = p)
    Jt          <- sample(1:len_sigma, 1, replace = T, prob = q)
    barpt       <- p
    Ot          <- c(It)
    if(It !=Jt)
    {
      Ot <- c(It,Jt)
    }
    for(i in Ot)
    {
      diff_S_i    <- sv_max_list[[i]] - trdata[order[tau], ]
      colsum_in_S <- colSums(diff_S_i*diff_S_i)
      sv_coef     <- svpara[i,1:ncol(sv_max_list[[i]])]
      
      qi          <- (len_sigma-1)/len_sigma*p[i]+1/len_sigma
      
      kvalue_S    <- exp(colsum_in_S/(-2*(sigma[i]^2)))
      fx          <- sv_coef%*% kvalue_S 
      
      hatyt       <-1 
      if(fx<0)
        hatyt <- -1
      if(ylabel[order[tau]]!=hatyt)
      {
        class_err[i]<-1
      }
      ct[i]       <- max(0,1-fx*ylabel[order[tau]])
      tilde_ct    <- ct[i]/(max_loss*qi)
      tilde_L[i]  <- tilde_L[i]+tilde_ct
      barpt[i]    <- p[i]*exp(-eta_EA*tilde_ct)
      if(i==Jt)
      {
        if(fx*ylabel[order[tau]]<1)
        {
          p_rho     <- len_sigma*para1_setting$B/(2*(1-nu)*length_tr^(1-nu)*tau^(nu))
          rho       <- rbinom(1,1,min(p_rho,1))
          if(rho==1 && k[i]< para1_setting$B/2)
          {
            ########### updating budget
            if(k[i]==0)
            {
              sv_max_list[[i]][,1]  <- trdata[order[tau],]
            }else{
              sv_max_list[[i]]      <- cbind(sv_max_list[[i]],trdata[order[tau],])
              svpara       <- cbind(svpara,c(rep(0, len_sigma)))
            }
            k[i]           <- k[i]+1
            nabla          <- -ylabel[order[tau]]
            update_p       <- p_rho/len_sigma
            svpara[i,k[i]] <- -lambda*nabla/update_p
            ###### compute the norm of f_t
            tem_f <- Norm_overlin_f_t
            Norm_overlin_f_t[i] <- sqrt(Norm_overlin_f_t[i]^2+(lambda/update_p)^2-2*lambda*nabla*fx/update_p+1e-7)
            if(Norm_overlin_f_t[i] > U)
            {
              svpara[i,]            <- svpara[i,]*U/Norm_overlin_f_t[i]
              Norm_overlin_f_t[i]   <- U
            }
          }
        }
      }
    }
    if(min(tilde_L)>min_tilde_L)
    {
      barpt       <- p_ini
      min_tilde_L <- min_tilde_L*4
      eta_EA      <- 4*sqrt(log(len_sigma))/sqrt(len_sigma*min_tilde_L)
    }
    err   <- err + class_err[It]        
    p     <- barpt/sum(barpt)
  }
  t2 <- proc.time()
  runtime[re] <- (t2 - t1)[3]
  all_error[re] <- err
  errorrate[re] <- err/length_tr
  all_K[re,]    <- k
  all_p[re,]    <- p
}

save_result <- list(
  note     = c(" the next term are:alg_name--dataname--eta--beta--Norm_f--Norm_X--ave_run_time--all_MSE--sd_time--sd_MSE"),
  alg_name = c("BATBooks"),
  dataname = paste0(Dataset[d_index], ".train"),
  B        =para1_setting$B,
  run_time = as.character(runtime),
  ave_run_time = sum(runtime)/reptimes,
  ave_err_rate = sum(errorrate)/reptimes,
  sd_time      = sd(runtime),
  sd_err       = sd(errorrate)
)

save_result2 <- list(
  all_supp   <- all_K,
  all_p      <- all_p 
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
sprintf("the average per-round running time is %.5f in dataset", sum(runtime)/reptimes/length_tr*10^4)
sprintf("standard deviation of running time is %.5f in dataset", sd(runtime)/length_tr*10^4)
