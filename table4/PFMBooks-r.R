setwd("F:/experiment/ML 2021/code/table4")
rm(list = ls())

dpath          <- file.path("F:/experiment/ML 2021/dataset/regression/")  

d_index <- 1

Dataset        <- c("Hardware","Twitter","slice")                    

savepath1      <- paste0("F:/experiment/ML 2021/Result/",paste0("PFMBooks-r-",Dataset[d_index],".txt"))
savepath2      <- paste0("F:/experiment/ML 2021/Result/",paste0("PFMBooks-r-all-",Dataset[d_index],".txt"))

traindatapath <- file.path(dpath, paste0(Dataset[d_index], ".train"))

traindatamatrix <- as.matrix(read.table(traindatapath))
trdata     <- traindatamatrix[ ,-1] 
ylabel     <- traindatamatrix[ ,1]                                             

length_tr  <- nrow(trdata)                                               
feature_tr <- ncol(trdata)              

para1_setting <- list(
  B      = 200, # B=50, 200, 400
  Y      = 0.1
)
nu         <- 1/3

reptimes   <- 20

power      <- seq(-4,4,1)
sigma      <- 2^(power)
len_sigma  <- length(sigma)

U_max      <- sqrt(para1_setting$B)
U_min      <- (para1_setting$B/length_tr)^(1/2)
M          <- ceiling(log(U_max))-ceiling(log(U_min))+1
eta_EA     <- sqrt(8*log(len_sigma*M*length_tr))/sqrt(length_tr)
U          <- c(rep(0, M))
g          <- c(rep(0, len_sigma*M))
lambda     <- c(rep(0, M))
all_K      <- c(rep(0, reptimes))
all_p      <- matrix(0,nrow=reptimes,ncol=len_sigma*M)

for(j in 1:M)
{
  U[j] <- exp(j+ceiling(log(U_min))-1)
  lambda[j] <- U[j]*sqrt((1+nu)*para1_setting$B)/(sqrt(2-2*nu)*length_tr) #### L=1
}
for (i in 1:len_sigma)
{
  for( j in 1:M)
  {
    g[(i-1)*M+j]<- U[j] + para1_setting$Y
  }
}
runtime    <- c(rep(0, reptimes))
errorrate  <- c(rep(0, reptimes))
all_error  <- c(rep(0, reptimes))
flag       <- c(rep(0, reptimes))

for(re in 1:reptimes)
{

  order      <- sample(1:length_tr,length_tr,replace = F)
  k          <- 0      ## record the number of support vector
  ite_binary <- c(rep(0, length_tr))
  haty       <- c(rep(0, length_tr))
  p          <- c(rep(1/(U_max*sqrt(length_tr)*len_sigma*M),len_sigma*M))
  for(i in 1:len_sigma)
  {
    p[(i-1)*M+1] = (1-1/(U_max*sqrt(length_tr)))*(1/len_sigma)+1/(len_sigma*M*U_max*sqrt(length_tr))
  }
  
  svpara   = matrix(0,nrow = len_sigma*M,ncol=1)
  sv_index = array(0,1)
  svmat    = matrix(0,nrow = feature_tr,ncol=1)
  barpt    <- c(rep(0, len_sigma*M))
  
  Norm_overlin_f_t   <- c(rep(0, len_sigma*M))                       ######## record the norm of \vertlone{f}_t
  err <- 0
  
  t1       <- proc.time()                                          ######## proc.time()
  for (tau in 1:length_tr)
  {
    ct       <- c(rep(0, len_sigma*M))
    It          <- sample(1:(len_sigma*M), 1, replace = T, prob = p)
    p_rho       <- para1_setting$B/(2*(1-nu)*length_tr^(1-nu)*tau^(nu))
    rho         <- rbinom(1,1,p_rho)
    
    diff_S_i    <- svmat - trdata[order[tau], ]
    colsum_in_S <- colSums(diff_S_i*diff_S_i)
    sv_coef     <- svpara
    #
    if(rho==1 && k<para1_setting$B/2)
    {
      ########### updating budget
      if(k==0)
      {
        svmat[,1]  <- trdata[order[tau],]
        svpara[,1] <- c(rep(0, len_sigma*M))
      }else{
        svmat      <- cbind(svmat,trdata[order[tau],])
        svpara     <- cbind(svpara,c(rep(0, len_sigma*M)))
      }
      sv_index[k+1]  <- order[tau]
    }
    sum <- 0
    for(i in 1:len_sigma)
    {
      kvalue_S    <- exp(colsum_in_S/(-2*(sigma[i]^2)))
      for(j in 1:M)
      {
        r   <- (i-1)*M+j
        fx  <- sv_coef[r,]%*% kvalue_S
        if(r==It)
          haty[tau] <- fx
        ct[r] <- abs(ylabel[order[tau]]-fx)
        barpt[r] <- p[r]*exp(-eta_EA*ct[r]/g[r]) 
        if(rho==1&& k<para1_setting$B/2)
        {
          nabla <- sign(fx-ylabel[order[tau]])
          svpara[r,k+1]   <- -lambda[j]*nabla/p_rho
          ###### compute the norm of f_t
          Norm_overlin_f_t[r] <- sqrt(Norm_overlin_f_t[r]^2+(lambda[j]/p_rho)^2-2*lambda[j]*nabla*fx/p_rho)
          if(Norm_overlin_f_t[r]>U[j])
          {
            svpara[r,]   <- svpara[r,]*U[j]/Norm_overlin_f_t[r]
            Norm_overlin_f_t[r]   <- U[j]
          }
        }
      }
    }
    if(rho==1 && k<para1_setting$B/2)
    {
      k            <- k+1
    }
    err <- err + ct[It]  

    ############################### updating pt by binary search
    upper_lambda <- 0    #### lambda^\ast < 0
    lambda_ast   <- upper_lambda
    tem          <- exp(-lambda_ast/g)
    sum_barpt    <- crossprod(barpt,tem)
    if(1-sum_barpt > 0.01 || (sum_barpt - 1)>1e-5)
    {
      lower_lambda <- -1
      lambda_ast   <- lower_lambda
      tem          <- exp(-lambda_ast/g)
      sum_barpt    <- crossprod(barpt,tem)
      while(sum_barpt<1)
      {
        flag[re]     <- flag[re] +1
        lower_lambda <- lower_lambda*para1_setting$B^(1/8)
        lambda_ast   <- lower_lambda
        tem          <- exp(-lambda_ast/g)
        sum_barpt    <- crossprod(barpt,tem)
      }
      while(1-sum_barpt > 0.01 || (sum_barpt - 1)>1e-5)
      {
        if(sum_barpt>1)
        {
          lower_lambda <- lambda_ast
        }else{
          upper_lambda <- lambda_ast
        }
        lambda_ast <- (lower_lambda+upper_lambda)/2
        tem <- exp(-lambda_ast/g)
        sum_barpt <- crossprod(barpt,tem)
        ite_binary[tau] <- ite_binary[tau]+1
      }
    }
    p     <- barpt*exp(-lambda_ast/g)
  }
  t2 <- proc.time()
  runtime[re] <- (t2 - t1)[3]
  all_error[re] <- err
  errorrate[re] <- err/length_tr
  all_K[re]     <- k
  all_p[re,]    <- p
}

save_result <- list(
  note     = c(" the next term are:alg_name--dataname--nu--B--ave_run_time--all_AAL--sd_time--sd_AAL"),
  alg_name = c("PFMBooks-r"),
  dataname = paste0(Dataset[d_index], ".train"),
  nu       = nu,
  B = para1_setting$B,
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
sprintf("the average AAL is %f", sum(errorrate)/reptimes)
sprintf("standard deviation of runing time is %.5f in dataset", sd(runtime))
sprintf("standard deviation of AAL is %.5f in dataset", sd(errorrate))
