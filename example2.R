library(network)
library(huge)
library(gglasso)
library(grpreg)
library(cvTools)
library(MASS)
library(glmnet)
library(arm)
library(sglfast)

source("FISTA_L12_fun.R")

###
p=100
Ntrain=120 #80/120
Nvalid=120 #80/120
Ntest=500



nlambda=20
lambda.min.ratio=1e-2
tol_out=1e-4
tol_in=1e-4
maxiter_out=1e2
maxiter_in=1e4


# graph structure ------------------------------------------------------------
opattern = diag(p)
diag(opattern[1:(p-1),2:p]) = 1
# diag(opattern[1:(p-2),3:p]) = 1
diag(opattern[2:p,1:(p-1)]) = 1
# diag(opattern[3:p,1:(p-2)]) = 1
huge.plot(opattern)


# generate multivariate normal data
Sigma = matrix(0,p,p)
for (i in 1:p) {
  for (j in 1:p) {
    Sigma[i,j] = (0.5)^(abs(i-j))
  }
}
beta=c(rep(1,15),rep(0,85))
NP=sum(beta!=0)
NN=sum(beta==0)

sim_fun <- function(Ntrain, Nvalid) {
  
  beta_est_all=matrix(0, p, 8)
  beta_L2_all=rep(0,8)
  cr_all=rep(0,8)
  FPR_all=rep(0,8)
  FNR_all=rep(0,8)
  
  ## Generate training data, validation data and testing data
  xdata = mvrnorm(Ntrain+Nvalid+Ntest, rep(0,p), Sigma)
  
  # xdata = rmvnorm(Ntrain+Nvalid+Ntest, mean = rep(0,p), sigma = Sigma)
  
  ## logistic
  feta = xdata[1:(Ntrain+Nvalid),]%*%beta;
  fprob = exp(feta)/(1+exp(feta))
  ydata = as.vector(rbinom(Ntrain+Nvalid, 1, fprob))
  
  
  train=cbind(ydata[1:Ntrain],xdata[1:Ntrain,])
  valid=cbind(ydata[(Ntrain+1):(Ntrain+Nvalid)],xdata[(Ntrain+1):(Ntrain+Nvalid),])
  xtest=xdata[(Ntrain+Nvalid+1):(Ntrain+Nvalid+Ntest),]
  ## logistic
  feta = c(xtest%*%beta)
  fprob = exp(feta)/(1+exp(feta))
  ytest = as.vector(rbinom(Ntest, 1, fprob))
  
  ## Use the training data to estimate the network 
  out.glasso = huge(as.matrix(train[,2:(p+1)]),method = "glasso",nlambda=100,verbose=F)
  out.select = huge.select(out.glasso, criterion = "stars", rep.num = 10) 
  icov=as.matrix(out.select$opt.icov)
  pattern=as.matrix.network(network(icov),matrix.type="adjacency")      
  pattern=pattern+diag(p)
  # huge.plot(pattern)
  
  ### DSGLMIG method
  ### L12 FISTA
  # weights=rep(1,p)
  weights = sqrt(as.vector(apply(pattern,2,sum)))
  # xi = 0
  # tt1=system.time({fista = FISTA_L12(X = train[,-1], Y = train[,1], pattern = pattern,
  #                                           weights = weights, xi = xi,
  #                   nlambda = nlambda, lambda.min.ratio = lambda.min.ratio,
  #                   tol_out = tol_out, tol_in = tol_in, maxiter_out = maxiter_out,
  #                   maxiter_in = maxiter_in)})
  # neg.logLik1 = apply(fista$beta_est, 2, nloglik,  x = valid[,-1], y = valid[,1])
  # lambdamin=fista$lambda[which.min(neg.logLik1)[1]]
  # beta_est_all[,1] =as.vector(fista$beta[,which.min(neg.logLik1)[1]])
  # beta_est_all[,1][abs(beta_est_all[,1])<0.001] = 0
  
  
  # print("Finish the estimated network")
  # cat(tt1, '\n')
  
  
  gradient = ngradient(x = train[,-1], y = train[, 1], mu = invlink(c(train[,-1] %*% rep(0,p))))
  lambda_max = sqrt(sum(gradient^2)) / (Ntrain *min(weights))
  lambda_all = exp(seq(log(lambda_max), log(lambda_max * lambda.min.ratio), 
                       length = nlambda))
  ##### SRIG method
  edges=which(pattern==1,arr.ind=T)
  index0=as.vector(edges[,1])
  groupindex=as.vector(edges[,2])
  y_star=train[,1]
  x_star=train[,1+index0]  
  yvalid=valid[,1]
  xvalid=valid[,1+index0]
  
  
  ### L12 via sparse group lasso
  data.train = list(x= x_star, y=y_star)
  data.validate = list(x=xvalid, y=yvalid)
  tt1 = system.time({isgl.fit = isgl_simple(data.train, data.validate, groupindex, type = "logit")})
  coefs = isgl.fit$beta
  for(i in 1:p){beta_est_all[i,1]=sum(coefs[index0==i])}
  beta_est_all[,1][abs(beta_est_all[,1])<0.001] = 0

  model_prediction = xtest%*%beta_est_all[,1]
  pre.pro = (1 + exp( -model_prediction ))^(-1)
  pre = ifelse( pre.pro > 0.5, 1, 0 )
  Prediction = factor(pre, levels = c(0,1))
  Reference = factor(ytest, levels = c(0,1))
  accu = table(Prediction, Reference)
  cr_all[1] = 1 - sum(diag(accu))/sum(accu)
  
  
  ##### SRIG method
  grouplasso=gglasso(x_star,2*y_star-1,group=groupindex,pf= weights,loss="logit",
                     lambda=lambda_all,
                     lambda.factor=lambda.min.ratio, intercept = FALSE)
  neg.logLik2 = apply(grouplasso$beta, 2, nloglik, x = xvalid, y = yvalid)
  lambdamin=grouplasso$lambda[which.min(neg.logLik2)[1]]
  coefs=as.vector(grouplasso$beta[,which.min(neg.logLik2)[1]])
  for(i in 1:p){beta_est_all[i,2]=sum(coefs[index0==i])}
  beta_est_all[,2][abs(beta_est_all[,2])<0.001] = 0
  
  model_prediction = xtest%*%beta_est_all[,2]
  pre.pro = (1 + exp( -model_prediction ))^(-1)
  pre = ifelse( pre.pro > 0.5, 1, 0 )
  Prediction = factor(pre, levels = c(0,1))
  Reference = factor(ytest, levels = c(0,1))
  accu = table(Prediction, Reference)
  cr_all[2] = 1 - sum(diag(accu))/sum(accu)
  
  
  ###  using true graph structure ###
  ### L12 FISTA
  # source("FISTA_L12_fun.R")
  xi = 5
  weights = sqrt(as.vector(apply(opattern,2,sum)))
  tt2=system.time({fista = FISTA_L12(X = train[,-1], Y = train[,1], pattern = opattern, weights = weights,
                                     xi = xi, nlambda = nlambda, lambda.min.ratio = lambda.min.ratio,
                                     tol_out = tol_out, tol_in = tol_in, maxiter_out = maxiter_out,
                                     maxiter_in = maxiter_in)})
  neg.logLik3 = apply(fista$beta_est, 2, nloglik,  x = valid[,-1], y = valid[,1])
  lambdamin=fista$lambda[which.min(neg.logLik3)[1]]
  beta_est_all[,3] =as.vector(fista$beta[,which.min(neg.logLik3)[1]])
  # beta_est_all[,3][abs(beta_est_all[,3])<0.001] = 0

  
  model_prediction = xtest%*%beta_est_all[,3]
  pre.pro = (1 + exp( -model_prediction ))^(-1)
  pre = ifelse( pre.pro > 0.5, 1, 0 )
  Prediction = factor(pre, levels = c(0,1))
  Reference = factor(ytest, levels = c(0,1))
  accu = table(Prediction, Reference)
  cr_all[3] = 1 - sum(diag(accu))/sum(accu)
  
  
  # print("Finish the true network")
  # cat(tt2, '\n')
  
  ### SRIG method
  ###  using true graph structure ###
  edges=which(opattern==1,arr.ind=T)
  index0=as.vector(edges[,1])
  groupindex=as.vector(edges[,2])
  y_star=train[,1]
  x_star=train[,1+index0]  
  yvalid=valid[,1]
  xvalid=valid[,1+index0]
  
  grouplasso=gglasso(x_star,2*y_star-1,group=groupindex,pf= weights,loss="logit",
                     lambda=lambda_all,
                     lambda.factor=lambda.min.ratio, intercept = FALSE)
  neg.logLik4 = apply(grouplasso$beta, 2, nloglik, x = xvalid, y = yvalid)
  lambdamin=grouplasso$lambda[which.min(neg.logLik4)[1]]
  coefs=as.vector(grouplasso$beta[,which.min(neg.logLik4)[1]])
  for(i in 1:p){beta_est_all[i,4]=sum(coefs[index0==i])}
  # beta_est_all[,4][abs(beta_est_all[,4])<0.001] = 0
  
  model_prediction = xtest%*%beta_est_all[,4]
  pre.pro = (1 + exp( -model_prediction ))^(-1)
  pre = ifelse( pre.pro > 0.5, 1, 0 )
  Prediction = factor(pre, levels = c(0,1))
  Reference = factor(ytest, levels = c(0,1))
  accu = table(Prediction, Reference)
  cr_all[4] = 1 - sum(diag(accu))/sum(accu)
  
  
  #### lasso by glmnet
  lasso = glmnet(x = train[,-1], y = train[,1], family = "binomial",
                 lambda=lambda_all,
                 lambda.min.ratio = lambda.min.ratio)
  neg.logLik5 = apply(lasso$beta, 2, nloglik, x = valid[,-1], y = valid[,1])
  lambdamin=lasso$lambda[which.min(neg.logLik5)[1]]
  beta_est_all[,5] =as.vector(lasso$beta[,which.min(neg.logLik5)[1]])
  # beta_est_all[,5][abs(beta_est_all[,5])<0.001] = 0
  
  model_prediction = xtest%*%beta_est_all[,5]
  pre.pro = (1 + exp( -model_prediction ))^(-1)
  pre = ifelse( pre.pro > 0.5, 1, 0 )
  Prediction = factor(pre, levels = c(0,1))
  Reference = factor(ytest, levels = c(0,1))
  accu = table(Prediction, Reference)
  cr_all[5] = 1 - sum(diag(accu))/sum(accu)
  
  
  ##ridge
  ridge = glmnet(x = train[,-1], y = train[,1], family = "binomial", alpha = 0,
                 lambda=lambda_all,
                 lambda.min.ratio = lambda.min.ratio)
  neg.logLik6 = apply(ridge$beta, 2, nloglik, x = valid[,-1], y = valid[,1])
  lambdamin=lasso$lambda[which.min(neg.logLik6)[1]]
  beta_est_all[,6] =as.vector(ridge$beta[,which.min(neg.logLik6)[1]])
  # beta_est_all[,6][abs(beta_est_all[,6])<0.001] = 0
  
  model_prediction = xtest%*%beta_est_all[,6]
  pre.pro = (1 + exp( -model_prediction ))^(-1)
  pre = ifelse( pre.pro > 0.5, 1, 0 )
  Prediction = factor(pre, levels = c(0,1))
  Reference = factor(ytest, levels = c(0,1))
  accu = table(Prediction, Reference)
  cr_all[6] = 1 - sum(diag(accu))/sum(accu)
  
  ## enet
  enet = glmnet(x = train[,-1], y = train[,1], family = "binomial", alpha = 1/2,
                lambda=lambda_all,
                lambda.min.ratio = lambda.min.ratio)
  neg.logLik7 = apply(enet$beta, 2, nloglik, x = valid[,-1], y = valid[,1])
  lambdamin=lasso$lambda[which.min(neg.logLik7)[1]]
  beta_est_all[,7] =as.vector(enet$beta[,which.min(neg.logLik7)[1]])
  # beta_est_all[,7][abs(beta_est_all[,7])<0.001] = 0
  
  model_prediction = xtest%*%beta_est_all[,7]
  pre.pro = (1 + exp( -model_prediction ))^(-1)
  pre = ifelse( pre.pro > 0.5, 1, 0 )
  Prediction = factor(pre, levels = c(0,1))
  Reference = factor(ytest, levels = c(0,1))
  accu = table(Prediction, Reference)
  cr_all[7] = 1 - sum(diag(accu))/sum(accu)
  
  ## Oracle
  library(arm)
  oracle = bayesglm.fit(x = train[,which(beta!=0)+1], y = train[,1], family = binomial(link = "logit"), intercept = FALSE,
                        control = list(maxit = 100))
  beta_est_all[,8] = c(coef(oracle), rep(0,p-15))
  
  model_prediction = xtest%*%beta_est_all[,8]
  pre.pro = (1 + exp( -model_prediction ))^(-1)
  pre = ifelse( pre.pro > 0.5, 1, 0 )
  Prediction = factor(pre, levels = c(0,1))
  Reference = factor(ytest, levels = c(0,1))
  accu = table(Prediction, Reference)
  cr_all[8] = 1 - sum(diag(accu))/sum(accu)
  
  
  ### Evaluation 
  for(i in 1:8) {
    beta_L2_all[i]=sqrt(sum((beta_est_all[,i]-beta)^2))
    FPR_all[i]=sum((beta==0)&(beta_est_all[,i]!=0))/NN
    FNR_all[i]=sum((beta!=0)&(beta_est_all[,i]==0))/NP
  }
  
  return(list( beta_est_all = beta_est_all, beta_L2_all = beta_L2_all, cr_all = cr_all, 
               FPR_all = FPR_all, FNR_all = FNR_all, tt1=tt1, tt2=tt2))
  
}


sim_exp2 <- sim_fun(Ntrain = Ntrain, Nvalid = Nvalid)

#### do parellels for 50 times replication
library(foreach)
library(doMC)
library(progress)

nsim = 50
numCores = 20
registerDoMC(numCores)  #change the 2 to your number of CPU cores 
ptm <- proc.time()
result = foreach(1:nsim) %dopar% {
  
  
  #loop contents here
  # set.seed(sim)
  sim_fun(Ntrain = Ntrain, Nvalid = Nvalid)
  
  # cat(sim, '\n')
  # pbTracker(pb, sim, numCores)
  
}
proc.time() - ptm

### simulation results
beta_L2_all <- do.call(rbind ,sapply(result, function(a){a$beta_L2_all}, simplify = FALSE))
misclass_all <- do.call(rbind ,sapply(result, function(a){a$cr_all}, simplify = FALSE))
FPR_all <- do.call(rbind ,sapply(result, function(a){a$FPR_all}, simplify = FALSE))
FNR_all <- do.call(rbind ,sapply(result, function(a){a$FNR_all}, simplify = FALSE))

means=rbind(apply(beta_L2_all,2,mean),apply(misclass_all,2,mean),apply(FPR_all,2,mean),apply(FNR_all,2,mean))
colnames(means) = c("DSRIG", "SRIG", "DSROG-O", "SRIG-O", "LASSO", "RIDGE", "E-NET", "GLM-O")
rownames(means) = c("L2", "ERROR", "FPR", "FNR")
sds=rbind(apply(beta_L2_all,2,sd),apply(misclass_all,2,sd),apply(FPR_all,2,sd),apply(FNR_all,2,sd))/sqrt(nsim)
colnames(sds) = c("DSRIG", "SRIG", "DSROG-O", "SRIG-O", "LASSO", "RIDGE", "E-NET", "GLM-O")
rownames(sds) = c("L2", "ERROR", "FPR", "FNR")
