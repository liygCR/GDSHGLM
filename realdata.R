library(network)
library(huge)
library(gglasso)
library(grpreg)
library(cvTools)
library(MASS)
library(glmnet)
library(ncvreg)
library(sglfast)
library(GGally)

## cross validation for group lasso, sourced from sglfast package
igl <- function (data.train, data.validate, index = NULL, group.length = NULL, 
                 type = "linear", standardize = F, momentum = 2) 
{
  if (standardize) {
    temp = sglfast:::standarize_inital_data(data.train, data.validate, 
                                            type)
    data.train = temp$data.train
    data.validate = temp$data.validate
    X.transform = temp$X.transform
    intercept = temp$intercept
  }
  else {
    X.transform = list(X.means = rep(0, ncol(data.train$x)), 
                       X.scale = rep(1, ncol(data.train$x)))
    intercept = 0
  }
  if (is.null(group.length)) {
    if (is.null(index)) {
      write("Error 1: You must provide valid group indices or lengths")
      return(1)
    }
    temp = sglfast:::index2group.length(index)
    group.length = temp$group.length
    ord = temp$ord
    data.train$x = data.train$x[, ord]
    data.validate$x = data.validate$x[, ord]
    unord = match(1:length(ord), ord)
  }
  else {
    unord = 1:ncol(data.train$x)
  }
  lambda.max = c(0, 0)
  gamma = rep(0, length(group.length))
  lambda.max[1] = 0
  lambda.max[2] = 1
  lambda.init = c(lambda.max[1] * 0.1, lambda.max[2])
  for (i in 1:length(gamma)) {
    gamma[i] = sglfast:::get_gammak.max(data.train, group.length, 
                                        i, type, lambda.init)
  }
  lambda.max[2] = max(gamma/sqrt(group.length))
  nparams = 2 + length(group.length)
  best_lambdas <- c(lambda.max * 0.1, sqrt(group.length))
  num_solves <- 0
  max_solves <- 50000
  model_params = sglfast:::solve_inner_problem(data.train, group.length, 
                                               best_lambdas, type)
  best_cost = sglfast:::get_validation_cost(data.validate$x, data.validate$y, 
                                            model_params, type)
  best_beta = model_params
  num_solves = num_solves + 1
  coord <- 1
  fixed = 0
  while ((num_solves < max_solves) && (fixed < nparams)) {
    old_lambdas <- best_lambdas
    if ((coord > 2) && (best_lambdas[2] == 0)) {
      coord = 1
      fixed = nparams - 2
      (next)()
    }
    curr_lambdas = best_lambdas
    t0 = best_lambdas[coord] * 0.1
    dir = 1
    t = t0
    # if (t == 0) {
    #   t = 0.01
    # }
    while (dir >= -1) {
      curr_lambdas[coord] = best_lambdas[coord] + dir * 
        runif(1, 0.1 * t, t)
      model_params <- sglfast:::solve_inner_problem(data.train, 
                                                    group.length, curr_lambdas, type)
      num_solves <- num_solves + 1
      cost <- sglfast:::get_validation_cost(data.validate$x, data.validate$y, 
                                            model_params, type)
      if (best_cost - cost > 1e-05) {
        best_cost = cost
        best_lambdas <- curr_lambdas
        best_beta = model_params
        if (dir == 1) {
          t = momentum * t
        }
        else {
          t = min(momentum * t, curr_lambdas[coord])
        }
      }
      else {
        dir = dir - 2
        t = t0
      }
    }
    if (old_lambdas[coord] == best_lambdas[coord]) {
      fixed = fixed + 1
    }
    else {
      fixed = 0
    }
    coord <- coord%%(nparams) + 1
  }
  solution = list(best_lambdas = best_lambdas, num_solves = num_solves, 
                  best_cost = best_cost, beta = best_beta$beta[unord], 
                  intercept = best_beta$intercept + intercept, X.transform = X.transform, 
                  type = type)
  class(solution) = "isgl"
  return(solution)
}



####
data = read.csv(file = "Matson.csv")

# the _ class
z=data[-1,]
z[,-c(1:6)] = as.data.frame(sapply(z[,-c(1:6)], function(x) {as.numeric(levels(x))[x]}))
z1= z[,-c(1,2,3,4,5)]
z2 = aggregate(. ~ X.4, data=z1, FUN=sum)
rownames(z2) = z2[,1]
z2 = z2[,-1]
z3 = do.call(rbind, apply(z2, 1, function(x){x/z2[1,]}))
z3 = z3[-1,]
z4=t(z3)
y = ifelse(data[1,-c(1:6)] == "R", 1, 0 )
rownames(y) = "response"
da = cbind(t(y), t(z3))


id = (which(apply(da[,-1], 2, function(x){sum(x == 0)/length(x) }) <=0.6))
p = length(id)
n= nrow(da)

## scaled mean 0, varaince 1.
dat = as.matrix(da[,c(1, id+1)])
dat[,-1] = apply(dat[,-1], 2, scale) 

## graphical
out.glasso = huge(as.matrix(dat[,2:(p+1)]),method = "glasso", nlambda = 100,verbose=F)
out.select = huge.select(out.glasso, criterion = "stars", rep.num = 10)
# out.select = huge.select(out.glasso, criterion = "ric", rep.num = 100) 
icov=as.matrix(out.select$opt.icov)
colnames(icov) = colnames(dat)[-1]
pattern=as.matrix.network(network(icov, directed = FALSE),matrix.type="adjacency")      
pattern=pattern+diag(p)
# huge.plot(pattern)
ggnet2(network(icov, directed = FALSE), label = TRUE, 
       node.color = 1, edge.color = 1, label.color = 1,
       size = 2,  vjust = -1,  label.size = 3)

## repeat 2-CV process 50 times.
N = 50
K = 2
FPR <- matrix(0,N,6)
TPR <- matrix(0,N,6)
PCC <- matrix(0,N,6)
beta_est = vector("list", N)

for (j in 1:N) {
  set.seed(j)
  #Randomly shuffle the data
  dat <- dat[sample(nrow(dat)),]
  #Create 5 equally size folds
  folds <- cut(seq(1,nrow(da)),breaks=5,labels=FALSE)
  #Perform 5 fold cross validation
  cvFPR <- matrix(0,K,6)
  cvTPR <- matrix(0,K,6)
  cvPCC <- matrix(0,K,6)
  beta_est[[j]] = vector("list", K)
  for(k in 1:K){
    beta_est[[j]][[k]] = matrix(0, p, 6)
    row.names(beta_est[[j]][[k]]) = colnames(dat)[-1]
    colnames(beta_est[[j]][[k]]) = c("L12", "L2", "Lasso", "Ridge", "Enet", "ALasso")
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==k,arr.ind=TRUE)
    test <- dat[testIndexes, ]
    train <- dat[-testIndexes, ]
    #Use the test and train data partitions to fit the model
    
    ## Use the training data to estimate the network 
    out.glasso = huge(as.matrix(train[,2:(p+1)]),method = "glasso", nlambda = 50,verbose=F)
    out.select = huge.select(out.glasso, criterion = "stars", rep.num = 10)
    # out.select = huge.select(out.glasso, criterion = "ric", rep.num = 100) 
    icov=as.matrix(out.select$opt.icov)
    pattern=as.matrix.network(network(icov),matrix.type="adjacency")      
    pattern=pattern+diag(p)
    # huge.plot(pattern)
    
    edges=which(pattern==1,arr.ind=T)
    index0=as.vector(edges[,1])
    groupindex=as.vector(edges[,2])
    y_star=train[,1]
    x_star=train[,1+index0]  
    ytest=test[,1]
    xtest=test[,1+index0]
    
    # ### L12 via sparse group lasso
    data.train = list(x= x_star, y=y_star)
    data.test = list(x=xtest, y=ytest)
    tt1 = system.time({isgl.fit = isgl_rs(data.train, data.test, groupindex,
                                       type = "logit",standardize = TRUE)})
    coefs = isgl.fit$beta
    for(i in 1:p){beta_est[[j]][[k]][i,1]=sum(coefs[index0==i])}
    beta_est[[j]][[k]][,1][abs(beta_est[[j]][[k]][,1])<0.001] = 0
    pre = ifelse( predict.isgl(isgl.fit, newX = xtest) > 0.5, 1, 0 )
    
    Prediction = factor(pre, levels = c(0,1))
    Reference = factor(ytest, levels = c(0,1))
    accu = table(Prediction, Reference)
    cvFPR[k,1] = accu[2,1]/(accu[1,1]+accu[2,1] )
    cvTPR[k,1] = accu[2,2]/(accu[1,2]+accu[2,2] )
    cvPCC[k,1] = sum(diag(accu))/sum(accu)
    
    
    ## SRIG method
    tt2 = system.time({igl.fit = igl(data.train, data.test, groupindex, 
                                     type = "logit", standardize = TRUE)})
    coefs = igl.fit$beta
    for(i in 1:p){beta_est[[j]][[k]][i,2]=sum(coefs[index0==i])}
    beta_est[[j]][[k]][,2][abs(beta_est[[j]][[k]][,2])<0.001] = 0
    pre = ifelse( predict.isgl(igl.fit, newX = xtest) > 0.5, 1, 0 )
    Prediction = factor(pre, levels = c(0,1))
    Reference = factor(ytest, levels = c(0,1))
    accu = table(Prediction, Reference)
    cvFPR[k,2] = accu[2,1]/(accu[1,1]+accu[2,1])
    cvTPR[k,2] = accu[2,2]/(accu[1,2]+accu[2,2])
    cvPCC[k,2] = sum(diag(accu))/sum(accu)
    
    
    # # # pf = sqrt(as.vector(apply(pattern,2,sum)))
    # pf = igl.fit$best_lambdas[-c(1,2)]
    # grouplasso = cv.grpreg(X = x_star, y = y_star, group = groupindex,
    #                        group.multiplier = pf,
    #                        penalty = "grLasso", family = "binomial")
    # coefs = coef(grouplasso$fit, s = grouplasso$lambda.min)[-1,]
    # for(i in 1:p){beta_est[[j]][[k]][i,2]=sum(coefs[index0==i])}
    # beta_est[[j]][[k]][,2][abs(beta_est[[j]][[k]][,2])<0.001] = 0
    # pre = predict(grouplasso$fit, type="class",  lambda = grouplasso$lambda.min,
    #               X = xtest)
    # Prediction = factor(pre, levels = c(0,1))
    # Reference = factor(ytest, levels = c(0,1))
    # accu = table(Prediction, Reference)
    # cvFPR[k,2] = accu[2,1]/(accu[1,1]+accu[2,1])
    # cvTPR[k,2] = accu[2,2]/(accu[1,2]+accu[2,2])
    # cvPCC[k,2] = sum(diag(accu))/sum(accu)
    
    
    
    ##### lasso case
    # cv.glmfit fit lasso model and use cross validation for lambda selection
    # alpha controls the degree of L1 penalty term, 
    #		alpha = 1, lasso, 
    #		alpha = 0, ridge, 
    #		alpha = (0,1), elastic net
    
    # nfolds = 5, five-fold cross validation (CV)
    cvfit_lasso <- cv.glmnet(x = train[,-1], y = train[,1], alpha = 1, 
                             family = "binomial", nfolds = 5)
    tmp <- cvfit_lasso$glmnet.fit$beta
    # the beta that correspondes to lambda.min selected by CV
    beta_est[[j]][[k]][,3] <- (as.matrix(tmp[,cvfit_lasso$lambda==cvfit_lasso$lambda.min]))
    beta_est[[j]][[k]][,3][abs(beta_est[[j]][[k]][,3])<0.001] = 0
    pre = predict(cvfit_lasso, s = "lambda.min", newx= test[,-1], type = "class")
    
    Prediction = factor(pre, levels = c(0,1))
    Reference = factor(ytest, levels = c(0,1))
    accu = table(Prediction, Reference)
    cvFPR[k,3] = accu[2,1]/(accu[1,1]+accu[2,1] )
    cvTPR[k,3] = accu[2,2]/(accu[1,2]+accu[2,2] )
    cvPCC[k,3] = sum(diag(accu))/sum(accu)
    
    ## ridge
    cvfit_ridge <- cv.glmnet(x = train[,-1], y = train[,1], alpha = 0, 
                             family = "binomial", nfolds = 5)
    tmp <- cvfit_ridge$glmnet.fit$beta
    # the beta that correspondes to lambda.min selected by CV
    beta_est[[j]][[k]][,4] <- (as.matrix(tmp[,cvfit_ridge$lambda==cvfit_ridge$lambda.min]))
    beta_est[[j]][[k]][,4][abs(beta_est[[j]][[k]][,4])<0.001] = 0
    pre = predict(cvfit_ridge, s = "lambda.min", newx= test[,-1], type = "class")
    
    Prediction = factor(pre, levels = c(0,1))
    Reference = factor(ytest, levels = c(0,1))
    accu = table(Prediction, Reference)
    cvFPR[k,4] = accu[2,1]/(accu[1,1]+accu[2,1] )
    cvTPR[k,4] = accu[2,2]/(accu[1,2]+accu[2,2] )
    cvPCC[k,4] = sum(diag(accu))/sum(accu)
    
    ## enet
    cvfit_enet <- cv.glmnet(x = train[,-1], y = train[,1], alpha = 1/2, 
                            family = "binomial", nfolds = 5)
    tmp <- cvfit_enet$glmnet.fit$beta
    # the beta that correspondes to lambda.min selected by CV
    beta_est[[j]][[k]][,5] <- (as.matrix(tmp[,cvfit_enet$lambda==cvfit_enet$lambda.min]))
    beta_est[[j]][[k]][,5][abs(beta_est[[j]][[k]][,5])<0.001] = 0
    pre = predict(cvfit_enet, s = "lambda.min", newx= test[,-1], type = "class")
    
    Prediction = factor(pre, levels = c(0,1))
    Reference = factor(ytest, levels = c(0,1))
    accu = table(Prediction, Reference)
    cvFPR[k,5] = accu[2,1]/(accu[1,1]+accu[2,1] )
    cvTPR[k,5] = accu[2,2]/(accu[1,2]+accu[2,2] )
    cvPCC[k,5] = sum(diag(accu))/sum(accu)
    
    ##
    weight = 1/(beta_est[[j]][[k]][,3])^2
    # weight = 1/(ridge_beta)^2
    # Some est. coef. is zero, the corresponding weight is Inf
    # to prevent numerical error, convert Inf to a large number (e.g. 1e6)
    weight[weight==Inf] = 1e6
    cvfit_adlasso <- cv.glmnet(x = train[,-1], y = train[,1], alpha = 1, 
                               family = "binomial", nfolds = 5, penalty.factor=weight)
    
    ## coefficients estimated by adaptive lasso
    tmp <- cvfit_adlasso$glmnet.fit$beta
    beta_est[[j]][[k]][,6] <- as.matrix(tmp[,cvfit_adlasso$lambda==cvfit_adlasso$lambda.min])
    beta_est[[j]][[k]][,6][abs(beta_est[[j]][[k]][,6])<0.001] = 0
    pre = predict(cvfit_adlasso, s = "lambda.min", newx= test[,-1], type = "class")
    
    Prediction = factor(pre, levels = c(0,1))
    Reference = factor(ytest, levels = c(0,1))
    accu = table(Prediction, Reference)
    cvFPR[k,6] = accu[2,1]/(accu[1,1]+accu[2,1] )
    cvTPR[k,6] = accu[2,2]/(accu[1,2]+accu[2,2] )
    cvPCC[k,6] = sum(diag(accu))/sum(accu)
    
  }
  
  FPR[j,] = colMeans(cvFPR, na.rm = FALSE, dims = 1)
  TPR[j,] = colMeans(cvTPR, na.rm = FALSE, dims = 1)
  PCC[j,] = colMeans(cvPCC, na.rm = FALSE, dims = 1)
  
  print(j)
}

ind = which(apply(TPR,1,function(x) any(is.nan(x))))
FPR = FPR[-ind, ]
TPR = TPR[-ind, ]
PCC = PCC[-ind, ]

result.mean = rbind(colMeans(1-FPR), colMeans(TPR), colMeans(PCC))
colnames(result.mean) = c("GDSGLM", "L2", "Lasso", "Ridge", "Enet", "ALasso")
rownames(result.mean) = c("Spec", "Sens", "PCC")

result.sd = rbind( apply(FPR, 2, sd),  apply(TPR, 2, sd), apply(PCC, 2, sd))
colnames(result.sd) = c("GDSGLM", "L2", "Lasso", "Ridge", "Enet", "ALasso")
rownames(result.sd) = c("Spec", "Sens", "PCC")


beta_est_all = lapply(beta_est, function (x) { sapply(x, function(y){y[,1]}) })
beta_est_all = do.call(cbind, beta_est_all)
beta_est_sub = beta_est_all[,which(apply(beta_est_all, 2, function(x){sum(x != 0)}) > 0)]
size = mean(apply(beta_est_sub, 2, function(x){sum(x != 0)}))
rownames(beta_est_sub)[(which(apply(beta_est_sub, 1, 
                                    function(x){sum(x != 0)/length(x) }) >0.75))]


