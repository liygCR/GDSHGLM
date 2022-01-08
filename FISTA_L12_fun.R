## Logistic Regression 
invlink   = function(eta) 1 / (1 + exp(-eta))
link      = function(mu) log(mu / (1 - mu))
nloglik   = function(y, x, beta, ...)
  -sum((y * (x%*%beta) - log(1 + exp(x%*%beta))))
ngradient = function(x, y, mu, ...)
  -crossprod(x, y - mu)
nhessian  = function(x, mu, ...)
  crossprod(x,  mu * (1 - mu) * x)

## sparse matrix
gsigmai<-function(p)
{
  set.seed(10)
  Omatrix=matrix(0,p,p)
  for(i in 1:(p-1)){
    for(j in (i+1):p){
      Omatrix[i,j]=0.5*rbinom(1,1,0.05)
    }
  }
  Omatrix=Omatrix+t(Omatrix)
  temp=eigen(Omatrix)$values
  delta=(temp[1]-p*temp[p])/(p-1)
  Omatrix=Omatrix+delta*diag(p)
  Omatrix=cov2cor(Omatrix)  
  Omatrix
}



# proximal of L1 norm
prox_L1 = function(x, lambda){
  return(sign(x) * pmax(0, abs(x) - lambda))
}

# project onto L12
project_L12 = function(x, lambda, group, weights, xi){
  x[group] = prox_L1(x[group], lambda = lambda*xi)
  l2norm = sqrt(sum(x[group] ^ 2))
  if (l2norm > lambda*weights) {
    x[group] = x[group] * lambda * weights / l2norm
  }
  x
}


#### Compute the proximal operator of the latent sparse group lasso #################
#### solve  argmin_{beta}lambda*(||beta||_{G,tau} + ||beta||_{S,xi})+0.5*||beta-r||^2
proximal_operator <-function(r, pattern, lambda, xi, weights, tolerance, maxiter){
  p = length(r)
  
  ## find active groups
  norms_1 = pattern %*% abs(r)
  norms_2 = sqrt(pattern %*% (r ^ 2))
  active_index = which(norms_2 > lambda * weights & norms_1 > lambda * xi)
  m = length(active_index)
  if (m == 0) {
    rep(0, p)
  }
  else{
    if (m == 1) {
      r - project_L12(
        x = r,
        lambda = lambda,
        group = which(pattern[active_index,] != 0),
        weights = weights[active_index],
        xi = xi
      ) 
    }
    else{
      # groups=NULL
      groups = list()
      for (tt in 1:m) {
        groups[[tt]] = which(pattern[active_index[tt], ] != 0)
      }
      x_last = r
      zn = t(matrix(r, p, m))
      znplus1 = matrix(NA, m, p)
      pmatrix = matrix(0, m, p)
      for (n in 1:maxiter)
      {
        for (i in 1:m) {
          pmatrix[i,] = project_L12(
            x = zn[i,],
            lambda = lambda,
            group = groups[[i]],
            weights = weights[active_index[i]],
            xi = xi
          )
        }
        x_new = apply(pmatrix, 2, mean)
        if (sqrt(sum((x_new - x_last) ^ 2)) <= tolerance * sqrt(sum(x_last^2))) {
          cat('prox convergence.', '\n')
          break
        }
        znplus1 = zn - pmatrix + matrix(rep(x_new, each = m), m, p)
        zn = znplus1
        x_last = x_new
      }
      r-x_new
    }
  }
}

### Use the FISTA method to compute the whole solution path of DSGLMIG method ###
FISTA_L12 <- function(X, Y, pattern, weights, xi, nlambda, lambda.min.ratio,
                      tol_out, tol_in, maxiter_out, maxiter_in) {
  n = nrow(X)
  p = ncol(X)
  
  ## logit
  # mu = invlink(c(X %*% rep(0,p)))
  # L = max(eigen(nhessian(x = X, mu = mu))$values)/n
  L = ((eigen(t(X) %*% X)$values)[1]) / (4*n)
  gradient = ngradient(x = X, y = Y, mu = invlink(c(X %*% rep(0,p))))
  lambda_max = sqrt(sum(gradient^2)) / (n*min(weights))
  lambda_all = exp(seq(log(lambda_max), log(lambda_max * lambda.min.ratio), 
                       length = nlambda))
  beta_all = matrix(0, p, (1 + nlambda)) ## first beta is for initiation
  for (i in 1:nlambda)
  {
    beta_last = beta_all[, i]
    z_last = beta_last
    t_last = 1
    for (k in 1:maxiter_out)
    {
      ## logit
      mu = invlink(c(X %*% z_last))
      need_project = z_last - ngradient(x = X, y = Y, mu = mu) / (n * L)
      # need_project=z_last-(xtx%*%z_last)/(n*L)+xty/(n*L)
      beta_new = proximal_operator(r = need_project, pattern = pattern, 
                                   lambda = lambda_all[i] / L, xi = xi, 
                                   weights = weights, tolerance = tol_in, 
                                   maxiter = maxiter_in)
      distance = beta_new - beta_last
      if (sqrt(sum(distance ^ 2)) <= tol_out * sqrt(sum(beta_last ^ 2))) {
        cat('FISTA convergence for lambda = ', lambda_all[i], '\n')
        # print("FISTA convergence.")
        break
      }
      t_new = (1 + sqrt(1 + 4 * t_last * t_last)) / 2
      z_new = beta_new + (t_last - 1) * distance / t_new
      beta_last = beta_new
      t_last = t_new
      z_last = z_new
    }
    beta_all[, (i + 1)] = beta_new
  }
  list(beta_est = beta_all[, 2:(1 + nlambda)], lambda = lambda_all)
}



