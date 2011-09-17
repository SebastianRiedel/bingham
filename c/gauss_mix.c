
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include "bingham.h"
#include "bingham/util.h"
#include "bingham/gauss_mix.h"




static void gauss_mix_init(gauss_mix_t *gm, double **X, int npoints)
{
  int i, k = gm->n, dims = gm->d;

  // initialize means with k randomly chosen data points
  int randIndex[k];
  randperm(randIndex, npoints, k);
  reorder_rows(gm->means, X, randIndex, k, dims);

  // the initial estimates of the mixing probabilities are set to 1/k
  for (i = 0; i < k; i++)
    gm->weights[i] = 1.0 / (double)k;

  // the covariances are initialized to 1/10 the global covariance.
  // this is a completely aribitrary choice and can be changed. Original
  // code set covariance to an isotropic gaussian with diagonal covariance
  // for most cases the newer version converges faster.
  double globMu[dims];
  mean(globMu, X, npoints, dims);
  double **globCovar = new_matrix2(dims, dims);
  cov(globCovar, X, globMu, npoints, dims);
  for (i = 0; i < k; i++) {
    mult(gm->covs[i][0], globCovar[0], .1, dims*dims);
  }

  free_matrix2(globCovar);
}


// Indicators are indicator functions and will contain the assignments of each data point to
// the mixture components, as result of the E-step
static void compute_indicators(double **Indicators, double **X, int npoints, gauss_mix_t *gm)
{
  int i, j, k = gm->n;

  // having the initial means, covariances, and probabilities, we can
  // initialize the (unnormalized) indicator functions following the standard EM equation
  for (i = 0; i < k; i++)
    for (j = 0; j < npoints; j++)
      Indicators[i][j] = mvnpdf(X[j], gm->means[i], gm->covs[i], gm->d);
}


static double compute_log_likelihood(double **Indicators, gauss_mix_t *gm, double *W, int npoints)
{
  double SumIndicators[npoints];
  wmean(SumIndicators, Indicators, gm->weights, gm->n, npoints);

  double loglike = 0.0;
  double epsilon = 1e-50;
  int i;
  for (i = 0; i < npoints; i++)
    loglike += W[i]*log(SumIndicators[i] + epsilon);

  return loglike;
}


static double compute_description_length(double loglike, gauss_mix_t *gm, double W_tot)
{
  int i, k = gm->n, dims = gm->d;
  double npars = dims/2.0 + dims*(dims+1)/4.0;  // actually npars/2
  double sum_log_weights = 0.0;
  for (i = 0; i < k; i++)
    sum_log_weights += log(gm->weights[i]);

  return -loglike + (npars*sum_log_weights) + (npars + 0.5)*k*log(W_tot);
}


static void remove_component(gauss_mix_t *gm, double **Indicators, int comp, int npoints)
{
  int i, k = gm->n, dims = gm->d;
  for (i = comp; i < k-1; i++) {
    memcpy(gm->means[i], gm->means[i+1], dims*sizeof(double));
    matrix_copy(gm->covs[i], gm->covs[i+1], dims, dims);
    gm->weights[i] = gm->weights[i+1];
    memcpy(Indicators[i], Indicators[i+1], npoints*sizeof(double));
  }
  free_matrix2(gm->covs[k-1]);
  gm->n--;
  mult(gm->weights, gm->weights, 1.0/sum(gm->weights, gm->n), gm->n);
}


static int update_component(gauss_mix_t *gm, int comp, double **Indicators, double **X, double *W, int npoints, double regularize)
{
  int i, k = gm->n, dims = gm->d;
  double epsilon = 1e-50;
  double npars = dims/2.0 + dims*(dims+1)/4.0;  // actually npars/2

  double SumIndicators[npoints];
  wmean(SumIndicators, Indicators, gm->weights, gm->n, npoints);

  double P[npoints];
  for (i = 0; i < npoints; i++)
    P[i] = W[i] * gm->weights[comp] * Indicators[comp][i] / (epsilon + SumIndicators[i]);

  // compute weighted mean and covariance
  wmean(gm->means[comp], X, P, npoints, dims);
  wcov(gm->covs[comp], X, P, gm->means[comp], npoints, dims);
  for (i = 0; i < dims; i++)
    gm->covs[comp][i][i] += regularize;

  // this is the special part of the M step that is able to kill components
  gm->weights[comp] = (sum(P,npoints) - npars) / sum(W, npoints);
  if (gm->weights[comp] < 0)
    gm->weights[comp] = 0;
  mult(gm->weights, gm->weights, 1.0/sum(gm->weights,k), k);

  // we now have to do some book-keeping if the current component was killed
  if (gm->weights[comp] == 0) {
    remove_component(gm, Indicators, comp, npoints);
    return 1;
  }

  // if the component was not killed, we update the corresponding indicator variables...
  for (i = 0; i < npoints; i++)
    Indicators[comp][i] = mvnpdf(X[i], gm->means[comp], gm->covs[comp], dims);

  return 0;
}



//----------------------------  EXTERNAL API  -------------------------------//



gauss_mix_t *new_gauss_mix(int dims, int k)
{
  int i;
  gauss_mix_t *gm;
  safe_calloc(gm, 1, gauss_mix_t);
  gm->n = k;
  gm->d = dims;
  safe_calloc(gm->weights, k, double);
  gm->means = new_matrix2(k, dims);
  safe_calloc(gm->covs, k, double**);
  for (i = 0; i < k; i++)
    gm->covs[i] = new_matrix2(dims, dims);

  return gm;
}


gauss_mix_t *gauss_mix_clone(gauss_mix_t *gm)
{
  int i;
  int k = gm->n;
  int dims = gm->d;

  gauss_mix_t *gm2;
  safe_calloc(gm2, 1, gauss_mix_t);
  gm2->n = k;
  gm2->d = dims;
  safe_calloc(gm2->weights, k, double);
  memcpy(gm2->weights, gm->weights, k*sizeof(double));
  gm2->means = matrix_clone(gm->means, k, dims);
  safe_calloc(gm2->covs, k, double**);
  for (i = 0; i < k; i++)
    gm2->covs[i] = matrix_clone(gm->covs[i], dims, dims);
  
  return gm2;
}


void free_gauss_mix(gauss_mix_t *gm)
{
  if (gm->weights)
    free(gm->weights);
  if (gm->means)
    free_matrix2(gm->means);
  if (gm->covs) {
    int i;
    for (i = 0; i < gm->n; i++)
      if (gm->covs[i])
	free_matrix2(gm->covs[i]);
    free(gm->covs);
  }
  free(gm);
}


gauss_mix_t *fit_gauss_mix(double **X, int npoints, int dims, double *w, unsigned int kmin, unsigned int kmax, double regularize, double th)
{
  int i;
  double W[npoints];
  mult(W, w, npoints/sum(w,npoints), npoints);  // normalize data weights

  gauss_mix_t *gm = new_gauss_mix(dims, kmax);  // kmax is the initial number of mixture components
  gauss_mix_init(gm, X, npoints);

  double **Indicators = new_matrix2(kmax, npoints);
  compute_indicators(Indicators, X, npoints, gm);

  double loglike = compute_log_likelihood(Indicators, gm, W, npoints);
  double dl = compute_description_length(loglike, gm, sum(W,npoints));

  // minimum description length seen so far, and corresponding parameter estimates
  double minDL = dl;
  gauss_mix_t *bestGM = gauss_mix_clone(gm);

  while (1) {

    while (1) {
      int comp = 0;
      // Since k may change during the process, we can not use a for loop
      while (comp < gm->n) {
	if (update_component(gm, comp, Indicators, X, W, npoints, regularize) == 0)
	  comp++;
	//printf("gm->n = %d\n", gm->n);
      }
      // compute description length
      double loglikeprev = loglike;
      loglike = compute_log_likelihood(Indicators, gm, W, npoints);
      dl = compute_description_length(loglike, gm, sum(W,npoints));
      // check if new mixture is best
      if (dl < minDL) {
	minDL = dl;
	free_gauss_mix(bestGM);
	bestGM = gauss_mix_clone(gm);
      }

      printf("k = %d, dl = %f, loglike = %f, loglikeprev = %f\n", gm->n, dl, loglike, loglikeprev); //dbug

      // check for convergence
      if (fabs(loglike-loglikeprev) / loglikeprev < th)
	break;
    }

    if (gm->n == kmin)
      break;

    // try removing smallest component
    int comp = 0;
    for (i = 1; i < gm->n; i++)
      if (gm->weights[i] < gm->weights[comp])
	comp = i;
    remove_component(gm, Indicators, comp, npoints);
    loglike = compute_log_likelihood(Indicators, gm, W, npoints);

    printf("*** killed smallest ***\n"); //dbug
  }

  free_gauss_mix(gm);
  free_matrix2(Indicators);

  return bestGM;
}


//---------------------------  TESTING  -----------------------//


#include "bingham/olf.h"


int main(int argc, char *argv[])
{
  if (argc < 3) {
    printf("usage: %s <pcd> <point_index>\n", argv[0]);
    return 1;
  }

  pcd_t *pcd = load_pcd(argv[1]);
  int point_index = atof(argv[2]);

  int i;
  int c = pcd->clusters[point_index];
  int I[pcd->num_points];
  int n = findeq(I, pcd->clusters, c, pcd->num_points);

  double **Q = new_matrix2(2*n, 4);
  double **X = new_matrix2(2*n, 3);
  int cnt=0;
  for (i = 0; i < pcd->num_points; i++) {
    if (pcd->clusters[i] == c) {
      memcpy(Q[cnt], pcd->quaternions[0][i], 4*sizeof(double));
      memcpy(Q[cnt+1], pcd->quaternions[1][i], 4*sizeof(double));
      X[cnt][0] = X[cnt+1][0] = pcd->points[0][i];
      X[cnt][1] = X[cnt+1][1] = pcd->points[1][i];
      X[cnt][2] = X[cnt+1][2] = pcd->points[2][i];
      cnt += 2;
    }
  }


  // compute weights
  n *= 2;
  double *q = pcd->quaternions[0][point_index];
  double dq, qdot;
  double w[n];
  double r = .2;

  for (i = 0; i < n; i++) {
    qdot = fabs(dot(q, Q[i], 4));
    dq = acos(MIN(qdot, 1.0));
    w[i] = exp(-(dq/r)*(dq/r));
  }

  gauss_mix_t* gm = fit_gauss_mix(X, n, 3, w, 1, 20, 1e-15, 1e-4);

  printf("n = %d\n", gm->n);
  for (i = 0; i < gm->n; i++) {
    printf("w[%d] = %f\n", i, gm->weights[i]);
    printf("mu[%d] = [%f, %f, %f]\n", i, gm->means[i][0], gm->means[i][1], gm->means[i][2]);
    printf("cov[%d] = [%f, %f, %f;  %f, %f, %f;  %f, %f, %f]\n", i, gm->covs[i][0][0], gm->covs[i][0][1], gm->covs[i][0][2],
	   gm->covs[i][1][0], gm->covs[i][1][1], gm->covs[i][1][2], gm->covs[i][2][0], gm->covs[i][2][1], gm->covs[i][2][2]);
  }

  return 0;
}

