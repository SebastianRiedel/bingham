
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "bingham/util.h"
#include "bingham/hll.h"





/*
 * Fills in a new HLL with a default prior.
 */
static void hll_default_prior(hll_t *hll)
{
  int dx = hll->dx;
  int n = hll->n;

  safe_calloc(hll->x0, dx, double);
  mean(hll->x0, hll->X, n, dx);  // x0 = mean(X)

  hll->S0 = new_matrix2(dx, dx);
  cov(hll->S0, hll->X, hll->x0, n, dx);

  /*
  int i;
  double d2_tot = 0.0;
  for (i = 0; i < n; i++) {
    double d = norm(hll->X[i], dx);
    d2_tot += d*d;
  }
  double sigma = d2_tot/(3.0*n);
  for (i = 0; i < dx; i++)
    hll->S0[i][i] = sigma;
  */

  hll->w0 = 2;

  /*
    x0 = zeros(1,3);
    S0 = mean(sum(X.^2,2))/3 * eye(3);
    w0 = 2;
  */
}


/*
 * Sample from an hll cache
 */
static void hll_sample_cache(double **X, double ***S, double **Q, hll_t *hll, int n)
{
  int i;
  for (i = 0; i < n; i++) {
    int j = kdtree_NN(hll->cache.Q_kdtree, Q[i]);
    memcpy(X[i], hll->cache.X[j], hll->dx * sizeof(double));
    matrix_copy(S[i], hll->cache.S[j], hll->dx, hll->dx);
  }
}



//-----------------------------  EXTERNAL API  -------------------------------//



/*
 * Make a new HLL model from Q->X, with a default prior.
 */
void hll_new(hll_t *hll, double **Q, double **X, int n, int dq, int dx)
{
  hll->Q = Q;
  hll->X = X;
  hll->n = n;
  hll->dq = dq;
  hll->dx = dx;

  hll->r = .2;  //dbug: is there a more principled way of setting this?

  hll_default_prior(hll);
}


/*
 * Free an HLL cache.
 */
void hll_free_cache(hll_t *hll)
{
  if (hll->cache.n > 0) {
    if (hll->cache.Q_kdtree)
      kdtree_free(hll->cache.Q_kdtree);
    if (hll->cache.Q)
      free_matrix2(hll->cache.Q);
    if (hll->cache.X)
      free_matrix2(hll->cache.X);
    int i;
    if (hll->cache.S) {
      for (i = 0; i < hll->cache.n; i++)
	if (hll->cache.S[i])
	  free_matrix2(hll->cache.S[i]);
      free(hll->cache.S);
    }
    hll->cache.n = 0;
  }
}


/*
 * Free an HLL model.
 */
void hll_free(hll_t *hll)
{
  if (hll->Q)
    free_matrix2(hll->Q);
  if (hll->X)
    free_matrix2(hll->X);
  if (hll->x0)
    free(hll->x0);
  if (hll->S0)
    free(hll->S0);

  hll_free_cache(hll);
}


/*
 * Cache the Local Likelihood distributions for Q
 */
void hll_cache(hll_t *hll, double **Q, int n)
{
  int i;

  // allocate space, build kdtree
  double **X = new_matrix2(n, hll->dx);
  double ***S;
  safe_calloc(S, n, double**);
  for (i = 0; i < n; i++)
    S[i] = new_matrix2(hll->dx, hll->dx);
  kdtree_t *Q_kdtree = kdtree(Q, n, hll->dq);

  // precompute LL distributions
  hll_sample(X, S, Q, hll, n);

  // cache in hll
  hll->cache.Q_kdtree = Q_kdtree;
  hll->cache.Q = matrix_clone(Q, n, hll->dq);
  hll->cache.X = X;
  hll->cache.S = S;
  hll->cache.n = n;
}


/*
 * Sample n Gaussians (with means X and covariances S) from HLL at sample points Q.
 */
void hll_sample(double **X, double ***S, double **Q, hll_t *hll, int n)
{
  int i, j;
  if (hll->cache.n > 0) {
    for (i = 0; i < n; i++) {
      j = kdtree_NN(hll->cache.Q_kdtree, Q[i]);
      memcpy(X[i], hll->cache.X[j], hll->dx * sizeof(double));
      matrix_copy(S[i], hll->cache.S[j], hll->dx, hll->dx);
    }
    return;
  }

  double r = hll->r;
  int nx = hll->dx;
  int nq = hll->dq;

  double **WS = new_matrix2(nx, nx);

  for (i = 0; i < n; i++) {

    //printf("q = [%.2f %.2f %.2f %.2f]\n", Q[0][0], Q[0][1], Q[0][2], Q[0][3]);

    //for (j = 0; j < hll->n; j++)
    //  printf("hll->Q[%d] = [%.2f %.2f %.2f %.2f]\n", j, hll->Q[j][0], hll->Q[j][1], hll->Q[j][2], hll->Q[j][3]);

    // compute weights
    double dq, qdot;
    double w[hll->n];
    //printf("w = [");
    for (j = 0; j < hll->n; j++) {
      qdot = fabs(dot(Q[i], hll->Q[j], nq));
      dq = acos(MIN(qdot, 1.0));
      w[j] = exp(-(dq/r)*(dq/r));
      //printf("%.2f ", w[j]);
    }
    //printf("]\n");

    // threshold weights
    double wmax = arr_max(w, hll->n);
    double wthresh = wmax/50;  //dbug: make this a parameter?
    for (j = 0; j < hll->n; j++)
      if (w[j] < wthresh)
	w[j] = 0;
    double wtot = hll->w0 + sum(w, hll->n);

    // compute posterior mean
    mult(X[i], hll->x0, hll->w0, nx);  // X[i] = w0*x0
    for (j = 0; j < hll->n; j++) {
      if (w[j] > 0) {
	double wx[nx];
	mult(wx, hll->X[j], w[j], nx);
	add(X[i], X[i], wx, nx);       // X[i] += wx
      }
    }
    mult(X[i], X[i], 1/wtot, nx);  // X[i] /= wtot

    // compute posterior covariance matrix
    mult(S[i][0], hll->S0[0], hll->w0, nx*nx);  // S[i] = w0*S0

    for (j = 0; j < hll->n; j++) {
      if (w[j] > 0) {
	double wdx[nx];
	sub(wdx, hll->X[j], X[i], nx);
	mult(wdx, wdx, w[j], nx);
	outer_prod(WS, wdx, wdx, nx, nx);    // WS = wdx'*wdx
	matrix_add(S[i], S[i], WS, nx, nx);  // S[i] += WS
      }
    }

    mult(S[i][0], S[i][0], 1/wtot, nx*nx);  // S[i] /= wtot
  }

  free_matrix2(WS);
}


/*
 * Load hlls from a file.
 *
 * .HLL format:
 *
 *    num_hlls
 *    <hll> <n> <ncache> <dq> <dx> <r> <w0> <x0> <S0>
 *      ...
 *    Q <hll> <i> <q> <x> 
 *      ...
 *    C <hll> <i> <q> <n> <x> <S>
 *      ...
 */
hll_t *load_hlls(char *fname, int *n)
{
  FILE *f = fopen(fname, "r");
  if (f == NULL) {
    fprintf(stderr, "Error: Can't open %s for reading\n", fname);
    return NULL;
  }

  int i, i2, j, j2, k, l;
  hll_t *hlls = NULL;

  // read num hlls
  if (fscanf(f, "%d\n", n) < 1) goto ERROR;

  // create hlls
  safe_calloc(hlls, *n, hll_t);

  //fprintf(stderr, "break 1\n"); //dbug

  // read hll headers
  for (i2 = 0; i2 < *n; i2++) {
    if (fscanf(f, "%d ", &i) < 1) goto ERROR;
    if (fscanf(f, "%d %d %d %d %lf %lf ", &hlls[i].n, &hlls[i].cache.n, &hlls[i].dq, &hlls[i].dx, &hlls[i].r, &hlls[i].w0) < 6) goto ERROR;

    //fprintf(stderr, "break 1.1\n"); //dbug

    // alloc Q,X,x0,S0
    hlls[i].Q = new_matrix2(hlls[i].n, hlls[i].dq);
    hlls[i].X = new_matrix2(hlls[i].n, hlls[i].dx);
    safe_calloc(hlls[i].x0, hlls[i].dx, double);
    hlls[i].S0 = new_matrix2(hlls[i].dx, hlls[i].dx);

    // alloc cache
    if (hlls[i].cache.n > 0) {
      hlls[i].cache.Q = new_matrix2(hlls[i].cache.n, hlls[i].dq);
      hlls[i].cache.X = new_matrix2(hlls[i].cache.n, hlls[i].dx);
      safe_calloc(hlls[i].cache.S, hlls[i].cache.n, double**);
      for (j = 0; j < hlls[i].cache.n; j++)
	hlls[i].cache.S[j] = new_matrix2(hlls[i].dx, hlls[i].dx);
    }

    //fprintf(stderr, "break 1.2\n"); //dbug

    // read x0,S0
    for (j = 0; j < hlls[i].dx; j++)
      if (fscanf(f, "%lf ", &hlls[i].x0[j]) < 1) goto ERROR;
    for (j = 0; j < hlls[i].dx; j++)
      for (k = 0; k < hlls[i].dx; k++)
	if (fscanf(f, "%lf ", &hlls[i].S0[j][k]) < 1) goto ERROR;
    //fscanf(f, "\n");

    //fprintf(stderr, "break 1.3\n"); //dbug
  }

  //fprintf(stderr, "break 2\n"); //dbug

  // read data
  for (i2 = 0; i2 < *n; i2++) {
    for (j2 = 0; j2 < hlls[i2].n; j2++) {
      if (fscanf(f, "Q %d %d ", &i, &j) < 2) goto ERROR;
      for (k = 0; k < hlls[i].dq; k++)
	if (fscanf(f, "%lf ", &hlls[i].Q[j][k]) < 1) goto ERROR;
      for (k = 0; k < hlls[i].dx; k++)
	if (fscanf(f, "%lf ", &hlls[i].X[j][k]) < 1) goto ERROR;
      //fscanf(f, "\n");
    }
  }

  //fprintf(stderr, "break 3\n"); //dbug

  // read cache
  for (i2 = 0; i2 < *n; i2++) {
    for (j2 = 0; j2 < hlls[i2].cache.n; j2++) {
      if (fscanf(f, "C %d %d ", &i, &j) < 2) goto ERROR;
      for (k = 0; k < hlls[i].dq; k++)
	if (fscanf(f, "%lf ", &hlls[i].cache.Q[j][k]) < 1) goto ERROR;
      int nx;
      if (fscanf(f, "%d ", &nx) < 1) goto ERROR;  //dbug: change this for mixtures
      for (k = 0; k < hlls[i].dx; k++)
	if (fscanf(f, "%lf ", &hlls[i].cache.X[j][k]) < 1) goto ERROR;
      for (k = 0; k < hlls[i].dx; k++)
	for (l = 0; l < hlls[i].dx; l++)
	  if (fscanf(f, "%lf ", &hlls[i].cache.S[j][k][l]) < 1) goto ERROR;
      //fscanf(f, "\n");
    }
  }

  //fprintf(stderr, "break 4\n"); //dbug

  // create cache kdtrees
  for (i = 0; i < *n; i++)
    hlls[i].cache.Q_kdtree = kdtree(hlls[i].cache.Q, hlls[i].cache.n, hlls[i].dq);

  fclose(f);

  //save_hlls("test.hll", hlls, *n); //dbug

  return hlls;

 ERROR:
  if (hlls) {
    for (i = 0; i < *n; i++)
      hll_free(&hlls[i]);
    free(hlls);
  }
  return NULL;
}


/*
 * Save hlls to a file.
 */
void save_hlls(char *fname, hll_t *hlls, int n)
{
  FILE *f = fopen(fname, "w");
  if (f == NULL) {
    fprintf(stderr, "Error: Can't open %s for writing\n", fname);
    return;
  }

  int i, j, k, l;

  // header
  fprintf(f, "%d\n", n);
  for (i = 0; i < n; i++) {
    fprintf(f, "%d %d %d %d %d %f %f ", i, hlls[i].n, hlls[i].cache.n, hlls[i].dq, hlls[i].dx, hlls[i].r, hlls[i].w0);
    for (j = 0; j < hlls[i].dx; j++)
      fprintf(f, "%f ", hlls[i].x0[j]);
    for (j = 0; j < hlls[i].dx; j++)
      for (k = 0; k < hlls[i].dx; k++)
	fprintf(f, "%f ", hlls[i].S0[j][k]);
    fprintf(f, "\n");
  }

  // data
  for (i = 0; i < n; i++) {
    for (j = 0; j < hlls[i].n; j++) {
      fprintf(f, "Q %d %d ", i, j);
      for (k = 0; k < hlls[i].dq; k++)
	fprintf(f, "%f ", hlls[i].Q[j][k]);
      for (k = 0; k < hlls[i].dx; k++)
	fprintf(f, "%f ", hlls[i].X[j][k]);
      fprintf(f, "\n");
    }
  }

  // cache
  for (i = 0; i < n; i++) {
    for (j = 0; j < hlls[i].cache.n; j++) {
      fprintf(f, "C %d %d ", i, j);
      for (k = 0; k < hlls[i].dq; k++)
	fprintf(f, "%f ", hlls[i].cache.Q[j][k]);
      fprintf(f, "1 ");  //dbug: change this for mixtures
      for (k = 0; k < hlls[i].dx; k++)
	fprintf(f, "%f ", hlls[i].cache.X[j][k]);
      for (k = 0; k < hlls[i].dx; k++)
	for (l = 0; l < hlls[i].dx; l++)
	  fprintf(f, "%f ", hlls[i].cache.S[j][k][l]);
      fprintf(f, "\n");
    }
  }

  fclose(f);
}
