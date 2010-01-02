
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "bingham.h"
#include "bingham/util.h"
#include "bingham/hypersphere.h"
#include "bingham/bingham_constants.h"


#define EPSILON 1e-8




//------------------- Bingham data log likelihood and partial derivatives -------------------//


/*
 * Computes the average log likelihood of the parameters B->Z and B->V given n samples X.
 */
double bingham_L(bingham_t *B, double **X, int n)
{
  int i, j, d = B->d;
  double dvx;
  double N = (double)n;

  double logf = 0;
  for (i = 0; i < n; i++) {
    for (j = 0; j < d-1; j++) {
      dvx = dot(B->V[j], X[i], d);
      logf += B->Z[j]*dvx*dvx;
    }
  }

  logf = logf/N - log(B->F);

  return logf;
}


/*
 * Computes the partial derivatives of B->F w.r.t. B->Z
 */
static void bingham_dF(double *dF, bingham_t *B)
{
  double *Z = B->Z;

  switch(B->d) {
  case 2:
    dF[0]= bingham_dF_1d(Z[0]);
    break;
  case 3:
    dF[0]= bingham_dF1_2d(Z[0], Z[1]);
    dF[1] = bingham_dF2_2d(Z[0], Z[1]);
    break;
  case 4:
    dF[0]= bingham_dF1_3d(Z[0], Z[1], Z[2]);
    dF[1] = bingham_dF2_3d(Z[0], Z[1], Z[2]);
    dF[2] = bingham_dF3_3d(Z[0], Z[1], Z[2]);
  }
}


/*
 * Computes the partial derivatives of the average log likelihood w.r.t. B->Z.
 */
static void bingham_dL(double *dL, bingham_t *B, double **X, int n)
{
  int i, j, d = B->d;
  double dvx;
  double F = B->F;
  double dF[d-1];
  bingham_dF(dF, B);

  memset(dL, 0, (d-1)*sizeof(double));
  for (i = 0; i < n; i++) {
    for (j = 0; j < d-1; j++) {
      dvx = dot(B->V[j], X[i], d);
      dL[j] += dvx*dvx;
    }
  }

  mult(dL, dL, 1/(double)n, d-1);
  mult(dF, dF, -1/F, d-1);
  add(dL, dL, dF, d-1);
}



//------------------- Bingham MLE of parameters -------------------//


/*
 * Compute MLE parameters Z given n samples X with principal components V
 * using gradient descent.
 *
static void bingham_MLE_2d_grad_desc(double *Z, double V[][3], double X[][3], int n)
{
  double YMIN = 0.0, YMAX = 25.0;
  int i, iter = 100;
  double L_prev, L, dL[2], Y[2], Y_next[2], Z_next[2], step = 1;

  // gradient descent w.r.t. y = -sqrt(z) is more stable
  Y[0] = sqrt(-Z[0]);
  Y[1] = sqrt(-Z[1]);

  L_prev = bingham_L_2d(Z, V, X, n);

  for (i = 0; i < iter; i++) {
    bingham_dL_2d(dL, Z, V, X, n);

    // gradient descent w.r.t. -sqrt(z)
    dL[0] = -2*Y[0]*dL[0];
    dL[1] = -2*Y[1]*dL[1];

    step *= 1.2;
    while (1) {
      //Z_next[0] = Z[0] + step*dL[0];
      //Z_next[1] = Z[1] + step*dL[1];

      // gradient descent w.r.t. -sqrt(z)
      Y_next[0] = Y[0] + step*dL[0];
      Y_next[1] = Y[1] + step*dL[1];

      // restrict Y's to reasonable range
      Y_next[0] = MAX(Y_next[0], YMIN);
      Y_next[0] = MIN(Y_next[0], YMAX);
      Y_next[1] = MAX(Y_next[1], YMIN);
      Y_next[1] = MIN(Y_next[1], YMAX);
      Y_next[1] = MIN(Y_next[0], Y_next[1]);

      Z_next[0] = -Y_next[0]*Y_next[0];
      Z_next[1] = -Y_next[1]*Y_next[1];

      //printf("  -->  Y_next = (%f, %f), Z_next = (%f, %f)\n", Y_next[0], Y_next[1], Z_next[0], Z_next[1]);
      L = bingham_L_2d(Z_next, V, X, n);
      if (L > L_prev - EPSILON) {
	break;
      }

      step *= .5;
      printf(".");
      fflush(0);
    }

    L_prev = L;
    printf("Y = (%f, %f), Z = (%f, %f) --> L = %f, dL = (%f, %f), step = %f\n", Y[0], Y[1], Z[0], Z[1], L, dL[0], dL[1], step);

    if (fabs(dL[0]) < 1e-5 && fabs(dL[1]) < 1e-5)
      break;

    //Z[0] += step*dL[0];
    //Z[1] += step*dL[1];

    // gradient descent w.r.t. -sqrt(z)
    Y[0] = Y_next[0];
    Y[1] = Y_next[1];
    Z[0] = Z_next[0];
    Z[1] = Z_next[1];
  }
}
*/

/*
 * Compute MLE parameters Z given n samples X with principal components V
 * using gradient descent.
 *
static void bingham_MLE_3d_grad_desc(double *Z, double V[][4], double X[][4], int n)
{
  double YMIN = 0.0, YMAX = 25.0;
  int i, iter = 100;
  double L_prev, L, dL[3], Y[3], Y_next[3], Z_next[3], step = 1;

  // gradient descent w.r.t. y = -sqrt(z) is more stable
  Y[0] = sqrt(-Z[0]);
  Y[1] = sqrt(-Z[1]);
  Y[2] = sqrt(-Z[2]);

  L_prev = bingham_L_3d(Z, V, X, n);

  for (i = 0; i < iter; i++) {
    bingham_dL_3d(dL, Z, V, X, n);

    // gradient descent w.r.t. -sqrt(z)
    dL[0] = -2*Y[0]*dL[0];
    dL[1] = -2*Y[1]*dL[1];
    dL[2] = -2*Y[2]*dL[2];

    step *= 1.2;
    while (1) {

      // gradient descent w.r.t. -sqrt(z)
      Y_next[0] = Y[0] + step*dL[0];
      Y_next[1] = Y[1] + step*dL[1];
      Y_next[2] = Y[2] + step*dL[2];

      // restrict Y's to reasonable range
      Y_next[0] = MAX(Y_next[0], YMIN);
      Y_next[0] = MIN(Y_next[0], YMAX);
      Y_next[1] = MAX(Y_next[1], YMIN);
      Y_next[1] = MIN(Y_next[1], YMAX);
      Y_next[1] = MIN(Y_next[0], Y_next[1]);
      Y_next[2] = MAX(Y_next[2], YMIN);
      Y_next[2] = MIN(Y_next[2], YMAX);
      Y_next[2] = MIN(Y_next[1], Y_next[2]);

      Z_next[0] = -Y_next[0]*Y_next[0];
      Z_next[1] = -Y_next[1]*Y_next[1];
      Z_next[2] = -Y_next[2]*Y_next[2];

      //printf("  -->  Y_next = (%f, %f), Z_next = (%f, %f)\n", Y_next[0], Y_next[1], Z_next[0], Z_next[1]);
      L = bingham_L_3d(Z_next, V, X, n);
      if (L > L_prev - EPSILON) {
	break;
      }

      step *= .5;
      printf(".");
      fflush(0);
    }

    L_prev = L;
    printf("Y = (%f, %f, %f), Z = (%f, %f, %f) --> L = %f, dL = (%f, %f, %f), step = %f\n",
	   Y[0], Y[1], Y[2], Z[0], Z[1], Z[2], L, dL[0], dL[1], dL[2], step);

    if (fabs(dL[0]) < 1e-5 && fabs(dL[1]) < 1e-5 && fabs(dL[2]) < 1e-5)
      break;

    // gradient descent w.r.t. -sqrt(z)
    Y[0] = Y_next[0];
    Y[1] = Y_next[1];
    Y[2] = Y_next[2];
    Z[0] = Z_next[0];
    Z[1] = Z_next[1];
    Z[2] = Z_next[2];
  }
}
*/


/*
 * Compute MLE parameters concentration parameters B->Z given n samples X with principal components B->V
 * using gradient descent.
 */
static void bingham_MLE_grad_desc(bingham_t *B, double **X, int n)
{
  double YMIN = 0.0, YMAX = 25.0;
  int i, j, d = B->d, iter = 20;
  double L_prev, L, dL[d-1], Y[d-1], Y_next[d-1], Z_next[d-1], step = 1;
  bingham_t B_next;
  memcpy(&B_next, B, sizeof(bingham_t));
  B_next.Z = Z_next;

  // gradient descent w.r.t. y = sqrt(-z) is more stable
  for (j = 0; j < d-1; j++)
    Y[j] = sqrt(-B->Z[j]);

  bingham_F(B);
  L_prev = bingham_L(B, X, n);

  for (i = 0; i < iter; i++) {

    bingham_dL(dL, B, X, n);

    vmult(dL, dL, Y, d-1);  // dL *= Y
    mult(dL, dL, -2, d-1);  // dL *= -2

    step *= 1.2;
    while (1) {

      mult(Y_next, dL, step, d-1);  // Y_next = step*dL
      add(Y_next, Y_next, Y, d-1);  // Y_next += Y

      // restrict Y's to reasonable range
      for (j = 0; j < d-1; j++) {
	Y_next[j] = MAX(Y_next[j], YMIN);
	Y_next[j] = MIN(Y_next[j], YMAX);
	if (j > 0)
	  Y_next[j] = MIN(Y_next[j-1], Y_next[j]);
      }

      vmult(Z_next, Y_next, Y_next, d-1);  // Z = Y_next.*Y_next
      mult(Z_next, Z_next, -1, d-1);       // Z = -Z

      bingham_F(&B_next);
      L = bingham_L(&B_next, X, n);
      if (L > L_prev - EPSILON) {
	break;
      }

      step *= .5;
      //printf(".");
      //fflush(0);
    }

    L_prev = L;
    //printf("Y = (%f, %f, %f), Z = (%f, %f, %f) --> L = %f, dL = (%f, %f, %f), step = %f\n",
    //	   Y[0], Y[1], Y[2], B->Z[0], B->Z[1], B->Z[2], L, dL[0], dL[1], dL[2], step);

    if (fabs(max(dL, d-1)) < 1e-5 && fabs(min(dL, d-1)) < 1e-5)
      break;

    memcpy(Y, Y_next, (d-1)*sizeof(double));
    memcpy(B->Z, Z_next, (d-1)*sizeof(double));
    bingham_F(B);
  }
}


/*
 * Compute MLE parameters concentration parameters B->Z given scatter matrix S with principal components B->V
 * using NN lookup.
 */
static void bingham_MLE_NN(bingham_t *B, double **S)
{
  int d = B->d;

  if (d != 4) {
    fprintf(stderr, "Error: bingham_MLE_NN() is only implemented for d = 4!  Exiting...\n");
    exit(1);
  }

  double sv0[d], sv1[d], sv2[d];

  sv0[0] = dot(S[0], B->V[0], d);
  sv0[1] = dot(S[1], B->V[0], d);
  sv0[2] = dot(S[2], B->V[0], d);
  sv0[3] = dot(S[3], B->V[0], d);

  sv1[0] = dot(S[0], B->V[1], d);
  sv1[1] = dot(S[1], B->V[1], d);
  sv1[2] = dot(S[2], B->V[1], d);
  sv1[3] = dot(S[3], B->V[1], d);

  sv2[0] = dot(S[0], B->V[2], d);
  sv2[1] = dot(S[1], B->V[2], d);
  sv2[2] = dot(S[2], B->V[2], d);
  sv2[3] = dot(S[3], B->V[2], d);

  double dY[d-1];
  dY[0] = dot(B->V[0], sv0, d);
  dY[1] = dot(B->V[1], sv1, d);
  dY[2] = dot(B->V[2], sv2, d);

  //printf("dY = [%f %f %f]\n", dY[0], dY[1], dY[2]);

  //mult(dY, dY, 1/(double)n, d-1);

  bingham_dY_params_3d(B->Z, &B->F, dY);

  //bingham_F(B);  //dbug
}



//------------------- Bingham API -------------------//


/*
 * Initialize the bingham library.
 */
void bingham_init()
{
  bingham_constants_init();
  hypersphere_init();
}


/*
 * Update the normalization constant in B (and return it).
 */
double bingham_F(bingham_t *B)
{
  double *Z = B->Z;

  switch (B->d) {
  case 2:
    B->F = bingham_F_1d(Z[0]);
    break;
  case 3:
    B->F = bingham_F_2d(Z[0], Z[1]);
    break;
  case 4:
    B->F = bingham_F_3d(Z[0], Z[1], Z[2]);
    break;
  default:
    B->F = 0;
    fprintf(stderr, "Warning: bingham_F() doesn't know how to handle Bingham distributions in %d dimensions\n", B->d);
  }

  return B->F;
}


/*
 * Copy the contents of one bingham distribution into another.
 * Note: bingham_copy() blindly allocates space for dst->V and dst->Z
 * without freeing them first, so it should only be called on a freshly
 * allocated bingham_t struct.
 */
void bingham_copy(bingham_t *dst, bingham_t *src)
{
  int d = src->d;

  dst->d = d;

  dst->V = new_matrix2(d-1, d);
  memcpy(dst->V[0], src->V[0], d*(d-1)*sizeof(double));

  safe_malloc(dst->Z, d-1, double);
  memcpy(dst->Z, src->Z, (d-1)*sizeof(double));

  dst->F = src->F;
}


/*
 * Free the contents of a bingham
 */
void bingham_free(bingham_t *B)
{
  free_matrix2(B->V);
  free(B->Z);
}


/*
 * Allocate the contents of a bingham
 */
void bingham_alloc(bingham_t *B, int d)
{
  B->d = d;
  B->V = new_matrix2(d-1, d);
  safe_malloc(B->Z, d-1, double);
}


/*
 * Create a new Bingham distribution.
 *
 * @param B Bingham distribution to create
 * @param d dimensionality of the distribution
 * @param V (d-1)-by-d matrix of principle axes
 * @param Z (d-1)-length array of concentration parameters
 */
void bingham_new(bingham_t *B, int d, double **V, double *Z)
{
  int i;

  B->d = d;

  B->V = new_matrix2(d-1, d);
  for (i = 0; i < d-1; i++)
    memcpy(B->V[i], V[i], d*sizeof(double));

  safe_malloc(B->Z, d-1, double);
  memcpy(B->Z, Z, (d-1)*sizeof(double));

  bingham_F(B);
}


/*
 * Create a new Bingham distribution on the unit circle S1.
 */
void bingham_new_S1(bingham_t *B, double *v1, double z1)
{
  double *V[1] = {v1};
  double Z[1] = {z1};

  bingham_new(B, 2, V, Z);
}


/*
 * Create a new Bingham distribution on the unit sphere S2.
 */
void bingham_new_S2(bingham_t *B, double *v1, double *v2, double z1, double z2)
{
  double *V[2] = {v1, v2};
  double Z[2] = {z1, z2};

  bingham_new(B, 3, V, Z);
}


/*
 * Create a new Bingham distribution on the unit 3-sphere S3.
 */
void bingham_new_S3(bingham_t *B, double *v1, double *v2, double *v3, double z1, double z2, double z3)
{
  double *V[3] = {v1, v2, v3};
  double Z[3] = {z1, z2, z3};

  bingham_new(B, 4, V, Z);
}


/*
 * Evaluate the PDF of a bingham.
 */
double bingham_pdf(double x[], bingham_t *B)
{
  return exp(bingham_L(B, &x, 1));
}


/*
 * Fit a bingham to a set of samples.
 */
void bingham_fit(bingham_t *B, double **X, int n, int d)
{
  double **Xt = new_matrix2(d, n);
  transpose(Xt, X, n, d);
  double **S = new_matrix2(d, d);
  matrix_mult(S, Xt, X, d, n, d);
  mult(S[0], S[0], 1/(double)n, d*d);

  bingham_fit_scatter(B, S, d);

  free_matrix2(S);
}


/*
 * Fit a bingham to the scatter matrix (X'*X) of a set of samples.
 */
void bingham_fit_scatter(bingham_t *B, double **S, int d)
{
  // use PCA to get B->V
  double *eigenvals;
  safe_malloc(eigenvals, d, double);
  double **V = new_matrix2(d, d);
  eigen_symm(eigenvals, V, S, d);

  //printf("eigenvals = [%f %f %f %f]\n", eigenvals[0], eigenvals[1], eigenvals[2], eigenvals[3]);
  //printf("V = [%f %f %f %f ; %f %f %f %f ; %f %f %f %f ; %f %f %f %f]\n",
  //	 V[0][0], V[0][1], V[0][2], V[0][3],
  //	 V[1][0], V[1][1], V[1][2], V[1][3],
  //	 V[2][0], V[2][1], V[2][2], V[2][3],
  //	 V[3][0], V[3][1], V[3][2], V[3][3]);

  B->d = d;
  B->V = V;
  safe_calloc(B->Z, d-1, double);

  // init B->Z
  //int i;
  //for (i = 0; i < d-1; i++)
  //  B->Z[i] = -1;

  //double t0, t1, L;
  //t0 = get_time_ms();
  bingham_MLE_NN(B, S);
  //t1 = get_time_ms();
  //L = bingham_L(B, X, n);
  //printf("Computed MLE (NN) in %.2f ms:  Z = (%f, %f, %f)  --> L = %f\n", t1-t0, B->Z[0], B->Z[1], B->Z[2], L);

  //t0 = get_time_ms();
  //bingham_MLE_grad_desc(B, X, n);
  //t1 = get_time_ms();
  //L = bingham_L(B, X, n);
  //printf("Computed MLE (grad) in %.2f ms:  Z = (%f, %f, %f) --> L = %f\n", t1-t0, B->Z[0], B->Z[1], B->Z[2], L);

  free(eigenvals);
}


/*
 * Discretize a Bingham distribution.
 */
void bingham_discretize(bingham_pmf_t *pmf, bingham_t *B, int ncells)
{
  int i, d = B->d;

  pmf->d = d;
  pmf->resolution = 1/(double)ncells;

  if (d == 4) {

    // mesh
    //octetramesh_t *oct = hypersphere_tessellation_octetra(ncells);
    //pmf->tetramesh = octetramesh_to_tetramesh(oct);
    //octetramesh_free(oct);
    //free(oct);
    pmf->tetramesh = tessellate_S3(ncells);

    double t0 = get_time_ms();  //dbug

    // points and volumes
    int n = pmf->tetramesh->nt;
    pmf->n = n;
    pmf->points = new_matrix2(n, d);
    safe_malloc(pmf->volumes, n, double);
    tetramesh_centroids(pmf->points, pmf->volumes, pmf->tetramesh);

    fprintf(stderr, "Created points and volumes in %.0f ms\n", get_time_ms() - t0);  //dbug

    t0 = get_time_ms();  //dbug

    // probability mass
    safe_malloc(pmf->mass, n, double);
    double tot_mass = 0;
    for (i = 0; i < n; i++) {
      mult(pmf->points[i], pmf->points[i], 1/norm(pmf->points[i], d), d);
      pmf->mass[i] = pmf->volumes[i] * exp(bingham_L(B, &pmf->points[i], 1));
      tot_mass += pmf->mass[i];
    }
    mult(pmf->mass, pmf->mass, 1/tot_mass, n);

    fprintf(stderr, "Computed probabilities in %.0f ms\n", get_time_ms() - t0);  //dbug

  }
  else {
    fprintf(stderr, "Warning: bingham_discretize() doesn't know how to discretize distributions in %d dimensions.\n", d);
    return;
  }
}


/*
 * Simulate samples from a discrete Bingham distribution.
 */
void bingham_sample(double **X, bingham_pmf_t *pmf, int n)
{
  int i;

  // compute the CDF
  double *cdf;  safe_malloc(cdf, pmf->n, double);

  memset(cdf, 0, pmf->n * sizeof(double));
  cdf[0] = pmf->mass[0];
  for (i = 1; i < pmf->n; i++)
    cdf[i] = cdf[i-1] + pmf->mass[i];

  // sample from the inverse CDF
  for (i = 0; i < n; i++) {
    double u = frand();
    int cell = binary_search(u, cdf, pmf->n);

    if (pmf->d == 4) {

      double *v0 = pmf->tetramesh->vertices[ pmf->tetramesh->tetrahedra[cell][0] ];
      double *v1 = pmf->tetramesh->vertices[ pmf->tetramesh->tetrahedra[cell][1] ];
      double *v2 = pmf->tetramesh->vertices[ pmf->tetramesh->tetrahedra[cell][2] ];
      double *v3 = pmf->tetramesh->vertices[ pmf->tetramesh->tetrahedra[cell][3] ];
      double *S[4] = {v0, v1, v2, v3};

      //double x1[4], x2[4];
      //avg(x1, v0, v1, 4);
      //avg(x2, v2, v3, 4);
      //avg(X[i], x1, x2, 4);

      sample_simplex(X[i], S, pmf->d, pmf->d);
    }
    else {
      fprintf(stderr, "Warning: bingham_discretize() doesn't know how to discretize distributions in %d dimensions.\n", pmf->d);
      free(cdf);
      return;
    }
  }

  free(cdf);
}


/*
 * Fits a Bingham distribution to the rows of X with MLESAC.
 * Fills in B and outliers, and returns the number of outliers.
 */
int bingham_fit_mlesac(bingham_t *B, int *outliers, double **X, int n, int d)
{
  //fprintf(stderr, "bingham_fit_mlesac()\n");

  int i, j, r[d], iter = 100;
  double p0 = 1 / surface_area_sphere(d-1);
  double logp0 = log(p0);
  double pmax = 0;
  int first = 1;

  //fprintf(stderr, "p0 = %f, logp0 = %f\n", p0, logp0);

  double **Xi = new_matrix2(d, d);

  bingham_t Bi;

  for (i = 0; i < iter; i++) {

    // pick d points at random from X (no replacement)
    randperm(r, n, d);
    for (j = 0; j < d; j++)
      memcpy(Xi[j], X[r[j]], d*sizeof(double));

    //fprintf(stderr, "r = [%d %d %d %d]\n", r[0], r[1], r[2], r[3]);

    // fit a Bingham to the d points
    bingham_fit(&Bi, Xi, d, d);

    // compute data log likelihood
    double logp = 0;
    for (j = 0; j < n; j++) {
      double p = bingham_pdf(X[j], &Bi);
      if (p > p0) {
	logp += log(p);
	//fprintf(stderr, "+");
	//fflush(stderr);
      }
      else {
	logp += logp0;
	//fprintf(stderr, ".");
	//fflush(stderr);
      }
    }
    //fprintf(stderr, "\n");

    //fprintf(stderr, "logp = %f\n", logp);

    if (first || (logp > pmax)) {

      //fprintf(stderr, " *** new best, logp = %f ***\n", logp);

      pmax = logp;
      if (first)
	first = 0;
      else
	bingham_free(B);
      memcpy(B, &Bi, sizeof(bingham_t));  // copy pointers from Bi to B
    }
    else
      bingham_free(&Bi);
  }
  free_matrix2(Xi);

  // find inliers/outliers
  int L[n];
  for (i = 0; i < n; i++) {
    double p = bingham_pdf(X[i], B);
    if (p > p0)
      L[i] = 1;
    else
      L[i] = 0;
  }
  int num_inliers = count(L, n);
  int num_outliers = n - num_inliers;
  int inliers[num_inliers];
  find(inliers, L, n);
  vnot(L, L, n);
  find(outliers, L, n);

  //fprintf(stderr, "inliers = [ ");
  //for (i = 0; i < num_inliers; i++)
  //  fprintf(stderr, "%d ", inliers[i]);
  //fprintf(stderr, "];\n");

  //fprintf(stderr, "outliers = [ ");
  //for (i = 0; i < num_outliers; i++)
  //  fprintf(stderr, "%d ", outliers[i]);
  //fprintf(stderr, "];\n");

  // fit B to all the inliers
  bingham_free(B);
  Xi = new_matrix2(num_inliers, d);
  for (j = 0; j < num_inliers; j++)
    memcpy(Xi[j], X[inliers[j]], d*sizeof(double));
  bingham_fit(B, Xi, num_inliers, d);
  free_matrix2(Xi);

  return num_outliers;
}


/*
 * Fits a mixture of bingham distributions to the rows of X using a sample consensus algorithm.
 * Fills in num_clusters and weights, and returns an array of bingham_t.
 */
void bingham_cluster(bingham_mix_t *BM, double **X, int n, int d)
{
  const int min_points = 20;  // TODO: make this a parameter
  const int iter = 100;
  int outliers[n];
  int num_outliers;
  int i, j;

  int capacity = 100;
  safe_calloc(BM->B, capacity, bingham_t);
  safe_calloc(BM->w, capacity, double);
  BM->n = 0;

  for (i = 0; i < iter; i++) {

    num_outliers = bingham_fit_mlesac(&BM->B[i], outliers, X, n, d);
    BM->w[i] = n - num_outliers;

    //fprintf(stderr, "num_outliers = %d, w[%d] = %.0f\n", num_outliers, i, BM->w[i]);

    if (BM->w[i] >= min_points)
      BM->n++;
    else
      break;

    if (num_outliers < min_points)
      break;

    n = num_outliers;
    double **X2 = new_matrix2(n, d);
    for (j = 0; j < n; j++)
      memcpy(X2[j], X[outliers[j]], d*sizeof(double));
    if (i > 0)
      free_matrix2(X);
    X = X2;
  }
  if (i > 0)
    free(X);

  safe_realloc(BM->B, BM->n, bingham_t);
  safe_realloc(BM->w, BM->n, double);
  mult(BM->w, BM->w, 1/sum(BM->w, BM->n), BM->n);
}


/*
 * Multiplies two bingham distributions, B1 and B2.  Assumes B is already allocated.
 *
 * TODO: Make this faster.
 */
void bingham_mult(bingham_t *B, bingham_t *B1, bingham_t *B2)
{
  if (B1->d != B2->d) {
    fprintf(stderr, "Error: B1->d != B2->d in bingham_mult()!\n");
    return;
  }

  int i, j;
  int d = B1->d;

  B->d = d;

  double **C1 = new_matrix2(d, d);
  double **C2 = new_matrix2(d, d);
  double **C = new_matrix2(d, d);

  double **Z = new_matrix2(d-1, d-1);
  double **Vt = new_matrix2(d, d-1);
  double **ZV = new_matrix2(d-1, d);

  // compute C1
  for (i = 0; i < d-1; i++)
    Z[i][i] = B1->Z[i];
  transpose(Vt, B1->V, d-1, d);
  matrix_mult(ZV, Z, B1->V, d-1, d-1, d);
  matrix_mult(C1, Vt, ZV, d, d-1, d);

  // compute C2
  for (i = 0; i < d-1; i++)
    Z[i][i] = B2->Z[i];
  transpose(Vt, B2->V, d-1, d);
  matrix_mult(ZV, Z, B2->V, d-1, d-1, d);
  matrix_mult(C2, Vt, ZV, d, d-1, d);

  // compute the principal components of C = C1 + C2
  matrix_add(C, C1, C2, d, d);
  double z[d];
  double **V = C1;  // save an alloc
  eigen_symm(z, V, C, d);
  //matrix_copy(B->V, V, d-1, d);
  for (i = 0; i < d-1; i++)
    for (j = 0; j < d; j++)
      B->V[i][j] = V[j][d-1-i];

  // set the smallest z[i] (in magnitude) to zero
  for (i = 0; i < d-1; i++)
    B->Z[i] = z[d-1-i] - z[0];

  // lookup F
  if (d == 4)
    B->F = bingham_F_lookup_3d(B->Z);
  else if (d == 3)
    B->F = bingham_F_2d(B->Z[0], B->Z[1]);
  else if (d == 2)
    B->F = bingham_F_1d(B->Z[0]);
  else {
    fprintf(stderr, "Error: bingham_mult() only supports 1D, 2D, and 3D binghams.\n");
    B->F = 0;
  }

  /*
  printf("z = [%f %f %f %f]\n", z[0], z[1], z[2], z[3]);
  printf("V[0] = [%f %f %f %f]\n", V[0][0], V[0][1], V[0][2], V[0][3]);
  printf("V[1] = [%f %f %f %f]\n", V[1][0], V[1][1], V[1][2], V[1][3]);
  printf("V[2] = [%f %f %f %f]\n", V[2][0], V[2][1], V[2][2], V[2][3]);
  printf("V[3] = [%f %f %f %f]\n", V[3][0], V[3][1], V[3][2], V[3][3]);
  */

  //printf("B->F = %f\n", B->F);
  //printf("B->Z = [%f %f %f]\n", B->Z[0], B->Z[1], B->Z[2]);
  //printf("B->V[0] = [%f %f %f %f]\n", B->V[0][0], B->V[0][1], B->V[0][2], B->V[0][3]);
  //printf("B->V[1] = [%f %f %f %f]\n", B->V[1][0], B->V[1][1], B->V[1][2], B->V[1][3]);
  //printf("B->V[2] = [%f %f %f %f]\n", B->V[2][0], B->V[2][1], B->V[2][2], B->V[2][3]);
  
  free_matrix2(Vt);
  free_matrix2(Z);
  free_matrix2(ZV);
  free_matrix2(C1);
  free_matrix2(C2);
  free_matrix2(C);
}



/*
 * Multiply two bingham mixtures, BM = BM1 * BM2.
 */
void bingham_mixture_mult(bingham_mix_t *BM, bingham_mix_t *BM1, bingham_mix_t *BM2)
{
  int n1 = BM1->n;
  int n2 = BM2->n;
  int d = BM1->B[0].d;
  int i, j;

  BM->n = n1*n2;
  safe_calloc(BM->w, BM->n, double);
  safe_calloc(BM->B, BM->n, bingham_t);

  // multiply bingham mixtures
  int n = 0;
  for (i = 0; i < n1; i++) {
    for (j = 0; j < n2; j++) {
      bingham_alloc(&BM->B[n], d);
      bingham_mult(&BM->B[n], &BM1->B[i], &BM2->B[j]);
      BM->w[n] = BM1->w[i] * BM2->w[j];
      n++;
    }
  }

  // sort mixture components
  bingham_t b[n];
  sortable_t wb[n];
  for (i = 0; i < n; i++) {
    wb[i].value = BM->w[i];
    wb[i].data = (void *)(&b[i]);
    memcpy(&b[i], &BM->B[i], sizeof(bingham_t));
  }
  sort_data(wb, n);
  for (i = 0; i < n; i++) {
    BM->w[i] = wb[i].value;
    memcpy(&BM->B[i], wb[i].data, sizeof(bingham_t));
  }
}


/*
 * Remove mixture components with peak less than pthresh.
 */
void bingham_mixture_thresh_peaks(bingham_mix_t *BM, double pthresh)
{
  int i, n = BM->n;
  double peaks[n];

  int cnt = 0;
  for (i = 0; i < n; i++) {
    peaks[i] = BM->w[i] / BM->B[i].F;
    if (peaks[i] >= pthresh) {  // keep the bingham
      if (cnt < i) {
	memcpy(&BM->B[cnt], &BM->B[i], sizeof(bingham_t));
	BM->w[cnt] = BM->w[i];
      }
      cnt++;
    }
    else {  // remove the bingham
      bingham_free(&BM->B[i]);
    }
  }
}




//------------ DEPRECATED ------------//


/*
 * Function pointer to bingham_pdf.
 *
static double bingham_pdf_callback(double *x, void *B)
{
  int d = ((bingham_t *)B)->d;
  double y[d];
  mult(y, x, 1/norm(x, d), d);

  return bingham_pdf(y, (bingham_t *)B);
}
*/

/*
 * Discretize a Bingham distribution into a multi-resolution grid.
 *
void bingham_discretize_mres(bingham_pmf_t *pmf, bingham_t *B, double resolution)
{
  int i, d = B->d;

  pmf->d = d;
  pmf->resolution = resolution;

  if (d == 4) {

    // mesh
    octetramesh_t *oct = hypersphere_tessellation_octetra_mres(bingham_pdf_callback, (void *)B, resolution);
    pmf->tetramesh = octetramesh_to_tetramesh(oct);
    octetramesh_free(oct);
    free(oct);

    // points and volumes
    int n = pmf->tetramesh->nt;
    pmf->n = n;
    pmf->points = new_matrix2(n, d);
    safe_malloc(pmf->volumes, n, double);
    tetramesh_centroids(pmf->points, pmf->volumes, pmf->tetramesh);

    // probability mass
    safe_malloc(pmf->mass, n, double);
    double tot_mass = 0;
    for (i = 0; i < n; i++) {
      pmf->mass[i] = pmf->volumes[i] * exp(bingham_L(B, &pmf->points[i], 1));
      tot_mass += pmf->mass[i];
    }
    //mult(pmf->mass, pmf->mass, 1/tot_mass, n);
  }
  else {
    printf("Warning: bingham_discretize() doesn't know how to discretize distributions in %d dimensions.\n", d);
    return;
  }    
}
*/

