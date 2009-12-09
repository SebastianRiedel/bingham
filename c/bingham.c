
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bingham.h"
#include "bingham/util.h"
#include "bingham/hypersphere.h"
#include "bingham/bingham_constants.h"


#define EPSILON 1e-8



/*
 * Binary search to find i s.t. A[i-1] <= x < A[i]
 */
static int binary_search_cdf(double x, double *A, int n)
{
  int i0 = 0;
  int i1 = n-1;
  int i;

  while (i0 <= i1) {
    i = (i0 + i1) / 2;
    if (x > A[i])
      i0 = i + 1;
    else if (i > 0 && x < A[i-1])
      i1 = i-1;
    else
      break;
  }

  if (i0 <= i1)
    return i;

  return n-1;
}



//------------------- Bingham data log likelihood and partial derivatives -------------------//


/*
 * Computes the average log likelihood of the parameters B->Z and B->V given n samples X.
 */
static double bingham_L(bingham_t *B, double **X, int n)
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
 * Compute MLE parameters concentration parameters B->Z given n samples X with principal components B->V
 * using NN lookup.
 */
static void bingham_MLE_NN(bingham_t *B, double **X, int n)
{
  int i, j, d = B->d;
  double dvx;

  if (d != 4) {
    printf("Error: bingham_MLE_NN() is only implemented for d = 4!  Exiting...\n");
    exit(1);
  }

  double dY[d-1];
  memset(dY, 0, (d-1)*sizeof(double));
  for (i = 0; i < n; i++) {
    for (j = 0; j < d-1; j++) {
      dvx = dot(B->V[j], X[i], d);
      dY[j] += dvx*dvx;
    }
  }

  mult(dY, dY, 1/(double)n, d-1);

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
 * Function pointer to bingham_pdf.
 */
static double bingham_pdf_callback(double *x, void *B)
{
  int d = ((bingham_t *)B)->d;
  double y[d];
  mult(y, x, 1/norm(x, d), d);

  return bingham_pdf(y, (bingham_t *)B);
}


/*
 * Fit a bingham to a set of samples.
 */
void bingham_fit(bingham_t *B, double **X, int n, int d)
{
  // use PCA to get B->V
  double **Xt = new_matrix2(d, n);
  transpose(Xt, X, n, d);
  double **X2 = new_matrix2(d, d);
  matrix_mult(X2, Xt, X, d, n, d);
  double *eigenvals;
  safe_malloc(eigenvals, d, double);
  double **V = new_matrix2(d, d);
  eigen_symm(eigenvals, V, X2, d);

  // use gradient descent to get B->Z
  B->d = d;
  B->V = V;
  safe_calloc(B->Z, d-1, double);

  // init B->Z
  int i;
  for (i = 0; i < d-1; i++)
    B->Z[i] = -1;

  double t0, t1, L;
  t0 = get_time_ms();
  bingham_MLE_NN(B, X, n);
  t1 = get_time_ms();
  L = bingham_L(B, X, n);
  printf("Computed MLE (NN) in %.2f ms:  Z = (%f, %f, %f)  --> L = %f\n", t1-t0, B->Z[0], B->Z[1], B->Z[2], L);

  t0 = get_time_ms();
  bingham_MLE_grad_desc(B, X, n);
  t1 = get_time_ms();
  L = bingham_L(B, X, n);
  printf("Computed MLE (grad) in %.2f ms:  Z = (%f, %f, %f) --> L = %f\n", t1-t0, B->Z[0], B->Z[1], B->Z[2], L);


  free(Xt);
  free(X2);
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
    octetramesh_t *oct = hypersphere_tessellation_octetra(ncells);
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


/*
 * Discretize a Bingham distribution into a multi-resolution grid.
 */
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


/*
 * Simulate samples from a discrete Bingham distribution.
 */
void bingham_sample(double **X, bingham_pmf_t *pmf, int n)
{
  int i;

  // compute the CDF
  double cdf[pmf->n];
  memset(cdf, 0, pmf->n * sizeof(double));
  cdf[0] = pmf->mass[0];
  for (i = 1; i < pmf->n; i++)
    cdf[i] = cdf[i-1] + pmf->mass[i];

  // sample from the inverse CDF
  for (i = 0; i < n; i++) {
    double u = frand();
    int cell = binary_search_cdf(u, cdf, pmf->n);

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
      printf("Warning: bingham_discretize() doesn't know how to discretize distributions in %d dimensions.\n", pmf->d);
      return;
    }
  }
}

