
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
 * Create a new uniform Bingham distribution.
 */
void bingham_new_uniform(bingham_t *B, int d)
{
  bingham_alloc(B, d);
  memset(B->V[0], 0, d*(d-1)*sizeof(double));
  memset(B->Z, 0, (d-1)*sizeof(double));
  B->F = surface_area_sphere(d);
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
  //return exp(bingham_L(B, &x, 1));

  double **V = B->V;
  double *Z = B->Z;

  if (B->d == 4) {
    double dvx0 = V[0][0]*x[0] + V[0][1]*x[1] + V[0][2]*x[2] + V[0][3]*x[3];
    double dvx1 = V[1][0]*x[0] + V[1][1]*x[1] + V[1][2]*x[2] + V[1][3]*x[3];
    double dvx2 = V[2][0]*x[0] + V[2][1]*x[1] + V[2][2]*x[2] + V[2][3]*x[3];
    return exp(Z[0]*dvx0*dvx0 + Z[1]*dvx1*dvx1 + Z[2]*dvx2*dvx2) / B->F;
  }

  int i, d = B->d;
  double dvx, logf = 0;
  for (i = 0; i < d-1; i++) {
    dvx = dot(V[i], x, d);
    logf += Z[i]*dvx*dvx;
  }

  return exp(logf) / B->F;
}


/*
 * Test whether a Bingham is uniform.
 */
int bingham_is_uniform(bingham_t *B)
{
  int i;
  for (i = 0; i < B->d-1; i++)
    if (B->Z[i] != 0.0)
      return 0;
  return 1;
}


/*
 * Computes the mode of a bingham distribution.
 */
void bingham_mode(double *mode, bingham_t *B)
{
  int i, j, k, d = B->d;
  double **V = B->V;

  // search for axis to cut the hypersphere in half
  double Vcut_raw[(d-1)*(d-1)];
  double *Vcut[d-1];
  for (i = 0; i < d-1; i++)
    Vcut[i] = Vcut_raw + (d-1)*i;
  int cut_axis = 0;
  double max_det = 0;
  for (i = 0; i < d; i++) {  // try cutting each axis, from i=0:d-1
    // set Vcut = V(uncut_axes,:)
    for (j = 0; j < d-1; j++) {
      for (k = 0; k < d-1; k++) {
	if (k < i)
	  Vcut[j][k] = V[j][k];
	else
	  Vcut[j][k] = V[j][k+1];
      }
    }

    double det_Vcut = fabs(det(Vcut, d-1));

    if (det_Vcut > max_det) {
      max_det = det_Vcut;
      cut_axis = i;
    }
  }
  // set Vcut = V(uncut_axes,:)
  for (j = 0; j < d-1; j++) {
    for (k = 0; k < d-1; k++) {
      if (k < cut_axis)
	Vcut[j][k] = V[j][k];
      else
	Vcut[j][k] = V[j][k+1];
    }
  }

  double b[d-1];
  for (i = 0; i < d-1; i++)
    b[i] = -V[i][cut_axis];

  // now solve Vcut * x = b
  double x[d-1];

  /*dbug
  printf("cut_axis = %d\n", cut_axis);
  printf("Vcut = [\n%f %f %f ; \n%f %f %f ; \n%f %f %f]\n",
	 Vcut[0][0], Vcut[0][1], Vcut[0][2],
	 Vcut[1][0], Vcut[1][1], Vcut[1][2],
	 Vcut[2][0], Vcut[2][1], Vcut[2][2]);
  printf("b = [%f %f %f]\n", b[0], b[1], b[2]);
  */

  solve(x, Vcut[0], b, d-1);

  //printf("x = [%f %f %f]\n\n", x[0], x[1], x[2]);

  for (i = 0; i < d-1; i++) {
    if (i < cut_axis)
      mode[i] = x[i];
    else
      mode[i+1] = x[i];
  }
  mode[cut_axis] = 1;
  
  normalize(mode, mode, d);
}


/*
 * Computes some statistics of a bingham.
 * Note: allocates space in stats.
 */
void bingham_stats(bingham_stats_t *stats, bingham_t *B)
{
  int i, j, d = B->d;
  double F = B->F;

  memset(stats, 0, sizeof(bingham_stats_t));
  stats->B = B;

  // look up dF
  safe_calloc(stats->dF, d-1, double);
  if (d == 4)
    bingham_dF_lookup_3d(stats->dF, B->Z);
  else if (d == 3) {
    stats->dF[0] = bingham_dF1_2d(B->Z[0], B->Z[1]);
    stats->dF[1] = bingham_dF2_2d(B->Z[0], B->Z[1]);
  }
  else if (d == 2)
    stats->dF[0] = bingham_dF_1d(B->Z[0]);
  else {
    fprintf(stderr, "Error: bingham_stats() only supports 1D, 2D, and 3D binghams.\n");
  }

  // compute the entropy
  stats->entropy = log(F);
  for (i = 0; i < d-1; i++)
    stats->entropy -= B->Z[i] * stats->dF[i] / F;

  if (!bingham_is_uniform(B)) {

    // compute the mode
    safe_calloc(stats->mode, d, double);
    bingham_mode(stats->mode, B);

    // compute the scatter matrix
    double **Si = new_matrix2(d, d);
    double **S = new_matrix2(d, d);
    double **v;
    double *vt[d];
    double sigma;
    v = &stats->mode;
    for (j = 0; j < d; j++)
      vt[j] = &v[0][j];
    matrix_mult(Si, vt, v, d, 1, d);
    sigma = 1 - sum(stats->dF, d-1)/F;
    mult(Si[0], Si[0], sigma, d*d);
    matrix_add(S, S, Si, d, d);
    for (i = 0; i < d-1; i++) {
      v = &B->V[i];
      for (j = 0; j < d; j++)
	vt[j] = &v[0][j];
      matrix_mult(Si, vt, v, d, 1, d);
      sigma = stats->dF[i]/F;
      mult(Si[0], Si[0], sigma, d*d);
      matrix_add(S, S, Si, d, d);
    }
    free_matrix2(Si);
    stats->scatter = S;
  }
}


/*
 * Frees a bingham_stats_t (but not the underlying bingham_t).
 */
void bingham_stats_free(bingham_stats_t *stats)
{
  if (stats->mode)
    free(stats->mode);
  if (stats->dF)
    free(stats->dF);
  if (stats->scatter)
    free_matrix2(stats->scatter);
}


/*
 * Computes the cross entropy H(B1,B2) between two binghams.
 */
double bingham_cross_entropy(bingham_stats_t *s1, bingham_stats_t *s2)
{
  int i, j;
  int d = s1->B->d;
  double F1 = s1->B->F;
  double **V1 = s1->B->V;
  double *dF1 = s1->dF;
  double F2 = s2->B->F;
  double *Z2 = s2->B->Z;
  double **V2 = s2->B->V;

  // compute the full, transposed V1
  double **V1_ft = new_matrix2(d, d);
  for (i = 0; i < d; i++)
    V1_ft[i][0] = s1->mode[i];
  for (i = 0; i < d; i++)
    for (j = 1; j < d; j++)
      V1_ft[i][j] = V1[j-1][i];

  // rotate B2 into B1's coordinate frame
  double **V1_ft_inv = new_matrix2(d, d);
  inv(V1_ft_inv, V1_ft, d);
  double **A = new_matrix2(d-1, d);
  for (i = 0; i < d-1; i++) {
    for (j = 0; j < d; j++) {
      A[i][j] = dot(V1_ft_inv[j], V2[i], d);
      A[i][j] *= A[i][j];
    }
  }

  /*
  printf("A = [%f %f %f %f ; %f %f %f %f ; %f %f %f %f]\n",
	 A[0][0], A[0][1], A[0][2], A[0][3],
	 A[1][0], A[1][1], A[1][2], A[1][3],
	 A[2][0], A[2][1], A[2][2], A[2][3]);
  */

  // compute H(B1,B2)
  double H = log(F2);
  for (i = 0; i < d-1; i++) {
    double H_i = A[i][0];
    for (j = 1; j < d; j++)
      H_i += (A[i][j] - A[i][0]) * (dF1[j-1]/F1);
    H_i *= Z2[i];
    H -= H_i;
  }

  free_matrix2(V1_ft);
  free_matrix2(V1_ft_inv);
  free_matrix2(A);

  return H;
}


/*
 * Computes the KL divergence D_KL(B1||B2) between two binghams.
 */
double bingham_KL_divergence(bingham_stats_t *s1, bingham_stats_t *s2)
{
  return bingham_cross_entropy(s1, s2) - s1->entropy;
}


/*
 * Merge two binghams: B = a*B1 + (1-a)*B2.
 */
void bingham_merge(bingham_t *B, bingham_stats_t *s1, bingham_stats_t *s2, double alpha)
{
  int d = s1->B->d;

  double **S = new_matrix2(d, d);
  matrix_copy(S, s2->scatter, d, d);
  mult(S[0], S[0], (1-alpha)/alpha, d*d);
  matrix_add(S, S, s1->scatter, d, d);
  mult(S[0], S[0], alpha, d*d);

  bingham_fit_scatter(B, S, d);

  free_matrix2(S);
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
  /*dbug
  int i, j;
  for (i = 0; i < d; i++) {
    printf("S[%d] = [ ", i);
    for (j = 0; j < d; j++)
      printf("%f ", S[i][j]);
    printf("]\n");
  }
  */

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
    pmf->tessellation = tessellate_S3(ncells);
    pmf->n = pmf->tessellation->n;

    double t0 = get_time_ms();  //dbug

    // probability mass
    safe_malloc(pmf->mass, pmf->n, double);
    double tot_mass = 0;
    for (i = 0; i < pmf->n; i++) {
      pmf->mass[i] = pmf->tessellation->volumes[i] * bingham_pdf(pmf->tessellation->centroids[i], B);
      tot_mass += pmf->mass[i];
    }
    mult(pmf->mass, pmf->mass, 1/tot_mass, pmf->n);

    fprintf(stderr, "Computed probabilities in %.0f ms\n", get_time_ms() - t0);  //dbug

  }
  else {
    fprintf(stderr, "Warning: bingham_discretize() doesn't know how to discretize distributions in %d dimensions.\n", d);
    return;
  }
}


/*
 * Bingham mixture sampler
 */
void bingham_mixture_sample(double **X, bingham_mix_t *BM, bingham_stats_t *BM_stats, int n)
{
  int i, j;

  if (n == 1) {
    i = pmfrand(BM->w, BM->n);
    bingham_sample(X, &BM->B[i], &BM_stats[i], 1);
  }
  else if (n < 100) {
    for (i = 0; i < n; i++)
      bingham_mixture_sample(&X[i], BM, BM_stats, 1);
  }
  else {
    // apportion samples to each mixture component
    int ni[BM->n];
    int ntot = 0;
    for (i = 0; i < BM->n; i++) {
      ni[i] = round(n * BM->w[i]);
      ntot += ni[i];
    }
    if (ntot < n) {  // too few samples
      for (i = ntot; i < n; i++) {
	j = pmfrand(BM->w, BM->n);
	ni[j]++;
      }
    }
    else {
      while (ntot > n) {  // too many samples
	j = pmfrand(BM->w, BM->n);
	if (ni[j] > 0) {
	  ni[j]--;
	  ntot--;
	}
      }
    }

    // get ni[i] samples from each component, BM->B[i]
    ntot = 0;
    for (i = 0; i < BM->n; i++) {
      bingham_sample(&X[ntot], &BM->B[i], &BM_stats[i], ni[i]);
      ntot += ni[i];
    }
  }
}


/*
 * Sample n points uniformly from the hypersphere in d dimensions, S^{d-1}.
 */
void bingham_sample_uniform(double **X, int d, int n)
{
  int i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < d; j++)
      X[i][j] = normrand(0,1);
    normalize(X[i], X[i], d);
  }
}


/*
 * Metroplis-Hastings sampler for the Bingham distribution.
 */
void bingham_sample(double **X, bingham_t *B, bingham_stats_t *stats, int n)
{
  if (bingham_is_uniform(B)) {
    bingham_sample_uniform(X, B->d, n);
    return;
  }

  int burn_in = 10;
  int sample_rate = 1;

  int i, j, d = B->d;
  double F = B->F;
  double x[d], x2[d];
  double pcs[d];
  double **V = new_matrix2(d,d);
  double mu[d];
  for (i = 0; i < d; i++)
    mu[i] = 0;

  eigen_symm(pcs, V, stats->scatter, d);
  //printf("pcs = [");
  for (i = 0; i < d; i++) {
    pcs[i] = sqrt(pcs[i]);
    //printf("%f, ", pcs[i]);
  }
  //printf("]\n");

  memcpy(x, stats->mode, d*sizeof(double));

  double t = bingham_pdf(x, B);              // target
  double p = acgpdf_pcs(x, pcs, V, d);   // proposal

  int num_accepts = 0;
  for (i = 0; i < n*sample_rate + burn_in; i++) {
    acgrand_pcs(x2, pcs, V, d);
    //double x2_norm = norm(x2, d);

    //if (x2_norm > .9 && x2_norm < 1.1) {
    //normalize(x2, x2, d);
      double t2 = bingham_pdf(x2, B);
      double p2 = acgpdf_pcs(x2, pcs, V, d);
      double a1 = t2 / t;
      double a2 = p / p2;
      double a = a1*a2;
      if (a > frand()) {
	memcpy(x, x2, d*sizeof(double));
	p = p2;
	t = t2;
	num_accepts++;
      }
      //}
    if (i >= burn_in && (i - burn_in) % sample_rate == 0) {
      j = (i - burn_in) / sample_rate;
      memcpy(X[j], x, d*sizeof(double));
    }
  }

  //printf("accept_rate = %f\n", num_accepts / (double)(n*sample_rate + burn_in));

  free_matrix2(V);
}


/*
 * Simulate samples from a discrete Bingham distribution.
 */
void bingham_sample_pmf(double **X, bingham_pmf_t *pmf, int n)
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

      double *v0 = pmf->tessellation->tetramesh->vertices[ pmf->tessellation->tetramesh->tetrahedra[cell][0] ];
      double *v1 = pmf->tessellation->tetramesh->vertices[ pmf->tessellation->tetramesh->tetrahedra[cell][1] ];
      double *v2 = pmf->tessellation->tetramesh->vertices[ pmf->tessellation->tetramesh->tetrahedra[cell][2] ];
      double *v3 = pmf->tessellation->tetramesh->vertices[ pmf->tessellation->tetramesh->tetrahedra[cell][3] ];
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
 * Samples deterministically from the ridges of a Bingham
 */
void bingham_sample_ridge(double **X, bingham_t *B, int n, double pthresh)
{
  const double eps = 1e-16;

  double *Z = B->Z;
  double **V = B->V;
  double F = B->F;
  int d = B->d;
  double p0 = pthresh;
  int i, j, k;

  double Y[d-1], sumlogy = 0;
  for (i = 0; i < d-1; i++) {
    Y[i] = 1/sqrt(-Z[i]);
    sumlogy += log(Y[i]);
  }
  double m = exp((log(n) - sumlogy)/(d-1));
  int grid_size;
  int pcs_grid_size[d-1];
  double cmax[d-1];
  double step[d-1];
  // generate a grid with at least n cells
  while (1) {
    grid_size = 1;
    for (i = 0; i < d-1; i++) {
      // calculate num samples in each pc direction (rounded to the nearest odd number)
      pcs_grid_size[i] = 2*ceil(m*Y[i]/2.0) - 1;
      cmax[i] = sqrt(log(F*p0)/Z[i]);
      if (cmax[i] > 1)
	cmax[i] = 1;
      step[i] = 2*cmax[i]/(pcs_grid_size[i]-1) - eps;
      grid_size *= pcs_grid_size[i];
    }
    if (grid_size >= n)
      break;
    m *= 1.1;
  }

  double **X2 = new_matrix2(grid_size, d);
  double *pcs_grid_coords[d-1];
  for (i = 0; i < d-1; i++) {
    safe_malloc(pcs_grid_coords[i], pcs_grid_size[i], double);
    if (pcs_grid_size[i] == 1)
      pcs_grid_coords[i][0] = 0;
    else {
      double c = 0;
      int j0 = pcs_grid_size[i]/2;
      for (j = 0; j < (pcs_grid_size[i]+1)/2; j++) {
	pcs_grid_coords[i][j0+j] = c;
	pcs_grid_coords[i][j0-j] = -c;
	c += step[i];
      }
    }
  }

  // search for axis to cut the hypersphere in half
  double Vcut_raw[(d-1)*(d-1)];
  double *Vcut[d-1];
  for (i = 0; i < d-1; i++)
    Vcut[i] = Vcut_raw + (d-1)*i;
  int cut_axis = 0;
  double max_det = 0;
  for (i = 0; i < d; i++) {  // try cutting each axis, from i=0:d-1
    // set Vcut = V(uncut_axes,:)
    for (j = 0; j < d-1; j++) {
      for (k = 0; k < d-1; k++) {
	if (k < i)
	  Vcut[j][k] = V[j][k];
	else
	  Vcut[j][k] = V[j][k+1];
      }
    }
    double det_Vcut = abs(det(Vcut, d-1));
    if (det_Vcut > max_det) {
      max_det = det_Vcut;
      cut_axis = i;
    }
  }
  // set Vcut = V(uncut_axes,:)
  for (j = 0; j < d-1; j++) {
    for (k = 0; k < d-1; k++) {
      if (k < cut_axis)
	Vcut[j][k] = V[j][k];
      else
	Vcut[j][k] = V[j][k+1];
    }
  }

  double c0[d-1];
  for (i = 0; i < d-1; i++)
    c0[i] = V[i][cut_axis];

  double W_raw[(d-1)*(d-1)];
  double *W[d-1];
  for (i = 0; i < d-1; i++)
    W[i] = W_raw + (d-1)*i;
  inv(W, Vcut, d-1);
 
  double x[d-1];
  double c[d-1];
  double dc[d-1];
  int cnt = 0;
  int idx[d-1];
  for (i = 0; i < d-1; i++)
    idx[i] = 0;

  while (1) {
    // sample uncut coords x = W*(c-c0) and normalize
    for (i = 0; i < d-1; i++)
      c[i] = pcs_grid_coords[i][idx[i]];
    sub(dc, c, c0, d-1);  // dc = c-c0
    for (i = 0; i < d-1; i++)
      x[i] = dot(W[i], dc, d-1);  // x = W*dc
    for (i = 0; i < d-1; i++) {
      if (i < cut_axis)
	X2[cnt][i] = x[i];
      else
	X2[cnt][i+1] = x[i];
    }
    X2[cnt][cut_axis] = 1;
    normalize(X2[cnt], X2[cnt], d);
    cnt++;

    // increment the grid indices
    for (i = d-2; i >= 0; i--) {
      if (idx[i] < pcs_grid_size[i] - 1) {
	idx[i]++;
	break;
      }
      idx[i] = 0;
    }
    if (i < 0)
      break;
  }

  // sort samples by pdf, and return the top n
  double *pdf;
  int *indices;
  safe_malloc(pdf, grid_size, double);
  safe_malloc(indices, grid_size, int);
  for (i = 0; i < grid_size; i++) {
    pdf[i] = -bingham_pdf(X2[i], B);
    //fprintf(stderr, "X2[%d] = (%f, %f, %f, %f), pdf[%d] = %f\n",
    //	    i, X2[i][0], X2[i][1], X2[i][2], X2[i][3], i, pdf[i]);
  }
  sort_indices(pdf, indices, grid_size);
  for (i = 0; i < n; i++)
    memcpy(X[i], X2[indices[i]], d*sizeof(double));

  // free temporary variables
  free_matrix2(X2);
  for (i = 0; i < d-1; i++)
    free(pcs_grid_coords[i]);
  free(pdf);
  free(indices);
}


/*
 * Computes the PDF of x with respect to a bingham mixture distribution
 */
double bingham_mixture_pdf(double x[], bingham_mix_t *BM)
{
  int i, n = BM->n;
  double p = 0;
  for (i = 0; i < n; i++)
    p += BM->w[i] * bingham_pdf(x, &BM->B[i]);

  return p;
}


/*
 * Samples deterministically from the ridges of a bingham mixture
 */
void bingham_mixture_sample_ridge(double **X, bingham_mix_t *BM, int n, double pthresh)
{
  int i, d = BM->B[0].d;
  int num_samples = n*BM->n;

  double **X2 = new_matrix2(num_samples, d);

  // sample n points from each component
  for (i = 0; i < BM->n; i++)
    bingham_sample_ridge(X2+i*n, &BM->B[i], n, pthresh);

  //dbug
  //printf("X2 = [ ...\n");
  //int j;
  //for (j = 0; j < num_samples; j++)
  //  printf("%f, %f, %f, %f ; ...\n", X2[j][0], X2[j][1], X2[j][2], X2[j][3]);
  //printf("];\n\n");

  // sort samples by pdf, and return the top n
  double *pdf;
  int *indices;
  safe_malloc(pdf, num_samples, double);
  safe_malloc(indices, num_samples, int);
  for (i = 0; i < num_samples; i++)
    pdf[i] = -bingham_mixture_pdf(X2[i], BM);
  sort_indices(pdf, indices, num_samples);
  for (i = 0; i < n; i++)
    memcpy(X[i], X2[indices[i]], d*sizeof(double));

  free_matrix2(X2);
  free(pdf);
  free(indices);
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

  int points_left = n;
  double **X_left = new_matrix2(n, d);
  matrix_copy(X_left, X, n, d);

  int capacity = 100;
  safe_calloc(BM->B, capacity, bingham_t);
  safe_calloc(BM->w, capacity, double);
  BM->n = 0;

  for (i = 0; i < iter; i++) {

    num_outliers = bingham_fit_mlesac(&BM->B[i], outliers, X_left, points_left, d);
    BM->w[i] = points_left - num_outliers;

    //fprintf(stderr, "num_outliers = %d, w[%d] = %.0f\n", num_outliers, i, BM->w[i]);

    if (BM->w[i] >= min_points)
      BM->n++;
    else
      break;

    if (num_outliers < min_points)
      break;

    points_left = num_outliers;
    for (j = 0; j < points_left; j++)
      memcpy(X_left[j], X_left[outliers[j]], d*sizeof(double));
  }

  // fit a uniform distribution to the remaining outliers
  num_outliers = n - (int)sum(BM->w, BM->n);
  if (num_outliers > 0) {
    BM->w[BM->n] = (double)num_outliers;
    bingham_new_uniform(&BM->B[BM->n], d);
    BM->n++;
  }

  //dbug
  //for (i = 0; i < BM->n; i++)
  //  fprintf(stderr, "w[%d] = %.0f\n", i, BM->w[i]);
  //fprintf(stderr, "w_tot = %.0f\n", sum(BM->w, BM->n));
  //fprintf(stderr, "num_outliers = %d\n", num_outliers);

  safe_realloc(BM->B, BM->n, bingham_t);
  safe_realloc(BM->w, BM->n, double);
  mult(BM->w, BM->w, 1/sum(BM->w, BM->n), BM->n);

  free_matrix2(X_left);
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

  //double **Z = new_matrix2(d-1, d-1);
  //double **ZV = new_matrix2(d-1, d);
  //double **Vt = new_matrix2(d, d-1);
  
  double **v;
  double *vt[d];
  

  // compute C1
  for (i = 0; i < d-1; i++) {
    v = &B1->V[i];
    for (j = 0; j < d; j++)
      vt[j] = &v[0][j];
    matrix_mult(C, vt, v, d, 1, d);
    mult(C[0], C[0], B1->Z[i], d*d);
    matrix_add(C1, C1, C, d, d);
  }
	
  // compute C2
  for (i = 0; i < d-1; i++) {
    v = &B2->V[i];
    for (j = 0; j < d; j++)
      vt[j] = &v[0][j];
    matrix_mult(C, vt, v, d, 1, d);
    mult(C[0], C[0], B2->Z[i], d*d);
    matrix_add(C2, C2, C, d, d);
  }


  //for (i = 0; i < d-1; i++)
  //  Z[i][i] = B1->Z[i];
  //transpose(Vt, B1->V, d-1, d);
  //matrix_mult(ZV, Z, B1->V, d-1, d-1, d);
  //matrix_mult(C1, Vt, ZV, d, d-1, d);

  //for (i = 0; i < d-1; i++)
  //  Z[i][i] = B2->Z[i];
  //transpose(Vt, B2->V, d-1, d);
  //matrix_mult(ZV, Z, B2->V, d-1, d-1, d);
  //matrix_mult(C2, Vt, ZV, d, d-1, d);


  // compute the principal components of C = C1 + C2
  matrix_add(C, C1, C2, d, d);
  double z[d];
  double **V = C1;  // save an alloc
  eigen_symm(z, V, C, d);
  //matrix_copy(B->V, V, d-1, d);
  for (i = 0; i < d-1; i++)
    for (j = 0; j < d; j++)
      B->V[i][j] = V[d-1-i][j];  //V[j][d-1-i];

  //printf("z = [ %f %f %f %f ]\n\n", z[0], z[1], z[2], z[3]);
  //printf("V[0] = [%f %f %f %f]\n", V[0][0], V[0][1], V[0][2], V[0][3]);
  //printf("V[1] = [%f %f %f %f]\n", V[1][0], V[1][1], V[1][2], V[1][3]);
  //printf("V[2] = [%f %f %f %f]\n", V[2][0], V[2][1], V[2][2], V[2][3]);
  //printf("V[2] = [%f %f %f %f]\n", V[3][0], V[3][1], V[3][2], V[3][3]);

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

  //free_matrix2(Vt);
  //free_matrix2(Z);
  //free_matrix2(ZV);
  free_matrix2(C1);
  free_matrix2(C2);
  free_matrix2(C);
}


/*
 * Add two bingham mixtures (dst += src)
 */
void bingham_mixture_add(bingham_mix_t *dst, bingham_mix_t *src)
{
  int n = dst->n;

  // make space for src
  safe_realloc(dst->w, dst->n + src->n, double);
  safe_realloc(dst->B, dst->n + src->n, bingham_t);
  dst->n += src->n;

  int i;
  for (i = 0; i < src->n; i++) {
    dst->w[n+i] = src->w[i];
    bingham_copy(&dst->B[n+i], &src->B[i]);
  }
}


/*
 * Copy the contents of one bingham mixture into another.
 *
 * Warning: allocates new space in 'dst' blindly.
 */
void bingham_mixture_copy(bingham_mix_t *dst, bingham_mix_t *src)
{
  int i, n = src->n;
  dst->n = n;
  safe_malloc(dst->w, n, double);
  safe_malloc(dst->B, n, bingham_t);
  for (i = 0; i < n; i++) {
    dst->w[i] = src->w[i];
    bingham_copy(&dst->B[i], &src->B[i]);
  }
}


/*
 * Free the contents of a bingham mixture.
 */
void bingham_mixture_free(bingham_mix_t *BM)
{
  int i;
  for (i = 0; i < BM->n; i++)
    bingham_free(&BM->B[i]);
  free(BM->w);
  free(BM->B);
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
  if (n > 1) {
    bingham_t b[n];
    sortable_t wb[n];
    for (i = 0; i < n; i++) {
      wb[i].value = -BM->w[i];  // descending order
      wb[i].data = (void *)(&b[i]);
      memcpy(&b[i], &BM->B[i], sizeof(bingham_t));
    }
    sort_data(wb, n);
    for (i = 0; i < n; i++) {
      BM->w[i] = -wb[i].value;  // descending order
      memcpy(&BM->B[i], wb[i].data, sizeof(bingham_t));
    }
  }
}


/*
 * Find the highest peak in a mixture.
 */
double bingham_mixture_peak(bingham_mix_t *BM)
{
  int i;
  double peak, max_peak = 0;
  for (i = 0; i < BM->n; i++) {
    peak = BM->w[i] / BM->B[i].F;
    if (peak > max_peak)
      max_peak = peak;
  }

  return max_peak;
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
  mult(BM->w, BM->w, 1/sum(BM->w, cnt), cnt);
  BM->n = cnt;
}


/*
 * Remove mixture components with weight less than wthresh.
 */
void bingham_mixture_thresh_weights(bingham_mix_t *BM, double wthresh)
{
  int i, n = BM->n;

  int cnt = 0;
  for (i = 0; i < n; i++) {
    if (BM->w[i] >= wthresh) {  // keep the bingham
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
  mult(BM->w, BM->w, 1/sum(BM->w, cnt), cnt);
  BM->n = cnt;
}



/*
 * Load a Bingham Mixtures (BMX) file.
 */
bingham_mix_t *load_bmx(char *f_bmx, int *k)
{
  FILE *f = fopen(f_bmx, "r");

  if (f == NULL) {
    fprintf(stderr, "Invalid filename: %s", f_bmx);
    return NULL;
  }

  // get the number of binghams mixtures in the bmx file
  *k = 0;
  char sbuf[1024], *s = sbuf;
  int c;
  while (!feof(f)) {
    fgets(s, 1024, f);
    if (s[0] == 'B' && sscanf(s, "B %d", &c) && c+1 > *k)
      *k = c+1;
  }
  rewind(f);

  bingham_mix_t *BM = (bingham_mix_t *)calloc(*k, sizeof(bingham_mix_t));

  // get the number of binghams in each mixture
  int i;
  while (!feof(f)) {
    fgets(s, 1024, f);
    if (s[0] == 'B' && sscanf(s, "B %d %d", &c, &i) == 2 && i+1 > BM[c].n)
      BM[c].n = i+1;
  }
  rewind(f);

  // allocate space for the binghams
  for (c = 0; c < *k; c++) {
    BM[c].w = (double *)calloc(BM[c].n, sizeof(double));
    BM[c].B = (bingham_t *)calloc(BM[c].n, sizeof(bingham_t));
  }

  // read in the binghams and corresponding weights
  int d, j, j2;
  double w, xxx;
  int line = 0;
  while (!feof(f)) {
    line++;
    s = sbuf;
    fgets(s, 1024, f);
    if (s[0] == 'B' && sscanf(s, "B %d %d %lf %d", &c, &i, &w, &d) == 4) {
      BM[c].w[i] = w;
      BM[c].B[i].d = d;
      BM[c].B[i].Z = (double *)calloc(d-1, sizeof(double));
      BM[c].B[i].V = new_matrix2(d-1, d);
      s = sword(s, " \t", 5);
      if (sscanf(s, "%lf", &BM[c].B[i].F) < 1)  // read F
	break;
      s = sword(s, " \t", 1);

      for (j = 0; j < d-1; j++) {  // read dF -- dbug: add dF to bingham_t
	if (sscanf(s, "%lf", &xxx) < 1)
	  break;
	s = sword(s, " \t", 1);
      }
      if (j < d-1)  // error
	break;

      for (j = 0; j < d-1; j++) {  // read Z
	if (sscanf(s, "%lf", &BM[c].B[i].Z[j]) < 1)
	  break;
	s = sword(s, " \t", 1);
      }
      if (j < d-1)  // error
	break;
      for (j = 0; j < d-1; j++) {  // read V
	for (j2 = 0; j2 < d; j2++) {
	  if (sscanf(s, "%lf", &BM[c].B[i].V[j][j2]) < 1)
	    break;
	  s = sword(s, " \t", 1);
	}
	if (j2 < d)  // error
	  break;
      }
      if (j < d-1)  // error
	break;
    }
  }
  if (!feof(f)) {  // error
    fprintf(stderr, "Error reading file %s at line %d.\n", f_bmx, line);
    return NULL;
  }
  fclose(f);

  return BM;
}


/*
 * Write bingham mixtures in the following format:
 *
 * B <c> <i> <w> <d> <F> <dF> <Z> <V>
 */
void save_bmx(bingham_mix_t *BM, int num_clusters, char *fout)
{
  fprintf(stderr, "saving BMM to %s\n", fout);

  FILE *f = fopen(fout, "w");
  int c, i, j, k;
  bingham_stats_t stats;

  for (c = 0; c < num_clusters; c++) {
    for (i = 0; i < BM[c].n; i++) {

      double w = BM[c].w[i];
      int d = BM[c].B[i].d;
      double F = BM[c].B[i].F;
      double *Z = BM[c].B[i].Z;
      double **V = BM[c].B[i].V;

      bingham_stats(&stats, &BM[c].B[i]);
      double *dF = stats.dF;

      fprintf(f, "B %d %d %f ", c, i, w);
      fprintf(f, "%d %f ", d, F);
      for (j = 0; j < d-1; j++)
	fprintf(f, "%f ", dF[j]);
      for (j = 0; j < d-1; j++)
	fprintf(f, "%f ", Z[j]);
      for (j = 0; j < d-1; j++)
	for (k = 0; k < d; k++)
	  fprintf(f, "%f ", V[j][k]);
      fprintf(f, "\n");

      bingham_stats_free(&stats);
    }
  }

  fclose(f);
}


/*
 * Print the fields of a Bingham (for debugging).
 */
void print_bingham(bingham_t *B)
{
  int i, j, d = B->d;

  printf("B->F = %f\n", B->F);
  printf("B->Z = [ ");
  for (i = 0; i < d-1; i++)
    printf("%f ", B->Z[i]);
  printf("]\n");
  for (i = 0; i < d-1; i++) {
    printf("B->V[%d] = [ ", i);
    for (j = 0; j < d; j++)
      printf("%f ", B->V[i][j]);
    printf("]\n");
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

