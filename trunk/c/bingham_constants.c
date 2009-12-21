
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bingham/util.h"
#include "bingham/bingham_constants.h"
#include "bingham/bingham_constant_tables.h"


/** Note: It is assumed that z1 < z2 < z3 < 0 for all the bingham functions. **/


#define EPSILON 1e-8
#define ITERATION_MULT 10
#define MIN_ITERATIONS 10



static kdtree_t *dY_tree_3d = NULL;  // dY = d(logF) = dF/F
static int **dY_indices_3d = NULL;  // map from the indices of dY to indices (i,j,k) of F, dF*, etc.


/*
 * Initialize the KD-trees for fast constant lookups.
 */
void bingham_constants_init()
{
  if (dY_tree_3d)  // already initialized
    return;

  double t0 = get_time_ms();

  int i, j, k;
  const int n = BINGHAM_TABLE_LENGTH;

  dY_indices_3d = new_matrix2i(n*n*n, 3);

  // build dY3d vectors
  double **dY3d = new_matrix2(n*n*n, 3);
  int cnt = 0;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      for (k = 0; k < n; k++) {
	if (i >= j && j >= k) {

	  dY3d[cnt][0] = bingham_dY1_table[i][j][k];
	  dY3d[cnt][1] = bingham_dY2_table[i][j][k];
	  dY3d[cnt][2] = bingham_dY3_table[i][j][k];

	  dY_indices_3d[cnt][0] = i;
	  dY_indices_3d[cnt][1] = j;
	  dY_indices_3d[cnt][2] = k;

	  cnt++;
	}
      }
    }
  }

  // create a KD-tree from the vectors in dY3d
  dY_tree_3d = kdtree(dY3d, cnt, 3);

  free_matrix2(dY3d);

  printf("Initialized bingham constants in %.0f ms\n", get_time_ms() - t0);
}


/*
 * Look up concentration params Z and normalization constant F given dY.
 */
void bingham_dY_params_3d(double *Z, double *F, double *dY)
{
  if (dY_tree_3d == NULL)
    bingham_constants_init();

  int nn_index = kdtree_NN(dY_tree_3d, dY);

  int i = dY_indices_3d[nn_index][0];
  int j = dY_indices_3d[nn_index][1];
  int k = dY_indices_3d[nn_index][2];

  double r0 = bingham_table_range[i];
  double r1 = bingham_table_range[j];
  double r2 = bingham_table_range[k];

  Z[0] = -r0*r0;
  Z[1] = -r1*r1;
  Z[2] = -r2*r2;

  *F = bingham_F_table[i][j][k];
}




///////////////////////////////////////////////////////////////////
//                                                               //
//****************  Hypergeometric 1F1 Functions  ***************//
//                                                               //
///////////////////////////////////////////////////////////////////



//-------------------  1F1 Canonical form  -------------------//


/*
 * Computes the hypergeometric function 1F1(a;b;z) in canonical form (z > 0)
 */
static double compute_1F1_1d_canon(double a, double b, double z, int iter)
{
  //printf("compute_1F1_1d_canon(%f, %f, %f, %d)\n", a, b, z, iter);

  int i;
  double g, F = 0.0, logz = log(z);

  for (i = 0; i < iter; i++) {
    g = lgamma(i+a) - lgamma(i+b) + i*logz - lfact(i);
    if (i > z && exp(g) < EPSILON * F)  // exp(g) < 1e-8 * F
      break;
    F += exp(g);
  }

  return 2*sqrt(M_PI)*F;
}


/*
 * Computes the hypergeometric function 1F1(a;b;z1,z2) in canonical form (z1 > z2 > 0)
 */
static double compute_1F1_2d_canon(double a, double b, double z1, double z2, int iter)
{
  int i, j;
  double g, F = 0.0, logz1 = log(z1), logz2 = log(z2);

  for (i = 0; i < iter; i++) {
    for (j = 0; j < iter; j++) {
      g = lgamma(i+a) + lgamma(j+a) - lgamma(i+j+b) + i*logz1 + j*logz2 - lfact(i) - lfact(j);
      if ((i > z1 || j > z2) && exp(g) < EPSILON * F)  // exp(g) < 2e-9
	break;
      F += exp(g);
    }
  }

  return 2*sqrt(M_PI)*F;
}


/*
 * Computes the hypergeometric function 1F1(a;b;z1,z2,z3) in canonical form (z1 > z2 > z3 > 0)
 */
static double compute_1F1_3d_canon(double a, double b, double z1, double z2, double z3, int iter)
{
  int i, j, k;
  double g, F = 0.0, logz1 = log(z1), logz2 = log(z2), logz3 = log(z3);

  for (i = 0; i < iter; i++) {
    for (j = 0; j < iter; j++) {
      for (k = 0; k < iter; k++) {
	g = lgamma(i+a) + lgamma(j+a) + lgamma(k+a) - lgamma(i+j+k+b) + i*logz1 + j*logz2 + k*logz3 - lfact(i) - lfact(j) - lfact(k);
	if ((i > z1 || j > z2 || k > z3) && exp(g) < EPSILON * F)  // exp(g) < 2e-9
	  break;
	F += exp(g);
      }
    }
  }

  return 2*sqrt(M_PI)*F;
}



//-------------------  1F1 General form  -------------------//


/*
 * Computes the hypergeometric function 1F1(a;b;z) with a = 1/2, b = (dim+1)/2
 */
static double compute_1F1_1d(int dim, double z, int iter)
{
  if (fabs(z) < EPSILON)  // uniform
    return surface_area_sphere(dim);

  if (z < 0)
    return exp(z)*compute_1F1_1d(dim, -z, iter);

  return compute_1F1_1d_canon(.5, .5*(dim+1), z, iter);
}


/*
 * Computes the hypergeometric function 1F1(a;b;z1,z2) with a = 1/2, b = (dim+1)/2
 */
static double compute_1F1_2d(int dim, double z1, double z2, int iter)
{
  if (fabs(z1) < EPSILON)  // uniform
    return surface_area_sphere(dim);

  if (fabs(z2) < EPSILON)  // z2 = 0
    return sqrt(M_PI)*compute_1F1_1d(dim, z1, iter);

  if (z1 < 0)
    return exp(z1)*compute_1F1_2d(dim, -z1, z2-z1, iter);

  return compute_1F1_2d_canon(.5, .5*(dim+1), z1, z2, iter);
}


/*
 * Computes the hypergeometric function 1F1(a;b;z1,z2,z3) with a = 1/2, b = (dim+1)/2
 */
static double compute_1F1_3d(int dim, double z1, double z2, double z3, int iter)
{
  if (fabs(z1) < EPSILON)  // uniform
    return surface_area_sphere(dim);

  if (fabs(z3) < EPSILON)  // z3 = 0
    return sqrt(M_PI)*compute_1F1_2d(dim, z1, z2, iter);

  if (z1 < 0)
    return exp(z1)*compute_1F1_3d(dim, -z1, z3-z1, z2-z1, iter);

  return compute_1F1_3d_canon(.5, .5*(dim+1), z1, z2, z3, iter);
}



//--------------  1F1 Partial derivatives (general form)  --------------//


//.....(((  1D  ))).....

/*
 * Computes the partial derivative w.r.t. z of the hypergeometric
 * function 1F1(a;b;z) with a = 1/2, b = (dim+1)/2
 */
static double compute_d1F1_dz_1d(int dim, double z, int iter)
{
  //printf("compute_d1F1_dz1_1d(%d, %f, %d)\n", dim, z, iter);

  if (fabs(z) < EPSILON)  // uniform
    return surface_area_sphere(dim)/(double)(dim+1);

  if (z < 0)
    return compute_1F1_1d(dim, z, iter) - exp(z)*compute_d1F1_dz_1d(dim, -z, iter);

  int i;
  double g, F = 0.0, logz = log(z);
  double a = .5, b = .5*(dim+1);

  for (i = 1; i < iter; i++) {
    g = lgamma(i+a) - lgamma(i+b) + (i-1)*logz - lfact(i-1);
    if (i > z && exp(g) < EPSILON * F)  // exp(g) < 2e-9
      break;
    F += exp(g);
  }

  //printf("  --> %f\n", 2*sqrt(M_PI)*F);

  return 2*sqrt(M_PI)*F;
}


//.....(((  2D  ))).....

static double compute_d1F1_dz2_2d(int dim, double z1, double z2, int iter);


/*
 * Computes the partial derivative w.r.t. z1 of the hypergeometric
 * function 1F1(a;b;z1,z2) with a = 1/2, b = (dim+1)/2
 */
static double compute_d1F1_dz1_2d(int dim, double z1, double z2, int iter)
{
  //printf("compute_d1F1_dz1_2d(%d, %f, %f, %d)\n", dim, z1, z2, iter);

  if (fabs(z1) < EPSILON)  // z1 = z2 = 0 --> uniform
    return surface_area_sphere(dim)/(double)(dim+1);

  if (fabs(z2) < EPSILON) {  // z2 = 0
    double retval = sqrt(M_PI)*compute_d1F1_dz_1d(dim, z1, iter);
    //printf("  --> %f\n", retval);
    return retval;
  }

  if (z1 < 0)
    return compute_1F1_2d(dim, z1, z2, iter) -
      exp(z1)*(compute_d1F1_dz1_2d(dim, -z1, z2-z1, iter) +
	       compute_d1F1_dz2_2d(dim, -z1, z2-z1, iter));

  int i, j;
  double g, F = 0.0, logz1 = log(z1), logz2 = log(z2);
  double a = .5, b = .5*(dim+1);

  for (i = 1; i < iter; i++) {
    for (j = 0; j < iter; j++) {
      g = lgamma(i+a) + lgamma(j+a) - lgamma(i+j+b) + (i-1)*logz1 + j*logz2 - lfact(i-1) - lfact(j);
      if ((i > z1 || j > z2) && exp(g) < EPSILON * F)  // exp(g) < 2e-9
	break;
      F += exp(g);
    }
  }

  //printf("  --> %f\n", 2*sqrt(M_PI)*F);

  return 2*sqrt(M_PI)*F;
}


/*
 * Computes the partial derivative w.r.t. z2 of the hypergeometric
 * function 1F1(a;b;z1,z2) with a = 1/2, b = (dim+1)/2
 */
static double compute_d1F1_dz2_2d(int dim, double z1, double z2, int iter)
{
  //printf("compute_d1F1_dz2_2d(%d, %f, %f, %d)\n", dim, z1, z2, iter);

  if (fabs(z1) < EPSILON)  // z1 = z2 = 0 --> uniform
    return surface_area_sphere(dim)/(double)(dim+1);

  if (fabs(z2) < EPSILON)  // z2 = 0
    return .5*sqrt(M_PI)*compute_1F1_1d(dim+2, z1, iter);

  if (z1 < 0)
    return exp(z1)*compute_d1F1_dz2_2d(dim, -z1, z2-z1, iter);

  int i, j;
  double g, F = 0.0, logz1 = log(z1), logz2 = log(z2);
  double a = .5, b = .5*(dim+1);

  for (i = 0; i < iter; i++) {
    for (j = 1; j < iter; j++) {
      g = lgamma(i+a) + lgamma(j+a) - lgamma(i+j+b) + i*logz1 + (j-1)*logz2 - lfact(i) - lfact(j-1);
      if ((i > z1 || j > z2) && exp(g) < EPSILON * F)  // exp(g) < 2e-9
	break;
      F += exp(g);
    }
  }

  return 2*sqrt(M_PI)*F;
}


//.....(((  3D  ))).....

static double compute_d1F1_dz2_3d(int dim, double z1, double z2, double z3, int iter);
static double compute_d1F1_dz3_3d(int dim, double z1, double z2, double z3, int iter);


/*
 * Computes the partial derivative w.r.t. z1 of the hypergeometric
 * function 1F1(a;b;z1,z2,z3) with a = 1/2, b = (dim+1)/2
 */
static double compute_d1F1_dz1_3d(int dim, double z1, double z2, double z3, int iter)
{
  //printf("compute_d1F1_dz1_3d(%d, %f, %f, %f, %d)\n", dim, z1, z2, z3, iter);

  if (fabs(z1) < EPSILON)  // z1 = z2 = z3 = 0 --> uniform
    return surface_area_sphere(dim)/(double)(dim+1);

  if (fabs(z3) < EPSILON) {  // z3 = 0
    double retval = sqrt(M_PI)*compute_d1F1_dz1_2d(dim, z1, z2, iter);
    //printf("  --> %f\n", retval);
    return retval;
  }

  if (z1 < 0)
    return compute_1F1_3d(dim, z1, z2, z3, iter) -
      exp(z1)*(compute_d1F1_dz1_3d(dim, -z1, z3-z1, z2-z1, iter) +
	       compute_d1F1_dz2_3d(dim, -z1, z3-z1, z2-z1, iter) +
	       compute_d1F1_dz3_3d(dim, -z1, z3-z1, z2-z1, iter));

  int i, j, k;
  double g, F = 0.0, logz1 = log(z1), logz2 = log(z2), logz3 = log(z3);
  double a = .5, b = .5*(dim+1);

  for (i = 1; i < iter; i++) {
    for (j = 0; j < iter; j++) {
      for (k = 0; k < iter; k++) {
	g = lgamma(i+a) + lgamma(j+a) + lgamma(k+a) - lgamma(i+j+k+b) + (i-1)*logz1 + j*logz2 + k*logz3 - lfact(i-1) - lfact(j) - lfact(k);
	if ((i > z1 || j > z2 || k > z3) && exp(g) < EPSILON * F)  // exp(g) < 2e-9
	  break;
	F += exp(g);
      }
    }
  }

  //printf("  --> %f\n", 2*sqrt(M_PI)*F);

  return 2*sqrt(M_PI)*F;
}


/*
 * Computes the partial derivative w.r.t. z2 of the hypergeometric
 * function 1F1(a;b;z1,z2,z3) with a = 1/2, b = (dim+1)/2
 */
static double compute_d1F1_dz2_3d(int dim, double z1, double z2, double z3, int iter)
{
  //printf("compute_d1F1_dz2_3d(%d, %f, %f, %f, %d)\n", dim, z1, z2, z3, iter);

  if (fabs(z1) < EPSILON)  // z1 = z2 = z3 = 0 --> uniform
    return surface_area_sphere(dim)/(double)(dim+1);

  if (fabs(z3) < EPSILON)  // z3 = 0
    return sqrt(M_PI)*compute_d1F1_dz2_2d(dim, z1, z2, iter);

  if (z1 < 0)
    return exp(z1)*compute_d1F1_dz3_3d(dim, -z1, z3-z1, z2-z1, iter);

  int i, j, k;
  double g, F = 0.0, logz1 = log(z1), logz2 = log(z2), logz3 = log(z3);
  double a = .5, b = .5*(dim+1);

  for (i = 0; i < iter; i++) {
    for (j = 1; j < iter; j++) {
      for (k = 0; k < iter; k++) {
	g = lgamma(i+a) + lgamma(j+a) + lgamma(k+a) - lgamma(i+j+k+b) + i*logz1 + (j-1)*logz2 + k*logz3 - lfact(i) - lfact(j-1) - lfact(k);
	if ((i > z1 || j > z2 || k > z3) && exp(g) < EPSILON * F)  // exp(g) < 2e-9
	  break;
	F += exp(g);
      }
    }
  }

  return 2*sqrt(M_PI)*F;
}


/*
 * Computes the partial derivative w.r.t. z3 of the hypergeometric
 * function 1F1(a;b;z1,z2,z3) with a = 1/2, b = (dim+1)/2
 */
static double compute_d1F1_dz3_3d(int dim, double z1, double z2, double z3, int iter)
{
  //printf("compute_d1F1_dz3_3d(%d, %f, %f, %f, %d)\n", dim, z1, z2, z3, iter);

  if (fabs(z1) < EPSILON)  // z1 = z2 = z3 = 0 --> uniform
    return surface_area_sphere(dim)/(double)(dim+1);

  if (fabs(z3) < EPSILON)  // z3 = 0
    return .5*sqrt(M_PI)*compute_1F1_2d(dim+2, z1, z2, iter);

  if (z1 < 0)
    return exp(z1)*compute_d1F1_dz2_3d(dim, -z1, z3-z1, z2-z1, iter);

  int i, j, k;
  double g, F = 0.0, logz1 = log(z1), logz2 = log(z2), logz3 = log(z3);
  double a = .5, b = .5*(dim+1);

  for (i = 0; i < iter; i++) {
    for (j = 0; j < iter; j++) {
      for (k = 1; k < iter; k++) {
	g = lgamma(i+a) + lgamma(j+a) + lgamma(k+a) - lgamma(i+j+k+b) + i*logz1 + j*logz2 + (k-1)*logz3 - lfact(i) - lfact(j) - lfact(k-1);
	if ((i > z1 || j > z2 || k > z3) && exp(g) < EPSILON * F)  // exp(g) < 2e-9
	  break;
	F += exp(g);
      }
    }
  }

  return 2*sqrt(M_PI)*F;
}






//---------------- Bingham normalizing constants F(z) and partial derivatives ------------------//

inline double bingham_F_1d(double z)
{
  int iter = MAX((int)fabs(z)*ITERATION_MULT, MIN_ITERATIONS);
  return compute_1F1_1d(1, z, iter);
}

inline double bingham_dF_1d(double z)
{
  int iter = MAX((int)fabs(z)*ITERATION_MULT, MIN_ITERATIONS);
  return compute_d1F1_dz_1d(1, z, iter);
}

inline double bingham_F_2d(double z1, double z2)
{
  int iter = MAX((int)MAX(fabs(z1), fabs(z2))*ITERATION_MULT, MIN_ITERATIONS);
  return compute_1F1_2d(2, z1, z2, iter);
}

inline double bingham_dF1_2d(double z1, double z2)
{
  int iter = MAX((int)MAX(fabs(z1), fabs(z2))*ITERATION_MULT, MIN_ITERATIONS);
  return compute_d1F1_dz1_2d(2, z1, z2, iter);
}

inline double bingham_dF2_2d(double z1, double z2)
{
  int iter = MAX((int)MAX(fabs(z1), fabs(z2))*ITERATION_MULT, MIN_ITERATIONS);
  return compute_d1F1_dz2_2d(2, z1, z2, iter);
}

inline double bingham_F_3d(double z1, double z2, double z3)
{
  int iter = MAX((int)MAX(MAX(fabs(z1), fabs(z2)), fabs(z3))*ITERATION_MULT, MIN_ITERATIONS);
  return compute_1F1_3d(3, z1, z2, z3, iter);
}

inline double bingham_dF1_3d(double z1, double z2, double z3)
{
  int iter = MAX((int)MAX(MAX(fabs(z1), fabs(z2)), fabs(z3))*ITERATION_MULT, MIN_ITERATIONS);
  return compute_d1F1_dz1_3d(3, z1, z2, z3, iter);
}

inline double bingham_dF2_3d(double z1, double z2, double z3)
{
  int iter = MAX((int)MAX(MAX(fabs(z1), fabs(z2)), fabs(z3))*ITERATION_MULT, MIN_ITERATIONS);
  return compute_d1F1_dz2_3d(3, z1, z2, z3, iter);
}

inline double bingham_dF3_3d(double z1, double z2, double z3)
{
  int iter = MAX((int)MAX(MAX(fabs(z1), fabs(z2)), fabs(z3))*ITERATION_MULT, MIN_ITERATIONS);
  return compute_d1F1_dz3_3d(3, z1, z2, z3, iter);
}





//----------------- Bingham F(z) "compute_all" tools --------------------//




void compute_all_bingham_F_2d(double z1_min, double z1_max, double z1_step,
			      double z2_min, double z2_max, double z2_step)
{
  double z1, z2;

  printf("F = [ ...\n");
  for (z1 = z1_min; z1 <= z1_max; z1 += z1_step) {
    for (z2 = z2_min; z2 <= z2_max; z2 += z2_step)
      printf("%f ", z1 <= z2 ? bingham_F_2d(z1, z2) : 0.0);
    printf("; ...\n");
  }
  printf("];\n\n");
}

void compute_all_bingham_dF1_2d(double z1_min, double z1_max, double z1_step,
				double z2_min, double z2_max, double z2_step)
{
  double z1, z2;

  printf("dF1 = [ ...\n");
  for (z1 = z1_min; z1 <= z1_max; z1 += z1_step) {
    for (z2 = z2_min; z2 <= z2_max; z2 += z2_step)
      printf("%f ", z1 <= z2 ? bingham_dF1_2d(z1, z2) : 0.0);
    printf("; ...\n");
  }
  printf("];\n\n");
}

void compute_all_bingham_dF2_2d(double z1_min, double z1_max, double z1_step,
			       double z2_min, double z2_max, double z2_step)
{
  double z1, z2;

  printf("dF2 = [ ...\n");
  for (z1 = z1_min; z1 <= z1_max; z1 += z1_step) {
    for (z2 = z2_min; z2 <= z2_max; z2 += z2_step)
      printf("%f ", z1 <= z2 ? bingham_dF2_2d(z1, z2) : 0.0);
    printf("; ...\n");
  }
  printf("];\n\n");
}

void compute_all_bingham_F_3d(double z1_min, double z1_max, double z1_step,
			      double z2_min, double z2_max, double z2_step,
			      double z3_min, double z3_max, double z3_step)
{
  double z1, z2, z3;
  int k=0;

  printf("F = [];\n\n");
  for (z3 = z3_min; z3 <= z3_max; z3 += z3_step) {
    k++;
    printf("F(:,:,%d) = [ ...\n", k);
    for (z1 = z1_min; z1 <= z1_max; z1 += z1_step) {
      for (z2 = z2_min; z2 <= z2_max; z2 += z2_step)
	printf("%f ", z1 <= z2 && z2 <= z3 ? bingham_F_3d(z1, z2, z3) : 0.0);
      printf("; ...\n");
    }
    printf("];\n\n");
  }
}

void compute_all_bingham_dF1_3d(double z1_min, double z1_max, double z1_step,
			      double z2_min, double z2_max, double z2_step,
			      double z3_min, double z3_max, double z3_step)
{
  double z1, z2, z3;
  int k=0;

  printf("dF1 = [];\n\n");
  for (z3 = z3_min; z3 <= z3_max; z3 += z3_step) {
    k++;
    printf("dF1(:,:,%d) = [ ...\n", k);
    for (z1 = z1_min; z1 <= z1_max; z1 += z1_step) {
      for (z2 = z2_min; z2 <= z2_max; z2 += z2_step)
	printf("%f ", z1 <= z2 && z2 <= z3 ? bingham_dF1_3d(z1, z2, z3) : 0.0);
      printf("; ...\n");
    }
    printf("];\n\n");
  }
}

void compute_all_bingham_dF2_3d(double z1_min, double z1_max, double z1_step,
			      double z2_min, double z2_max, double z2_step,
			      double z3_min, double z3_max, double z3_step)
{
  double z1, z2, z3;
  int k=0;

  printf("dF2 = [];\n\n");
  for (z3 = z3_min; z3 <= z3_max; z3 += z3_step) {
    k++;
    printf("dF2(:,:,%d) = [ ...\n", k);
    for (z1 = z1_min; z1 <= z1_max; z1 += z1_step) {
      for (z2 = z2_min; z2 <= z2_max; z2 += z2_step)
	printf("%f ", z1 <= z2 && z2 <= z3 ? bingham_dF2_3d(z1, z2, z3) : 0.0);
      printf("; ...\n");
    }
    printf("];\n\n");
  }
}

void compute_all_bingham_dF3_3d(double z1_min, double z1_max, double z1_step,
			      double z2_min, double z2_max, double z2_step,
			      double z3_min, double z3_max, double z3_step)
{
  double z1, z2, z3;
  int k=0;

  printf("dF3 = [];\n\n");
  for (z3 = z3_min; z3 <= z3_max; z3 += z3_step) {
    k++;
    printf("dF3(:,:,%d) = [ ...\n", k);
    for (z1 = z1_min; z1 <= z1_max; z1 += z1_step) {
      for (z2 = z2_min; z2 <= z2_max; z2 += z2_step)
	printf("%f ", z1 <= z2 && z2 <= z3 ? bingham_dF3_3d(z1, z2, z3) : 0.0);
      printf("; ...\n");
    }
    printf("];\n\n");
  }
}


void compute_range_bingham_F_2d(double *y, int n)
{
  int i, j;
  double z1, z2;

  printf("F = [ ...\n");
  fprintf(stderr, "F");
  for (i = 0; i < n; i++) {
    fprintf(stderr, ".");
    fflush(stderr);
    z1 = -y[i]*y[i];
    for (j = 0; j < n; j++) {
      z2 = -y[j]*y[j];
      printf("%f ", z1 <= z2 ? bingham_F_2d(z1, z2) : 0.0);
    }
    printf("; ...\n");
  }
  printf("];\n\n");
  fprintf(stderr, "\n");
}

void compute_range_bingham_dF1_2d(double *y, int n)
{
  int i, j;
  double z1, z2;

  printf("dF1 = [ ...\n");
  fprintf(stderr, "dF1");
  for (i = 0; i < n; i++) {
    fprintf(stderr, ".");
    fflush(stderr);
    z1 = -y[i]*y[i];
    for (j = 0; j < n; j++) {
      z2 = -y[j]*y[j];
      printf("%f ", z1 <= z2 ? bingham_dF1_2d(z1, z2) : 0.0);
    }
    printf("; ...\n");
  }
  printf("];\n\n");
  fprintf(stderr, "\n");
}

void compute_range_bingham_dF2_2d(double *y, int n)
{
  int i, j;
  double z1, z2;

  printf("dF2 = [ ...\n");
  fprintf(stderr, "dF2");
  for (i = 0; i < n; i++) {
    fprintf(stderr, ".");
    fflush(stderr);
    z1 = -y[i]*y[i];
    for (j = 0; j < n; j++) {
      z2 = -y[j]*y[j];
      printf("%f ", z1 <= z2 ? bingham_dF1_2d(z1, z2) : 0.0);
    }
    printf("; ...\n");
  }
  printf("];\n\n");
  fprintf(stderr, "\n");
}


void compute_range_bingham_F_3d(double *y, int n, int k0, int k1)
{
  int i, j, k;
  double z1, z2, z3;

  //printf("F = [];\n\n");
  fprintf(stderr, "F");
  for (k = k0; k < k1; k++) {
    z3 = -y[k]*y[k];
    printf("F(:,:,%d) = [ ...\n", k);
    for (i = 0; i < n; i++) {
      fprintf(stderr, ".");
      fflush(stderr);
      z1 = -y[i]*y[i];
      for (j = 0; j < n; j++) {
	z2 = -y[j]*y[j];
	printf("%f ", z1 <= z2 && z2 <= z3 ? bingham_F_3d(z1, z2, z3) : 0.0);
      }
      printf("; ...\n");
    }
    printf("];\n\n");
    fprintf(stderr, "\n");
  }
}

void compute_range_bingham_dF1_3d(double *y, int n, int k0, int k1)
{
  int i, j, k;
  double z1, z2, z3;

  //printf("dF1 = [];\n\n");
  fprintf(stderr, "dF1");
  for (k = k0; k < k1; k++) {
    z3 = -y[k]*y[k];
    printf("F(:,:,%d) = [ ...\n", k);
    for (i = 0; i < n; i++) {
      fprintf(stderr, ".");
      fflush(stderr);
      z1 = -y[i]*y[i];
      for (j = 0; j < n; j++) {
	z2 = -y[j]*y[j];
	printf("%f ", z1 <= z2 && z2 <= z3 ? bingham_dF1_3d(z1, z2, z3) : 0.0);
      }
      printf("; ...\n");
    }
    printf("];\n\n");
    fprintf(stderr, "\n");
  }
}

void compute_range_bingham_dF2_3d(double *y, int n, int k0, int k1)
{
  int i, j, k;
  double z1, z2, z3;

  //printf("dF2 = [];\n\n");
  fprintf(stderr, "dF2");
  for (k = k0; k < k1; k++) {
    z3 = -y[k]*y[k];
    printf("F(:,:,%d) = [ ...\n", k);
    for (i = 0; i < n; i++) {
      fprintf(stderr, ".");
      fflush(stderr);
      z1 = -y[i]*y[i];
      for (j = 0; j < n; j++) {
	z2 = -y[j]*y[j];
	printf("%f ", z1 <= z2 && z2 <= z3 ? bingham_dF2_3d(z1, z2, z3) : 0.0);
      }
      printf("; ...\n");
    }
    printf("];\n\n");
    fprintf(stderr, "\n");
  }
}

void compute_range_bingham_dF3_3d(double *y, int n, int k0, int k1)
{
  int i, j, k;
  double z1, z2, z3;

  //printf("dF3 = [];\n\n");
  fprintf(stderr, "dF3");
  for (k = k0; k < k1; k++) {
    z3 = -y[k]*y[k];
    printf("F(:,:,%d) = [ ...\n", k);
    for (i = 0; i < n; i++) {
      fprintf(stderr, ".");
      fflush(stderr);
      z1 = -y[i]*y[i];
      for (j = 0; j < n; j++) {
	z2 = -y[j]*y[j];
	printf("%f ", z1 <= z2 && z2 <= z3 ? bingham_dF3_3d(z1, z2, z3) : 0.0);
      }
      printf("; ...\n");
    }
    printf("];\n\n");
    fprintf(stderr, "\n");
  }
}

