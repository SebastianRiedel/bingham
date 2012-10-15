
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "bingham/util.h"



void print_kdtree(kdtree_t *tree, int depth)
{
  if (tree == NULL)
    return;

  int i;
  for (i = 0; i < depth; i++)
    printf("  ");
  printf("node %d {d=%d, axis=%d, x=(%.2f", tree->i, tree->d, tree->axis, tree->x[0]);
  for (i = 1; i < tree->d; i++)
    printf(", %.2f", tree->x[i]);
  printf("), bbox_min = (%.2f", tree->bbox_min[0]);
  for (i = 1; i < tree->d; i++)
    printf(", %.2f", tree->bbox_min[i]);
  printf("), bbox_max = (%.2f", tree->bbox_max[0]);
  for (i = 1; i < tree->d; i++)
    printf(", %.2f", tree->bbox_max[i]);
  printf(")}\n");

  print_kdtree(tree->left, depth+1);
  print_kdtree(tree->right, depth+1);
}


void test_kdtree(int argc, char *argv[])
{
  if (argc < 3) {
    printf("usage: %s <n> <d> <x> -- where x is a d-vector\n", argv[0]);
    return;
  }

  int i, j, n = atoi(argv[1]), d = atoi(argv[2]);

  if (argc < d+3) {
    printf("usage: %s <n> <d> <x> -- where x is a d-vector\n", argv[0]);
    return;
  }

  double *x;
  safe_malloc(x, d, double);
  for (i = 0; i < d; i++)
    x[i] = atof(argv[i+3]);

  /*
  double X_data[6][2] = {{-1, -1},
			 {-1, 1},
			 {1, -1},
			 {1, 1},
			 {-2, 0},
			 {2, 0}};
  */

  double **X = new_matrix2(n, d);
  //memcpy(X[0], X_data, n*d*sizeof(double));
  int nn = 0;
  double dmin = DBL_MAX;
  for (i = 0; i < n; i++) {
    for (j = 0; j < d; j++)
      X[i][j] = 100*frand();
    double dx = dist(x, X[i], d);
    if (dx < dmin) {
      dmin = dx;
      nn = i;
    }
  }

  double t = get_time_ms();

  kdtree_t *tree = kdtree(X, n, d);

  double tree_time = get_time_ms() - t;

  //print_kdtree(tree, 0);

  t = get_time_ms();

  printf("\n------------ NN query -------------\n");
  int kdnn = kdtree_NN(tree, x);
  printf("  KD-tree NN to x is node %d\n", kdnn);
  printf("  Real NN to x is node %d\n", nn);

  if (kdnn != nn)
    printf("\n*****  Error: Kd-tree NN != Real NN  *****\n");

  double nn_time = get_time_ms() - t;

  printf("\n --> Built KD-tree with %d nodes in %.2f ms\n", n, tree_time);
  printf(" --> Found NN to x in %.2f ms\n", nn_time);
  printf("\n");
}


void test_normrand(int argc, char *argv[])
{
  if (argc < 4) {
    printf("usage: %s <n> <mu> <sigma>\n", argv[0]);
    return;
  }

  int i, n = atoi(argv[1]);
  double mu = atof(argv[2]);
  double sigma = atof(argv[3]);

  printf("x = [ ");
  double x[n];
  for (i = 0; i < n; i++) {
    x[i] = normrand(mu, sigma);
    printf("%f, ", x[i]);
  }
  printf("]\n");

  double sample_mu = sum(x,n) / (double)n;

  // set x = (x - sample_mu)^2
  for (i = 0; i < n; i++)
    x[i] = (x[i] - sample_mu)*(x[i] - sample_mu);

  double sample_sigma = sqrt(sum(x,n) / (double)n);

  printf("sample mu = %f, sample_sigma = %f\n", sample_mu, sample_sigma);
}


void test_safe_alloc()
{
  int i;
  double *x;
  for (i = 0; i < 10000; i++) {
    printf(".");
    fflush(0);
    safe_malloc(x, 10000000, double);
  }
}


void test_sort_indices()
{
  double x[10] = {0, 90, 70, 40, 20, 10, 30, 80, 50, 60};
  int idx[10];

  sort_indices(x, idx, 10);

  printf("x[idx] = [ ");
  int i;
  for (i = 0; i < 10; i++)
    printf("%f ", x[idx[i]]);
  printf("]\n");
}


void test_mink()
{
  double x[10] = {0, 90, 70, 40, 20, 10, 30, 80, 50, 60};
  int idx[5];

  mink(x, idx, 10, 5);

  printf("idx = [ ");
  int i;
  for (i = 0; i < 5; i++)
    printf("%d ", idx[i]);
  printf("]\n");


  // get timing info
  int n = 100000;
  double y[n];
  for (i = 0; i < n; i++)
    y[i] = normrand(0,1);
  int yi[n];

  double t = get_time_ms();
  sort_indices(y, yi, n);
  printf("sorted %d numbers in %f ms\n", n, get_time_ms() - t);
  
  t = get_time_ms();
  mink(y, yi, n, 100);
  printf("got the min %d numbers in %f ms\n", 100, get_time_ms() - t);
}


void test_pmfrand(int argc, char *argv[])
{
  if (argc < 3) {
    printf("usage: %s <n> <w1> ... <wn>\n", argv[0]);
    return;
  }

  int n = atoi(argv[1]);
  
  if (argc < n+2) {
    printf("usage: %s <n> <w1> ... <wn>\n", argv[0]);
    return;
  }

  int i;
  double w[n];
  for (i = 0; i < n; i++)
    w[i] = atof(argv[i+2]);

  int nsamples = 10000;
  int x[nsamples];

  for (i = 0; i < nsamples; i++)
    x[i] = pmfrand(w, n);

  for (i = 0; i < n; i++)
    w[i] = 0;
  for (i = 0; i < nsamples; i++)
    w[x[i]]++;
  mult(w, w, 1/(double)nsamples, n);

  printf("w2 = [");
  for (i = 0; i < n; i++)
    printf("%.2f, ", w[i]);
  printf("]\n");
}


void test_mvnrand_pcs(int argc, char *argv[])
{
  if (argc < 7) {
    printf("usage: %s <n> <m1> <m2> <z1> <z2> <v11>\n", argv[0]);
    return;
  }

  int d = 2;
  int i, n = atoi(argv[1]);
  double m1 = atof(argv[2]);
  double m2 = atof(argv[3]);
  double z1 = atof(argv[4]);
  double z2 = atof(argv[5]);
  double v11 = atof(argv[6]);

  double mu[2] = {m1, m2};
  double z[2] = {z1, z2};
  double **V = new_matrix2(d, d);
  V[0][0] = v11;
  V[0][1] = sqrt(1 - v11*v11);
  V[1][0] = V[0][1];
  V[1][1] = -V[0][0];

  //printf("X = [ ");
  double **X = new_matrix2(n, d);
  for (i = 0; i < n; i++) {
    mvnrand_pcs(X[i], mu, z, V, d);
    //printf("%f %f, ", X[i][0], X[i][1]);
  }
  //printf("]\n");

  double sample_mu[2] = {0,0};
  for (i = 0; i < n; i++) {
    sample_mu[0] += X[i][0];
    sample_mu[1] += X[i][1];
  }
  sample_mu[0] /= (double)n;
  sample_mu[1] /= (double)n;

  // set X = (X - sample_mu)
  for (i = 0; i < n; i++)
    sub(X[i], X[i], sample_mu, d);

  double **Xt = new_matrix2(d, n);
  transpose(Xt, X, n, d);
  double **S = new_matrix2(d, d);
  matrix_mult(S, Xt, X, d, n, d);
  mult(S[0], S[0], 1/(double)n, d*d);

  eigen_symm(z, V, S, d);

  printf("sample mu = (%f, %f), z = (%f, %f), V = [%f %f; %f %f]\n", sample_mu[0], sample_mu[1], z[0], z[1], V[0][0], V[0][1], V[1][0], V[1][1]);
}


void test_mvnpdf_pcs(int argc, char *argv[])
{
  if (argc < 8) {
    printf("usage: %s <x1> <x2> <m1> <m2> <z1> <z2> <v11>\n", argv[0]);
    return;
  }

  int d = 2;
  double x[2];
  double mu[2];
  double z[2];
  double **V = new_matrix2(2,2);

  x[0] = atof(argv[1]);
  x[1] = atof(argv[2]);
  mu[0] = atof(argv[3]);
  mu[1] = atof(argv[4]);
  z[0] = atof(argv[5]);
  z[1] = atof(argv[6]);
  V[0][0] = atof(argv[7]);
  V[0][1] = sqrt(1 - V[0][0]*V[0][0]);
  V[1][0] = V[0][1];
  V[1][1] = -V[0][0];

  printf("pdf = %f\n", mvnpdf_pcs(x, mu, z, V, d));
}


void test_regression(int argc, char *argv[])
{
  if (argc < 2) {
    printf("usage: %s <b0> [b1] [b2] ...\n", argv[0]);
    return;
  }

  int i, j, d = argc-1, n = 100;
  double b[argc-1];
  for (i = 0; i < d; i++)
    b[i] = atof(argv[i+1]);

  // y(x) = b[0] + b[1]*x + b[2]*x^2 + ...
  double x[n], y[n];
  for (i = 0; i < n; i++) {
    x[i] = i;
    y[i] = b[0];
    for (j = 1; j < d; j++)
      y[i] += b[j]*pow(x[i],j);
  }

  polynomial_regression(b, x, y, n, d);

  printf("b = [ ");
  for (i = 0; i < d; i++)
    printf("%f ", b[i]);
  printf("]\n");
}

/*void test_repmat() {
  // TODO(sanja): silly sanity check, write a better test
  double **A = new_matrix2(2, 3);
  int i, j;
  for (i = 0; i < 2; ++i)
    for (j = 0; j < 3; ++j)
      A[i][j] = (3 * i + (j+1));
  double **B;
  B = repmat(A, 2, 3, 2, 3);
  print_matrix(B, 4, 9);
  }*/

int main(int argc, char *argv[])
{
  //test_regression(argc, argv);
  //test_repmat();
  //test_kdtree(argc, argv);
  //test_normrand(argc, argv);
  //test_safe_alloc();
  //test_sort_indices();
  //test_mvnrand_pcs(argc, argv);
  //test_mvnpdf_pcs(argc, argv);
  //test_pmfrand(argc, argv);
  //test_mink();

  return 0;
}

