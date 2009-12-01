
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


int main(int argc, char *argv[])
{
  //test_kdtree(argc, argv);
  //test_normrand(argc, argv);
  test_safe_alloc();

  return 0;
}

