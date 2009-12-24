
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bingham.h"
#include "bingham/util.h"
#include "bingham/bingham_constants.h"
#include "bingham/hypersphere.h"


/*
 * Fits a mixture of bingham distributions to data in files of the form:
 *
 * <n> <d>
 * <x11> ... <x1d>
 * .
 * .
 * .
 * <xn1> ... <xnd>
 */


double **load_data(char *filename, int *n_ptr, int *d_ptr)
{
  FILE *f = fopen(filename, "r");
  int n, d, i, j;

  fscanf(f, "%d %d\n", &n, &d);

  double **X = new_matrix2(n, d);

  for (i = 0; i < n; i++)
    for (j = 0; j < d; j++)
      fscanf(f, "%lf", &X[i][j]);

  *n_ptr = n;
  *d_ptr = d;

  return X;
}

void usage(int argc, char *argv[])
{
  printf("usage: %s <filename>\n", argv[0]);
  exit(1);
}

int main(int argc, char *argv[])
{
  double t0 = get_time_ms();
  bingham_init();
  double t1 = get_time_ms();
  fprintf(stderr, "Initialized bingham library in %.0f ms\n", t1-t0);

  if (argc < 2)
    usage(argc, argv);

  char *filename = argv[1];

  int n, d, i, j, c;
  double **X = load_data(filename, &n, &d);

  bingham_mix_t BM;
  bingham_cluster(&BM, X, n, d);

  printf("B_num = %d\n\n", BM.n);
  printf("B_weights = [ ");
  for (c = 0; c < BM.n; c++)
    printf("%f ", BM.w[c]);
  printf("]\n\n");

  for (c = 0; c < BM.n; c++) {
    printf("B(%d).V = [ ", c+1);
    for (i = 0; i < d; i++) {
      for (j = 0; j < d-1; j++)
	printf("%f ", BM.B[c].V[j][i]);
      printf("; ");
    }
    printf("];\n\n");

    printf("B(%d).Z = [ ", c+1);
    for (i = 0; i < d-1; i++)
      printf("%f ", BM.B[c].Z[i]);
    printf("];\n\n");

    printf("B(%d).F = %f;\n\n", c+1, BM.B[c].F);
  }

  return 0;
}
