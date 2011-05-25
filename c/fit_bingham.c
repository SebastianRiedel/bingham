
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bingham.h"
#include "bingham/util.h"
#include "bingham/bingham_constants.h"
#include "bingham/hypersphere.h"


/*
 * Fits a bingham distribution to data in files of the form:
 *
 * <n> <d>
 * <x11> ... <x1d>
 * .
 * .
 * <xn1> ... <xnd>
 *
 *
 * or to scatter matrix in files of the form:
 *
 * <d> <d>
 * <s11> ... <s1d>
 * .
 * .
 * <sd1> ... <sdd>
 */


/*
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
*/

void usage(int argc, char *argv[])
{
  printf("usage: %s [-s] <fin> <fout>\n", argv[0]);
  exit(1);
}

int main(int argc, char *argv[])
{
  double t0 = get_time_ms();
  bingham_init();
  double t1 = get_time_ms();
  fprintf(stderr, "Initialized bingham library in %.0f ms\n", t1-t0);

  if (argc < 3)
    usage(argc, argv);

  int load_scatter = 0;
  if (argc >= 4) {
    if (!strcmp(argv[1], "-s"))
      load_scatter = 1;
    else
      usage(argc, argv);
  }

  char *fin = (load_scatter ? argv[2] : argv[1]);
  char *fout = (load_scatter ? argv[3] : argv[2]);

  int n, d;
  bingham_mix_t BM;

  if (load_scatter) {
    double **S = load_matrix(fin, &n, &d);
    if (n == d) {
      bingham_t B;
      bingham_fit_scatter(&B, S, d);
      double w = 1.0;
      BM.B = &B;
      BM.w = &w;
      BM.n = 1;
    }
    else if (n > d) {  // fit multiple scatter matrices
      BM.n = n/d;
      safe_calloc(BM.B, BM.n, bingham_t);
      safe_calloc(BM.w, BM.n, double);
      int i;
      for (i = 0; i < BM.n; i++) {
	if (norm(S[d*i], d*d) < .00001)  // S = [0]  -->  uniform
	  bingham_new_uniform(&BM.B[i], d);
	else
	  bingham_fit_scatter(&BM.B[i], &S[d*i], d);
	BM.w[i] = 1;
      }
    }
  }
  else {
    double **X = load_matrix(fin, &n, &d);
    bingham_t B;
    bingham_fit(&B, X, n, d);
    double w = 1.0;
    BM.B = &B;
    BM.w = &w;
    BM.n = 1;
  }

  save_bmx(&BM, 1, fout);

  return 0;
}
