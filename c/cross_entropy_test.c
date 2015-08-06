
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bingham.h"
//#include "bingham/bingham_constants.h"


int main(int argc, char *argv[])
{
  // if (argc < 5) {
  //   printf("usage: %s <z1> <z2> <z3> <num_samples>\n", argv[0]);
  //   exit(1);
  // }

  // double z1 = atof(argv[1]);
  // double z2 = atof(argv[2]);
  // double z3 = atof(argv[3]);
  // int nsamples = atoi(argv[4]);

  bingham_t *B1;
  bingham_new_uniform(B, 4);

  
  double q[4];
  q[0] = 0.87851;
  q[1] = 0.36758;
  q[2] = 0.29688;
  q[3] = 0.07044;
  double q_norm[4];
  normalize(q_norm, q, 4);

  bingham t *B2;
  bingham_new_random(bingham_t *B, int -900);  

  double z1 = -900;
  double z2 = atof(argv[2]);
  double z3 = atof(argv[3]);

  double Z[3] = {z1, z2, z3};
  double V[3][4] = {{0,1,0,0}, {0,0,1,0}, {0,0,0,1}};
  double *Vp[3] = {&V[0][0], &V[1][0], &V[2][0]};

  bingham_t B;
  bingham_new(&B, 4, Vp, Z);

  printf("Sampling from bingham:\n");
  print_bingham(&B);

  double **X = new_matrix2(nsamples, 4);
  bingham_sample(X, &B, nsamples);

  int i;
  for (i = 0; i < nsamples; i++)
    printf("%f %f %f %f\n", X[i][0], X[i][1], X[i][2], X[i][3]);

  return 0;
}
