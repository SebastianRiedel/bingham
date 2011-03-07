
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bingham.h"
#include "bingham/util.h"
#include "bingham/hll.h"



void test_hll_sample(int argc, char *argv[])
{
  if (argc < 3) {
    printf("usage: %s <theta> <n>\n", argv[0]);
    return;
  }

  double sample_theta = atof(argv[1]);
  int n = atoi(argv[2]);

  // generate noisy samples along a circle
  double **Q = new_matrix2(n,4);
  double **X = new_matrix2(n,3);
  int i;
  for (i = 0; i < n; i++) {
    double theta = i*2*M_PI/n;
    Q[i][0] = cos(theta/2);
    Q[i][1] = 0;
    Q[i][2] = 0;
    Q[i][3] = sin(theta/2);
    X[i][0] = 10*cos(theta) + normrand(0,1);
    X[i][1] = 10*sin(theta) + normrand(0,1);
    X[i][2] = normrand(0,1);
  }

  // build hll model
  hll_t hll;
  hll_new(&hll, Q, X, n, 4, 3);

  // sample from the hll
  double *q, *x, **S;
  safe_calloc(q, 4, double);
  safe_calloc(x, 3, double);
  S = new_matrix2(3,3);
  q[0] = cos(sample_theta/2);
  q[1] = 0;
  q[2] = 0;
  q[3] = sin(sample_theta/2);
  hll_sample(&x, &S, &q, &hll, 1);

  // print out the sample mean and covariance
  printf("x = [%.2f %.2f, %.2f]\n", x[0], x[1], x[2]);
  printf("S = [%.2f %.2f, %.2f; %.2f %.2f %.2f; %.2f %.2f %.2f]\n", S[0][0], S[0][1], S[0][2], S[1][0], S[1][1], S[1][2], S[2][0], S[2][1], S[2][2]);

  // get timing info
  double t = get_time_ms();
  for (i = 0; i < 1000; i++)
    hll_sample(&x, &S, &q, &hll, 1);
  printf("Got %d samples from p(x|q) in %f ms\n", 1000, get_time_ms() - t);

  // get timing info for bingham sampling
  bingham_t B;
  bingham_stats_t stats;
  bingham_fit(&B, Q, n, 4);
  bingham_stats(&stats, &B);
  t = get_time_ms();
  for (i = 0; i < 1000; i++)
    bingham_sample(&q, &B, &stats, 1);
  printf("Got %d samples from p(q) in %f ms\n", 1000, get_time_ms() - t);
}


int main(int argc, char *argv[])
{
  test_hll_sample(argc, argv);

  return 0;
}
