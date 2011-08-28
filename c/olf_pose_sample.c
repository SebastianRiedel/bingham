
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bingham.h"
#include "bingham/util.h"
#include "bingham/olf.h"




int main(int argc, char *argv[])
{
  if (argc < 5) {
    printf("usage: %s <olf> <samples_file> <nsamples> <pcd_1> [... <pcd_k>]\n", argv[0]);
    return 1;
  }

  olf_t *olf = load_olf(argv[1]);
  if (olf == NULL) {
    printf("Error loading olf\n");
    return 1;
  }

  FILE *f = fopen(argv[2], "w");
  if (f == NULL) {
    printf("Can't open %s for writing\n", argv[2]);
    return 1;
  }
  fprintf(f, "X = []; Q = []; W = [];\n");

  int num_samples = atof(argv[3]);

  int num_pcds = argc - 4;
  int j;
  for (j = 0; j < num_pcds; j++) {
    pcd_t *pcd = load_pcd(argv[j+4]);
    if (pcd == NULL) {
      printf("Error loading pcd: %s\n", argv[j+4]);
      return 1;
    }
    olf_classify_points(pcd, olf);
    olf_pose_samples_t *poses = olf_pose_sample(olf, pcd, num_samples);
    poses = olf_aggregate_pose_samples(poses, olf);

    // write pose samples to file
    double **X = poses->X;
    double **Q = poses->Q;
    double *W = poses->W;
    int n;
    for (n = 1; n < poses->n; n++)
      if (W[n] < .01 * W[0])
	break;

    int i;
    fprintf(f, "X{%d} = [", j+1);
    for (i = 0; i < n; i++)
      fprintf(f, "%f, %f, %f;  ", X[i][0], X[i][1], X[i][2]);
    fprintf(f, "];\n");

    fprintf(f, "Q{%d} = [", j+1);
    for (i = 0; i < n; i++)
      fprintf(f, "%f, %f, %f, %f;  ", Q[i][0], Q[i][1], Q[i][2], Q[i][3]);
    fprintf(f, "];\n");

    fprintf(f, "W{%d} = [", j+1);
    for (i = 0; i < n; i++)
      fprintf(f, "%f ", W[i]);
    fprintf(f, "];\n");
  }

  fclose(f);

  return 0;
}


