
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bingham.h"
#include "bingham/util.h"
#include "bingham/olf.h"


void set_proposal_weights(olf_t *olf, int proposal_cluster)
{
  if (proposal_cluster < 0) {
    if (olf->proposal_weights)
      free(olf->proposal_weights);
    olf->proposal_weights = NULL;
    return;
  }

  int n = olf->num_clusters;

  if (olf->proposal_weights == NULL)
    safe_calloc(olf->proposal_weights, n, double);

  double eps = .001;
  olf->proposal_weights[proposal_cluster] = 1.0 - eps;

  int i;
  for (i = 0; i < n; i++)
    if (i != proposal_cluster)
      olf->proposal_weights[i] = eps/(double)(n-1);
}


void load_params(olf_t *olf, int *num_samples, char *param_file)
{
  FILE *f = fopen(param_file, "r");
  if (f == NULL) {
    fprintf(stderr, "Error loading param file: %s\n", param_file);
    return;
  }

  char sbuf[128];

  while (!feof(f)) {
    char *s = sbuf;
    if (fgets(s, 1024, f)) {
      if (!wordcmp(s, "num_samples", " \t\n")) {
	s = sword(s, " \t", 1);
	sscanf(s, "%d", num_samples);
      }
      else if (!wordcmp(s, "rot_symm", " \t\n")) {
	s = sword(s, " \t", 1);
	sscanf(s, "%d", &olf->rot_symm);
      }
      else if (!wordcmp(s, "num_validators", " \t\n")) {
	s = sword(s, " \t", 1);
	sscanf(s, "%d", &olf->num_validators);
      }
      else if (!wordcmp(s, "lambda", " \t\n")) {
	s = sword(s, " \t", 1);
	sscanf(s, "%lf", &olf->lambda);
      }
      else if (!wordcmp(s, "pose_agg_x", " \t\n")) {
	s = sword(s, " \t", 1);
	sscanf(s, "%lf", &olf->pose_agg_x);
      }
      else if (!wordcmp(s, "pose_agg_q", " \t\n")) {
	s = sword(s, " \t", 1);
	sscanf(s, "%lf", &olf->pose_agg_q);
      }
      else if (!wordcmp(s, "proposal_cluster", " \t\n")) {
	s = sword(s, " \t", 1);
	int proposal_cluster;
	sscanf(s, "%d", &proposal_cluster);
	set_proposal_weights(olf, proposal_cluster);
      }
    }
  }
}



int main(int argc, char *argv[])
{
  if (argc < 5) {
    printf("usage: %s <olf> <samples_file> [-p <param_file>] <pcd_1> [... <pcd_k>]\n", argv[0]);
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

  int num_samples = 500;  // default
  
  int pcd_offset = 3;
  if (argv[3][0] == '-' && argv[3][1] == 'p') {
    load_params(olf, &num_samples, argv[4]);
    pcd_offset += 2;
  }

  int num_pcds = argc - pcd_offset;
  int j;
  for (j = 0; j < num_pcds; j++) {
    pcd_t *pcd = load_pcd(argv[j+pcd_offset]);
    if (pcd == NULL) {
      printf("Error loading pcd: %s\n", argv[j+pcd_offset]);
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
    mult(W, W, 1/sum(W,n), n);

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


