#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bingham.h"
#include "bingham/util.h"
#include "bingham/olf.h"


void load_true_pose(char *pose_file, simple_pose_t *true_pose) {
  FILE *f = fopen(pose_file, "r");
  if (f == NULL) {
    fprintf(stderr, "Error loading true pose file: %s\n", pose_file);
    return;
  }
  char sbuf[1024];
  char *s = sbuf;
  if (fgets(s, 1024, f)) {
    //s = sword(s, " ", 1);
    int i;
    for (i = 0; i < 3; ++i) {
      s = sword(s, " ", 1);
      true_pose->X[i] = atof(s);
    }
    for (i = 0; i < 4; ++i) {
      s = sword(s, " ", 1);
      true_pose->Q[i] = atof(s);
    }
  }
  fclose(f);
}

void load_params(scope_params_t *params, char *param_file)
{
  FILE *f = fopen(param_file, "r");
  if (f == NULL) {
    fprintf(stderr, "Error loading param file: %s\n", param_file);
    return;
  }

  char sbuf[1024];

  int cnt = 0;
  while (!feof(f)) {
    char *s = sbuf;
    if (fgets(s, 1024, f)) {
      cnt++;
      
      // remove comments
      char *comment_pos = strchr(s, '#');
      if (comment_pos)
	*comment_pos = '\n';

      // skip leading whitespace
      s += strspn(s, " \t");

      // skip empty lines
      if (*s == '\n')
	continue;

      char *name = s;
      char *value = sword(s, " \t", 1);

      if (!wordcmp(name, "num_samples", " \t\n"))
	sscanf(value, "%d", &params->num_samples);
      else if (!wordcmp(name, "num_samples_init", " \t\n"))
	sscanf(value, "%d", &params->num_samples_init);
      else if (!wordcmp(name, "num_correspondences", " \t\n"))
	sscanf(value, "%d", &params->num_correspondences);
      else if (!wordcmp(name, "branching_factor", "\t\n"))
	sscanf(value, "%d", &params->branching_factor);
      else if (!wordcmp(name, "knn", " \t\n"))
	sscanf(value, "%d", &params->knn);
      else if (!wordcmp(name, "num_validation_points", " \t\n"))
	sscanf(value, "%d", &params->num_validation_points);
      else if (!wordcmp(name, "use_range_image", "\t\n"))
	sscanf(value, "%d", &params->use_range_image);
      else if (!wordcmp(name, "do_icp", "\t\n"))
	sscanf(value, "%d", &params->do_icp);
      else if (!wordcmp(name, "do_final_icp", "\t\n"))
	sscanf(value, "%d", &params->do_final_icp);

      else if (!wordcmp(name, "dispersion_weight", "\t\n"))
	sscanf(value, "%d", &params->dispersion_weight);
      else if (!wordcmp(name, "sift_dthresh", " \t\n"))
	sscanf(value, "%lf", &params->sift_dthresh);
      else if (!wordcmp(name, "xyz_weight", " \t\n"))
	sscanf(value, "%lf", &params->xyz_weight);
      else if (!wordcmp(name, "normal_weight", " \t\n"))
	sscanf(value, "%lf", &params->normal_weight);
      else if (!wordcmp(name, "L_weight", " \t\n"))
	sscanf(value, "%lf", &params->L_weight);
      else if (!wordcmp(name, "range_sigma", " \t\n"))
	sscanf(value, "%lf", &params->range_sigma);
      else if (!wordcmp(name, "range_weight", " \t\n"))
	sscanf(value, "%lf", &params->range_weight);
      else if (!wordcmp(name, "f_sigma", " \t\n"))
	sscanf(value, "%lf", &params->f_sigma);
      else if (!wordcmp(name, "lab_sigma", " \t\n"))
	sscanf(value, "%lf", &params->lab_sigma);
      else if (!wordcmp(name, "xyz_sigma", " \t\n"))
	sscanf(value, "%lf", &params->xyz_sigma);
      else if (!wordcmp(name, "vis_weight", " \t\n"))
	sscanf(value, "%lf", &params->vis_weight);
      else if (!wordcmp(name, "f_weight", " \t\n"))
	sscanf(value, "%lf", &params->f_weight);

      else if (!wordcmp(name, "pose_clustering", " \t\n"))
	sscanf(value, "%d", &params->pose_clustering);
      else if (!wordcmp(name, "x_cluster_thresh", " \t\n"))
	sscanf(value, "%lf", &params->x_cluster_thresh);
      else if (!wordcmp(name, "q_cluster_thresh", " \t\n"))
	sscanf(value, "%lf", &params->q_cluster_thresh);

      else if (!wordcmp(name, "range_edge_weight", " \t\n"))
	sscanf(value, "%lf", &params->range_edge_weight);
      else if (!wordcmp(name, "curv_edge_weight", " \t\n"))
	sscanf(value, "%lf", &params->curv_edge_weight);
      else if (!wordcmp(name, "img_edge_weight", " \t\n"))
	sscanf(value, "%lf", &params->img_edge_weight);
      else if (!wordcmp(name, "edge_blur", " \t\n"))
	sscanf(value, "%d", &params->edge_blur);
      else if (!wordcmp(name, "edge_weight", " \t\n"))
	sscanf(value, "%lf", &params->edge_weight);

      /*
      else if (!wordcmp(name, "surfdist_weight", " \t\n"))
	sscanf(value, "%lf", &params->surfdist_weight);
      else if (!wordcmp(name, "surfwidth_weight", " \t\n"))
	sscanf(value, "%lf", &params->surfwidth_weight);
      else if (!wordcmp(name, "surfdist_thresh", " \t\n"))
	sscanf(value, "%lf", &params->surfdist_thresh);
      else if (!wordcmp(name, "surfwidth_thresh", " \t\n"))
	sscanf(value, "%lf", &params->surfwidth_thresh);
      else if (!wordcmp(name, "surfdist_sigma", " \t\n"))
	sscanf(value, "%lf", &params->surfdist_sigma);
      else if (!wordcmp(name, "surfwidth_sigma", " \t\n"))
	sscanf(value, "%lf", &params->surfwidth_sigma);
      else if (!wordcmp(name, "fsurf_sigma", " \t\n"))
	sscanf(value, "%lf", &params->fsurf_sigma);
      */

      else {
	fprintf(stderr, "Error: bad parameter ''%s'' at line %d of %s\n", s, cnt, param_file);
	exit(1);
      }
    }
  }
  fclose(f);
}


int main(int argc, char *argv[])
{
  short have_true_pose = 0;
  simple_pose_t true_pose;
  if (argc < 9) {
    printf("usage: %s <pcd_obs> <pcd_obs_fg> <pcd_obs_sift> <pcd_model> <pcd_model_sift> <pcd_model_range_edges> <param_file> <samples_output>\n", argv[0]);
    printf("or\n");
    printf("usage: %s <pcd_obs> <pcd_obs_fg> <pcd_obs_sift> <pcd_model> <pcd_model_sift> <pcd_model_range_edges> <param_file> <samples_output> <ground_truth_file>\n", argv[0]);
    return 1;
  } else if (argc == 10) {
    have_true_pose = 1;
  }

  olf_obs_t obs;
  //obs.range_image = pcd_to_range_image(load_pcd(argv[1]), 0, M_PI/1000.0);
  obs.bg_pcd = load_pcd(argv[1]);
  obs.fg_pcd = load_pcd(argv[2]);
  obs.sift_pcd = load_pcd(argv[3]);

  olf_model_t model;
  model.obj_pcd = load_pcd(argv[4]);
  model.sift_pcd = load_pcd(argv[5]);
  model.range_edges_pcd = load_pcd(argv[6]);

  scope_params_t params;
  load_params(&params, argv[7]);

  if (have_true_pose) {
    load_true_pose(argv[9], &true_pose);
  }

  FILE *f = fopen(argv[8], "w");
  if (f == NULL) {
    printf("Can't open %s for writing\n", argv[2]);
    return 1;
  }

  olf_pose_samples_t *poses;
  if (have_true_pose) {
    poses = scope(&model, &obs, &params, have_true_pose, &true_pose);
  } else {
    poses = scope(&model, &obs, &params, have_true_pose, NULL);
  }

  // write pose samples to file
  double **X = poses->X;
  double **Q = poses->Q;
  double *W = poses->W;
  int n = poses->n;
  double **vis_probs = poses->vis_probs; //dbug
  /*
  for (n = 1; n < poses->n; n++)
    if (W[n] < .01 * W[0])
      break;
  mult(W, W, 1/sum(W,n), n);
  */

  fprintf(f, "X = [");
  int i;
  for (i = 0; i < n; i++)
    fprintf(f, "%f, %f, %f;  ", X[i][0], X[i][1], X[i][2]);
  fprintf(f, "];\n");

  fprintf(f, "Q = [");
  for (i = 0; i < n; i++)
    fprintf(f, "%f, %f, %f, %f;  ", Q[i][0], Q[i][1], Q[i][2], Q[i][3]);
  fprintf(f, "];\n");

  fprintf(f, "W = [");
  for (i = 0; i < n; i++)
    fprintf(f, "%f ", W[i]);
  fprintf(f, "];\n");

  //dbug
  fprintf(f, "vis_probs = [");
  int j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < model.obj_pcd->num_points; j++)
      fprintf(f, "%f ", vis_probs[i][j]);
    fprintf(f, "; ");
  }
  fprintf(f, "];\n");


  fclose(f);

  return 0;
}
