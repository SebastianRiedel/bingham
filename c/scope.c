#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bingham.h"
#include "bingham/util.h"
#include "bingham/olf.h"


int load_true_pose(char *pose_file, simple_pose_t *true_pose) {
  FILE *f = fopen(pose_file, "r");
  if (f == NULL) {
    fprintf(stderr, "Error loading true pose file: %s\n", pose_file);
    return 0;
  }
  char sbuf[1024];
  char *s = sbuf;
  if (fgets(s, 1024, f)) {
    int n = sscanf(s, "%lf %lf %lf %lf %lf %lf %lf", &true_pose->X[0], &true_pose->X[1], &true_pose->X[2],
		   &true_pose->Q[0], &true_pose->Q[1], &true_pose->Q[2], &true_pose->Q[3]);
    if (n < 7) {
      fprintf(stderr, "Error reading true pose file: %s\n", pose_file);
    }

    //s = sword(s, " ", 1);
    /*
    int i;
    for (i = 0; i < 3; ++i) {
      s = sword(s, " ", 1);
      true_pose->X[i] = atof(s);
    }
    for (i = 0; i < 4; ++i) {
      s = sword(s, " ", 1);
      true_pose->Q[i] = atof(s);
    }
    */
  }
  fclose(f);

  return 1;
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
      else if (!wordcmp(name, "range_weight", " \t\n"))
	sscanf(value, "%lf", &params->range_weight);
      else if (!wordcmp(name, "f_sigma", " \t\n"))
	sscanf(value, "%lf", &params->f_sigma);
      else if (!wordcmp(name, "xyz_sigma", " \t\n"))
	sscanf(value, "%lf", &params->xyz_sigma);
      else if (!wordcmp(name, "vis_thresh", " \t\n"))
	sscanf(value, "%lf", &params->vis_thresh);
      else if (!wordcmp(name, "f_weight", " \t\n"))
	sscanf(value, "%lf", &params->f_weight);

      else if (!wordcmp(name, "range_sigma", " \t\n"))
	sscanf(value, "%lf", &params->range_sigma);
      else if (!wordcmp(name, "normal_sigma", " \t\n"))
	sscanf(value, "%lf", &params->normal_sigma);
      else if (!wordcmp(name, "lab_sigma", " \t\n"))
	sscanf(value, "%lf", &params->lab_sigma);

      else if (!wordcmp(name, "score2_xyz_weight", " \t\n"))
	sscanf(value, "%lf", &params->score2_xyz_weight);
      else if (!wordcmp(name, "score2_normal_weight", " \t\n"))
	sscanf(value, "%lf", &params->score2_normal_weight);
      else if (!wordcmp(name, "score2_vis_weight", " \t\n"))
	sscanf(value, "%lf", &params->score2_vis_weight);
      else if (!wordcmp(name, "score2_segment_weight", " \t\n"))
	sscanf(value, "%lf", &params->score2_segment_weight);
      else if (!wordcmp(name, "score2_edge_weight", " \t\n"))
	sscanf(value, "%lf", &params->score2_edge_weight);
      else if (!wordcmp(name, "score2_edge_occ_weight", " \t\n"))
	sscanf(value, "%lf", &params->score2_edge_occ_weight);
      else if (!wordcmp(name, "score2_edge_vis_weight", " \t\n"))
	sscanf(value, "%lf", &params->score2_edge_vis_weight);
      else if (!wordcmp(name, "score2_L_weight", " \t\n"))
	sscanf(value, "%lf", &params->score2_L_weight);
      else if (!wordcmp(name, "score2_A_weight", " \t\n"))
	sscanf(value, "%lf", &params->score2_A_weight);
      else if (!wordcmp(name, "score2_B_weight", " \t\n"))
	sscanf(value, "%lf", &params->score2_B_weight);

      else if (!wordcmp(name, "score3_xyz_weight", " \t\n"))
	sscanf(value, "%lf", &params->score3_xyz_weight);
      else if (!wordcmp(name, "score3_normal_weight", " \t\n"))
	sscanf(value, "%lf", &params->score3_normal_weight);
      else if (!wordcmp(name, "score3_vis_weight", " \t\n"))
	sscanf(value, "%lf", &params->score3_vis_weight);
      else if (!wordcmp(name, "score3_segment_weight", " \t\n"))
	sscanf(value, "%lf", &params->score3_segment_weight);
      else if (!wordcmp(name, "score3_edge_weight", " \t\n"))
	sscanf(value, "%lf", &params->score3_edge_weight);
      else if (!wordcmp(name, "score3_edge_occ_weight", " \t\n"))
	sscanf(value, "%lf", &params->score3_edge_occ_weight);
      else if (!wordcmp(name, "score3_edge_vis_weight", " \t\n"))
	sscanf(value, "%lf", &params->score3_edge_vis_weight);
      else if (!wordcmp(name, "score3_L_weight", " \t\n"))
	sscanf(value, "%lf", &params->score3_L_weight);
      else if (!wordcmp(name, "score3_A_weight", " \t\n"))
	sscanf(value, "%lf", &params->score3_A_weight);
      else if (!wordcmp(name, "score3_B_weight", " \t\n"))
	sscanf(value, "%lf", &params->score3_B_weight);


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
  if (argc < 10 || argc > 11) {
    printf("usage: %s <pcd_obs> <pcd_obs_fg> <pcd_obs_sift> <pcd_model> <pcd_model_fpfh> <pcd_model_sift> <pcd_model_range_edges> <param_file> <samples_output>\n", argv[0]);
    printf("or\n");
    printf("usage: %s <pcd_obs> <pcd_obs_fg> <pcd_obs_sift> <pcd_model> <pcd_model_fpfh> <pcd_model_sift> <pcd_model_range_edges> <param_file> <samples_output> <ground_truth_file>\n", argv[0]);
    return 1;
  }

  olf_obs_t obs;
  //obs.range_image = pcd_to_range_image(load_pcd(argv[1]), 0, M_PI/1000.0);
  obs.bg_pcd = load_pcd(argv[1]);
  obs.fg_pcd = load_pcd(argv[2]);
  obs.sift_pcd = load_pcd(argv[3]);

  olf_model_t model;
  model.obj_pcd = load_pcd(argv[4]);
  model.fpfh_pcd = load_pcd(argv[5]);
  model.sift_pcd = load_pcd(argv[6]);
  model.range_edges_pcd = load_pcd(argv[7]);

  scope_params_t params;
  load_params(&params, argv[8]);

  FILE *f = fopen(argv[9], "w");
  if (f == NULL) {
    printf("Can't open %s for writing\n", argv[9]);
    return 1;
  }

  if (argc > 10)
    have_true_pose = load_true_pose(argv[10], &true_pose);

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

  double **scores = poses->scores;
  int num_scores = poses->num_scores;

  /*
  for (n = 1; n < poses->n; n++)
    if (W[n] < .01 * W[0])
      break;
  mult(W, W, 1/sum(W,n), n);
  */

  fprintf(f, "X = [");
  int i, j;
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

  if (scores) {
    fprintf(f, "scores = [");
    for (i = 0; i < n; i++) {
      for (j = 0; j < num_scores; j++)
	fprintf(f, "%f ", scores[i][j]);
      fprintf(f, "; ");
    }
    fprintf(f, "];\n");
  }


  //**************************************************

  //dbug
  if (poses->C_obs) {
    fprintf(f, "C_obs = [");
    for (i = 0; i < n; i++) {
      for (j = 0; j < params.num_correspondences; j++)
	fprintf(f, "%d ", 1 + poses->C_obs[i][j]);
      fprintf(f, "; ");
    }
    fprintf(f, "];\n");
  }
  if (poses->C_model) {
    fprintf(f, "C_model = [");
    for (i = 0; i < n; i++) {
      for (j = 0; j < params.num_correspondences; j++)
	fprintf(f, "%d ", 1 + poses->C_model[i][j]);
      fprintf(f, "; ");
    }
    fprintf(f, "];\n");
  }
  
  

  //***************************************************

  //dbug
  double **vis_probs = poses->vis_probs;
  double **xyz_dists = poses->xyz_dists;
  double **normal_dists = poses->normal_dists;
  int **range_edge_pixels = poses->range_edge_pixels;
  double **range_edge_vis_prob = poses->range_edge_vis_prob;
  int *num_range_edge_points = poses->num_range_edge_points;
  int **occ_edge_pixels = poses->occ_edge_pixels;
  int *num_occ_edge_points = poses->num_occ_edge_points;

  //dbug
  fprintf(f, "vis_probs = [");
  for (i = 0; i < n; i++) {
    for (j = 0; j < model.obj_pcd->num_points; j++)
      fprintf(f, "%f ", vis_probs[i][j]);
    fprintf(f, "; ");
  }
  fprintf(f, "];\n");

  //dbug
  fprintf(f, "xyz_dists = [");
  for (i = 0; i < n; i++) {
    for (j = 0; j < model.obj_pcd->num_points; j++)
      fprintf(f, "%f ", xyz_dists[i][j]);
    fprintf(f, "; ");
  }
  fprintf(f, "];\n");

  //dbug
  fprintf(f, "normal_dists = [");
  for (i = 0; i < n; i++) {
    for (j = 0; j < model.obj_pcd->num_points; j++)
      fprintf(f, "%f ", normal_dists[i][j]);
    fprintf(f, "; ");
  }
  fprintf(f, "];\n");

  //dbug
  extern double **obs_edge_image_;
  extern double **obs_edge_image_orig_;
  extern int obs_edge_image_width_;
  extern int obs_edge_image_height_;
  fprintf(f, "obs_edge_image = [");
  for (i = 0; i < obs_edge_image_width_ * obs_edge_image_height_; i++)
    fprintf(f, "%f ", obs_edge_image_[0][i]);
  fprintf(f, "];\n");
  fprintf(f, "obs_edge_image = reshape(obs_edge_image, [%d,%d]);\n\n", obs_edge_image_height_, obs_edge_image_width_);
  fprintf(f, "obs_edge_image_orig = [");
  for (i = 0; i < obs_edge_image_width_ * obs_edge_image_height_; i++)
    fprintf(f, "%f ", obs_edge_image_orig_[0][i]);
  fprintf(f, "];\n");
  fprintf(f, "obs_edge_image_orig = reshape(obs_edge_image_orig, [%d,%d]);\n\n", obs_edge_image_height_, obs_edge_image_width_);

  //dbug
  //extern double **range_edge_points_;
  //extern int **range_edge_pixels_;
  //extern double *range_edge_vis_prob_;
  //extern int num_range_edge_points_;
  //fprintf(f, "range_edge_points = [");
  //for (i = 0; i < num_range_edge_points_; i++)
  //  fprintf(f, "%f %f %f ; ", range_edge_points_[i][0], range_edge_points_[i][1], range_edge_points_[i][2]);
  //fprintf(f, "];\n");
  fprintf(f, "range_edge_pixels = {};\n");
  for (i = 0; i < n; i++) {
    fprintf(f, "range_edge_pixels{%d} = [", i+1);
    for (j = 0; j < num_range_edge_points[i]; j++)
      fprintf(f, "%d %d ; ", range_edge_pixels[i][2*j] + 1, range_edge_pixels[i][2*j+1] + 1);
    fprintf(f, "];\n");
  }
  fprintf(f, "range_edge_vis_prob = {};\n");
  for (i = 0; i < n; i++) {
    fprintf(f, "range_edge_vis_prob{%d} = [", i+1);
    for (j = 0; j < num_range_edge_points[i]; j++)
      fprintf(f, "%f ", range_edge_vis_prob[i][j]);
    fprintf(f, "];\n");
  }

  fprintf(f, "occ_edge_pixels = {};\n");
  for (i = 0; i < n; i++) {
    fprintf(f, "occ_edge_pixels{%d} = [", i+1);
    for (j = 0; j < num_occ_edge_points[i]; j++)
      fprintf(f, "%d %d ; ", occ_edge_pixels[i][2*j] + 1, occ_edge_pixels[i][2*j+1] + 1);
    fprintf(f, "];\n");
  }

  //*****************************************************/

  fclose(f);

  return 0;
}
