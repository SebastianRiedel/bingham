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


char *get_dirname(char *path)
{
  // get directory name
  char *s = strrchr(path, '/');
  char *dirname;
  if (s != NULL) {
    int n = s - path;
    safe_calloc(dirname, n+1, char);
    if (n > 0)
      memcpy(dirname, path, n);
    dirname[n] = '\0';
  }
  else {
    safe_calloc(dirname, 2, char);
    sprintf(dirname, ".");
  }

  return dirname;
}

olf_model_t load_olf_model(char *model_file)
{
  char *dirname = get_dirname(model_file);

  FILE *f = fopen(model_file, "r");
  if (f == NULL) {
    fprintf(stderr, "Error loading model file: %s\n", model_file);
    return NULL;
  }

  char line[1024];
  if (!fgets(line, 1024, f)) {
    fprintf(stderr, "Error parsing model file: %s\n", model_file);
    return NULL;
  }
  fclose(f);

  char model_name[1024], obj_pcd[1024], fpfh_pcd[1024], sift_pcd[1024], range_edges_pcd[1024];
  if (sscanf(line, "%s %s %s %s %s", model_name, obj_pcd, fpfh_pcd, sift_pcd, range_edges_pcd) < 5) {
    fprintf(stderr, "Error parsing model file: %s\n", model_file);
    return NULL;
  }

  olf_model_t model;
  safe_calloc(model.name, strlen(model_name)+1, char);
  strcpy(model.name, model_name);

  sprintf(line, "%s/%s", dirname, obj_pcd);
  model.obj_pcd = load_pcd(line);

  sprintf(line, "%s/%s", dirname, fpfh_pcd);
  model.fpfh_pcd = load_pcd(line);

  sprintf(line, "%s/%s", dirname, sift_pcd);
  model.sift_pcd = load_pcd(line);

  sprintf(line, "%s/%s", dirname, range_edges_pcd);
  model.range_edges_pcd = load_pcd(line);

  //cleanup
  free(dirname);

  return model;
}


olf_model_t *load_olf_models(int *n, char *models_file)
{
  char *dirname = get_dirname(models_file);

  FILE *f = fopen(model_files, "r");
  if (f == NULL) {
    fprintf(stderr, "Error loading models file: %s\n", models_file);
    return NULL;
  }

  // get the number of non-empty lines in models_file
  int num_models = 0;
  char line[1024];
  whlie (!feof(f)) {
    if (!fgets(line, 1024, f)) {
      fprintf(stderr, "Error parsing model file: %s\n", model_file);
      return NULL;
    }
    strcbrk(
  }

  rewind(f);

  fclose(f);

  //cleanup
  free(dirname);

  return models;
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
  memset(&params, 0, sizeof(scope_params_t));
  load_scope_params(&params, argv[8]);

  FILE *f = fopen(argv[9], "w");
  if (f == NULL) {
    printf("Can't open %s for writing\n", argv[9]);
    return 1;
  }

  if (argc > 10)
    have_true_pose = load_true_pose(argv[10], &true_pose);


  // get data
  scope_model_data_t *model_data = get_scope_model_data(&model, &params);
  scope_obs_data_t *obs_data = get_scope_obs_data(&obs, &params);


  scope_samples_t *S = scope(model_data, obs_data, &params, (have_true_pose ? &true_pose : NULL));
  int n = S->num_samples;


  // cleanup
  free_scope_model_data(model_data);
  free_scope_obs_data(obs_data);
  



  fprintf(f, "X = [");
  int i, j;
  for (i = 0; i < n; i++)
    fprintf(f, "%f, %f, %f;  ", S->samples[i].x[0], S->samples[i].x[1], S->samples[i].x[2]);
  fprintf(f, "];\n");

  fprintf(f, "Q = [");
  for (i = 0; i < n; i++)
    fprintf(f, "%f, %f, %f, %f;  ", S->samples[i].q[0], S->samples[i].q[1], S->samples[i].q[2], S->samples[i].q[3]);
  fprintf(f, "];\n");

  fprintf(f, "W = [");
  for (i = 0; i < n; i++)
    fprintf(f, "%f ", S->W[i]);
  fprintf(f, "];\n");


  //**************************************************

  //dbug
  fprintf(f, "scores = [");
  for (i = 0; i < n; i++) {
    for (j = 0; j < S->samples[i].num_scores; j++)
      fprintf(f, "%f ", S->samples[i].scores[j]);
    fprintf(f, "; ");
  }
  fprintf(f, "];\n");

  fprintf(f, "C_obs = [");
  for (i = 0; i < n; i++) {
    for (j = 0; j < params.num_correspondences; j++)
      fprintf(f, "%d ", 1 + S->samples[i].c_obs[j]);
    fprintf(f, "; ");
  }
  fprintf(f, "];\n");

  fprintf(f, "C_model = [");
  for (i = 0; i < n; i++) {
    for (j = 0; j < params.num_correspondences; j++)
      fprintf(f, "%d ", 1 + S->samples[i].c_model[j]);
    fprintf(f, "; ");
  }
  fprintf(f, "];\n");

  fprintf(f, "C_type = [");
  for (i = 0; i < n; i++) {
    for (j = 0; j < params.num_correspondences; j++)
      fprintf(f, "%d ", S->samples[i].c_type[j]);
    fprintf(f, "; ");
  }
  fprintf(f, "];\n");
  
  fprintf(f, "C_num = [");
  for (i = 0; i < n; i++)
    fprintf(f, "%d ", S->samples[i].nc);
  fprintf(f, "];\n");
  
  fprintf(f, "vis_probs = [");
  for (i = 0; i < n; i++) {
    for (j = 0; j < model.obj_pcd->num_points; j++)
      fprintf(f, "%f ", S->samples[i].vis_probs[j]);
    fprintf(f, "; ");
  }
  fprintf(f, "];\n");

  fprintf(f, "labdist_p_ratios = [");
  for (i = 0; i < n; i++) {
    for (j = 0; j < model.obj_pcd->num_points; j++)
      fprintf(f, "%f ", S->samples[i].labdist_p_ratios[j]);
    fprintf(f, "; ");
  }
  fprintf(f, "];\n");

  /***************************************************

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

  *****************************************************/

  fclose(f);

  return 0;
}
