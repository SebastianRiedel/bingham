#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bingham.h"
#include "bingham/cuda_wrapper.h"
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



int main(int argc, char *argv[])
{
  short have_true_pose = 0;
  simple_pose_t true_pose;
  if (argc < 9 || argc > 10) {
    printf("usage: %s <pcd_obs> <pcd_obs_fpfh> <pcd_obs_shot> <pcd_obs_sift> <obs_table> <model> <param_file> <samples_output>\n", argv[0]);
    printf("or\n");
    printf("usage: %s <pcd_obs> <pcd_obs_fpfh> <pcd_obs_shot> <pcd_obs_sift> <obs_table> <model> <param_file> <samples_output> <ground_truth_file>\n", argv[0]);
    return 1;
  }

  // load params
  scope_params_t params;
  memset(&params, 0, sizeof(scope_params_t));
  load_scope_params(&params, argv[7]);

  // load obs data
  olf_obs_t obs;
  obs.bg_pcd = load_pcd(argv[1]);
  if (params.use_fpfh)
    obs.fpfh_pcd = load_pcd(argv[2]);
  if (params.use_shot)
    obs.shot_pcd = load_pcd(argv[3]);
  if (params.use_sift)
    obs.sift_pcd = load_pcd(argv[4]);
  int xxx,yyy;

  double **table_plane;
  if (params.use_table) {
    table_plane = load_matrix(argv[5], &xxx, &yyy);
    obs.table_plane = table_plane[0];
  }

  scope_obs_data_t obs_data;
  get_scope_obs_data(&obs_data, &obs, &params);

  // load model data
  olf_model_t model;
  load_olf_model(&model, argv[6], &params);
  scope_model_data_t model_data;
  get_scope_model_data(&model_data, &model, &params);

  FILE *f = fopen(argv[8], "w");
  if (f == NULL) {
    printf("Can't open %s for writing\n", argv[8]);
    return 1;
  }

  if (argc > 9)
    have_true_pose = load_true_pose(argv[9], &true_pose);


  //dbug
  if (params.test_bpa) {
    if (!have_true_pose) {
      printf("Error: must have true pose to test BPA!\n");
      return -1;
    }
    test_bpa(&model_data, &obs_data, &params, &true_pose);
    return 0;
  }


  scope_samples_t *S;

  if (params.use_cuda) {
    cu_model_data_t cu_model;
    cu_obs_data_t cu_obs;
    scope_params_t *cu_params;
    printf("Initializing CUDA\n");
    cu_init();
    printf("CUDA initialized\n");
    cu_init_scoring(&model_data, &obs_data, &cu_model, &cu_obs, &cu_params, &params);
    printf("Data copied\n");
    
    S = scope(&model_data, &obs_data, &params, (have_true_pose ? &true_pose : NULL), &cu_model, &cu_obs, cu_params, NULL);

    cu_free_all_the_things(&cu_model, &cu_obs, cu_params, &params);
  }
  else
    S = scope(&model_data, &obs_data, &params, (have_true_pose ? &true_pose : NULL), NULL, NULL, NULL, NULL);

  // cleanup
  free_scope_obs_data(&obs_data);
  free_scope_model_data(&model_data);


  //**************************************************

  int n = S->num_samples;

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

  fprintf(f, "B = {");
  for (i = 0; i < n; i++) {
    fprintf(f, "struct('d',4, 'F',%f, 'Z',[%f,%f,%f], 'V',[%f,%f,%f,%f; %f,%f,%f,%f; %f,%f,%f,%f]'), ",
	    S->samples[i].B.F, S->samples[i].B.Z[0], S->samples[i].B.Z[1], S->samples[i].B.Z[2],
	    S->samples[i].B.V[0][0], S->samples[i].B.V[0][1], S->samples[i].B.V[0][2], S->samples[i].B.V[0][3],
	    S->samples[i].B.V[1][0], S->samples[i].B.V[1][1], S->samples[i].B.V[1][2], S->samples[i].B.V[1][3],
	    S->samples[i].B.V[2][0], S->samples[i].B.V[2][1], S->samples[i].B.V[2][2], S->samples[i].B.V[2][3]);
  }
  fprintf(f, "};\n");

  fprintf(f, "X0 = [");
  for (i = 0; i < n; i++)
    fprintf(f, "%f, %f, %f;  ", S->samples[i].x0[0], S->samples[i].x0[1], S->samples[i].x0[2]);
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
  
  fprintf(f, "segments = {");
  for (i = 0; i < n; i++) {
    fprintf(f, "[");
    for (j = 0; j < S->samples[i].num_segments; j++)
      fprintf(f, "%d ", S->samples[i].segments_idx[j]);
    fprintf(f, "], ");
  }
  fprintf(f, "};\n");


  extern double t[4];
  fprintf(f, "timings = [");
  for (i = 0; i < 4; ++i)
    fprintf(f, "%lf ", t[i]);
  fprintf(f, "];\n");
    

  /*fprintf(f, "C_obs = {");
  for (i = 0; i < n; i++) {
    fprintf(f, "[");
    for (j = 0; j < S->samples[i].nc; j++)
      fprintf(f, "%d ", 1 + S->samples[i].c_obs[j]);
    fprintf(f, "], ");
  }
  fprintf(f, "};\n");

  fprintf(f, "C_model = {");
  for (i = 0; i < n; i++) {
    fprintf(f, "[");
    for (j = 0; j < S->samples[i].nc; j++)
      fprintf(f, "%d ", 1 + S->samples[i].c_model[j]);
    fprintf(f, "], ");
  }
  fprintf(f, "};\n");

  fprintf(f, "C_type = {");
  for (i = 0; i < n; i++) {
    fprintf(f, "[");
    for (j = 0; j < S->samples[i].nc; j++)
      fprintf(f, "%d ", S->samples[i].c_type[j]);
    fprintf(f, "], ");
  }
  fprintf(f, "};\n");
  
  fprintf(f, "C_num = [");
  for (i = 0; i < n; i++)
    fprintf(f, "%d ", S->samples[i].nc);
  fprintf(f, "];\n");

  */


  /*  fprintf(f, "obs_edge_image = [");
  for (i = 0; i < obs_edge_image_width_; ++i) {
    for (j = 0; j < obs_edge_image_height_; ++j) {
      fprintf(f, "%f ", obs_edge_image_orig_[i][j]);
    }
    fprintf(f, "; ");
  }
  fprintf(f, "]; \n");

  fprintf(f, "xyz_dists = [");
  for (i = 0; i < n; i++) {
    for (j = 0; j < S->samples[i].dists.n; j++)
      fprintf(f, "%f ", S->samples[i].dists.xyz_dists[j]);
    fprintf(f, "; ");
  }
  fprintf(f, "];\n");

  fprintf(f, "fpfh_dists = [");
  for (i = 0; i < n; i++) {
    for (j = 0; j < S->samples[i].dists.n; j++)
      fprintf(f, "%f ", S->samples[i].dists.fpfh_dists[j]);
    fprintf(f, "; ");
  }
  fprintf(f, "];\n");

  fprintf(f, "normal_dists = [");
  for (i = 0; i < n; i++) {
    for (j = 0; j < S->samples[i].dists.n; j++)
      fprintf(f, "%f ", S->samples[i].dists.normal_dists[j]);
    fprintf(f, "; ");
  }
  fprintf(f, "];\n");

  fprintf(f, "vis_probs = [");
  for (i = 0; i < n; i++) {
    for (j = 0; j < S->samples[i].dists.n; j++)
      fprintf(f, "%f ", S->samples[i].dists.vis_probs[j]);
    fprintf(f, "; ");
  }
  fprintf(f, "];\n");
  
  fprintf(f, "range_edge_pixels = {};\n");
  for (i = 0; i < n; ++i) {
    fprintf(f, "range_edge_pixels{%d} = [", i+1);
    for (j = 0; j < S->samples[i].dists.num_range_edge_points; j++)
      fprintf(f, "%d, %d; ", S->samples[i].dists.range_edge_pixels[j][0], S->samples[i].dists.range_edge_pixels[j][1]);
    fprintf(f, "];\n");    
  }

  fprintf(f, "range_edge_points = {};\n");
  for (i = 0; i < n; ++i) {
    fprintf(f, "range_edge_points{%d} = [", i+1);
    for (j = 0; j < S->samples[i].dists.num_range_edge_points; j++)
      fprintf(f, "%f, %f, %f; ", S->samples[i].dists.range_edge_points[j][0], S->samples[i].dists.range_edge_points[j][1], S->samples[i].dists.range_edge_points[j][2]);
    fprintf(f, "];\n");    
    }*/

  /*
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
  */

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
