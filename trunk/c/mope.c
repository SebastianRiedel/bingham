#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bingham.h"
#include "bingham/util.h"
#include "bingham/olf.h"

#include "bingham/cuda_wrapper.h"

int main(int argc, char *argv[])
{
  if (argc < 9) {
    printf("usage: %s <pcd_obs> <pcd_obs_fpfh> <pcd_obs_sift> <pcd_obs_shot> <model> <scope_param_file> <mope_param_file> <output_file> <r/s/w flag> <if flag is r or w, sample_file>\n", argv[0]);
    return 1;
  }
  if (argc == 10 && (argv[9][0] == 'r' || argv[9][0] == 'w')) {
    printf("Missing samples file\n");
  }

  char flag;
  char *fl = NULL;
  if (argc == 10) {
    flag = 's';
  } else {  
    fl = argv[9];
    flag = fl[0];
    if (flag != 'r' && flag != 'w' && flag != 's') {
      printf("Invalid read/write flag\n");
      return 1;
    }
  }

  FILE *f = fopen(argv[8], "w");
  if (f == NULL) {
    printf("Can't open %s for writing\n", argv[8]);
    return 1;
  }
  FILE *sample = NULL;
  if (flag != 's') {
    sample = fopen(argv[10], fl);
    if (sample == NULL) {
      printf("Can't open %s", argv[10]);
      return 1;
    }
  }

  int i;
  mope_samples_t *M;
  int num_models;

  // load params
  scope_params_t scope_params;
  memset(&scope_params, 0, sizeof(scope_params_t));
  load_scope_params(&scope_params, argv[6]);

  mope_params_t mope_params;
  memset(&mope_params, 0, sizeof(mope_params_t));
  load_mope_params(&mope_params, argv[7]);

  int num_obs_segments;
  int *segment_cnts = NULL;
  
  if (flag == 'w' || flag == 's') {

    // load obs data
    olf_obs_t obs;
    obs.bg_pcd = load_pcd(argv[1]);
    if (scope_params.use_fpfh)
      obs.fpfh_pcd = load_pcd(argv[2]);
    if (scope_params.use_shot)
      obs.shot_pcd = load_pcd(argv[3]);
    if (scope_params.use_sift)
      obs.sift_pcd = load_pcd(argv[4]);
    scope_obs_data_t obs_data;
    get_scope_obs_data(&obs_data, &obs, &scope_params);

    num_obs_segments = obs_data.num_obs_segments;
    safe_calloc(segment_cnts, num_obs_segments, int);
  
    // load model data
    olf_model_t *models = load_olf_models(&num_models, argv[5], &scope_params);
    scope_model_data_t model_data[num_models];
    for (i = 0; i < num_models; i++)
      get_scope_model_data(&model_data[i], &models[i], &scope_params);

    //mope_sample_t *M = mope_greedy(model_data, num_models, &obs_data, &scope_params, &mope_params, cu_model, &cu_obs);
    if (scope_params.use_cuda) {
      cu_init();
      cu_model_data_t cu_model[num_models];
      cu_obs_data_t cu_obs;
      cu_init_scoring_mope(model_data, &obs_data, num_models, cu_model, &cu_obs, &scope_params);
       
      M = annealing_with_scope(model_data, num_models, segment_cnts, &obs_data, &scope_params, &mope_params, cu_model, &cu_obs, NULL, sample, 1, NULL);
      cu_free_all_the_things_mope(cu_model, &cu_obs, num_models, &scope_params);
    } else {
      M = annealing_with_scope(model_data, num_models, segment_cnts, &obs_data, &scope_params, &mope_params, NULL, NULL, NULL, sample, 1, NULL);
    }

    /*//dbug
    for (i = 0; i < M->num_objects; i++)
    printf("%s\n", models[ M->model_ids[i] ].name);*/

    // cleanup
    for (i = 0; i < num_models; i++)
      free_scope_model_data(&model_data[i]);
    free_scope_obs_data(&obs_data);

  } else {
    char sbuf[1024];
    char *s = sbuf;

    fgets(s, 1024, sample);
    sscanf(s, "%d", &num_models);
    fgets(s, 1024, sample);
    sscanf(s, "%d", &num_obs_segments);
    
    safe_calloc(segment_cnts, num_obs_segments, int);
    for (i = 0; i < num_obs_segments; ++i) {
      fgets(s, 1024, sample);
      sscanf(s, "%d", &segment_cnts[i]);
    }

    scope_model_data_t model_data[num_models];
    scope_obs_data_t obs_data;
    if (mope_params.num_rounds == 2) {
      // load model data
      olf_model_t *models = load_olf_models(&num_models, argv[5], &scope_params);
      for (i = 0; i < num_models; i++)
	get_scope_model_data(&model_data[i], &models[i], &scope_params);

      // load obs data
      olf_obs_t obs;
      obs.bg_pcd = load_pcd(argv[1]);
      if (scope_params.use_fpfh)
	obs.fpfh_pcd = load_pcd(argv[2]);
      if (scope_params.use_shot)
	obs.shot_pcd = load_pcd(argv[3]);
      if (scope_params.use_sift)
	obs.sift_pcd = load_pcd(argv[4]);
      get_scope_obs_data(&obs_data, &obs, &scope_params);
    }

    //num_models = 35;
    M = annealing_existing_samples(model_data, num_models, segment_cnts, &obs_data, num_obs_segments, &scope_params, &mope_params, sample, 1);
    
    /*//dbug
    for (i = 0; i < M->num_objects; i++)
    printf("%s\n", models[ M->model_ids[i] ].name);*/
  } 
 
  int idx; // I hate when I run out of loop letters...
  fprintf(f, "results = cell(%d, 1);\n", M->num_samples);
  fprintf(f, "num_obs_segments = %d;\n", num_obs_segments);
  
  fprintf(f, "segment_cnts = [");
  for (idx = 0; idx < num_obs_segments; ++idx) {
    fprintf(f, "%d, ", segment_cnts[idx]);
  }
  fprintf(f, "];\n");

  for (idx = 0; idx < M->num_samples; ++idx) {
    
    fprintf(f, "results{%d} = struct();\n", idx + 1);
    
    int n = M->samples[idx].num_objects;
    fprintf(f, "X = [");
    for (i = 0; i < n; i++)
      fprintf(f, "%f, %f, %f;  ", M->samples[idx].objects[i].x[0], M->samples[idx].objects[i].x[1], M->samples[idx].objects[i].x[2]);
    fprintf(f, "];\n");
  
    fprintf(f, "Q = [");
    for (i = 0; i < n; i++)
      fprintf(f, "%f, %f, %f, %f;  ", M->samples[idx].objects[i].q[0], M->samples[idx].objects[i].q[1], M->samples[idx].objects[i].q[2], M->samples[idx].objects[i].q[3]);
    fprintf(f, "];\n");
  
    fprintf(f, "IDs = [");
    for (i = 0; i < n; i++)
      fprintf(f, "%d ", M->samples[idx].model_ids[i] + 1);
    fprintf(f, "];\n");
  
    fprintf(f, "scope_scores = [");
    int j;
    for (i = 0; i < n; i++) {
      for (j = 0; j < M->samples[idx].objects[i].num_scores; j++)
	fprintf(f, "%f ", M->samples[idx].objects[i].scores[j]);
      fprintf(f, "; ");
    }
    fprintf(f, "];\n");
    
    double weights[18] = {mope_params.scope_xyz_weight, mope_params.scope_normal_weight, mope_params.scope_vis_weight, mope_params.scope_random_walk_weight, mope_params.scope_edge_weight, 
			  mope_params.scope_edge_vis_weight, mope_params.scope_edge_occ_weight, mope_params.scope_L_weight, mope_params.scope_A_weight, mope_params.scope_B_weight, mope_params.scope_fpfh_weight,
			  mope_params.scope_specularity_weight, mope_params.scope_segment_affinity_weight, mope_params.scope_segment_weight, mope_params.scope_table_weight,
			  0, 0, 0};
    
    fprintf(f, "scope_W = [");
    for (i = 0; i < n; i++) {
      double scope_W = dot(weights, M->samples[idx].objects[i].scores, 15);
      fprintf(f, "%f ", scope_W);
    }
    fprintf(f, "];\n");

    fprintf(f, "segments_idx = {");
    for (i = 0; i < n; i++) {
      fprintf(f, "[");
      for (j = 0; j < M->samples[idx].objects[i].num_segments; j++)
	fprintf(f, "%d ", M->samples[idx].objects[i].segments_idx[j]);
      fprintf(f, "], ");
    }
    fprintf(f, "};\n");


    fprintf(f, "mope_scores = [");
    for (i = 0; i < M->samples[idx].num_scores; i++)
      fprintf(f, "%f ", M->samples[idx].scores[i]);
    fprintf(f, "];\n");

    fprintf(f, "num_segments = [");
    for (i = 0; i < n; i++) {
      fprintf(f, "%d ", M->samples[idx].objects[i].num_segments);
    }
    fprintf(f, "];\n");

    fprintf(f, "num_objects = %d;\n", M->samples[idx].num_objects);

    fprintf(f, "results{%d}.W = %f;\n", idx+1, M->W[idx]);

    fprintf(f, "results{%d}.X = X;\n", idx+1);
    fprintf(f, "results{%d}.Q = Q;\n", idx+1);
    fprintf(f, "results{%d}.IDs = IDs;\n", idx+1);
    fprintf(f, "results{%d}.scope_scores = scope_scores;\n", idx+1);
    fprintf(f, "results{%d}.mope_scores = mope_scores;\n", idx+1);
    fprintf(f, "results{%d}.num_segments = num_segments;\n", idx+1);
    fprintf(f, "results{%d}.segments_idx = segments_idx;\n", idx+1);
    fprintf(f, "results{%d}.scope_W = scope_W;\n", idx+1);	
    fprintf(f, "results{%d}.num_objects = num_objects;\n", idx+1);
    fprintf(f, "results{%d}.num_obs_segments = num_obs_segments;\n", idx+1);
    fprintf(f, "results{%d}.segment_cnts = segment_cnts;\n", idx+1);
    
  }
  fclose(f);
  
  //free(segment_cnts);
  /*for (idx = 0; idx < 10; ++idx) {
    printf("Mope score[%d] = %lf", idx, M->W[idx]);
    printf("\t Object IDs:");
    for (i = 0; i < M->samples[idx].num_objects; ++i)
      printf(" %d", M->samples[idx].model_ids[i]);
    printf("\n");
    }*/

  return 0;
}
