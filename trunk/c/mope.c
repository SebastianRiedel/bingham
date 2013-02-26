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
  if (argc < 7) {
    printf("usage: %s <pcd_obs> <pcd_obs_fg> <pcd_obs_sift> <pcd_obs_shot> <model> <param_file> <samples_output>\n", argv[0]);
    return 1;
  }
  double tTotal = get_time_ms();

  double t0 = get_time_ms(); //dbug
  // load params
  scope_params_t params;
  memset(&params, 0, sizeof(scope_params_t));
  load_scope_params(&params, argv[6]);

  // load obs data
  olf_obs_t obs;
  obs.bg_pcd = load_pcd(argv[1]);
  obs.fg_pcd = load_pcd(argv[2]);
  obs.shot_pcd = load_pcd(argv[3]);
  obs.sift_pcd = load_pcd(argv[4]);
  scope_obs_data_t obs_data;
  get_scope_obs_data(&obs_data, &obs, &params);

  // load model data
  int num_models;
  olf_model_t *models = load_olf_models(&num_models, argv[5]);
  scope_model_data_t model_data[num_models];
  int i;
  for (i = 0; i < num_models; i++)
    get_scope_model_data(&model_data[i], &models[i], &params);

  printf("CPU load models: %f\n", (get_time_ms() - t0) / 1000.0);  //dbug
  t0 = get_time_ms();

  /*
  FILE *f = fopen(argv[7], "w");
  if (f == NULL) {
    printf("Can't open %s for writing\n", argv[7]);
    return 1;
  }
  */

  cu_init();
  printf("GPU init: %f\n", (get_time_ms() - t0) / 1000.0);  //dbug
  t0 = get_time_ms();

  cu_model_data_t cu_model[num_models];
  cu_obs_data_t cu_obs;
  cu_init_scoring_mope(model_data, &obs_data, num_models, cu_model, &cu_obs);
    
  printf("GPU model upload: %f\n", (get_time_ms() - t0) / 1000.0);  //dbug
  t0 = get_time_ms();


  /*for (i = 0; i < num_models; ++i)
    scope(&model_data[i], &obs_data, &params, NULL, &cu_model[i], &cu_obs);

  printf("18 scopes running time: %f\n", (get_time_ms() - t0) / 1000.0);  //dbug
  t0 = get_time_ms();*/


  //mope_sample_t *M = mope_greedy(model_data, num_models, &obs_data, &params, cu_model, &cu_obs);
  mope_sample_t *M = mope_annealing(model_data, num_models, &obs_data, &params, cu_model, &cu_obs);

  //dbug
  for (i = 0; i < M->num_objects; i++)
  printf("%s\n", models[ M->model_ids[i] ].name);

  // cleanup
  for (i = 0; i < num_models; i++)
    free_scope_model_data(&model_data[i]);
  free_scope_obs_data(&obs_data);

  FILE *f = fopen(argv[7], "w");
  if (f == NULL) {
    printf("Can't open %s for writing\n", argv[7]);
    return 1;
  }

  printf("CPU cleanup: %f\n", (get_time_ms() - t0) / 1000.0);  //dbug
  t0 = get_time_ms();

  cu_free_all_the_things_mope(cu_model, &cu_obs, num_models);

  printf("GPU cleanup: %f\n", (get_time_ms() - t0) / 1000.0);  //dbug
  printf("Total time: %f\n", (get_time_ms() - tTotal) / 1000.0);

  int n = M->num_objects;
  fprintf(f, "X = [");
  for (i = 0; i < n; i++)
    fprintf(f, "%f, %f, %f;  ", M->objects[i].x[0], M->objects[i].x[1], M->objects[i].x[2]);
  fprintf(f, "];\n");

  fprintf(f, "Q = [");
  for (i = 0; i < n; i++)
    fprintf(f, "%f, %f, %f, %f;  ", M->objects[i].q[0], M->objects[i].q[1], M->objects[i].q[2], M->objects[i].q[3]);
  fprintf(f, "];\n");

  fprintf(f, "IDs = [");
  for (i = 0; i < n; i++)
    fprintf(f, "%d ", M->model_ids[i] + 1);
  fprintf(f, "];\n");


  return 0;
}

