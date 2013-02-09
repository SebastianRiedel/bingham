#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bingham.h"
#include "bingham/util.h"
#include "bingham/olf.h"


int main(int argc, char *argv[])
{
  if (argc < 7) {
    printf("usage: %s <pcd_obs> <pcd_obs_fg> <pcd_obs_sift> <model> <param_file> <samples_output>\n", argv[0]);
    return 1;
  }

  // load params
  scope_params_t params;
  memset(&params, 0, sizeof(scope_params_t));
  load_scope_params(&params, argv[5]);

  // load obs data
  olf_obs_t obs;
  obs.bg_pcd = load_pcd(argv[1]);
  obs.fg_pcd = load_pcd(argv[2]);
  obs.sift_pcd = load_pcd(argv[3]);
  scope_obs_data_t obs_data;
  get_scope_obs_data(&obs_data, &obs, &params);

  // load model data
  int num_models;
  olf_model_t *models = load_olf_models(&num_models, argv[4]);
  scope_model_data_t model_data[num_models];
  int i;
  for (i = 0; i < num_models; i++)
    get_scope_model_data(&model_data[i], &models[i], &params);

  /*
  FILE *f = fopen(argv[6], "w");
  if (f == NULL) {
    printf("Can't open %s for writing\n", argv[6]);
    return 1;
  }
  */

  mope_sample_t *M = mope_greedy(model_data, num_models, &obs_data, &params);

  //dbug
  for (i = 0; i < M->num_objects; i++)
    printf("%s\n", models[ M->model_ids[i] ].name);

  // cleanup
  for (i = 0; i < num_models; i++)
    free_scope_model_data(&model_data[i]);
  free_scope_obs_data(&obs_data);


  return 0;
}

