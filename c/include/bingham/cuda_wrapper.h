#ifndef CUDA_WRAPPER_H_
#define CUDA_WRAPPER_H_

#include "olf.h"

#ifdef __cplusplus
extern "C" {
#endif

struct scope_model_data_struct;
struct scope_obs_data_struct;
struct scope_sample_struct;
struct scope_params_struct;
struct score_comp_models_struct;

typedef struct {
  int *ptr;
  size_t n;
} cu_int_arr_t;

typedef struct {
  double *ptr;
  size_t n;
} cu_double_arr_t;

typedef struct {
  double *ptr;
  //size_t pitch;
  size_t n, m;
} cu_double_matrix_t;

typedef struct {
  int *ptr;
  //size_t pitch;
  size_t n, m;
} cu_int_matrix_t;

typedef struct {
  //cudaPitchedPtr ptr;
  //cudaExtent extent;
  double *ptr;
  size_t n, m, p;
} cu_double_matrix3d_t;

typedef struct cu_model_data_struct {
  cu_double_matrix_t points, normals, lab, ved, color_avg_cov, color_means1, color_means2, fpfh_shapes, range_edges_model_views, range_edges_points;
  cu_double_matrix3d_t color_cov1, color_cov2;
  
  cu_double_arr_t normalvar;
  cu_int_arr_t color_cnts1, color_cnts2, range_edges_view_idx, range_edges_view_cnt;

  int num_points, num_views, max_num_edges;
  struct score_comp_models_struct *score_comp_models;
  
} cu_model_data_t;

typedef struct {
  double res, min0, min1;
  int w, h;
} cu_range_image_data_t;

typedef struct cu_obs_data_struct {
  cu_double_matrix_t range_image, range_image_pcd_obs_lab, pcd_obs_fpfh, edge_image, segment_affinities;
  cu_double_matrix3d_t range_image_points, range_image_normals, obs_lab_image;

  cu_range_image_data_t range_image_data;

  cu_int_matrix_t range_image_cnt, range_image_idx;
  int num_obs_segments;
} cu_obs_data_t;


  void cu_init();

  void cu_init_scoring(struct scope_model_data_struct *model_data, struct scope_obs_data_struct *obs_data,
		       cu_model_data_t *cu_model, cu_obs_data_t *cu_obs, struct scope_params_struct **cu_params, struct scope_params_struct *params);
  void cu_init_scoring_mope(struct scope_model_data_struct *model_data, struct scope_obs_data_struct *obs_data, int num_models, cu_model_data_t cu_model[], cu_obs_data_t *cu_obs, 
			    struct scope_params_struct *params);
  void cu_free_all_the_things(cu_model_data_t *cu_model, cu_obs_data_t *cu_obs, struct scope_params_struct *cu_params, struct scope_params_struct *params);
  void cu_free_all_the_things_mope(cu_model_data_t cu_model[], cu_obs_data_t *cu_obs, int num_models, struct scope_params_struct *params);

//void cu_noise_models_sigmas(double *range_sigma, double *normal_sigma, double *l_sigma, double *a_sigma, double *b_sigma, const double *surface_angles, const double *edge_dists, int n);
  void cu_score_samples(double *scores, struct scope_sample_struct *samples, int num_samples, cu_model_data_t *cu_model, cu_obs_data_t *cu_obs, struct scope_params_struct *params, int score_round, 
			int num_validation_points, int num_obs_segments);

  void score_samples(double *scores, struct scope_sample_struct *samples, int num_samples, cu_model_data_t *cu_model, cu_obs_data_t *cu_obs, struct scope_params_struct *cu_params, 
		     struct scope_params_struct *params, int num_validation_points, int model_points, int num_obs_segments, int edge_scoring, int num_round);
  void align_models_gradient(struct scope_sample_struct *samples, int num_samples, cu_model_data_t *cu_model, cu_obs_data_t *cu_obs, struct scope_params_struct *cu_params, struct scope_params_struct *params, 
			     int num_points, int model_points, int round);
  void init_curand();

#ifdef __cplusplus
}
#endif

#endif
