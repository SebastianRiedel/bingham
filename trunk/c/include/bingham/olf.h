
#ifndef BINGHAM_OLF_H
#define BINGHAM_OLF_H


#ifdef __cplusplus
extern "C" {
#endif 



#include "bingham.h"
#include "hll.h"
#include <flann/flann.h>


  // point cloud data structure
  typedef struct {
    int num_channels;
    int num_points;
    char **channels;  // e.g. {"x", "y", "z"}
    double **data;    // e.g. data[0] = {x1,x2,x3}, data[1] = {y1,y2,y3}, etc.

    // data pointers
    double *pc1;
    double *pc2;
    double *range_edge;
    double *curv_edge;
    double *img_edge;

    // transposed data
    double **points;
    double **views;
    double **colors;
    double **normals;
    double **principal_curvatures;
    double **shapes;
    double **sift;
    //double **sdw;

    // computed data
    int *clusters;
    double **quaternions[2];
    double **lab;

    // data sizes
    int shape_length;
    int sift_length;
    int sdw_length;

    //kdtree_t *points_kdtree;
    //pcd_balls_t *balls;

  } pcd_t;


  // range image
  typedef struct {
    double res;
    double min[2];
    double vp[7];
    int w;
    int h;
    double **image;  // stored column-wise: image[x][y]
    int **idx;
    double ***points;   // average point in each cell
    double ***normals;  // average normal in each cell
    int **cnt;          // number of points in each cell
  } range_image_t;


  typedef struct {
    pcd_t *pcd;      // must be sorted by viewpoint!
    int num_views;
    double **views;  // Nx3 matrix of the unique viewpoints
    int *view_idx;   // view_idx[i] = index of first pcd point with views[i] as its viewpoint
    int *view_cnt;   // view_cnt[i] = number of pcd points with views[i] as their viewpoint
  } multiview_pcd_t;


  typedef struct {
    double **X;
    double **Q;
    double *W;
    int n;
    double **vis_probs;  //dbug
    double **xyz_dists;  //dbug
  } olf_pose_samples_t;


  typedef struct {
    double X[3];
    double Q[4];
  } simple_pose_t;



  pcd_t *load_pcd(char *f_pcd);                        // loads a pcd
  void pcd_free(pcd_t *pcd);                           // frees the contents of a pcd_t, but not the pointer itself
  int pcd_channel(pcd_t *pcd, char *channel_name);     // gets the index of a channel by name
  int pcd_add_channel(pcd_t *pcd, char *channel);      // adds a channel to pcd

  range_image_t *pcd_to_range_image(pcd_t *pcd, double *vp, double res, int padding);



  // BPA (Bingham Procrustean Alignment)

  typedef struct {
    pcd_t *obj_pcd;
    pcd_t *sift_pcd;
    pcd_t *range_edges_pcd;
  } olf_model_t;

  typedef struct {
    pcd_t *fg_pcd;
    pcd_t *sift_pcd;
    pcd_t *bg_pcd;
    //range_image_t *range_image;
  } olf_obs_t;


  typedef struct {  // scope_params_t

    // GENERAL PARAMS
    int verbose;
    int num_samples_init;
    int num_samples;
    int num_correspondences;
    int branching_factor;
    int num_validation_points;
    int knn;
    int use_range_image;
    int do_icp;
    int do_final_icp;

    // WEIGHT / SIGMA PARAMS
    int dispersion_weight;
    double xyz_weight;
    double normal_weight;
    double range_sigma;
    double range_weight;
    double sift_dthresh;
    double L_weight;
    double f_sigma;
    double lab_sigma;
    double xyz_sigma;
    double vis_weight;
    double f_weight;
    double edge_weight;
    double normal_sigma;

    // POSE CLUSTERING PARAMS
    int pose_clustering;
    double x_cluster_thresh;
    double q_cluster_thresh;

    // EDGE IMAGE PARAMS
    double range_edge_weight;
    double curv_edge_weight;
    double img_edge_weight;
    int edge_blur;

    //double surfdist_weight;
    //double surfwidth_weight;
    //double surfdist_thresh;
    //double surfwidth_thresh;
    //double surfdist_sigma;
    //double surfwidth_sigma;
    //double fsurf_sigma;

  } scope_params_t;


  olf_pose_samples_t *scope(olf_model_t *model, olf_obs_t *obs, scope_params_t *params, short have_true_pose, simple_pose_t *true_pose);


  /*  
  int sample_model_point_given_model_pose(double *X, double *Q, int *c_model_prev, int n_model_prev, double *model_pmf, pcd_t *pcd_model);

  int sample_obs_correspondence_given_model_pose(double *X, double *Q, int model, pcd_t *pcd_model, int shape_length, double **obs_fxyzn, flann_index_t obs_xyzn_index, struct FLANNParameters *obs_xyzn_params, scope_params_t *params);
  void get_model_pose_distribution_from_correspondences(pcd_t *pcd_obs, pcd_t *pcd_model, int *c_obs, int n, int *c_model, double xyz_sigma, double *x, bingham_t *B);
  void sample_model_pose(pcd_t *pcd_model, int *c_model, int c, double *x0, bingham_t *B, double *x, double *q);
  void model_pose_likelihood(pcd_t *pcd_obs, pcd_t *pcd_model, int *c_obs, int *c_model, int n, double *x, double *q, bingham_t *B, double xyz_sigma, double f_sigma, double dispersion_weight, 
			     double *logp, double logp_comp[4]);
  int sample_model_correspondence_given_model_pose(pcd_t *pcd_obs, double **model_fxyzn, scope_params_t *params, struct FLANNParameters *model_xyzn_params, flann_index_t model_xyzn_index, double *x, double *q, int c2_obs, int sample_nn, int use_f);

  void get_point(double p[3], pcd_t *pcd, int idx);
  void get_normal(double p[3], pcd_t *pcd, int idx);
  void get_shape(double p[33], pcd_t *pcd, int idx);
  */



  //
  // DEPRECATED
  //

  /* heirarchical segmentation and balls model
  typedef struct {
    int num_segments;
    int *num_balls;
    int *segment_labels;
    int *ball_labels;
    double **segment_centers;
    double *segment_radii;
    double ***ball_centers;
    double **ball_radii;
    double mean_segment_radius;
    double mean_ball_radius;
  } pcd_balls_t;
  */

  /* oriented local feature model
  typedef struct {
    pcd_t *pcd;                   // point cloud
    bingham_mix_t *bmx;           // bingham mixture models (one per cluster)
    hll_t *hll;                   // hyperspherical local likelihood models (one per cluster)
    int num_clusters;             // # local feature clusters
    double *cluster_weights;      // cluster weights
    int shape_length;             // local shape descriptor length
    double **mean_shapes;         // cluster means
    double *shape_variances;      // cluster variances

    // params
    int rot_symm;
    int num_validators;
    double lambda;
    double pose_agg_x;
    double pose_agg_q;
    double *proposal_weights;
    int cluttered;
    int num_proposal_segments;
    int *proposal_segments;
  } olf_t;
  */

  //olf_t *load_olf(char *fname);                       // loads an olf from fname.pcd and fname.bmx
  //void olf_free(olf_t *olf);                          // frees the contents of an olf_t, but not the pointer itself
  //void olf_classify_points(pcd_t *pcd, olf_t *olf);   // classify pcd points (add channel "cluster") using olf shapes

  // computes the pdf of pose (x,q) given n points from pcd w.r.t. olf (assumes points are classified)
  //double olf_pose_pdf(double *x, double *q, olf_t *olf, pcd_t *pcd, int *indices, int n);

  // samples n weighted poses (X,Q,W) using olf model "olf" and point cloud "pcd"
  //olf_pose_samples_t *olf_pose_sample(olf_t *olf, pcd_t *pcd, int n);

  // aggregate the weighted pose samples, (X,Q,W)
  //olf_pose_samples_t *olf_aggregate_pose_samples(olf_pose_samples_t *poses, olf_t *olf);

  //olf_pose_samples_t *olf_pose_samples_new(int n);          // create a new olf_pose_samples_t
  //void olf_pose_samples_free(olf_pose_samples_t *poses);    // free pose samples



#ifdef __cplusplus
}
#endif 


#endif
