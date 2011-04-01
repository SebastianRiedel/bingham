
#ifndef BINGHAM_OLF_H
#define BINGHAM_OLF_H


#ifdef __cplusplus
extern "C" {
#endif 



#include "bingham.h"


  typedef struct {
    int num_channels;
    int num_points;
    char **channels;  // e.g. {"x", "y", "z"}
    double **data;    // e.g. data[0] = {x1,x2,x3}, data[1] = {y1,y2,y3}, etc.

    // data pointers
    double *clusters;
    double **points;
    double **normals;
    double **principal_curvatures;
    double **shapes;

    // computed data
    double **quaternions[2];

  } pcd_t;


  typedef struct {
    pcd_t *pcd;                   // point cloud
    bingham_mix_t *bmx;           // bingham mixture models (one per cluster)
    hll_t *hll;                   // hyperspherical local likelihood models (one per cluster)
    int num_clusters;             // # local feature clusters
    double *cluster_weights;      // cluster weights
    int shape_length;             // local shape descriptor length
    double **mean_shapes;         // cluster means
    double *shape_variances;      // cluster variances
  } olf_t;



  pcd_t *load_pcd(char *f_pcd);                        // loads a pcd
  void pcd_free(pcd_t *pcd);                           // frees the contents of a pcd_t, but not the pointer itself
  int pcd_channel(pcd_t *pcd, char *channel_name);     // gets the index of a channel by name
  int pcd_add_channel(pcd_t *pcd, char *channel);      // adds a channel to pcd

  olf_t *load_olf(char *fname);                       // loads an olf from fname.pcd and fname.olf
  void olf_free(olf_t *olf);                          // frees the contents of an olf_t, but not the pointer itself
  void olf_classify_points(pcd_t *pcd, olf_t *olf);   // classify pcd points (add channel "cluster") using olf shapes

  // computes the pdf of pose (x,q) given n points from pcd w.r.t. olf (assumes points are classified)
  double olf_pose_pdf(double *x, double *q, olf_t *olf, pcd_t *pcd, int *indices, int n);

  // samples n weighted poses (X,Q,W) using olf model "olf" and point cloud "pcd"
  void olf_pose_sample(double **X, double **Q, double *W, olf_t *olf, pcd_t *pcd, int n);


#ifdef __cplusplus
}
#endif 


#endif
