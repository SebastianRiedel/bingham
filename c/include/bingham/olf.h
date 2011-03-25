
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
    int num_clusters;             // # local feature clusters
    double *cluster_weights;      // cluster weights
    int shape_length;             // local shape descriptor length
    double **mean_shapes;         // cluster means
    double *shape_variances;      // cluster variances
  } olf_t;



  pcd_t *load_pcd(char *f_pcd);                        // loads a pcd
  void pcd_free(pcd_t *pcd);                           // frees the contents of a pcd_t, but not the pointer itself
  int pcd_channel(pcd_t *pcd, char *channel_name);     // gets the index of a channel by name

  olf_t *load_olf(char *fname);     // loads an olf from fname.pcd and fname.olf
  void olf_free(olf_t *olf);        // frees the contents of an olf_t, but not the pointer itself
  //double olf_pdf


#ifdef __cplusplus
}
#endif 


#endif
