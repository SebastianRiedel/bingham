
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "bingham/util.h"
#include "bingham/olf.h"



//--------------------------- dbug global data ---------------------------//

double **obs_edge_image_;
int obs_edge_image_width_;
int obs_edge_image_height_;
double **obs_edge_image_orig_;
double mps_xyz_dists_[100000];
double xyz_score_;
double mps_normal_dists_[100000];
double normal_score_;
double lab_scores_[3];
double vis_score_;
int **occ_edge_pixels_ = NULL;
int num_occ_edge_points_;
double **range_edge_points_ = NULL;
int **range_edge_pixels_ = NULL;
double *range_edge_vis_prob_ = NULL;
int num_range_edge_points_;
double edge_score_, edge_vis_score_, edge_occ_score_;
double segment_score_;
int mps_idx_[100000];
double mps_vis_prob_[100000];
double *x_true_, *q_true_;
int have_true_pose_ = 0;

//scope_params_t **params_ptr_;



int is_good_pose(double *x, double *q)
{
  double dx[3];
  sub(dx, x, x_true_, 3);
  double cos_dq = fabs(dot(q, q_true_, 4));
  cos_dq = MIN(cos_dq, 1.0);
  double dq = acos(cos_dq);
  return (norm(dx,3) < 0.05) && (dq < M_PI/8.0);
}

int is_great_pose(double *x, double *q)
{
  double dx[3];
  sub(dx, x, x_true_, 3);
  double cos_dq = fabs(dot(q, q_true_, 4));
  cos_dq = MIN(cos_dq, 1.0);
  double dq = acos(cos_dq);
  return (norm(dx,3) < 0.025) && (dq < M_PI/16.0);
}

int is_perfect_pose(double *x, double *q)
{
  double dx[3];
  sub(dx, x, x_true_, 3);
  double cos_dq = fabs(dot(q, q_true_, 4));
  cos_dq = MIN(cos_dq, 1.0);
  double dq = acos(cos_dq);
  return (norm(dx,3) < 0.015) && (dq < M_PI/24.0);
}

void get_good_poses(scope_samples_t *S, int good_pose_idx[], int great_pose_idx[], int perfect_pose_idx[],
		    int *num_good_poses, int *num_great_poses, int *num_perfect_poses)
{
  *num_good_poses = 0;
  *num_great_poses = 0; 
  *num_perfect_poses = 0;
  int i;
  for (i = 0; i < S->num_samples; ++i) {
    if (is_good_pose(S->samples[i].x, S->samples[i].q))
      good_pose_idx[(*num_good_poses)++] = i;
    if (is_great_pose(S->samples[i].x, S->samples[i].q))
      great_pose_idx[(*num_great_poses)++] = i;
    if (is_perfect_pose(S->samples[i].x, S->samples[i].q))
      perfect_pose_idx[(*num_perfect_poses)++] = i;
  }
}

void print_good_poses(scope_samples_t *S)
{
  int n = S->num_samples;
  int num_good_poses = 0, num_great_poses = 0, num_perfect_poses = 0;
  int good_pose_idx[n], great_pose_idx[n], perfect_pose_idx[n];
  get_good_poses(S, good_pose_idx, great_pose_idx, perfect_pose_idx, &num_good_poses, &num_great_poses, &num_perfect_poses);
  printf("Found %d/%d good poses, %d/%d great poses, %d/%d perfect poses\n", num_good_poses, n, num_great_poses, n, num_perfect_poses, n);
}

void print_good_poses_verbose(scope_samples_t *S, double w_sigma)
{
  int n = S->num_samples;
  int num_good_poses = 0, num_great_poses = 0, num_perfect_poses = 0;
  int good_pose_idx[n], great_pose_idx[n], perfect_pose_idx[n];
  get_good_poses(S, good_pose_idx, great_pose_idx, perfect_pose_idx, &num_good_poses, &num_great_poses, &num_perfect_poses);
  printf("Found %d/%d good poses, %d/%d great poses, %d/%d perfect poses\n", num_good_poses, n, num_great_poses, n, num_perfect_poses, n);

  int i;
  for (i = 0; i < n; i++) {
    if (ismemberi(i, perfect_pose_idx, num_perfect_poses))
      printf("*** W[%d] = %.2f +- %.2f\n", i, S->W[i], w_sigma);
    else if (ismemberi(i, great_pose_idx, num_great_poses))
      printf("** W[%d] = %.2f +- %.2f\n", i, S->W[i], w_sigma);
    else if (ismemberi(i, good_pose_idx, num_good_poses))
      printf("* W[%d] = %.2f +- %.2f\n", i, S->W[i], w_sigma);
    else
      printf("W[%d] = %.2f +- %.2f\n", i, S->W[i], w_sigma);
  }
}







//---------------------------- STATIC HELPER FUNCTIONS ---------------------------//

/*
 * reverse principal curvature
 */
void quaternion_flip(double *q2, double *q)
{
  if (q == q2) {
    double q1[4];
    q1[0] = -q[1];
    q1[1] = q[0];
    q1[2] = q[3];
    q1[3] = -q[2];
    memcpy(q2, q1, 4*sizeof(double));
  }
  else {
    q2[0] = -q[1];
    q2[1] = q[0];
    q2[2] = q[3];
    q2[3] = -q[2];
  }
}

/*
 * compute viewpoint (in model coordinates) for model placement (x,q) assuming observed viewpoint = (0,0,0)
 */
void model_pose_to_viewpoint(double *vp, double *x, double *q)
{
  double q_inv[4];
  quaternion_inverse(q_inv, q);
  double **R_inv = new_matrix2(3,3);  
  quaternion_to_rotation_matrix(R_inv,q_inv);
  matrix_vec_mult(vp, R_inv, x, 3, 3);
  mult(vp, vp, -1, 3);
  free_matrix2(R_inv);
}

/*
 * get the plane equation coefficients (c[0]*x + c[1]*y + c[2]*z + c[3] = 0) from (point,normal)
 */
void xyzn_to_plane(double *c, double *point, double *normal)
{
  c[0] = normal[0];
  c[1] = normal[1];
  c[2] = normal[2];
  c[3] = -dot(point, normal, 3);
}

void dilate_matrix(double **Y, double **X, int n, int m)
{
  int i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      if (X[i][j] > 0.0)
	Y[i][j] = X[i][j];
      else {
	int cnt = 0;
	double p = 0.0;
	if (i > 0 && X[i-1][j] > 0.0) {  p += X[i-1][j]; cnt++;  }
	if (i < n-1 && X[i+1][j] > 0.0) {  p += X[i+1][j]; cnt++;  }
	if (j > 0 && X[i][j-1] > 0.0) {  p += X[i][j-1]; cnt++;  }
	if (j < m-1 && X[i][j+1] > 0.0) {  p += X[i][j+1]; cnt++;  }
	if (cnt > 0)
	  Y[i][j] = p / (double)cnt;
      }
    }
  }
}

double **get_sub_matrix(double **X, int x0, int y0, int x1, int y1)
{
  int w = x1-x0+1;
  int h = y1-y0+1;
  double **Y = new_matrix2(w,h);

  int x;
  for (x = x0; x <= x1; x++)
    memcpy(Y[x-x0], &X[x][y0], h*sizeof(double));

  return Y;
}

/*
 * blur matrix with a 3x3 gaussian filter
 */
void blur_matrix(double **dst, double **src, int n, int m)
{
  double G[3] = {.6193, .0838, .0113};

  double **I = (dst==src ? new_matrix2(n,m) : dst);
  memcpy(dst[0], src[0], n*m*sizeof(double));

  int i,j;
  for (i = 1; i < n-1; i++)
    for (j = 1; j < m-1; j++)
      I[i][j] = G[0]*src[i][j] + G[1]*(src[i+1][j] + src[i-1][j] + src[i][j+1] + src[i][j-1]) + G[2]*(src[i+1][j+1] + src[i+1][j-1] + src[i-1][j+1] + src[i-1][j-1]);

  if (dst==src) {
    memcpy(dst[0], I[0], n*m*sizeof(double));
    free_matrix2(I);
  }
}

void matrix_cell_gradient(double *g, int i, int j, double **X, int n, int m)
{
  if (i == 0)
    g[0] = X[1][j] - X[0][j];
  else if (i == n-1)
    g[0] = X[n-1][j] - X[n-2][j];
  else
    g[0] = (X[i+1][j] - X[i-1][j]) / 2.0;

  if (j == 0)
    g[1] = X[i][1] - X[i][0];
  else if (j == m-1)
    g[1] = X[i][m-1] - X[i][m-2];
  else
    g[1] = (X[i][j+1] - X[i][j-1]) / 2.0;
}


/*
 * compute quaternions given normals and principal curvatures
 */
void compute_orientation_quaternions(double ***Q, double **N, double **PCS, int num_points)
{
  int i;
  double nx, ny, nz, pcx, pcy, pcz, pcx2, pcy2, pcz2;
  double **R = new_matrix2(3,3);

  for (i = 0; i < num_points; i++) {
    nx = N[i][0];
    ny = N[i][1];
    nz = N[i][2];
    pcx = PCS[i][0];
    pcy = PCS[i][1];
    pcz = PCS[i][2];
    pcx2 = ny*pcz - nz*pcy;
    pcy2 = nz*pcx - nx*pcz;
    pcz2 = nx*pcy - ny*pcx;

    // compute "up" quaternion
    R[0][0] = nx;  R[0][1] = pcx;  R[0][2] = pcx2;
    R[1][0] = ny;  R[1][1] = pcy;  R[1][2] = pcy2;
    R[2][0] = nz;  R[2][1] = pcz;  R[2][2] = pcz2;
    rotation_matrix_to_quaternion(Q[0][i], R);

    // compute "down" quaternion
    R[0][0] = nx;  R[0][1] = -pcx;  R[0][2] = -pcx2;
    R[1][0] = ny;  R[1][1] = -pcy;  R[1][2] = -pcy2;
    R[2][0] = nz;  R[2][1] = -pcz;  R[2][2] = -pcz2;
    rotation_matrix_to_quaternion(Q[1][i], R);
  }

  free_matrix2(R);
}


/*
 * add data pointers to a pcd
 */
static void pcd_add_data_pointers(pcd_t *pcd)
{
  int i, j, num_points = pcd->num_points;
  int ch_cluster = pcd_channel(pcd, "cluster");
  int ch_x = pcd_channel(pcd, "x");
  int ch_y = pcd_channel(pcd, "y");
  int ch_z = pcd_channel(pcd, "z");
  int ch_vx = pcd_channel(pcd, "vx");
  int ch_vy = pcd_channel(pcd, "vy");
  int ch_vz = pcd_channel(pcd, "vz");
  int ch_red = pcd_channel(pcd, "red");
  int ch_green = pcd_channel(pcd, "green");
  int ch_blue = pcd_channel(pcd, "blue");
  int ch_nx = pcd_channel(pcd, "nx");
  int ch_ny = pcd_channel(pcd, "ny");
  int ch_nz = pcd_channel(pcd, "nz");
  int ch_pcx = pcd_channel(pcd, "pcx");
  int ch_pcy = pcd_channel(pcd, "pcy");
  int ch_pcz = pcd_channel(pcd, "pcz");
  int ch_pc1 = pcd_channel(pcd, "pc1");
  int ch_pc2 = pcd_channel(pcd, "pc2");
  int ch_f1 = pcd_channel(pcd, "f1");
  int ch_f33 = pcd_channel(pcd, "f33");
  //int ch_balls = pcd_channel(pcd, "balls");
  int ch_sift1 = pcd_channel(pcd, "sift1");
  int ch_sift128 = pcd_channel(pcd, "sift128");
  //int ch_surfdist = pcd_channel(pcd, "surfdist");
  //int ch_surfwidth = pcd_channel(pcd, "surfwidth");
  int ch_range_edge = pcd_channel(pcd, "range_edge");
  int ch_curv_edge = pcd_channel(pcd, "curv_edge");
  int ch_img_edge = pcd_channel(pcd, "img_edge");
  int ch_ved1 = pcd_channel(pcd, "ved1");
  int ch_ved66 = pcd_channel(pcd, "ved66");

  if (ch_cluster>=0) {
    //pcd->clusters = pcd->data[ch_cluster];
    safe_malloc(pcd->clusters, num_points, int);
    for (i = 0; i < num_points; i++)
      pcd->clusters[i] = (int)(pcd->data[ch_cluster][i]);
  }
  if (ch_x>=0 && ch_y>=0 && ch_z>=0) {
    pcd->points = new_matrix2(num_points, 3);
    for (i = 0; i < num_points; i++) {
      pcd->points[i][0] = pcd->data[ch_x][i];
      pcd->points[i][1] = pcd->data[ch_y][i];
      pcd->points[i][2] = pcd->data[ch_z][i];
    }
  }
  if (ch_vx>=0 && ch_vy>=0 && ch_vz>=0) {
    pcd->views = new_matrix2(num_points, 3);
    for (i = 0; i < num_points; i++) {
      pcd->views[i][0] = pcd->data[ch_vx][i];
      pcd->views[i][1] = pcd->data[ch_vy][i];
      pcd->views[i][2] = pcd->data[ch_vz][i];
    }
  }
  if (ch_red>=0 && ch_green>=0 && ch_blue>=0) {
    pcd->colors = new_matrix2(num_points, 3);
    pcd->lab = new_matrix2(num_points, 3);
    for (i = 0; i < num_points; i++) {
      pcd->colors[i][0] = pcd->data[ch_red][i];
      pcd->colors[i][1] = pcd->data[ch_green][i];
      pcd->colors[i][2] = pcd->data[ch_blue][i];
      rgb2lab(pcd->lab[i], pcd->colors[i]);
    }
  }
  if (ch_nx>=0 && ch_ny>=0 && ch_nz>=0) {
    pcd->normals = new_matrix2(num_points, 3);
    for (i = 0; i < num_points; i++) {
      pcd->normals[i][0] = pcd->data[ch_nx][i];
      pcd->normals[i][1] = pcd->data[ch_ny][i];
      pcd->normals[i][2] = pcd->data[ch_nz][i];
    }
  }
  if (ch_pcx>=0 && ch_pcy>=0 && ch_pcz>=0) {
    pcd->principal_curvatures = new_matrix2(num_points, 3);
    for (i = 0; i < num_points; i++) {
      pcd->principal_curvatures[i][0] = pcd->data[ch_pcx][i];
      pcd->principal_curvatures[i][1] = pcd->data[ch_pcy][i];
      pcd->principal_curvatures[i][2] = pcd->data[ch_pcz][i];
    }
  }
  if (ch_f1>=0 && ch_f33>=0) {
    pcd->shape_length = 33;
    pcd->shapes = new_matrix2(num_points, pcd->shape_length);
    for (i = 0; i < num_points; i++)
      for (j = 0; j < pcd->shape_length; j++)
	pcd->shapes[i][j] = pcd->data[ch_f1 + j][i];
  }
  if (ch_sift1>=0 && ch_sift128>=0) {
    pcd->sift_length = 128;
    pcd->sift = new_matrix2(num_points, pcd->sift_length);
    for (i = 0; i < num_points; i++)
      for (j = 0; j < pcd->sift_length; j++)
	pcd->sift[i][j] = pcd->data[ch_sift1 + j][i];
  }
  if (ch_ved1>=0 && ch_ved66>=0) {
    pcd->ved_length = 66;
    pcd->ved = new_matrix2(num_points, pcd->ved_length);
    for (i = 0; i < num_points; i++)
      for (j = 0; j < pcd->ved_length; j++)
	pcd->ved[i][j] = pcd->data[ch_ved1 + j][i];
  }
  /*
  if (ch_surfdist>=0 && ch_surfwidth>=0) {
    pcd->sdw_length = 2;
    pcd->sdw = new_matrix2(num_points, pcd->sdw_length);
    for (i = 0; i < num_points; i++) {
      pcd->sdw[i][0] = pcd->data[ch_surfdist][i];
      pcd->sdw[i][1] = pcd->data[ch_surfwidth][i];
    }
  }
  */

  // data pointers
  if (ch_pc1>=0 && ch_pc2>=0) {
    pcd->pc1 = pcd->data[ch_pc1];
    pcd->pc2 = pcd->data[ch_pc2];
  }
  if (ch_range_edge>=0)
    pcd->range_edge = pcd->data[ch_range_edge];
  if (ch_curv_edge>=0)
    pcd->curv_edge = pcd->data[ch_curv_edge];
  if (ch_img_edge>=0)
    pcd->img_edge = pcd->data[ch_img_edge];


  // add quaternion orientation features
  if (ch_nx>=0 && ch_ny>=0 && ch_nz>=0 && ch_pcx>=0 && ch_pcy>=0 && ch_pcz>=0) {
    pcd->quaternions[0] = new_matrix2(pcd->num_points, 4);
    pcd->quaternions[1] = new_matrix2(pcd->num_points, 4);
    compute_orientation_quaternions(pcd->quaternions, pcd->normals, pcd->principal_curvatures, pcd->num_points);
  }

  // add points kdtree
  //double **X = new_matrix2(pcd->num_points, 3);
  //transpose(X, pcd->points, 3, pcd->num_points);
  //pcd->points_kdtree = kdtree(X, pcd->num_points, 3);
  //free_matrix2(X);

  // add balls model
  //if (ch_balls>=0)
  //  pcd_add_balls(pcd);
}


/*
 * free data pointers in a pcd
 */
static void pcd_free_data_pointers(pcd_t *pcd)
{
  if (pcd->points)
    free_matrix2(pcd->points);
  if (pcd->views)
    free_matrix2(pcd->views);
  if (pcd->colors)
    free_matrix2(pcd->colors);
  if (pcd->lab)
    free_matrix2(pcd->lab);
  if (pcd->normals)
    free_matrix2(pcd->normals);
  if (pcd->principal_curvatures)
    free_matrix2(pcd->principal_curvatures);
  if (pcd->shapes)
    free_matrix2(pcd->shapes);
  if (pcd->sift)
    free_matrix2(pcd->sift);
  if (pcd->ved)
    free_matrix2(pcd->ved);
  //if (pcd->sdw)
  //  free_matrix2(pcd->sdw);
  if (pcd->clusters)
    free(pcd->clusters);

  if (pcd->quaternions[0])
    free_matrix2(pcd->quaternions[0]);
  if (pcd->quaternions[1])
    free_matrix2(pcd->quaternions[1]);

  //if (pcd->points_kdtree)
  //  kdtree_free(pcd->points_kdtree);
  //if (pcd->balls)
  //  pcd_free_balls(pcd);
}





//---------------------------- EXTERNAL API ---------------------------//



/*
 * loads a pcd
 */
pcd_t *load_pcd(char *f_pcd)
{
  int i, j;
  FILE *f = fopen(f_pcd, "r");

  if (f == NULL) {
    fprintf(stderr, "Invalid filename: %s\n", f_pcd);
    return NULL;
  }

  pcd_t *pcd;
  safe_calloc(pcd, 1, pcd_t);

  char sbuf[10000], *s = sbuf;
  while (!feof(f)) {
    s = sbuf;
    if (fgets(s, 10000, f)) {

      if (!wordcmp(s, "COLUMNS", " \t\n") || !wordcmp(s, "FIELDS", " \t\n")) {
	s = sword(s, " \t", 1);
	pcd->channels = split(s, " \t\n", &pcd->num_channels);


	/* TODO: make a file converter?
	   replace_word(pcd->channels, pcd->num_channels, "normal_x", "nx");
	   replace_word(pcd->channels, pcd->num_channels, "normal_y", "ny");
	   replace_word(pcd->channels, pcd->num_channels, "normal_z", "nz");
	   replace_word(pcd->channels, pcd->num_channels, "principal_curvature_x", "pcx");
	   replace_word(pcd->channels, pcd->num_channels, "principal_curvature_y", "pcy");
	   replace_word(pcd->channels, pcd->num_channels, "principal_curvature_z", "pcz");
	   //s = strrep(s, 'fpfh', ['f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 ' ...
	   //                       'f15 f16 f17 f18 f19 f20 f21 f22 f23 f24 f25 f26 f27 f28 f29 f30 f31 f32 f33']);
	   */

      }
      else if (!wordcmp(s, "POINTS", " \t\n")) {
	s = sword(s, " \t", 1);
	sscanf(s, "%d", &pcd->num_points);
      }
      else if (!wordcmp(s, "DATA", " \t\n")) {
	s = sword(s, " \t", 1);
	if (wordcmp(s, "ascii", " \t\n")) {  // NOTE(sanja): should this be !wordcmp?
	  fprintf(stderr, "Error: only ascii pcd files are supported.\n");
	  pcd_free(pcd);
	  free(pcd);
	  return NULL;
	}

	safe_calloc(pcd->data, pcd->num_channels, double *);
	for (i = 0; i < pcd->num_channels; i++)
	  safe_calloc(pcd->data[i], pcd->num_points, double);

	for (i = 0; i < pcd->num_points; i++) {
	  s = sbuf;
	  if (fgets(s, 10000, f) == NULL)
	    break;
	  for (j = 0; j < pcd->num_channels; j++) {
	    if (sscanf(s, "%lf", &pcd->data[j][i]) < 1)
	      break;
	    s = sword(s, " \t", 1);
	  }
	  if (j < pcd->num_channels)
	    break;
	}

	if (i < pcd->num_points) {
	  fprintf(stderr, "Error: corrupt pcd data at row %d\n", i);
	  pcd_free(pcd);
	  free(pcd);
	  return NULL;
	}
      }
    }
  }

  pcd_add_data_pointers(pcd);

  return pcd;
}


/*
 * frees the contents of a pcd_t, but not the pointer itself
 */
void pcd_free(pcd_t *pcd)
{
  int i;

  if (pcd == NULL)
    return;

  if (pcd->channels) {
    for (i = 0; i < pcd->num_channels; i++)
      if (pcd->channels[i])
	free(pcd->channels[i]);
    free(pcd->channels);
  }

  if (pcd->data) {
    for (i = 0; i < pcd->num_channels; i++)
      free(pcd->data[i]);
    free(pcd->data);
  }

  pcd_free_data_pointers(pcd);
}


/*
 * gets the index of a channel by name
 */
int pcd_channel(pcd_t *pcd, char *channel_name)
{
  int i;
  for (i = 0; i < pcd->num_channels; i++)
    if (!strcmp(pcd->channels[i], channel_name))
      return i;

  return -1;
}


/*
 * adds a channel to pcd
 */
int pcd_add_channel(pcd_t *pcd, char *channel)
{
  int ch = pcd_channel(pcd, channel);
  if (ch >= 0) {
    printf("Warning: channel %s already exists\n", channel);
    return ch;
  }

  // add channel name
  ch = pcd->num_channels;
  pcd->num_channels++;
  safe_realloc(pcd->channels, pcd->num_channels, char *);
  safe_calloc(pcd->channels[ch], strlen(channel) + 1, char);
  strcpy(pcd->channels[ch], channel);
  
  // add space for data
  safe_realloc(pcd->data, pcd->num_channels, double *);
  safe_calloc(pcd->data[ch], pcd->num_points, double);

  return ch;
}


multiview_pcd_t *get_multiview_pcd(pcd_t *pcd)
{
  multiview_pcd_t *mvp;
  safe_calloc(mvp, 1, multiview_pcd_t);
  mvp->pcd = pcd;

  const double epsilon = .0000001;

  // get the number of views
  mvp->num_views = 1;
  int i;
  for (i = 1; i < pcd->num_points; i++)
    if (fabs(pcd->views[i][0] - pcd->views[i-1][0]) > epsilon || fabs(pcd->views[i][1] - pcd->views[i-1][1]) > epsilon || fabs(pcd->views[i][2] - pcd->views[i-1][2]) > epsilon)
      mvp->num_views++;

  // allocate stuff
  mvp->views = new_matrix2(mvp->num_views, 3);
  safe_malloc(mvp->view_idx, mvp->num_views, int);
  safe_malloc(mvp->view_cnt, mvp->num_views, int);

  // fill in data
  memcpy(mvp->views[0], pcd->views[0], 3*sizeof(double));
  mvp->view_idx[0] = 0;
  int cnt = 1;
  for (i = 1; i < pcd->num_points; i++) {
    if (fabs(pcd->views[i][0] - pcd->views[i-1][0]) > epsilon || fabs(pcd->views[i][1] - pcd->views[i-1][1]) > epsilon || fabs(pcd->views[i][2] - pcd->views[i-1][2]) > epsilon) {
      memcpy(mvp->views[cnt], pcd->views[i], 3*sizeof(double));
      mvp->view_idx[cnt++] = i;
    }
  }
  for (i = 0; i < mvp->num_views - 1; i++)
    mvp->view_cnt[i] = mvp->view_idx[i+1] - mvp->view_idx[i];
  mvp->view_cnt[i] = pcd->num_points - mvp->view_idx[i];

  return mvp;
}

void free_multiview_pcd(multiview_pcd_t *mvp)
{
  free_matrix2(mvp->views);
  free(mvp->view_idx);
  free(mvp->view_cnt);
  free(mvp);
}





 //==============================================================================================//

 //---------------------------------------  Range Image  ----------------------------------------//

 //==============================================================================================//


range_image_t *pcd_to_range_image_from_template(pcd_t *pcd, range_image_t *R0)
{
  double *vp = R0->vp;
  double **Q;
  Q = new_matrix2(3,3);
  quaternion_to_rotation_matrix(Q, &vp[3]);

  int i, j, n = pcd->num_points;
  double X[n], Y[n], D[n];
  double **P = new_matrix2(n,3);  // points
  double **N = new_matrix2(n,3);  // normals
  for (i = 0; i < n; i++) {
    // get point and normal (w.r.t. viewpoint)
    double *p = P[i];
    sub(p, pcd->points[i], vp, 3);
    matrix_vec_mult(p, Q, p, 3, 3);
    matrix_vec_mult(N[i], Q, pcd->normals[i], 3, 3);
    // compute range image coordinates
    D[i] = norm(p,3);
    X[i] = atan2(p[0], p[2]);
    Y[i] = acos(p[1]/D[i]);
  }

  range_image_t *R;
  safe_calloc(R, 1, range_image_t);
  *R = *R0;  // shallow copy

  int w = R->w, h = R->h;
  R->image = new_matrix2(w,h);
  R->idx = new_matrix2i(w,h);
  R->points = new_matrix3(w,h,3);
  R->normals = new_matrix3(w,h,3);
  R->cnt = new_matrix2i(w,h);

  for (i = 0; i < w; i++) {
    for (j = 0; j < h; j++) {
      R->image[i][j] = -1.0;
      R->idx[i][j] = -1;
    }
  }

  // fill in range image
  for (i = 0; i < n; i++) {
    int cx = (int)floor( (X[i] - R->min[0]) / R->res);
    int cy = (int)floor( (Y[i] - R->min[1]) / R->res);
    if (cx < 0 || cy < 0 || cx >= w || cy >= h)
      continue;
    double d = D[i];
    double r = R->image[cx][cy];
    if (r < 0 || r > d) {
      R->image[cx][cy] = d;
      R->idx[cx][cy] = i;
    }
    double *avg_point = R->points[cx][cy];
    double *avg_normal = R->normals[cx][cy];
    int cnt = R->cnt[cx][cy];
    for (j = 0; j < 3; j++) {
      avg_point[j] = (cnt*avg_point[j] + P[i][j]) / (double)(cnt+1);
      avg_normal[j] = (cnt*avg_normal[j] + N[i][j]) / (double)(cnt+1);
    }
    normalize(avg_normal, avg_normal, 3);
    R->cnt[cx][cy]++;
  }

  free_matrix2(Q);
  free_matrix2(P);
  free_matrix2(N);

  return R;
}


range_image_t *pcd_to_range_image(pcd_t *pcd, double *vp, double res, int padding)
{
  // create range image
  range_image_t *R;
  safe_calloc(R, 1, range_image_t);
  if (vp != NULL)
    memcpy(R->vp, vp, 7*sizeof(double));
  else {
    R->vp[3] = 1.0;  // identity transform: (0,0,0,1,0,0,0)
    vp = R->vp;
  }
  R->res = res;

  int i, j, n = pcd->num_points;
  double **Q;
  Q = new_matrix2(3,3);
  quaternion_to_rotation_matrix(Q, &vp[3]);

  double X[n], Y[n], D[n];
  double **P = new_matrix2(n,3);  // points
  double **N = new_matrix2(n,3);  // normals
  for (i = 0; i < n; i++) {
    // get point and normal (w.r.t. viewpoint)
    double *p = P[i];
    sub(p, pcd->points[i], vp, 3);
    matrix_vec_mult(p, Q, p, 3, 3);
    matrix_vec_mult(N[i], Q, pcd->normals[i], 3, 3);
    // compute range image coordinates
    D[i] = norm(p,3);
    X[i] = atan2(p[0], p[2]);
    Y[i] = acos(p[1]/D[i]);
  }

  R->min[0] = min(X,n) - res/2.0 - res*padding;
  R->min[1] = min(Y,n) - res/2.0 - res*padding;
  int w0 = (int)ceil(2*M_PI/res);
  double **image = new_matrix2(w0,w0);
  int **idx = new_matrix2i(w0,w0);

  for (i = 0; i < w0; i++) {
    for (j = 0; j < w0; j++) {
      image[i][j] = -1.0;
      idx[i][j] = -1;
    }
  }

  // fill in range image
  for (i = 0; i < n; i++) {
    int cx = (int)floor( (X[i] - R->min[0]) / res);
    int cy = (int)floor( (Y[i] - R->min[1]) / res);
    double d = D[i];

    double r = image[cx][cy];
    if (r < 0 || r > d) {
      image[cx][cy] = d;
      idx[cx][cy] = i;
      R->w = MAX(R->w, cx+1);
      R->h = MAX(R->h, cy+1);
    }
  }
  R->w = MIN(R->w + padding, w0);
  R->h = MIN(R->h + padding, w0);

  // crop empty range pixels
  R->image = new_matrix2(R->w, R->h);
  R->idx = new_matrix2i(R->w, R->h);
  for (i = 0; i < R->w; i++) {
    for (j = 0; j < R->h; j++) {
      R->image[i][j] = image[i][j];
      R->idx[i][j] = idx[i][j];
    }
  }

  // add average cell points and normals
  int w = R->w;
  int h = R->h;
  R->points = new_matrix3(w,h,3);
  R->normals = new_matrix3(w,h,3);
  R->cnt = new_matrix2i(w,h);
  for (i = 0; i < n; i++) {
    int cx = (int)floor( (X[i] - R->min[0]) / R->res);
    int cy = (int)floor( (Y[i] - R->min[1]) / R->res);
    if (cx < 0 || cy < 0 || cx >= w || cy >= h)
      continue;
    double *avg_point = R->points[cx][cy];
    double *avg_normal = R->normals[cx][cy];
    int cnt = R->cnt[cx][cy];
    for (j = 0; j < 3; j++) {
      avg_point[j] = (cnt*avg_point[j] + P[i][j]) / (double)(cnt+1);
      avg_normal[j] = (cnt*avg_normal[j] + N[i][j]) / (double)(cnt+1);
    }
    normalize(avg_normal, avg_normal, 3);
    R->cnt[cx][cy]++;
  }

  // cleanup
  free_matrix2(image);
  free_matrix2i(idx);
  free_matrix2(Q);
  free_matrix2(P);
  free_matrix2(N);

  return R;
}

void free_range_image(range_image_t *range_image)
{
  free_matrix2(range_image->image);
  free_matrix2i(range_image->idx);
  free_matrix3(range_image->points);
  free_matrix3(range_image->normals);
  free_matrix2i(range_image->cnt);
  free(range_image);
}

int range_image_xyz2sub(int *i, int *j, range_image_t *range_image, double *xyz)
{
  //TODO: use range image viewpoint

  double d = norm(xyz, 3);
  double x = atan2(xyz[0], xyz[2]);
  double y = acos(xyz[1] / d);

  int cx = (int)floor((x - range_image->min[0]) / range_image->res);
  int cy = (int)floor((y - range_image->min[1]) / range_image->res);

  *i = cx;
  *j = cy;

  return cx>=0 && cy>=0 && (cx < range_image->w) && (cy < range_image->h);
}


void range_image_find_nn(int *nn_idx, double *nn_d2, double **query_xyz, double **query, int query_num, int query_len,
			 double **data, range_image_t *range_image, int search_radius_pixels)
{
  int i;
  for (i = 0; i < query_num; i++) {
    int imin = 0;
    double d2min = dist2(query[i], data[0], query_len);
    int x, y;
    int inbounds = range_image_xyz2sub(&x, &y, range_image, query_xyz[i]);
    if (inbounds) {
      int r = search_radius_pixels;
      int x0 = MAX(0, x-r);
      int x1 = MIN(x+r, range_image->w - 1);
      int y0 = MAX(0, y-r);
      int y1 = MIN(y+r, range_image->h - 1);
      for (x = x0; x <= x1; x++) {
	for (y = y0; y <= y1; y++) {
	  int idx = range_image->idx[x][y];
	  if (idx < 0)
	    continue;
	  double d2 = dist2(query[i], data[idx], query_len);
	  if (imin < 0 || d2 < d2min) {
	    imin = idx;
	    d2min = d2;
	  }
	}
      }
    }
    nn_idx[i] = imin;
    nn_d2[i] = d2min;
  }
}



double **get_edge_feature_image(pcd_t *pcd, range_image_t *range_image, scope_params_t *params)
{
  double min_edge_prob = .05;  // TODO: make this a param

  double range_edge_weight = params->range_edge_weight;
  double curv_edge_weight = params->curv_edge_weight;
  double img_edge_weight = params->img_edge_weight;
  int edge_blur = params->edge_blur;

  int w = range_image->w;
  int h = range_image->h;

  double **RE = new_matrix2(w,h);  // range_edge image
  double **CE = new_matrix2(w,h);  // curv_edge image
  double **IE = new_matrix2(w,h);  // img_edge image

  int i, x, y;
  for (i = 0; i < pcd->num_points; i++) {
    int inbounds = range_image_xyz2sub(&x, &y, range_image, pcd->points[i]);
    if (inbounds) {
      if (pcd->range_edge[i] > 0)
	RE[x][y] = MAX(RE[x][y], pcd->range_edge[i]);
      if (pcd->curv_edge[i] > 0)
	CE[x][y] = MAX(CE[x][y], pcd->curv_edge[i]);
      if (pcd->img_edge[i] > 0)
	IE[x][y] = MAX(IE[x][y], pcd->img_edge[i]);
    }
  }

  mult(RE[0], RE[0], range_edge_weight, w*h);
  mult(CE[0], CE[0], curv_edge_weight, w*h);
  mult(IE[0], IE[0], img_edge_weight, w*h);

  double **I = RE;  // save an alloc
  matrix_add(I, RE, CE, w, h);
  matrix_add(I, I, IE, w, h);
  
  //dbug
  obs_edge_image_orig_ = matrix_clone(I, w, h);

  // blur the edge image
  for (i = 0; i < edge_blur; i++)
    blur_matrix(I, I, w, h);

  // cap edge image pixels between min_edge_prob and 1, then take a log
  for (i = 0; i < w*h; i++) {
    double p = MIN(I[0][i], 1.0);
    I[0][i] = log(MAX(p, min_edge_prob));
  }

  //free_matrix2(RE);
  free_matrix2(CE);
  free_matrix2(IE);

  return I;
}


/*
int **compute_range_image_edges(range_image_t *range_image, double dthresh)
{
  int x,y;
  int w = range_image->w;
  int h = range_image->h;

  int **E = new_matrix2i(w,h);
  double **RI = range_image->image;

  for (x = 1; x < w-1; x++) {
    for (y = 1; y < h-1; y++) {
      
      double r = RI[x][y];

      //if (r 
    }
  }

  return E;
}
*/






 //==============================================================================================//

 //-----------------------------------  PCD Helper Functions  -----------------------------------//

 //==============================================================================================//


inline void get_point(double p[3], pcd_t *pcd, int idx)
{
  memcpy(p, pcd->points[idx], 3*sizeof(double));
}

inline void get_normal(double n[3], pcd_t *pcd, int idx)
{
  memcpy(n, pcd->normals[idx], 3*sizeof(double));
}

void orthogonal_vector(double *w, double *v, int n)
{
  int i;
  double w_proj[n];
  for (i = 0; i < n; i++)
    w[i] = normrand(0,1);
  proj(w_proj, w, v, n);
  sub(w, w, w_proj, n);
  normalize(w, w, n);
}

void vector_to_possible_quaternion(double *q, double *v)
{
  if (find_first_non_zero(v, 3) == -1) {
    q[0] = 1;
    q[1] = q[2] = q[3] = 0;
    return;
  }

  double **r = new_matrix2(3, 3);
  normalize(r[0], v, 3);
  orthogonal_vector(r[1], r[0], 3);
  cross(r[2], r[0], r[1]);
  transpose(r, r, 3, 3);
  
  rotation_matrix_to_quaternion(q, r);
  free_matrix2(r);
}

void get_validation_points(int *idx, pcd_t *pcd_model, int num_validation_points)
{
  int i;
  if (num_validation_points == pcd_model->num_points)  // use all the points
    for (i = 0; i < pcd_model->num_points; i++)
      idx[i] = i;
  else
    randperm(idx, pcd_model->num_points, num_validation_points);
}

double compute_visibility_prob(double *point, double *normal, range_image_t *obs_range_image, double vis_thresh, int search_radius)
{
  double V[3];
  normalize(V, point, 3);

  if (normal != NULL && dot(V, normal, 3) >= -.1)  // normals pointing away
    return 0.0;

  int x, y;
  int inbounds = range_image_xyz2sub(&x, &y, obs_range_image, point);
  if (!inbounds)
    return 0.0;

  double model_range = norm(point, 3);
  double obs_range = obs_range_image->image[x][y];

  if (search_radius > 0) {
    int x0 = MAX(x - search_radius, 0);
    int x1 = MIN(x + search_radius, obs_range_image->w - 1);
    int y0 = MAX(y - search_radius, 0);
    int y1 = MIN(y + search_radius, obs_range_image->h - 1);
    for (x = x0; x <= x1; x++)
      for (y = y0; y <= y1; y++)
	obs_range = MAX(obs_range, obs_range_image->image[x][y]);
  }

  double dR = model_range - obs_range;
  return (dR < 0 ? 1.0 : normpdf(dR/vis_thresh, 0, 1) / .3989);  // .3989 = normpdf(0,0,1)
}

void transform_cloud(double **cloud2, double **cloud, int n, double *x, double *q)
{
  double **R = new_matrix2(3,3);
  quaternion_to_rotation_matrix(R,q);
  int i;
  for (i = 0; i < n; i++) {
    matrix_vec_mult(cloud2[i], R, cloud[i], 3, 3);
    if (x != NULL)
      add(cloud2[i], cloud2[i], x, 3);
  }
  free_matrix2(R);
}

double **get_sub_cloud_at_pose(pcd_t *pcd, int *idx, int n, double *x, double *q)
{
  double **cloud = new_matrix2(n,3);
  double **R = new_matrix2(3,3);
  quaternion_to_rotation_matrix(R,q);
  int i;
  for (i = 0; i < n; i++) {
    memcpy(cloud[i], pcd->points[idx[i]], 3*sizeof(double));
    matrix_vec_mult(cloud[i], R, cloud[i], 3, 3);
    add(cloud[i], cloud[i], x, 3);
  }
  free_matrix2(R);

  return cloud;
}

double **get_sub_cloud_normals_rotated(pcd_t *pcd, int *idx, int n, double *q)
{
  double **normals = new_matrix2(n,3);
  double **R = new_matrix2(3,3);
  quaternion_to_rotation_matrix(R,q);
  int i;
  for (i = 0; i < n; i++) {
    memcpy(normals[i], pcd->normals[idx[i]], 3*sizeof(double));
    matrix_vec_mult(normals[i], R, normals[i], 3, 3);
  }
  free_matrix2(R);

  return normals;
}

double **get_sub_cloud_fpfh(pcd_t *pcd, int *idx, int n)
{
  double **F = new_matrix2(n, pcd->shape_length);
  reorder_rows(F, pcd->shapes, idx, n, pcd->shape_length);
  return F;
}

/*
double **get_sub_cloud_sdw(pcd_t *pcd, int *idx, int n, scope_params_t *params)
{
  double surfdist_thresh = params->surfdist_thresh;
  double surfwidth_thresh = params->surfwidth_thresh;

  double **sdw = new_matrix2(n, 2);
  reorder_rows(sdw, pcd->sdw, idx, n, 2);

  int i;
  for (i = 0; i < n; i++) {
    sdw[i][0] = MIN(sdw[i][0], surfdist_thresh);
    sdw[i][1] = MIN(sdw[i][1], surfwidth_thresh);
  }

  return sdw;
}
*/

double **get_sub_cloud_lab(pcd_t *pcd, int *idx, int n)
{
  double **lab = new_matrix2(n,3);
  reorder_rows(lab, pcd->lab, idx, n, 3);
  return lab;
}

double **get_xyzn_features(double **points, double **normals, int n, scope_params_t *params)
{
  int i, j;
  double xyz_weight = params->xyz_weight;
  double normal_weight = params->normal_weight;
  double **xyzn = new_matrix2(n,6);
  for (i = 0; i <n; i++) {
    for (j = 0; j < 3; j++) {
      xyzn[i][j] = xyz_weight * points[i][j];
      xyzn[i][j+3] = normal_weight * normals[i][j];
    }
  }

  return xyzn;
}

double **get_fxyzn_features(double **fpfh, double **points, double **normals, int n, int fpfh_length, scope_params_t *params)
{
  int i, j;
  double xyz_weight = params->xyz_weight;
  double normal_weight = params->normal_weight;
  double **fxyzn = new_matrix2(n, fpfh_length + 6);
  for (i = 0; i <n; i++) {
    memcpy(fxyzn[i], fpfh[i], fpfh_length*sizeof(double));
    for (j = 0; j < 3; j++) {
      fxyzn[i][j+fpfh_length] = xyz_weight * points[i][j];
      fxyzn[i][j+3+fpfh_length] = normal_weight * normals[i][j];
    }
  }

  return fxyzn;
}

/*
float **get_fsurf_features(double **fpfh, double **sdw, int n, int fpfh_length, scope_params_t *params)
{
  int i, j;
  double surfdist_thresh = params->surfdist_thresh;
  double surfwidth_thresh = params->surfwidth_thresh;
  double surfdist_weight = params->surfdist_weight;
  double surfwidth_weight = params->surfwidth_weight;

  float **fsurf = new_matrix2f(n, fpfh_length + 2);
  for (i = 0; i < n; i++) {
    for (j = 0; j < fpfh_length; j++)
      fsurf[i][j] = fpfh[i][j];
    double surfdist, surfwidth;
    if (sdw) {
      surfdist = MIN(sdw[i][0], surfdist_thresh);
      surfwidth = MIN(sdw[i][1], surfwidth_thresh);
    } else {
      surfdist = surfdist_thresh;
      surfwidth = surfwidth_thresh;
    }
    fsurf[i][fpfh_length] = surfdist_weight * surfdist;
    fsurf[i][fpfh_length+1] = surfwidth_weight * surfwidth;
  }

  return fsurf;
}
*/


/*
 * returns the number of points
 */
double **get_range_edge_points(int *n_ptr, double *x, double *q, multiview_pcd_t *range_edges_model, range_image_t *obs_range_image)
{
  // compute viewpoint for model placement (x,q) assuming observed viewpoint = (0,0,0)
  double vp[3];
  model_pose_to_viewpoint(vp, x, q);

  // get model range edges for nearest stored viewpoint
  double cos_vp_angles[range_edges_model->num_views];
  matrix_vec_mult(cos_vp_angles, range_edges_model->views, vp, range_edges_model->num_views, 3);
  int i = find_max(cos_vp_angles, range_edges_model->num_views);
  int vp_idx = range_edges_model->view_idx[i];
  int num_edge_points = range_edges_model->view_cnt[i];

  //printf("vp = [%f, %f, %f], closest stored vp = [%f, %f, %f]\n", vp[0], vp[1], vp[2],
  //	 range_edges_model->views[i][0], range_edges_model->views[i][1], range_edges_model->views[i][2]);  //dbug

  // sample edge points to validate
  int idx[num_edge_points];
  int n = *n_ptr;
  if (n >= num_edge_points || n == 0) {
    n = num_edge_points;
    for (i = 0; i < n; i++)
      idx[i] = i;
  }
  else
    randperm(idx, num_edge_points, n);

  // make idx be pcd point indices
  for (i = 0; i < n; i++)
    idx[i] += vp_idx;

  //printf("n = %d, idx[0] = %d, idx[n-1] = %d\n", n, idx[0], idx[n-1]); //dbug

  // get the actual points in the correct pose
  double **P = new_matrix2(n,3);
  reorder_rows(P, range_edges_model->pcd->points, idx, n, 3);

  *n_ptr = n;
  return P;
}


void edge_image_random_walk(int *xi_ptr, int *yi_ptr, double **obs_edge_image, int w, int h, int num_steps, int hops)
{
  int xi = *xi_ptr;
  int yi = *yi_ptr;

  int i;
  for (i = 0; i < num_steps; i++) {
    // shoot a ray from (xi,yi) in a random direction
    double theta = 2*M_PI*frand();
    double rx = cos(theta);
    double ry = sin(theta);

    // at each cell along the ray, stop with prob. = exp(obs_edge_image[xi][yi])
    int xi2 = xi;
    int yi2 = yi;
    double xf = xi + .5;
    double yf = yi + .5;
    int hi = 0;
    while (1) {
      if (hi++ > hops) {
	double p = exp(obs_edge_image[xi2][yi2]);
	if (p > .2 && frand() < p)
	  break;
      }
      xf += rx;
      yf += ry;
      if (xf < 0 || yf < 0 || xf >= w || yf >= h)
	break;
      xi2 = floor(xf);
      yi2 = floor(yf);
    }

    // pick a random observed point along the ray
    double r = frand();
    xi = floor(r*xi + (1-r)*xi2);
    yi = floor(r*yi + (1-r)*yi2);
  }

  *xi_ptr = xi;
  *yi_ptr = yi;
}


double *compute_model_saliency(pcd_t *pcd_model)
{
  double epsilon = 1e-50;

  int i;
  double *model_pmf;
  int n = pcd_model->num_points;
  safe_calloc(model_pmf, n, double);
  for (i = 0; i < pcd_model->num_points; i++) {
    double pc1 = MAX(pcd_model->pc1[i], epsilon);
    double pc2 = MAX(pcd_model->pc2[i], epsilon);
    model_pmf[i] = pc1/pc2 - 1.0;
    model_pmf[i] = MIN(model_pmf[i], 10);
  }
  mult(model_pmf, model_pmf, 1.0/sum(model_pmf, n), n);

  return model_pmf;
}


int find_obs_sift_matches(int *obs_idx, int *model_idx, pcd_t *sift_obs, pcd_t *sift_model, double sift_dthresh)
{
  const int sift_len = 128;

  int n1 = sift_obs->num_points;
  int n2 = sift_model->num_points;

  double **desc1 = sift_obs->sift;  //new_matrix2(n1, sift_len);
  double **desc2 = sift_model->sift;  //new_matrix2(n2, sift_len);
  //transpose(desc1, sift_obs->sift, sift_len, n1);
  //transpose(desc2, sift_model->sift, sift_len, n2);

  int num_matches = 0;
  int i,j;
  for (i = 0; i < n1; i++) {
    double dmin = 100000.0;
    int jmin = 0;
    for (j = 0; j < n2; j++) {
      double d = acos(dot(desc1[i], desc2[j], sift_len));
      if (d < dmin) {
	dmin = d;
	jmin = j;
      }
    }
    if (dmin < sift_dthresh) {
      obs_idx[num_matches] = i;
      model_idx[num_matches] = jmin;
      num_matches++;
    }
  }

  //free_matrix2(desc1);
  //free_matrix2(desc2);

  return num_matches;
}



//==============================================================================================//

//------------------------------------  SCOPE Data Processing ----------------------------------//

//==============================================================================================//


scope_model_data_t *get_scope_model_data(olf_model_t *model, scope_params_t *params)
{
  scope_model_data_t *data;
  safe_calloc(data, 1, scope_model_data_t);

  // unpack model arguments
  data->pcd_model = model->obj_pcd;
  data->fpfh_model = model->fpfh_pcd;
  data->sift_model = model->sift_pcd;
  data->range_edges_model = get_multiview_pcd(model->range_edges_pcd);

  // compute model feature saliency (for fpfh correspondences)
  data->fpfh_model_pmf = compute_model_saliency(data->fpfh_model);
  safe_calloc(data->fpfh_model_cmf, data->fpfh_model->num_points, double);
  cumsum(data->fpfh_model_cmf, data->fpfh_model_pmf, data->fpfh_model->num_points);

  // get combined feature matrices
  data->model_xyzn = get_xyzn_features(data->pcd_model->points, data->pcd_model->normals, data->pcd_model->num_points, params);
  data->fpfh_model_fxyzn = get_fxyzn_features(data->fpfh_model->shapes, data->fpfh_model->points, data->fpfh_model->normals, data->fpfh_model->num_points, data->fpfh_model->shape_length, params);
  data->fpfh_model_xyzn = get_xyzn_features(data->fpfh_model->points, data->fpfh_model->normals, data->fpfh_model->num_points, params);

  // flann params
  struct FLANNParameters flann_params_single = DEFAULT_FLANN_PARAMETERS;
  flann_params_single.algorithm = FLANN_INDEX_KDTREE_SINGLE;
  struct FLANNParameters flann_params = DEFAULT_FLANN_PARAMETERS;
  flann_params.algorithm = FLANN_INDEX_KDTREE;
  //flann_params.trees = 8;
  //flann_params.checks = 64;
  data->model_xyz_params = flann_params_single;
  data->fpfh_model_f_params = flann_params;
  data->fpfh_model_xyzn_params = flann_params_single;

  // build flann indices
  float speedup;
  data->model_xyz_index = flann_build_index_double(data->pcd_model->points[0], data->pcd_model->num_points, 3, &speedup, &data->model_xyz_params);
  data->fpfh_model_f_index = flann_build_index_double(data->fpfh_model->shapes[0], data->fpfh_model->num_points, data->fpfh_model->shape_length, &speedup, &data->fpfh_model_f_params);
  data->fpfh_model_xyzn_index = flann_build_index_double(data->fpfh_model_xyzn[0], data->fpfh_model->num_points, 6, &speedup, &data->fpfh_model_xyzn_params);

  return data;
}

scope_obs_data_t *get_scope_obs_data(olf_obs_t *obs, scope_params_t *params)
{
  scope_obs_data_t *data;
  safe_calloc(data, 1, scope_obs_data_t);

  // unpack obs arguments
  data->pcd_obs = obs->fg_pcd;
  data->sift_obs = obs->sift_pcd;
  data->pcd_obs_bg = obs->bg_pcd;

  // compute range images
  data->obs_range_image = pcd_to_range_image(data->pcd_obs_bg, 0, M_PI/1000.0, 4);  //TODO: make this a parameter?
  data->obs_fg_range_image = pcd_to_range_image_from_template(data->pcd_obs, data->obs_range_image);

  // compute blurred edge feature image
  data->obs_edge_image = get_edge_feature_image(data->pcd_obs, data->obs_fg_range_image, params);

  //dbug
  obs_edge_image_width_ = data->obs_fg_range_image->w;
  obs_edge_image_height_ = data->obs_fg_range_image->h;
  //printf("obs_edge_image_width_ = %d\n", data->obs_edge_image_width_);
  //printf("obs_edge_image_height_ = %d\n", data->obs_edge_image_height_);
  obs_edge_image_ = matrix_clone(data->obs_edge_image, obs_edge_image_width_, obs_edge_image_height_);  //dbug

  // get combined feature matrices
  data->obs_xyzn = get_xyzn_features(data->pcd_obs->points, data->pcd_obs->normals, data->pcd_obs->num_points, params);
  data->obs_bg_xyzn = get_xyzn_features(data->pcd_obs_bg->points, data->pcd_obs_bg->normals, data->pcd_obs_bg->num_points, params);
  data->obs_fxyzn = get_fxyzn_features(data->pcd_obs->shapes, data->pcd_obs->points, data->pcd_obs->normals, data->pcd_obs->num_points, data->pcd_obs->shape_length, params);

  // flann params
  struct FLANNParameters flann_params_single = DEFAULT_FLANN_PARAMETERS;
  flann_params_single.algorithm = FLANN_INDEX_KDTREE_SINGLE;
  data->obs_xyzn_params = flann_params_single;

  // build flann indices
  float speedup;
  data->obs_xyzn_index = flann_build_index_double(data->obs_xyzn[0], data->pcd_obs->num_points, 6, &speedup, &data->obs_xyzn_params);
  
  return data;
}

void free_scope_model_data(scope_model_data_t *data)
{
  free_multiview_pcd(data->range_edges_model);
  free(data->fpfh_model_pmf);
  free(data->fpfh_model_cmf);
  free_matrix2(data->model_xyzn);
  free_matrix2(data->fpfh_model_fxyzn);
  free_matrix2(data->fpfh_model_xyzn);
  flann_free_index(data->model_xyz_index, &data->model_xyz_params);
  flann_free_index(data->fpfh_model_f_index, &data->fpfh_model_f_params);
  flann_free_index(data->fpfh_model_xyzn_index, &data->fpfh_model_xyzn_params);

  free(data);
}

void free_scope_obs_data(scope_obs_data_t *data)
{
  free_range_image(data->obs_range_image);
  free_range_image(data->obs_fg_range_image);
  free_matrix2(data->obs_edge_image);
  free_matrix2(data->obs_xyzn);
  free_matrix2(data->obs_bg_xyzn);
  free_matrix2(data->obs_fxyzn);
  flann_free_index(data->obs_xyzn_index, &data->obs_xyzn_params);

  free(data);
}

void copy_olf(olf_t *olf2, olf_t *olf1)
{
  if (olf1->x) {
    if (olf2->x == NULL)
      safe_calloc(olf2->x, 3, double);
    memcpy(olf2->x, olf1->x, 3*sizeof(double));
  }

  if (olf1->q) {
    if (olf2->q == NULL)
      safe_calloc(olf2->q, 4, double);
    memcpy(olf2->q, olf1->q, 4*sizeof(double));
  }

  if (olf1->B) {
    if (olf2->B == NULL) {
      safe_calloc(olf2->B, 1, bingham_t);
      bingham_alloc(olf2->B, 4);
    }
    bingham_copy(olf2->B, olf1->B);
  }
}

void free_olf(olf_t *olf)
{
  if (olf->x)
    free(olf->x);
  if (olf->q)
    free(olf->q);
  if (olf->B) {
    bingham_free(olf->B);
    free(olf->B);
  }
}

void scope_sample_alloc(scope_sample_t *sample, int nc)
{
  safe_calloc(sample->c_obs, nc, int);
  safe_calloc(sample->c_model, nc, int);
  safe_calloc(sample->c_type, nc, int);
  safe_calloc(sample->c_score, nc, double);
  safe_calloc(sample->obs_olfs, nc, olf_t);
  safe_calloc(sample->model_olfs, nc, olf_t);
  bingham_alloc(&sample->B, 4);
}

void scope_sample_free(scope_sample_t *sample)
{
  free(sample->c_obs);
  free(sample->c_model);
  free(sample->c_score);
  free(sample->c_type);
  bingham_free(&sample->B);
  int j;
  for (j = 0; j < sample->nc; j++) {
    free_olf(&sample->obs_olfs[j]);
    free_olf(&sample->model_olfs[j]);
  }
}

// assumes s2 is already allocated
void scope_sample_copy(scope_sample_t *s2, scope_sample_t *s1)
{
  memcpy(s2->x, s1->x, 3 * sizeof(double));
  memcpy(s2->q, s1->q, 4 * sizeof(double));
  bingham_copy(&s2->B, &s1->B);
  s2->nc = s1->nc;
  memcpy(s2->c_obs, s1->c_obs, s1->nc * sizeof(int));
  memcpy(s2->c_model, s1->c_model, s1->nc * sizeof(int));
  memcpy(s2->c_type, s1->c_type, s1->nc * sizeof(int));
  memcpy(s2->c_score, s1->c_score, s1->nc * sizeof(double));
  int j;
  for (j = 0; j < s1->nc; j++) {
    copy_olf(&s2->obs_olfs[j], &s1->obs_olfs[j]);
    copy_olf(&s2->model_olfs[j], &s1->model_olfs[j]);
  }
}

scope_samples_t *create_scope_samples(int num_samples, int num_correspondences)
{
  scope_samples_t *S;
  safe_calloc(S, 1, scope_samples_t);
  S->num_samples_allocated = num_samples;
  safe_calloc(S->samples, num_samples, scope_sample_t);
  safe_calloc(S->W, num_samples, double);
  int i;
  for (i = 0; i < num_samples; i++)
    scope_sample_alloc(&S->samples[i], num_correspondences);

  return S;
}

void free_scope_samples(scope_samples_t *S)
{
  int i;
  for (i = 0; i < S->num_samples_allocated; i++)
    scope_sample_free(&S->samples[i]);
  free(S->samples);
  free(S->W);
  free(S);
}

void remove_redundant_pose_samples(scope_samples_t *S, scope_params_t *params)
{
  double dx2_thresh = params->x_cluster_thresh * params->x_cluster_thresh;
  double dq_thresh = params->q_cluster_thresh;
  int idx[S->num_samples];
  idx[0] = 0;
  int cnt = 1;
  int i, j;

  for (i = 1; i < S->num_samples; i++) {
    int unique = 1;
    for (j = 0; j < cnt; j++) {
      double dx2 = dist2(S->samples[i].x, S->samples[idx[j]].x, 3);
      double dq = acos(fabs(dot(S->samples[i].q, S->samples[idx[j]].q, 4)));
      if (dx2 < dx2_thresh && dq < dq_thresh) {
	unique = 0;
	break;
      }
    }
    if (unique)
      idx[cnt++] = i;
  }

  for (i = 0; i < cnt; i++)
    scope_sample_copy(&S->samples[i], &S->samples[idx[i]]);
  reorder(S->W, S->W, idx, cnt);
  S->num_samples = cnt;
}

void sort_pose_samples(scope_samples_t *S)
{
  int n = S->num_samples;
  int idx[n];
  sort_indices(S->W, idx, n);  // smallest to biggest
  reversei(idx, idx, n);  // reverse idx (biggest to smallest)

  reorder(S->W, S->W, idx, n);

  scope_sample_t new_samples[n];
  int i;
  for (i = 0; i < n; i++)
    new_samples[i] = S->samples[idx[i]];
  memcpy(S->samples, new_samples, n*sizeof(scope_sample_t));
}





 //==============================================================================================//

 //---------------------------------------  Noise Model  ----------------------------------------//

 //==============================================================================================//


inline double sigmoid(double x, const double *b)
{
  return b[0] + (1 - b[0]) / (1 + exp(-b[1]-b[2]*x));
}

scope_noise_model_t *get_noise_models(double *x, double *q, int *idx, int n, pcd_t *pcd_model, multiview_pcd_t *range_edges_model)
{
  const double b_SR[3] = {0.2878,    -5.6214,      7.7247};
  const double b_SN[3] = {0.1521,    -7.1290,     10.7090};
  const double b_SL[3] = {0.2238,    -5.1827,      6.8242};
  const double b_SA[3] = {0.1618,    -6.3992,      8.0207};
  const double b_SB[3] = {0.2313,    -6.3463,      8.0651};

  const double b_ER[3] = {0.3036,     0.2607,   -125.8843};
  const double b_EN[3] = {0.1246,     1.4406,   -185.8350};
  const double b_EL[3] = {0.2461,     0.2624,   -140.0192};
  const double b_EA[3] = {0.1494,     0.2114,   -139.4324};
  const double b_EB[3] = {0.2165,     0.2600,   -135.5203};

  int i;

  // compute surface angles
  double surface_angles[n];
  double **V = get_sub_cloud_at_pose(pcd_model, idx, n, x, q);
  double **N = get_sub_cloud_normals_rotated(pcd_model, idx, n, q);
  for (i = 0; i < n; i++) {
    normalize(V[i], V[i], 3);
    surface_angles[i] = 1 + dot(V[i], N[i], 3);
  }
  free_matrix2(V);
  free_matrix2(N);
  
  // lookup edge distances for closest model viewpoint
  double edge_dists[n];
  double vp[3];
  model_pose_to_viewpoint(vp, x, q);
  double cos_vp_angles[range_edges_model->num_views];
  matrix_vec_mult(cos_vp_angles, range_edges_model->views, vp, range_edges_model->num_views, 3);
  int vi = find_max(cos_vp_angles, range_edges_model->num_views);
  for (i = 0; i < n; i++)
    edge_dists[i] = pcd_model->ved[idx[i]][vi];

  // compute sigmas
  scope_noise_model_t *noise_models;
  safe_calloc(noise_models, n, scope_noise_model_t);
  for (i = 0; i < n; i++) {
    noise_models[i].range_sigma = .5*sigmoid(surface_angles[i], b_SR) + .5*sigmoid(edge_dists[i], b_ER);
    noise_models[i].normal_sigma = .5*sigmoid(surface_angles[i], b_SN) + .5*sigmoid(edge_dists[i], b_EN);
    noise_models[i].lab_sigma[0] = .5*sigmoid(surface_angles[i], b_SL) + .5*sigmoid(edge_dists[i], b_EL);
    noise_models[i].lab_sigma[1] = .5*sigmoid(surface_angles[i], b_SA) + .5*sigmoid(edge_dists[i], b_EA);
    noise_models[i].lab_sigma[2] = .5*sigmoid(surface_angles[i], b_SB) + .5*sigmoid(edge_dists[i], b_EB);

    //dbug
    //if (i%100==0) {
    //  printf("sigmoid(S=%.2f, b_SL) = %.2f, sigmoid(E=%.2f, b_ER) = %.2f\n", surface_angles[i], sigmoid(surface_angles[i], b_SL), edge_dists[i], sigmoid(edge_dists[i], b_EL));
    //  printf("sigmoid(S=%.2f, b_SA) = %.2f, sigmoid(E=%.2f, b_EA) = %.2f\n", surface_angles[i], sigmoid(surface_angles[i], b_SA), edge_dists[i], sigmoid(edge_dists[i], b_EA));
    //  printf("sigmoid(S=%.2f, b_SB) = %.2f, sigmoid(E=%.2f, b_EB) = %.2f\n", surface_angles[i], sigmoid(surface_angles[i], b_SB), edge_dists[i], sigmoid(edge_dists[i], b_EB));
    //}
  }

  return noise_models;
}




 //==============================================================================================//

 //---------------------------------  Model Placement Scoring  ----------------------------------//

 //==============================================================================================//


/*
double compute_xyzn_score(double *nn_d2, double *vis_prob, int n, scope_params_t *params)
{
  int i;
  double score = 0.0;
  double range_sigma = params->range_sigma;
  double xyz_weight = params->xyz_weight;
  for (i = 0; i < n; i++) {
    double d = sqrt(nn_d2[i]) / xyz_weight;  // TODO: make this a param
    d = MIN(d, 3*range_sigma);               // TODO: make this a param
    score += vis_prob[i] * log(normpdf(d, 0, range_sigma));
  }

  score -= log(normpdf(0,0,range_sigma));
  return score;
}
*/


double compute_xyz_score(double **cloud, double *vis_pmf, scope_noise_model_t *noise_models, int num_validation_points, range_image_t *obs_range_image, scope_params_t *params, int score_round)
{
  double score = 0.0;
  //double range_sigma = params->range_sigma;
  //double dmax = 2*range_sigma;
  int i;
  for (i = 0; i < num_validation_points; i++) {
    if (vis_pmf[i] > .01/(double)num_validation_points) {
      int xi,yi;
      range_image_xyz2sub(&xi, &yi, obs_range_image, cloud[i]);
      double range_sigma = params->range_sigma * noise_models[i].range_sigma;
      double dmax = 2*range_sigma;
      double d = dmax;
      if (obs_range_image->cnt[xi][yi] > 0) {
	// get distance from model point to range image cell plane
	double c[4];
	xyzn_to_plane(c, obs_range_image->points[xi][yi], obs_range_image->normals[xi][yi]);
	d = fabs(dot(c, cloud[i], 3) + c[3]);
	//d /= noise_models[i].range_sigma;
	d = MIN(d, dmax);
      }
      score += vis_pmf[i] * log(normpdf(d, 0, range_sigma));

      //dbug
      if (params->verbose)
	mps_xyz_dists_[i] = d / dmax;
    }
  }
  score -= log(normpdf(0, 0, params->range_sigma));

  //dbug
  if (params->verbose)
    xyz_score_ = score;

  double w = 0;
  if (score_round == 2)
    w = params->score2_xyz_weight;
  else
    w = params->score3_xyz_weight;

  return w * score;
}


double compute_normal_score(double **cloud, double **cloud_normals, double *vis_pmf, scope_noise_model_t *noise_models, int num_validation_points, range_image_t *obs_range_image, scope_params_t *params, int score_round)
{
  double score = 0.0;
  //double normal_sigma = params->normal_sigma;
  //double dmax = 2*normal_sigma;  // TODO: make this a param
  int i;
  for (i = 0; i < num_validation_points; i++) {
    if (vis_pmf[i] > .01/(double)num_validation_points) {
      int xi,yi;
      range_image_xyz2sub(&xi, &yi, obs_range_image, cloud[i]);
      double normal_sigma = params->normal_sigma * noise_models[i].normal_sigma;
      double dmax = 2*normal_sigma;
      double d = dmax;
      if (obs_range_image->cnt[xi][yi] > 0) {
	// get distance from model normal to range image cell normal
	d = 1.0 - dot(cloud_normals[i], obs_range_image->normals[xi][yi], 3);
	//d /= noise_models[i].normal_sigma;
	d = MIN(d, dmax);
      }
      score += vis_pmf[i] * log(normpdf(d, 0, normal_sigma));

      //dbug
      if (params->verbose)
	mps_normal_dists_[i] = d / dmax;
    }
  }
  score -= log(normpdf(0, 0, params->normal_sigma));

  //dbug
  if (params->verbose)
    normal_score_ = score;

  double w = 0;
  if (score_round == 2)
    w = params->score2_normal_weight;
  else
    w = params->score3_normal_weight;

  return w * score;
}

/*
double compute_xyz_score(double **xyz, int *nn_idx, double *vis_prob, int n, pcd_t *pcd_obs, scope_params_t *params)
{
  double score = 0.0;
  double range_sigma = params->range_sigma;
  int i;
  for (i = 0; i < n; i++) {
    double *obs_xyz = pcd_obs->points[nn_idx[i]];
    double d = dist(xyz[i], obs_xyz, 3);
    d = MIN(d, 2*range_sigma);  // TODO: make this a param
    score += vis_prob[i] * log(normpdf(d, 0, range_sigma));
  }

  score -= log(normpdf(0, 0, range_sigma));
  return score;
}
*/

double compute_fpfh_score(double **fpfh, int *nn_idx, double *vis_prob, int n, pcd_t *pcd_obs, scope_params_t *params)
{
  double score = 0.0;
  double f_sigma = params->f_sigma;
  int i;
  for (i = 0; i < n; i++) {
    double *obs_f = pcd_obs->shapes[nn_idx[i]];
    double df = dist(fpfh[i], obs_f, pcd_obs->shape_length);
    df = MIN(df, 4*f_sigma);  // TODO: make this a param
    score += vis_prob[i] * log(normpdf(df, 0, 2*f_sigma));
  }

  score -= log(normpdf(0,0,2*f_sigma));
  return params->f_weight * score;
}

/*
double compute_sdw_score(double **sdw, int *nn_idx, double *vis_prob, int n, pcd_t *pcd_obs, scope_params_t *params)
{
  double surfdist_thresh = params->surfdist_thresh;
  double surfwidth_thresh = params->surfwidth_thresh;
  double surfdist_sigma = params->surfdist_sigma;
  double surfwidth_sigma = params->surfwidth_sigma;

  double score = 0.0;
  int i;
  for (i = 0; i < n; i++) {
    double *obs_sdw = pcd_obs->sdw[nn_idx[i]];
    double obs_surfdist = MIN(obs_sdw[0], surfdist_thresh);
    double obs_surfwidth = MIN(obs_sdw[1], surfwidth_thresh);
    score += vis_prob[i] * log(normpdf(sdw[i][0] - obs_surfdist, 0, surfdist_sigma));
    score += vis_prob[i] * log(normpdf(sdw[i][1] - obs_surfwidth, 0, surfwidth_sigma));
  }

  score -= log(normpdf(0,0,surfdist_sigma));
  score -= log(normpdf(0,0,surfwidth_sigma));

  return score;
}
*/


double compute_lab_score(double **cloud, double **lab, double *vis_pmf, scope_noise_model_t *noise_models, int n, range_image_t *obs_range_image, pcd_t *pcd_obs, scope_params_t *params, int score_round)
{
  double scores[3] = {0, 0, 0};
  int i, j;
  //double L_weight = params->L_weight;
  //double lab_sigma = params->lab_sigma;
  //double dmax = 2*lab_sigma; // * sqrt(2.0 + L_weight*L_weight);  // TODO: make this a param
  for (i = 0; i < n; i++) {
    if (vis_pmf[i] > .01/(double)n) {
      int xi,yi;
      range_image_xyz2sub(&xi, &yi, obs_range_image, cloud[i]);
      //double d = dmax;
      double dlab[3], dmax[3], lab_sigma[3];
      for (j = 0; j < 3; j++) {
	lab_sigma[j] = params->lab_sigma * noise_models[i].lab_sigma[j];
	dmax[j] = 2*lab_sigma[j];
	dlab[j] = dmax[j];
      }

      int obs_idx = obs_range_image->idx[xi][yi];
      if (obs_idx > 0) {
	double *obs_lab = pcd_obs->lab[obs_idx];
	sub(dlab, lab[i], obs_lab, 3);
	//dlab[0] = L_weight * (lab[i][0] - obs_lab[0]) / noise_models[i].lab_sigma[0];
	//dlab[1] = (lab[i][1] - obs_lab[1]) / noise_models[i].lab_sigma[1];
	//dlab[2] = (lab[i][2] - obs_lab[2]) / noise_models[i].lab_sigma[2];
	//dlab[2] = 0;
	//d = norm(dlab, 3);
	//d = MIN(d, dmax);
	for (j = 0; j < 3; j++)	  
	  dlab[j] = MIN(dlab[j], dmax[j]);

	//dbug
	//if (params->verbose && (i%100==0)) {
	//  printf("model lab[%d] = [%.2f, %.2f, %.2f], obs_lab = [%.2f, %.2f, %.2f]\n", i, lab[i][0], lab[i][1], lab[i][2], obs_lab[0], obs_lab[1], obs_lab[2]);
	//}
      }
      //score += vis_pmf[i] * log(normpdf(d, 0, lab_sigma));
      for (j = 0; j < 3; j++)
	scores[j] += vis_pmf[i] * log(normpdf(dlab[j], 0, lab_sigma[j]));

      //dbug
      //if (params->verbose && (i%100==0)) {
      //  printf("dlab[%d] = [%.2f, %.2f, %.2f], lab_sigma = [%.2f, %.2f, %.2f]\n", i, dlab[0], dlab[1], dlab[2], lab_sigma[0], lab_sigma[1], lab_sigma[2]);
      //}
    }
  }
  //score -= log(normpdf(0, 0, params->lab_sigma));

  //dbug
  if (params->verbose) {
    lab_scores_[0] = scores[0];
    lab_scores_[1] = scores[1];
    lab_scores_[2] = scores[2];    
  }

  double lab_weights2[3] = {params->score2_L_weight, params->score2_A_weight, params->score2_B_weight};
  double lab_weights3[3] = {params->score3_L_weight, params->score3_A_weight, params->score3_B_weight};

  double *w = NULL;
  if (score_round == 2)
    w = lab_weights2;
  else
    w = lab_weights3;

  return dot(scores, w, 3);
}


double compute_vis_score(double *vis_prob, int n, scope_params_t *params, int score_round)
{
  double score = log(sum(vis_prob, n) / (double) n);

  //dbug
  if (params->verbose)
    vis_score_ = score;

  double w = 0;
  if (score_round == 2)
    w = params->score2_vis_weight;
  else
    w = params->score3_vis_weight;

  return w * score;
}


int **compute_occ_edges(int *num_occ_edges, double **cloud, double *vis_prob, int n, range_image_t *obs_range_image, scope_params_t *params)
{
  // create vis_prob image, V
  int i;
  int w = obs_range_image->w;
  int h = obs_range_image->h;
  double **V = new_matrix2(w,h);
  int xi, yi;
  int x0 = w, y0 = h, x1 = 0, y1 = 0;  // bounding box for model points in vis_prob_image
  for (i = 0; i < n; i++) {
    if (range_image_xyz2sub(&xi, &yi, obs_range_image, cloud[i]) && vis_prob[i] > V[xi][yi]) {
      V[xi][yi] = vis_prob[i];
      if (xi < x0)
	x0 = xi;
      if (xi > x1)
	x1 = xi;
      if (yi < y0)
	y0 = yi;
      if (yi > y1)
	y1 = yi;
    }
  }

  // downsample vis_prob sub matrix (loses a row or column if w2 or h2 is odd)
  double **V2 = get_sub_matrix(V, x0, y0, x1, y1);
  int w2 = (x1-x0+1)/2;
  int h2 = (y1-y0+1)/2;
  int x,y;
  for (x = 0; x < w2; x++) {
    for (y = 0; y < h2; y++) {
      double v2 = MAX(V2[2*x][2*y], V2[2*x+1][2*y]);
      v2 = MAX(v2, V2[2*x][2*y+1]);
      V2[x][y] = MAX(v2, V2[2*x+1][2*y+1]);
    }
  }

  // dilate vis_prob sub matrix
  dilate_matrix(V, V2, w2, h2);
  //dilate_matrix(V2, V, w2, h2);
  //dilate_matrix(V, V2, w2, h2);
  //dilate_matrix(V2, V, w2, h2);
  //dilate_matrix(V, V2, w2, h2);

  // compute edges where vis_prob crosses .5 threshold
  int px[n], py[n], cnt=0;
  for (x = 0; x < w2-1; x++) {
    for (y = 0; y < h2-1; y++) {
      if (V[x][y] >= .5) {
	if ((x > 0 && V[x-1][y] > 0.0 && V[x-1][y] < .5) || (x < w2-1 && V[x+1][y] > 0.0 && V[x+1][y] < .5) ||
	    (y > 0 && V[x][y-1] > 0.0 && V[x][y-1] < .5) || (y < h2-1 && V[x][y+1] > 0.0 && V[x][y+1] < .5)) {
	  px[cnt] = x0 + 2*x;
	  py[cnt++] = y0 + 2*y;
	}
      }
    }
  }
  *num_occ_edges = cnt;

  free_matrix2(V);
  free_matrix2(V2);

  if (cnt==0)
    return NULL;

  int **occ_edges = new_matrix2i(cnt,2);
  for (i = 0; i < cnt; i++) {
    occ_edges[i][0] = px[i];
    occ_edges[i][1] = py[i];
  }

  //dbug
  if (params->verbose) {
    if (occ_edge_pixels_ != NULL)
      free_matrix2i(occ_edge_pixels_);
    occ_edge_pixels_ = new_matrix2i(MAX(cnt,1), 2);  //dbug
    memcpy(occ_edge_pixels_[0], occ_edges[0], 2*cnt*sizeof(int));
    num_occ_edge_points_ = cnt;
  }

  return occ_edges;
}



/*
double compute_occ_edge_score(double **cloud, double *vis_prob, int n, range_image_t *obs_range_image, double **obs_edge_image, scope_params_t *params)
{
  int i;
  int num_occ_edges;
  int **occ_edges = compute_occ_edges(&num_occ_edges, cloud, vis_prob, n, obs_range_image);

  //dbug
  if (params->verbose) {
    if (occ_edge_pixels_ != NULL)
      free_matrix2i(occ_edge_pixels_);
    occ_edge_pixels_ = new_matrix2i(num_occ_edges, 2);
    memcpy(occ_edge_pixels_[0], occ_edges[0], 2*num_occ_edges*sizeof(int));
    num_occ_edge_points_ = num_occ_edges;
  }

  double score = 0.0;
  for (i = 0; i < num_occ_edges; i++) {
    int x = occ_edges[i][0];
    int y = occ_edges[i][1];
    score += obs_edge_image[x][y];
  }
  score /= (double) num_occ_edges;

  free_matrix2i(occ_edges);

  return params->score_edge_occ_weight * score;
}
*/



double compute_edge_score(double **P, int n, int **occ_edges, int num_occ_edges, range_image_t *obs_range_image, double **obs_edge_image, scope_params_t *params, int score_round)
{
  if (n == 0)
    return 0.0;

  // compute visibility of sampled model edges
  int vis_pixel_radius = 2;
  double vis_prob[n];
  int i;
  for (i = 0; i < n; i++)
    vis_prob[i] = compute_visibility_prob(P[i], NULL, obs_range_image, params->vis_thresh, vis_pixel_radius);
  double vis_pmf[n];
  normalize_pmf(vis_pmf, vis_prob, n);

  //dbug
  if (params->verbose) {
    if (range_edge_points_ == NULL) {
      range_edge_points_ = new_matrix2(10000, 3);
      range_edge_pixels_ = new_matrix2i(10000, 2);
      safe_calloc(range_edge_vis_prob_, 10000, double);
    }
    matrix_copy(range_edge_points_, P, n, 3);
    memset(range_edge_pixels_[0], 0, n*2*sizeof(int));
    num_range_edge_points_ = n;
    memcpy(range_edge_vis_prob_, vis_prob, n*sizeof(double));
  }

  // compute obs_edge_image score for sampled model edges
  double score = 0;
  int xi,yi;
  for (i = 0; i < n; i++) {
    if (range_image_xyz2sub(&xi, &yi, obs_range_image, P[i])) {
      score += vis_pmf[i] * obs_edge_image[xi][yi];

      //dbug
      if (params->verbose) {
	range_edge_pixels_[i][0] = xi;
	range_edge_pixels_[i][1] = yi;
      }
    }
  }
  double vis_score = log(sum(vis_prob, n) / (double) n);

  // add occlusion edges to score
  double occ_score = 0.0;
  if (num_occ_edges > 0) {

    for (i = 0; i < num_occ_edges; i++) {
      int x = occ_edges[i][0];
      int y = occ_edges[i][1];
      occ_score += obs_edge_image[x][y];
    }
    occ_score /= (double) num_occ_edges;
    occ_score = num_occ_edges*occ_score / (double)(n + num_occ_edges);
    score = n*score / (double)(n + num_occ_edges);
  }

  //dbug
  if (params->verbose) {
    edge_score_ = score;
    edge_vis_score_ = vis_score;
    edge_occ_score_ = occ_score;
  }

  double w1=0, w2=0, w3=0;
  if (score_round == 2) {
    w1 = params->score2_edge_weight;
    w2 = params->score2_edge_vis_weight;
    w3 = params->score2_edge_occ_weight;
  }
  else {
    w1 = params->score3_edge_weight;
    w2 = params->score3_edge_vis_weight;
    w3 = params->score3_edge_occ_weight;
  }

  return (w1 * score) + (w2 * vis_score) + (w3 * occ_score);
}





double compute_segment_score(double *x, double *q, double **cloud, flann_index_t model_xyz_index, struct FLANNParameters *model_xyz_params,
			     double *vis_prob, int n, range_image_t *obs_range_image, double **obs_edge_image, scope_params_t *params, int score_round)
{
  double range_sigma = params->range_sigma;
  double dmax = 2*range_sigma;

  int w = obs_range_image->w;
  int h = obs_range_image->h;
  
  double q_inv[4];
  quaternion_inverse(q_inv, q);
  double **R_inv = new_matrix2(3,3);
  quaternion_to_rotation_matrix(R_inv, q_inv);

  double score = 0.0;
  int cnt = 0;
  int i, xi, yi;
  for (i = 0; i < n; i++) {
    if (vis_prob[i] > .1 && range_image_xyz2sub(&xi, &yi, obs_range_image, cloud[i])) {
      
      edge_image_random_walk(&xi, &yi, obs_edge_image, w, h, 1, 0);

      if (obs_range_image->cnt[xi][yi] > 0) {

	// transform observed point into model coordinates
	double p[3];
	sub(p, obs_range_image->points[xi][yi], x, 3);
	matrix_vec_mult(p, R_inv, p, 3, 3);

	// find the xyz-distance to the closest model point
	int nn_idx;
	double nn_d2;
	flann_find_nearest_neighbors_index_double(model_xyz_index, p, 1, &nn_idx, &nn_d2, 1, model_xyz_params);

	// add score
	double d = MIN(sqrt(nn_d2), dmax);
	score += log(normpdf(d, 0, range_sigma));
	cnt++;
      }
    }
  }
  if (cnt > 0)
    score /= (double)cnt;

  free_matrix2(R_inv);

  if (params->verbose)
    segment_score_ = score;

  double weight = 0;
  if (score_round == 2)
    weight = params->score2_segment_weight;
  else
    weight = params->score3_segment_weight;

  return weight * score;
}



double model_placement_score(scope_sample_t *sample, scope_model_data_t *model_data, scope_obs_data_t *obs_data, scope_params_t *params, int score_round)
{
  double *x = sample->x;
  double *q = sample->q;

  //double t0 = get_time_ms();  //dbug

  // get model validation points
  int i;
  int num_validation_points = (params->num_validation_points > 0 ? params->num_validation_points : model_data->pcd_model->num_points);
  int idx[num_validation_points];
  get_validation_points(idx, model_data->pcd_model, num_validation_points);

  //printf("break 0, %.2f ms\n", get_time_ms() - t0);  //dbug
  //t0 = get_time_ms();

  if (params->verbose)
    memcpy(mps_idx_, idx, num_validation_points*sizeof(int));

  //printf("break 1, %.2f ms\n", get_time_ms() - t0);  //dbug
  //t0 = get_time_ms();

  // extract transformed model validation features
  double **cloud = get_sub_cloud_at_pose(model_data->pcd_model, idx, num_validation_points, x, q);

  if (score_round == 1) {  // after c=1, just use free space to score
    double dthresh = .05;  //TODO: make this a param
    double score = 0;
    int xi, yi;
    for (i = 0; i < num_validation_points; i++)
      if (range_image_xyz2sub(&xi, &yi, obs_data->obs_range_image, cloud[i]) && obs_data->obs_range_image->image[xi][yi] > dthresh + norm(cloud[i],3))
	score -= 1.0;
    score /= (double)num_validation_points;
    free_matrix2(cloud);
    return score;
  }



  double **cloud_normals = get_sub_cloud_normals_rotated(model_data->pcd_model, idx, num_validation_points, sample->q);
  //double **cloud_f = get_sub_cloud_fpfh(model_data->pcd_model, idx, num_validation_points);
  //double **cloud_sdw = get_sub_cloud_sdw(model_data->pcd_model, idx, num_validation_points, params);
  double **cloud_lab = get_sub_cloud_lab(model_data->pcd_model, idx, num_validation_points);
  //double **cloud_xyzn = get_xyzn_features(cloud, cloud_normals, num_validation_points, params);

  //printf("break 2, %.2f ms\n", get_time_ms() - t0);  //dbug
  //t0 = get_time_ms();

  // compute p(visibile)
  double vis_prob[num_validation_points];
  for (i = 0; i < num_validation_points; i++)
    vis_prob[i] = compute_visibility_prob(cloud[i], cloud_normals[i], obs_data->obs_range_image, params->vis_thresh, 0);
  double vis_pmf[num_validation_points];
  normalize_pmf(vis_pmf, vis_prob, num_validation_points);

  if (params->verbose)
    memcpy(mps_vis_prob_, vis_prob, num_validation_points*sizeof(double));

  //printf("break 3, %.2f ms\n", get_time_ms() - t0);  //dbug
  //t0 = get_time_ms();

  // compute noise models
  scope_noise_model_t *noise_models = get_noise_models(x, q, idx, num_validation_points, model_data->pcd_model, model_data->range_edges_model);

  //printf("break 4, %.2f ms\n", get_time_ms() - t0);  //dbug
  //t0 = get_time_ms();

  // compute nearest neighbors
  //int nn_idx[num_validation_points];  memset(nn_idx, 0, num_validation_points*sizeof(int));
  //double nn_d2[num_validation_points];  memset(nn_d2, 0, num_validation_points*sizeof(double));
  //int search_radius = 0;  // pixels
  //for (i = 0; i < num_validation_points; i++)
  //if (vis_prob[i] > .01)
  //range_image_find_nn(&nn_idx[i], &nn_d2[i], &cloud[i], &cloud_xyzn[i], 1, 6, obs_xyzn, obs_range_image, search_radius);
  //range_image_find_nn(&nn_idx[i], &nn_d2[i], &cloud[i], &cloud[i], 1, 3, pcd_obs->points, obs_range_image, search_radius);

  double xyz_score = compute_xyz_score(cloud, vis_pmf, noise_models, num_validation_points, obs_data->obs_range_image, params, score_round);
  double normal_score = compute_normal_score(cloud, cloud_normals, vis_pmf, noise_models, num_validation_points, obs_data->obs_range_image, params, score_round);
  double lab_score = compute_lab_score(cloud, cloud_lab, vis_pmf, noise_models, num_validation_points, obs_data->obs_range_image, obs_data->pcd_obs_bg, params, score_round);
  double vis_score = compute_vis_score(vis_prob, num_validation_points, params, score_round);
  //double xyzn_score = compute_xyzn_score(nn_d2, vis_pmf, num_validation_points, params);
  //double xyz_score = compute_xyz_score(cloud, nn_idx, vis_pmf, num_validation_points, pcd_obs, params);
  //double f_score = compute_fpfh_score(cloud_f, nn_idx, vis_pmf, num_validation_points, pcd_obs, params);
  //double sdw_score = compute_sdw_score(cloud_sdw, nn_idx, vis_pmf, num_validation_points, pcd_obs, params);
  //double lab_score = compute_lab_score(cloud_lab, nn_idx, vis_pmf, num_validation_points, pcd_obs, params);

  //printf("break 5, %.2f ms\n", get_time_ms() - t0);  //dbug
  //t0 = get_time_ms();

  //TODO: move this to compute_edge_score()

  double edge_score = 0.0;
  if (obs_data->obs_edge_image) {
    int n = params->num_validation_points;
    double **P = get_range_edge_points(&n, x, q, model_data->range_edges_model, obs_data->obs_range_image);
    transform_cloud(P, P, n, x, q);
    if (params->num_validation_points == 0) {
      int num_occ_edges;
      int **occ_edges = compute_occ_edges(&num_occ_edges, cloud, vis_prob, num_validation_points, obs_data->obs_range_image, params);
      edge_score = compute_edge_score(P, n, occ_edges, num_occ_edges, obs_data->obs_range_image, obs_data->obs_edge_image, params, score_round);
      if (occ_edges)
	free_matrix2i(occ_edges);
    }
    else
      edge_score = compute_edge_score(P, n, NULL, 0, obs_data->obs_range_image, obs_data->obs_edge_image, params, score_round);
    free_matrix2(P);
  }

  //printf("break 6, %.2f ms\n", get_time_ms() - t0);  //dbug
  //t0 = get_time_ms();

  double segment_score = 0;
  if (score_round >= 3)
    segment_score = compute_segment_score(x, q, cloud, model_data->model_xyz_index, &model_data->model_xyz_params, vis_prob,
					  num_validation_points, obs_data->obs_range_image, obs_data->obs_edge_image, params, score_round);

  double score = xyz_score + normal_score + edge_score + lab_score + vis_score + segment_score;

  //dbug
  //if (sample->c_type[0] == C_TYPE_SIFT)
  //  score += 100;

  //if (params->verbose) {
  //  printf("score_comp = (%f, %f, %f, %f, %f, %f)\n", xyz_score, normal_score, edge_score, lab_score, vis_score, segment_score);
  //  printf("score = %f\n", score);
  //}

  //printf("break 7, %.2f ms\n", get_time_ms() - t0);  //dbug

  
  if (params->verbose) {
    double scores[10] = {xyz_score_, normal_score_, vis_score_, segment_score_, edge_score_, edge_vis_score_, edge_occ_score_, lab_scores_[0], lab_scores_[1], lab_scores_[2]};
    if (sample->scores == NULL) {
      sample->num_scores = 10;
      safe_calloc(sample->scores, sample->num_scores, double);
    }
    memcpy(sample->scores, scores, sample->num_scores*sizeof(double));
  }


  // cleanup
  free_matrix2(cloud);
  free_matrix2(cloud_normals);
  //free_matrix2(cloud_f);
  //free_matrix2(cloud_sdw);
  free_matrix2(cloud_lab);
  //free_matrix2(cloud_xyzn);
  free(noise_models);

  return score;
}





//==============================================================================================//

//----------------------------  Bingham Procrustean Alignment (BPA)  ---------------------------//

//==============================================================================================//


// assumes B is already allocated
void olf_to_bingham(bingham_t *B, int idx, pcd_t *pcd)
{
  double epsilon = 1e-50;

  double r1[3] = {pcd->normals[idx][0], pcd->normals[idx][1], pcd->normals[idx][2]};
  double r2[3] = {pcd->principal_curvatures[idx][0], pcd->principal_curvatures[idx][1], pcd->principal_curvatures[idx][2]};
  double r3[3];
  cross(r3, r1, r2);
  double *R[3] = {r1,r2,r3};

  // v1: mode
  double v1[4];
  rotation_matrix_to_quaternion(v1, R);
  quaternion_inverse(v1, v1);

  // B->V[2]: rotation about the normal vector
  int i;
  for (i = 0; i < 3; i++) {
    r2[i] = -r2[i];
    r3[i] = -r3[i];
  }
  rotation_matrix_to_quaternion(B->V[2], R);
  quaternion_inverse(B->V[2], B->V[2]);

  double **V1 = new_matrix2(4,4);  // v1's projection matrix 
  outer_prod(V1, v1, v1, 4, 4);
  mult(V1[0], V1[0], -1, 16);
  for (i = 0; i < 4; i++)
    V1[i][i] += 1.0;
  double **V2 = new_matrix2(4,4);  // B->V[2]'s projection matrix
  outer_prod(V2, B->V[2], B->V[2], 4, 4);
  mult(V2[0], V2[0], -1, 16);
  for (i = 0; i < 4; i++)
    V2[i][i] += 1.0;
  double **V12 = new_matrix2(4,4);
  matrix_mult(V12, V1, V2, 4, 4, 4);

  // find a third orthogonal vector, B->V[1]
  double v3[4];
  for (i = 0; i < 4; i++)
    v3[i] = frand() - 0.5;
  matrix_vec_mult(B->V[1], V12, v3, 4, 4);
  normalize(B->V[1], B->V[1], 4);

  double **V3 = V1;  // B->V[1]'s projection matrix
  outer_prod(V3, B->V[1], B->V[1], 4, 4);
  mult(V3[0], V3[0], -1, 16);
  for (i = 0; i < 4; i++)
    V3[i][i] += 1.0;
  double **V123 = V2;
  matrix_mult(V123, V12, V3, 4, 4, 4);

  // find a fourth orthogonal vector, B->V[0]
  double v4[4];
  for (i = 0; i < 4; i++)
    v4[i] = frand() - 0.5;
  matrix_vec_mult(B->V[0], V123, v4, 4, 4);
  normalize(B->V[0], B->V[0], 4);

  // compute concentration params
  double pc1 = MAX(pcd->pc1[idx], epsilon);
  double pc2 = MAX(pcd->pc2[idx], epsilon);
  double z3 = MIN(10*(pc1/pc2 - 1), 100);
  B->Z[0] = -100;
  B->Z[1] = -100;
  B->Z[2] = -z3;

  // look up the normalization constant
  bingham_F(B);

  free_matrix2(V1);
  free_matrix2(V2);
  free_matrix2(V12);
}

void get_olf(olf_t *olf, pcd_t *pcd, int idx, int bingham)
{
  if (pcd->points) {
    if (olf->x == NULL)
      safe_calloc(olf->x, 3, double);
    memcpy(olf->x, pcd->points[idx], 3*sizeof(double));
  }

  if (pcd->quaternions) {
    if (olf->q == NULL)
      safe_calloc(olf->q, 4, double);
    memcpy(olf->q, pcd->quaternions[0][idx], 4*sizeof(double));

    if (bingham) {
      if (olf->B == NULL) {
	safe_calloc(olf->B, 1, bingham_t);
	bingham_alloc(olf->B, 4);
      }
      olf_to_bingham(olf->B, idx, pcd);
    }
  }
}

/*
void model_pose_from_one_correspondence(double *x, double *q, int c_obs, int c_model, pcd_t *pcd_obs, pcd_t *pcd_model)
{
  // compute rotation
  bingham_t B;
  bingham_alloc(&B, 4);
  olf_to_bingham(&B, c_model, pcd_model);
  int flip = (frand() < .5);
  double *q_feature_to_world = pcd_obs->quaternions[flip][c_obs];
  double q_feature_to_model[4], q_model_to_feature[4];
  double *q_feature_to_model_ptr[1] = {q_feature_to_model};

  bingham_sample(q_feature_to_model_ptr, &B, 1);
  //bingham_mode(q_feature_to_model, &B);

  quaternion_inverse(q_model_to_feature, q_feature_to_model);
  quaternion_mult(q, q_feature_to_world, q_model_to_feature);
  double **R = new_matrix2(3,3);
  quaternion_to_rotation_matrix(R, q);

  // compute translation
  double *obs_point = pcd_obs->points[c_obs];
  double *model_point = pcd_model->points[c_model];
  double model_point_rot[3];
  matrix_vec_mult(model_point_rot, R, model_point, 3, 3);
  sub(x, obs_point, model_point_rot, 3);

  // cleanup
  free_matrix2(R);
}
*/

/*
 * Use only least-squares information (not orientations)
 */
void get_model_pose_distribution_from_olf_correspondences_LS(double *x, bingham_t *B, olf_t *model_olfs, olf_t *obs_olfs, int n, double xyz_sigma)
{
  // compute olf centroids
  int i;
  double mean_model[3] = {0,0,0};
  double mean_obs[3] = {0,0,0};
  for (i = 0; i < n; i++) {
    add(mean_model, mean_model, model_olfs[i].x, 3);
    add(mean_obs, mean_obs, obs_olfs[i].x, 3);
  }
  mult(mean_model, mean_model, 1/(double)n, 3);
  mult(mean_obs, mean_obs, 1/(double)n, 3);

  // compute model position
  sub(x, mean_obs, mean_model, 3);
  
  // compute LS Bingham distributions for each point correspondence
  bingham_t B_ls[n];
  for (i = 0; i < n; ++i) {
    double v_obs[3], v_model[3];
    sub(v_obs, obs_olfs[i].x, mean_obs, 3);
    sub(v_model, model_olfs[i].x, mean_model, 3);
    double q_obs[4], q_model[4];
    vector_to_possible_quaternion(q_obs, v_obs);
    vector_to_possible_quaternion(q_model, v_model);
    double k = norm(v_obs, 3) * norm(v_model, 3) / (xyz_sigma * xyz_sigma);
    bingham_alloc(&B_ls[i], 4);
    B->d = 4;
    B->Z[0] = -2 * k;
    B->Z[1] = -2 * k;
    B->Z[2] = 0;
    double V_data[12] = {0, 0, 1, 0,   0, 0, 0, 1,   0, 1, 0, 0};
    memcpy(B->V[0], V_data, 12*sizeof(double));
    double q_inv[4];
    quaternion_inverse(q_inv, q_model);
    bingham_pre_rotate_3d(B, B, q_inv);
    bingham_post_rotate_3d(&B_ls[i], B, q_obs);
  }  

  bingham_mult_array(B, B_ls, n, 1);
  
  for (i = 0; i < n; ++i)
    bingham_free(&B_ls[i]);
}


/*
 * Compute x and p(q|x) by combining Bingham distributions from least-squares
 * alignment (LS) with Bingham distributions from OLFs.  The tuple (x,q)
 * where q is sampled from B indicates that one should rotate model_olfs
 * about its centroid by q, then shift by x.
 */
void get_model_pose_distribution_from_olf_correspondences(double *x, bingham_t *B, olf_t *model_olfs, olf_t *obs_olfs, int n, double xyz_sigma)
{
  bingham_t B_array[n+1];
  bingham_t *B_ls = &B_array[0];
  bingham_alloc(B_ls, 4);
  get_model_pose_distribution_from_olf_correspondences_LS(x, B_ls, model_olfs, obs_olfs, n, xyz_sigma);

  bingham_t *B_olf = &B_array[1];
  int i;
  for (i = 0; i < n; ++i)
    bingham_alloc(&B_olf[i], 4);
  bingham_t B_model_to_feature;
  bingham_alloc(&B_model_to_feature, 4);

  for (i = 0; i < n; ++i) {
    if (model_olfs[i].B == NULL)
      bingham_set_uniform(&B_olf[i]);
    else {
      bingham_invert_3d(&B_model_to_feature, model_olfs[i].B);
      double *q_feature_to_world = obs_olfs[i].q;
      bingham_post_rotate_3d(&B_olf[i], &B_model_to_feature, q_feature_to_world);

      // greedy check for principal curvature flips
      double q1[4], q2[4];
      quaternion_flip(q2, obs_olfs[i].q);
      bingham_post_rotate_3d(B, &B_model_to_feature, q2);
      bingham_mode(q1, &B_olf[i]);  // original
      bingham_mode(q2, B);          // flipped
	
      // pick whichever one has a higher likelihood w.r.t. LS distribution
      if (bingham_pdf(q1, B_ls) < bingham_pdf(q2, B_ls))
	bingham_copy(&B_olf[i], B);
    }
  }  

  bingham_mult_array(B, B_array, n+1, 1);

  for (i = 0; i < n+1; i++)
    bingham_free(&B_array[i]);
  bingham_free(&B_model_to_feature);
}

void sample_model_pose(pcd_t *pcd_model, int *c_model, int c, double *x0, bingham_t *B, double *x, double *q)
{
  double mu[3] = {0, 0, 0};

  int i;
  double point[3];
  for (i = 0; i < c; ++i) {
    get_point(point, pcd_model, c_model[i]);
    add(mu, mu, point, 3);
  }
  mult(mu, mu, 1/(double)c, 3);
  
  //bingham_sample(&q, B, 1);
  bingham_mode(q, B);
  double **R = new_matrix2(3,3);
  quaternion_to_rotation_matrix(R, q);

  // x = x0 + mu - R*mu
  matrix_vec_mult(x, R, mu, 3, 3);
  sub(x, mu, x, 3);
  add(x, x0, x, 3);

  free_matrix2(R);
}


void sample_model_pose_given_correspondences(scope_sample_t *sample, scope_model_data_t *model_data, scope_obs_data_t *obs_data, scope_params_t *params)
{
  // get model pose distribution
  double x0[3] = {0,0,0};
  bingham_t *B = &sample->B;
  get_model_pose_distribution_from_olf_correspondences(x0, B, sample->model_olfs, sample->obs_olfs, sample->nc, params->xyz_sigma);

  // sample a model orientation
  double *q = sample->q;
  //bingham_sample(q, B, 1);
  bingham_mode(q, B);
  double **R = new_matrix2(3,3);
  quaternion_to_rotation_matrix(R, q);

  // compute model position, x = x0 + mu - R*mu
  double *x = sample->x;
  double mu[3] = {0,0,0};
  int i;
  for (i = 0; i < sample->nc; i++)
    add(mu, mu, sample->model_olfs[i].x, 3);
  mult(mu, mu, 1/(double)sample->nc, 3);
  matrix_vec_mult(x, R, mu, 3, 3);
  sub(x, mu, x, 3);
  add(x, x0, x, 3);

  // cleanup
  free_matrix2(R);
}


double compute_olf_correspondence_score(scope_sample_t *sample, scope_params_t *params)
{
  double *x = sample->x;
  double *q = sample->q;
  bingham_t *B = &sample->B;
  //int *c_model = sample->c_model;
  //int *c_obs = sample->c_obs;
  int n = sample->nc;
  double xyz_sigma = params->xyz_sigma;
  double dispersion_weight = params->dispersion_weight;

  double **R = new_matrix2(3, 3);
  quaternion_to_rotation_matrix(R, q);
  double **xyz_obs = new_matrix2(n, 3);
  double **xyz_model = new_matrix2(n, 3);
  int i;
  for (i = 0; i < n; ++i) {
    memcpy(xyz_obs[i], sample->obs_olfs[i].x, 3*sizeof(double));
    memcpy(xyz_model[i], sample->model_olfs[i].x, 3*sizeof(double));
  }
  
  // compute the least squares likelihood
  double logp_ls = 0.;
  for (i = 0; i < n; ++i) {
    // dx = R*xyz_model[i] + x - xyz_obs[i]
    double dx[3];
    matrix_vec_mult(dx, R, xyz_model[i], 3, 3);
    add(dx, dx, x, 3);
    sub(dx, dx, xyz_obs[i], 3);
    logp_ls += log(normpdf(norm(dx,3), 0, xyz_sigma));
  }
  logp_ls /= (double)n;
  
  // compute the OLF orientation likelihood
  double logp_olf = log(bingham_pdf(q, B));
  
  /* compute FPFH likelihood
  double logp_fpfh = 0.;
  if (f_sigma > 0) {
    for (i = 0; i < n; ++i) {
      double d = dist(pcd_obs->shapes[c_obs[i]], pcd_model->shapes[c_model[i]], shape_length);
      logp_fpfh += log(normpdf(d, 0, f_sigma));
    }
    logp_fpfh /= (double)n;
  }
  */

  // compute dispersion likelihood
  double logp_disp = 0;
  if (dispersion_weight > 0) {
    double vars[3];
    variance(vars, xyz_model, n, 3);
    logp_disp = log(dispersion_weight * sum(vars, 3));
  }

  // compute the feature correspondence likelihood
  double logp_c = sum(sample->c_score, n) / (double)n;

  double logp = logp_ls + logp_olf + logp_disp + logp_c;

  /*
  if (logp_comp) {
    logp_comp[0] = logp_ls;
    logp_comp[1] = logp_olf;
    logp_comp[2] = logp_fpfh;
    logp_comp[3] = logp_disp;
  }
  */
  
  free_matrix2(xyz_obs);
  free_matrix2(xyz_model);
  free_matrix2(R);

  return logp;
}






//==============================================================================================//

//-------------------------------------  Correspondences  --------------------------------------//

//==============================================================================================//



int sample_obs_point_given_model_pose(scope_sample_t *sample, scope_model_data_t *model_data, scope_obs_data_t *obs_data)
{
  // TODO: make these params
  int num_steps = 3;  
  int hops = 2;
  int num_samples = 5;

  int nc = sample->nc;
  range_image_t *obs_fg_range_image = obs_data->obs_fg_range_image;
  double ** obs_edge_image = obs_data->obs_edge_image;
  int w = obs_fg_range_image->w;
  int h = obs_fg_range_image->h;

  // sample several new obs points candidates, and pick the one that's furthest away from the last obs point
  int c_obs_new[num_samples];
  double d2_obs[num_samples];  // dist. b/w obs points
  double *p_obs = obs_data->pcd_obs->points[sample->c_obs[nc-1]];
  int i, xi, yi;
  range_image_xyz2sub(&xi, &yi, obs_fg_range_image, p_obs);
  for (i = 0; i < num_samples; i++) {
    while (1) {
      int xi2=xi, yi2=yi;
      edge_image_random_walk(&xi2, &yi2, obs_edge_image, w, h, num_steps, hops);
      if (obs_fg_range_image->idx[xi2][yi2] >= 0) {
	c_obs_new[i] = obs_fg_range_image->idx[xi2][yi2];
	break;
      }
    }
    d2_obs[i] = dist2(p_obs, obs_data->pcd_obs->points[c_obs_new[i]], 3);
  }
  i = find_max(d2_obs, num_samples);

  return c_obs_new[i];
}


int sample_fpfh_model_point_given_model_pose(scope_sample_t *sample, scope_model_data_t *model_data, scope_obs_data_t *obs_data)
{
  pcd_t *fpfh_model = model_data->fpfh_model;
  double *model_cmf = model_data->fpfh_model_cmf;
  range_image_t *obs_range_image = obs_data->obs_range_image;

  double **R = new_matrix2(3,3);
  quaternion_to_rotation_matrix(R, sample->q);

  int i, xi, yi, c_model, found_point = 0;
  for (i = 0; i < 20; i++) {

    if (i < 10)  // try sampling from model_cmf
      c_model = cmfrand(model_cmf, fpfh_model->num_points);
    else  // if that doesn't work, try uniform sampling
      c_model = irand(fpfh_model->num_points);

    if (!ismemberi(c_model, sample->c_model, sample->nc)) {  // if a point is new
      double n[3], xyz[3];
      matrix_vec_mult(n, R, fpfh_model->normals[c_model], 3, 3);
      matrix_vec_mult(xyz, R, fpfh_model->points[c_model], 3, 3);
      add(xyz, xyz, sample->x, 3);

      if (dot(n, xyz, 3) < 0) {  // if normals are pointing towards camera
	if (obs_range_image == NULL ||
	    (range_image_xyz2sub(&xi, &yi, obs_range_image, xyz) && obs_range_image->image[xi][yi] > norm(xyz,3))) {  // if visible in range image
	  found_point = 1;
	  break;
	}
      }
    }
  }

  free_matrix2(R);

  return (found_point ? c_model : -1);
}

int sample_fpfh_obs_correspondence_given_model_pose(scope_sample_t *sample, int c_model, scope_model_data_t *model_data, scope_obs_data_t *obs_data, scope_params_t *params)
{
  pcd_t *fpfh_model = model_data->fpfh_model;
  double **obs_fxyzn = obs_data->obs_fxyzn;
  flann_index_t obs_xyzn_index = obs_data->obs_xyzn_index;
  struct FLANNParameters *obs_xyzn_params = &obs_data->obs_xyzn_params;
  int shape_length = fpfh_model->shape_length;

  double **R = new_matrix2(3,3);
  quaternion_to_rotation_matrix(R, sample->q);

  double mp_pos[3];
  get_point(mp_pos, fpfh_model, c_model);
  matrix_vec_mult(mp_pos, R, mp_pos, 3, 3);
  add(mp_pos, mp_pos, sample->x, 3);

  double mp_norm[3];
  get_normal(mp_norm, fpfh_model, c_model);
  matrix_vec_mult(mp_norm, R, mp_norm, 3, 3);

  double *mp_shape = fpfh_model->shapes[c_model];
  
  // Look for k-NN in xyz-normal space
  double xyzn_query[6];
  mult(xyzn_query, mp_pos, params->xyz_weight, 3);
  mult(&xyzn_query[3], mp_norm, params->normal_weight, 3);
  int nn_idx[params->knn];
  double nn_d2[params->knn];
  flann_find_nearest_neighbors_index_double(obs_xyzn_index, xyzn_query, 1, nn_idx, nn_d2, params->knn, obs_xyzn_params);
  
  // then compute full feature distance on just those k-NN
  double query[shape_length + 6];
  memcpy(query, mp_shape, shape_length*sizeof(double));
  memcpy(&query[shape_length], xyzn_query, 6*sizeof(double));

  int i;
  double p[params->knn];
  for (i = 0; i < params->knn; i++) {
    double d2 = dist2(query, obs_fxyzn[nn_idx[i]], shape_length + 6);
    p[i] = exp(.5*d2 / params->f_sigma);
  }
  normalize_pmf(p, p, params->knn);
  int c_obs = nn_idx[pmfrand(p, params->knn)];

  free_matrix2(R);
 
  return c_obs;
}


int sample_fpfh_model_correspondence_given_model_pose(scope_sample_t *sample, int c_obs, scope_model_data_t *model_data, scope_obs_data_t *obs_data, scope_params_t *params, int sample_nn, int use_f)
{
  pcd_t *pcd_obs = obs_data->pcd_obs;
  //double **fpfh_model_xyzn = model_data->fpfh_model_xyzn;
  double **fpfh_model_fxyzn = model_data->fpfh_model_fxyzn;
  struct FLANNParameters *fpfh_model_xyzn_params = &model_data->fpfh_model_xyzn_params;
  flann_index_t fpfh_model_xyzn_index = model_data->fpfh_model_xyzn_index;
  int shape_length = pcd_obs->shape_length;

  int i;
  double q_inv[4];
  quaternion_inverse(q_inv, sample->q);
  double **inv_R_model = new_matrix2(3, 3);
  quaternion_to_rotation_matrix(inv_R_model, q_inv);

  double xyz_obs_point2[3];
  get_point(xyz_obs_point2, pcd_obs, c_obs);
  sub(xyz_obs_point2, xyz_obs_point2, sample->x, 3);
  matrix_vec_mult(xyz_obs_point2, inv_R_model, xyz_obs_point2, 3, 3);

  double nxyz_obs_point2[3];
  get_normal(nxyz_obs_point2, pcd_obs, c_obs);
  matrix_vec_mult(nxyz_obs_point2, inv_R_model, nxyz_obs_point2, 3, 3);
  
  // look for k-NN in xyz-normal space
  double xyzn_query[6];
  mult(xyzn_query, xyz_obs_point2, params->xyz_weight, 3);
  mult(&xyzn_query[3], nxyz_obs_point2, params->normal_weight, 3);
  int nn_idx[params->knn];
  double nn_d2[params->knn];

  flann_find_nearest_neighbors_index_double(fpfh_model_xyzn_index, xyzn_query, 1, nn_idx, nn_d2, params->knn, fpfh_model_xyzn_params);
  //flann_find_nearest_neighbors_index_double(fpfh_model_xyz_index, xyz_obs_point2, 1, nn_idx, nn_d2, params->knn, fpfh_model_xyz_params);
  //for (i = 0; i < params->knn; i++)
  //  nn_d2[i] = dist2(xyzn_query, fpfh_model_xyzn[nn_idx[i]], 6);

  if (use_f) {
    // then compute full feature distance on just those k-NN
    double query[shape_length + 6];
    memcpy(query, pcd_obs->shapes[c_obs], shape_length * sizeof(double));
    for (i = 0; i < 3; i++) {
      query[shape_length + i] = params->xyz_weight * xyz_obs_point2[i];
      query[shape_length + 3 + i] = params->normal_weight * nxyz_obs_point2[i];
    }
    for (i = 0; i < params->knn; i++)
      nn_d2[i] = dist2(query, fpfh_model_fxyzn[nn_idx[i]], shape_length + 6);
  }

  int c_model;
  if (sample_nn)
    c_model = nn_idx[find_min(nn_d2, params->knn)];
  else {
    double p[params->knn];
    for (i = 0; i < params->knn; i++)
      p[i] = exp(-.5*nn_d2[i] / (params->f_sigma * params->f_sigma));
    normalize_pmf(p, p, params->knn);
    c_model = nn_idx[pmfrand(p, params->knn)];
  }

  return c_model;
}


void sample_fpfh_correspondence_given_model_pose(scope_sample_t *sample, scope_model_data_t *model_data, scope_obs_data_t *obs_data, scope_params_t *params)
{
  // TODO: make this a param
  double model_first_ratio = .5;

  int nc = sample->nc;
  int c_model, c_obs;
  int correspondence_found = 0;

  // find a correspondence
  if (frand() < model_first_ratio) {
    c_model = sample_fpfh_model_point_given_model_pose(sample, model_data, obs_data);
    if (c_model >= 0) {
      c_obs = sample_fpfh_obs_correspondence_given_model_pose(sample, c_model, model_data, obs_data, params);
      correspondence_found = 1;
    }
  }
  if (!correspondence_found) {
    c_obs = sample_obs_point_given_model_pose(sample, model_data, obs_data);
    c_model = sample_fpfh_model_correspondence_given_model_pose(sample, c_obs, model_data, obs_data, params, 1, 0);  //nn, fpfh
  }

  // compute correspondence score
  double d = dist(obs_data->pcd_obs->shapes[c_obs], model_data->fpfh_model->shapes[c_model], model_data->fpfh_model->shape_length);
  double c_score = log(normpdf(d, 0, params->f_sigma));

  sample->c_obs[nc] = c_obs;
  sample->c_model[nc] = c_model;
  sample->c_score[nc] = c_score;
  sample->c_type[nc] = C_TYPE_FPFH;
  sample->nc++;
}


void sample_edge_correspondence_given_model_pose(int *c_obs, int *c_model, int nc, double *x, double *q)
{
  //STUB

  // TODO: make these params
  //double model_first_ratio = .5;

  //if (frand() < model_first_ratio) {
  //  c_model[nc] = sample_model_edge_point_given_model_pose(x, q, pcd_model, obs_range_image);
  //}
}


void sample_correspondence_given_model_pose(scope_sample_t *sample, scope_model_data_t *model_data, scope_obs_data_t *obs_data, scope_params_t *params)
{
  // fpfh
  sample_fpfh_correspondence_given_model_pose(sample, model_data, obs_data, params);
  int i = sample->nc - 1;

  get_olf(&sample->model_olfs[i], model_data->fpfh_model, sample->c_model[i], 1);
  get_olf(&sample->obs_olfs[i], obs_data->pcd_obs, sample->c_obs[i], 0);

  // sift
  //get_olf(&model_olfs[i], model_data->sift_model, sample->c_model[i], 1);
  //get_olf(&obs_olfs[i], obs_data->sift_obs, sample->c_obs[i], 0);

  // edge

}


void resample_model_correspondences(scope_sample_t *sample, scope_model_data_t *model_data, scope_obs_data_t *obs_data, scope_params_t *params)
{
  int i;
  for (i = 0; i < sample->nc; i++) {
    if (sample->c_type[i] == C_TYPE_FPFH) {
      sample->c_model[i] = sample_fpfh_model_correspondence_given_model_pose(sample, sample->c_obs[i], model_data, obs_data, params, 1, 1);  //nn, fpfh
      get_olf(&sample->model_olfs[i], model_data->fpfh_model, sample->c_model[i], 1);
    }
    else if (sample->c_type[i] == C_TYPE_SIFT) {
      //STUB
    }
    else if (sample->c_type[i] == C_TYPE_EDGE) {
      //STUB
    }
  }
}






//==============================================================================================//

//-------------------------------------  Dense Alignments  -------------------------------------//

//==============================================================================================//

/*
void align_model_icp_dense(double *x, double *q, pcd_t *pcd_model, pcd_t *pcd_obs, range_image_t *obs_range_image,
			   range_image_t *obs_fg_range_image, scope_params_t *params, int max_iter, int num_icp_points)
{
  int i, iter;
  //int max_iter = 20;  //TODO: make this a param
  //int num_icp_points = 500;  // TODO: make this a param
  double x0[3];
  bingham_t B;
  bingham_alloc(&B, 4);

  for (iter = 0; iter < max_iter; iter++) {

    // get model icp points
    int c_model[num_icp_points];
    get_validation_points(c_model, pcd_model, num_icp_points);

    // extract transformed model icp features
    double **cloud = get_sub_cloud_at_pose(pcd_model, c_model, num_icp_points, x, q);
    double **cloud_normals = get_sub_cloud_normals_rotated(pcd_model, c_model, num_icp_points, q);

    // compute visibility mask and remove hidden model points
    int vis_mask[num_icp_points];
    for (i = 0; i < num_icp_points; i++) {
      double vis_prob = compute_visibility_prob(cloud[i], cloud_normals[i], obs_range_image, params->vis_thresh, 0);
      vis_mask[i] = (vis_prob > .1);
    }
    int idx[num_icp_points];
    int n = find(idx, vis_mask, num_icp_points);

    if (n < 4) {
      printf("n < 4 in ICP...aborting\n");
      free_matrix2(cloud);
      free_matrix2(cloud_normals);
      break;
    }

    reorder_rows(cloud, cloud, idx, n, 3);
    reorder_rows(cloud_normals, cloud_normals, idx, n, 3);
    reorderi(c_model, c_model, idx, n);

    // compute nearest neighbors
    int c_obs[n];
    double nn_d2[n];
    int search_radius = 5;  // pixels
    range_image_find_nn(c_obs, nn_d2, cloud, cloud, n, 3, pcd_obs->points, obs_fg_range_image, search_radius);

    // update model pose
    get_model_pose_distribution_from_correspondences_LS(pcd_obs, pcd_model, c_obs, c_model, n, params->xyz_sigma, x0, &B);
    sample_model_pose(pcd_model, c_model, n, x0, &B, x, q);

    free_matrix2(cloud);
    free_matrix2(cloud_normals);
  }
}


// get the jacobian of R*x w.r.t. q
double **point_rotation_jacobian(double *q, double *x)
{
  double q1 = q[0];
  double q2 = q[1];
  double q3 = q[2];
  double q4 = q[3];
  
  double v1 = x[0];
  double v2 = x[1];
  double v3 = x[2];

  double dp_dq_data[12] = {2*(q1*v1 + q3*v3 - q4*v2),  2*(q2*v1 + q3*v2 + q4*v3),  2*(q1*v3 + q2*v2 - q3*v1),  2*(q2*v3 - q1*v2 - q4*v1),
			   2*(q1*v2 - q2*v3 + q4*v1),  2*(q3*v1 - q2*v2 - q1*v3),  2*(q2*v1 + q3*v2 + q4*v3),  2*(q1*v1 + q3*v3 - q4*v2),
			   2*(q1*v3 + q2*v2 - q3*v1),  2*(q1*v2 - q2*v3 + q4*v1),  2*(q4*v2 - q3*v3 - q1*v1),  2*(q2*v1 + q3*v2 + q4*v3)};

  return new_matrix2_data(3, 4, dp_dq_data);
}

double **range_image_pixel_pose_jacobian(range_image_t *range_image, double *model_point, double *x, double *q)
{
  // transform model point by model pose (x,q)
  double p[3];
  double **R = new_matrix2(3,3);
  quaternion_to_rotation_matrix(R,q);
  matrix_vec_mult(p, R, model_point, 3, 3);
  add(p, p, x, 3);
  free_matrix2(R);
  double p1 = p[0];
  double p2 = p[1];
  double p3 = p[2];
  double r = norm(p,3);

  // compute jacobian of (i,j) w.r.t. p (and x)
  double ci = 1.0 / (p1*p1 + p3*p3);
  double cj = r * sqrt(ci);
  double dij_dp_data[6] = {ci*p3, 0, -ci*p1,  cj*p1*p2, cj*(p2*p2-r*r), cj*p2*p3};
  double **dij_dp = new_matrix2_data(2, 3, dij_dp_data);

  // compute jacobian of (i,j) w.r.t. q
  double **dp_dq = point_rotation_jacobian(q, model_point);
  double **dij_dq = new_matrix2(2,4);
  matrix_mult(dij_dq, dij_dp, dp_dq, 2, 3, 4);
  free_matrix2(dp_dq);

  // copy jacobians into dij_dxq
  double **dij_dxq = new_matrix2(2,7);
  memcpy(&dij_dxq[0][0], dij_dp[0], 3*sizeof(double));
  memcpy(&dij_dxq[1][0], dij_dp[1], 3*sizeof(double));
  memcpy(&dij_dxq[0][3], dij_dq[0], 4*sizeof(double));
  memcpy(&dij_dxq[1][3], dij_dq[1], 4*sizeof(double));

  // divide by range image resolution
  mult(dij_dxq[0], dij_dxq[0], 1/range_image->res, 14);

  free_matrix2(dij_dp);
  free_matrix2(dij_dq);

  return dij_dxq;
}

double edge_score_gradient(double *G, double *x, double *q, double **P, int n, range_image_t *obs_range_image, double **obs_edge_image, scope_params_t *params)
{
  double **P2 = new_matrix2(n,3);
  transform_cloud(P2, P, n, x, q);

  // compute visibility of sampled model edges
  int vis_pixel_radius = 2;
  double vis_prob[n];
  int i;
  for (i = 0; i < n; i++)
    vis_prob[i] = compute_visibility_prob(P2[i], NULL, obs_range_image, params->vis_thresh, vis_pixel_radius);
  double vis_pmf[n];
  normalize_pmf(vis_pmf, vis_prob, n);

  double score = 0.0;
  memset(G, 0, 7*sizeof(double));

  for (i = 0; i < n; i++) {

    if (vis_prob[i] < .01)
      continue;

    // get edge pixel
    int xi, yi;
    if (!range_image_xyz2sub(&xi, &yi, obs_range_image, P2[i]))
      continue;

    // add pixel edge score to total score
    score += vis_pmf[i] * obs_edge_image[xi][yi];

    // get edge image gradient at current pixel
    double dI_dij[2];
    matrix_cell_gradient(dI_dij, xi, yi, obs_edge_image, obs_range_image->w, obs_range_image->h);

    // get gradient of pixel location w.r.t. model pose (x,q)
    double **dij_dxq = range_image_pixel_pose_jacobian(obs_range_image, P[i], x, q);

    // get gradient of this point's edge score w.r.t. model pose (x,q)
    double dI_dxq[7];
    vec_matrix_mult(dI_dxq, dI_dij, dij_dxq, 2, 7);
    mult(dI_dxq, dI_dxq, vis_pmf[i], 7);

    // add this pixel's gradient to total gradient
    add(G, G, dI_dxq, 7);

    // cleanup
    free_matrix2(dij_dxq);
  }
  
  free_matrix2(P2);

  mult(G, G, params->score2_edge_weight, 7);

  return params->score2_edge_weight * score;
}

double xyzn_score_gradient(double *G, double *x, double *q, double **cloud, double **normals, scope_noise_model_t *noise_models, int n, range_image_t *obs_range_image, scope_params_t *params)
{
  //double range_sigma = params->range_sigma;
  //double normal_sigma = params->normal_sigma;
  //double dmax_xyz = 2*range_sigma;  // TODO: make this a param
  //double dmax_normal = 2*normal_sigma;  // TODO: make this a param

  double **cloud2 = new_matrix2(n,3);
  double **normals2 = new_matrix2(n,3);

  // transform surface points by (x,q)
  transform_cloud(cloud2, cloud, n, x, q);
  transform_cloud(normals2, normals, n, NULL, q);

  // compute p(visibile) for surface points
  double vis_prob[n];
  int i;
  for (i = 0; i < n; i++)
    vis_prob[i] = compute_visibility_prob(cloud2[i], normals2[i], obs_range_image, params->vis_thresh, 0);
  double vis_pmf[n];
  normalize_pmf(vis_pmf, vis_prob, n);

  double score = 0.0;
  memset(G, 0, 7*sizeof(double));

  for (i = 0; i < n; i++) {

    if (vis_prob[i] < .01)
      continue;

    // get range image cell
    int xi, yi;
    if (!range_image_xyz2sub(&xi, &yi, obs_range_image, cloud2[i]))
      continue;

    double range_sigma = params->range_sigma * noise_models[i].range_sigma;
    double normal_sigma = params->normal_sigma * noise_models[i].normal_sigma;
    double dmax_xyz = 2*range_sigma;
    double dmax_normal = 2*normal_sigma;
    double d_xyz = dmax_xyz;
    double d_normal = dmax_normal;
    double c[4];    // range image cell plane coeffs
    if (obs_range_image->cnt[xi][yi] > 0) {
      // get distance from model point to range image cell plane
      xyzn_to_plane(c, obs_range_image->points[xi][yi], obs_range_image->normals[xi][yi]);
      d_xyz = fabs(dot(c, cloud2[i], 3) + c[3]);
      //d_xyz /= noise_models[i].range_sigma;
      d_xyz = MIN(d_xyz, dmax_xyz);
      d_normal = 1.0 - dot(normals2[i], obs_range_image->normals[xi][yi], 3);
      //d_normal /= noise_models[i].normal_sigma;
      d_normal = MIN(d_normal, dmax_normal);
    }
    score += params->score2_xyz_weight * vis_pmf[i] * log(normpdf(d_xyz, 0, range_sigma));
    score += params->score2_normal_weight * vis_pmf[i] * log(normpdf(d_normal, 0, normal_sigma));

    // get gradient of this point's xyz score w.r.t. model pose (x,q)
    if (d_xyz < dmax_xyz) {
      double **dp_dq = point_rotation_jacobian(q, cloud[i]);
      double df_dp[3];
      memcpy(df_dp, c, 3*sizeof(double));
      //double rs = range_sigma * noise_models[i].range_sigma;
      mult(df_dp, df_dp, -(c[3] + dot(cloud2[i], c, 3)) / (range_sigma*range_sigma), 3);
      double G_xyz[7];
      memcpy(&G_xyz[0], df_dp, 3*sizeof(double));
      vec_matrix_mult(&G_xyz[3], df_dp, dp_dq, 3, 4);
      free_matrix2(dp_dq);
      mult(G_xyz, G_xyz, params->score2_xyz_weight * vis_pmf[i], 7);
      add(G, G, G_xyz, 7);
    }

    // get gradient of this point's normal score w.r.t. model pose (x,q)
    if (d_normal < dmax_normal) {
      double **dpn_dq = point_rotation_jacobian(q, normals[i]);
      double df_dpn[3];
      memcpy(df_dpn, c, 3*sizeof(double));
      //double ns = normal_sigma * noise_models[i].normal_sigma;
      mult(df_dpn, df_dpn, (1 - dot(normals2[i], c, 3)) / (normal_sigma*normal_sigma), 3);
      double G_normal[7] = {0,0,0,0,0,0,0};
      vec_matrix_mult(&G_normal[3], df_dpn, dpn_dq, 3, 4);
      free_matrix2(dpn_dq);
      mult(G_normal, G_normal, params->score2_normal_weight * vis_pmf[i], 7);
      add(G, G, G_normal, 7);
    }
  }

  free_matrix2(cloud2);
  free_matrix2(normals2);

  return score;
}


void align_model_gradient(double *x, double *q, pcd_t *pcd_model, multiview_pcd_t *range_edges_model,
			  range_image_t *obs_range_image, double **obs_edge_image, scope_params_t *params, int max_iter, int num_points)
{
  //double t0 = get_time_ms();  //dbug

  // compute model edge points
  int num_edge_points = num_points;
  double **P = get_range_edge_points(&num_edge_points, x, q, range_edges_model, obs_range_image);
  double **P2 = new_matrix2(num_edge_points, 3);
  
  // get model surface points
  int num_surface_points = (num_points > 0 ? num_points : pcd_model->num_points);
  int idx[num_surface_points];
  get_validation_points(idx, pcd_model, num_surface_points);
  double **cloud = new_matrix2(num_surface_points, 3);
  double **normals = new_matrix2(num_surface_points, 3);
  reorder_rows(cloud, pcd_model->points, idx, num_surface_points, 3);
  reorder_rows(normals, pcd_model->normals, idx, num_surface_points, 3);
  double **cloud2 = new_matrix2(num_surface_points, 3);
  double **normals2 = new_matrix2(num_surface_points, 3);

  //dbug
  //printf("break 1, %.2f ms\n", get_time_ms() - t0);
  //t0 = get_time_ms();

  double step = .01;  // step size in gradient ascent
  int i, j, iter;
  for (iter = 0; iter < max_iter; iter++) {
    
    // compute noise models
    scope_noise_model_t *noise_models = get_noise_models(x, q, idx, num_surface_points, pcd_model, range_edges_model);

    //dbug
    //printf("break 1.5, %.2f ms\n", get_time_ms() - t0);
    //t0 = get_time_ms();

    // compute score and its gradient w.r.t. model pose
    double G_edge[7], G_xyzn[7], G[7];
    double edge_score = edge_score_gradient(G_edge, x, q, P, num_edge_points, obs_range_image, obs_edge_image, params);
    double xyzn_score = xyzn_score_gradient(G_xyzn, x, q, cloud, normals, noise_models, num_surface_points, obs_range_image, params);
    //double current_score = edge_score + xyzn_score;
    add(G, G_edge, G_xyzn, 7);

    //dbug
    //printf("break 2, %.2f ms\n", get_time_ms() - t0);
    //t0 = get_time_ms();

    //dbug: disable orientation gradients
    //G[3] = G[4] = G[5] = G[6] = 0;

    //printf("edge_score = %f, xyzn_score = %f\n", edge_score, xyzn_score);  //dbug
    //printf("G_edge = [%f, %f, %f, %f, %f, %f, %f]\n", G_edge[0], G_edge[1], G_edge[2], G_edge[3], G_edge[4], G_edge[5], G_edge[6]);
    //printf("G_xyzn = [%f, %f, %f, %f, %f, %f, %f]\n", G_xyzn[0], G_xyzn[1], G_xyzn[2], G_xyzn[3], G_xyzn[4], G_xyzn[5], G_xyzn[6]);

    // line search
    double step_mult[3] = {.6, 1, 1.6};
    double best_score = -10000000.0;
    double best_step=0.0, best_x[3], best_q[4];
    for (j = 0; j < 3; j++) {

      //double t1 = get_time_ms();  //dbug

      // take a step in the direction of the gradient
      double x2[3], q2[4], dxq[7];
      normalize(G, G, 7);
      mult(dxq, G, step*step_mult[j], 7);
      add(x2, x, &dxq[0], 3);
      add(q2, q, &dxq[3], 4);
      normalize(q2, q2, 4);

      //printf("x2 = [%f, %f, %f], q2 = [%f, %f, %f]\n", x2[0], x2[1], x2[2], q2[0], q2[1], q2[2], q2[3]); //dbug

      // transform surface points by (x2,q2)
      transform_cloud(cloud2, cloud, num_surface_points, x2, q2);
      transform_cloud(normals2, normals, num_surface_points, NULL, q2);

      //printf("transform_cloud: %.2f ms\n", get_time_ms() - t1); //dbug
      //t1 = get_time_ms();  //dbug

      // compute p(visibile) for surface points
      double vis_prob[num_surface_points];
      for (i = 0; i < num_surface_points; i++)
	vis_prob[i] = compute_visibility_prob(cloud2[i], normals2[i], obs_range_image, params->vis_thresh, 0);
      double vis_pmf[num_surface_points];
      normalize_pmf(vis_pmf, vis_prob, num_surface_points);

      //printf("vis_prob: %.2f ms\n", get_time_ms() - t1); //dbug
      //t1 = get_time_ms();  //dbug

      // transform edge points
      transform_cloud(P2, P, num_edge_points, x2, q2);

      //printf("transform edge points: %.2f ms\n", get_time_ms() - t1); //dbug
      //t1 = get_time_ms();  //dbug

      // evaluate the score
      edge_score = compute_edge_score(P2, num_edge_points, NULL, 0, obs_range_image, obs_edge_image, params, 1);
      
      //printf("compute_edge_score: %.2f ms\n", get_time_ms() - t1); //dbug
      //t1 = get_time_ms();  //dbug

      double xyz_score = compute_xyz_score(cloud2, vis_pmf, noise_models, num_surface_points, obs_range_image, params, 1);
      double normal_score = compute_normal_score(cloud2, normals2, vis_pmf, noise_models, num_surface_points, obs_range_image, params, 1);
      double score = edge_score + xyz_score + normal_score;

      //printf("xyz/normal scores: %.2f ms\n", get_time_ms() - t1); //dbug
      //t1 = get_time_ms();  //dbug

      if (score > best_score) {
	best_score = score;
	memcpy(best_x, x2, 3*sizeof(double));
	memcpy(best_q, q2, 4*sizeof(double));
	best_step = step*step_mult[j];
      }

      //dbug
      //printf("break 3, %.2f ms\n", get_time_ms() - t0);
      //t0 = get_time_ms();
    }

    //printf("best_score = %f, current_score = %f\n", best_score, current_score); //dbug

    // termination criterion
    //if (best_score <= current_score)
    //  break;

    memcpy(x, best_x, 3*sizeof(double));
    memcpy(q, best_q, 4*sizeof(double));
    step = best_step;

    //printf("iter = %d, best_score > current_score, step = %f\n", iter, step); //dbug
    //printf("step = %f\n", step);  //dbug

    free(noise_models);
  }


  // cleanup
  free_matrix2(P);
  free_matrix2(P2);
  free_matrix2(cloud);
  free_matrix2(cloud2);
  free_matrix2(normals);
  free_matrix2(normals2);
}
*/




//==============================================================================================//

//------------------------------------------  SCOPE  -------------------------------------------//

//==============================================================================================//


scope_samples_t *scope_round1(scope_model_data_t *model_data, scope_obs_data_t *obs_data, scope_params_t *params)
{
  double t0 = get_time_ms(); //dbug

  // find sift matches
  int sift_match_obs_idx[obs_data->sift_obs->num_points], sift_match_model_idx[obs_data->sift_obs->num_points];
  int num_sift_matches = find_obs_sift_matches(sift_match_obs_idx, sift_match_model_idx, obs_data->sift_obs, model_data->sift_model, params->sift_dthresh);

  // get the max number of samples
  int num_samples_init = num_sift_matches + (params->num_samples_init == 0 ? obs_data->pcd_obs->num_points : params->num_samples_init);
  int max_num_samples = MAX(num_samples_init, params->num_samples * params->branching_factor);

  // create scope samples
  scope_samples_t *S = create_scope_samples(max_num_samples, params->num_correspondences);
  S->num_samples = num_samples_init;

  // sample poses from one correspondence
  int i,j;
  for (i = 0; i < num_samples_init; i++) {

    if (i < num_sift_matches) {  // sift correspondence

      int c_obs = sift_match_obs_idx[i];
      int c_model = sift_match_model_idx[i];
      S->samples[i].c_obs[0] = c_obs;
      S->samples[i].c_model[0] = c_model;
      S->samples[i].c_score[0] = 1.0;            //TODO: compute SIFT correspondence likelihood
      S->samples[i].c_type[0] = C_TYPE_SIFT;
      S->samples[i].nc = 1;

      get_olf(&S->samples[i].model_olfs[0], model_data->sift_model, c_model, 1);
      get_olf(&S->samples[i].obs_olfs[0], obs_data->sift_obs, c_obs, 0);
    }

    else {  // fpfh correspondence

      // get obs point
      int c_obs = irand(obs_data->pcd_obs->num_points);
      S->samples[i].c_obs[0] = c_obs;

      // get model point
      int nn_idx[params->knn];
      double nn_d2[params->knn];
      flann_find_nearest_neighbors_index_double(model_data->fpfh_model_f_index, obs_data->pcd_obs->shapes[c_obs], 1, nn_idx, nn_d2, params->knn, &model_data->fpfh_model_f_params);
      double p[params->knn];
      for (j = 0; j < params->knn; j++)
	p[j] = exp(-.5*nn_d2[j] / (params->f_sigma * params->f_sigma));
	//p[j] = exp(-.5*nn_d2[j] / (params->fsurf_sigma * params->fsurf_sigma));
      normalize_pmf(p, p, params->knn);
      int j = pmfrand(p, params->knn);
      int c_model = nn_idx[j];

      S->samples[i].c_obs[0] = c_obs;
      S->samples[i].c_model[0] = c_model;
      S->samples[i].c_type[0] = C_TYPE_FPFH;
      S->samples[i].nc = 1;

      // compute correspondence score
      S->samples[i].c_score[0] = log(normpdf(sqrt(nn_d2[j]), 0, params->f_sigma));

      get_olf(&S->samples[i].model_olfs[0], model_data->fpfh_model, c_model, 1);
      get_olf(&S->samples[i].obs_olfs[0], obs_data->pcd_obs, c_obs, 0);
      if (frand() > .5)
	quaternion_flip(S->samples[i].obs_olfs[0].q, S->samples[i].obs_olfs[0].q);
    }

    // sample a pose
    sample_model_pose_given_correspondences(&S->samples[i], model_data, obs_data, params);
  }

  //dbug
  printf("Sampled c=1 poses in %.3f seconds\n", (get_time_ms() - t0) / 1000.0);  //dbug
  if (have_true_pose_)
    print_good_poses(S);

  // score hypotheses
  t0 = get_time_ms();
  for (i = 0; i < num_samples_init; i++)
    S->W[i] = model_placement_score(&S->samples[i], model_data, obs_data, params, 2);

  // sort hypotheses
  sort_pose_samples(S);

  //TODO: make this a param
  //double round1_score_thresh = -.2;
  //S->num_samples = find_first_lt(S->W, round1_score_thresh, S->num_samples);
  S->num_samples = MIN(S->num_samples, params->num_samples);

  //dbug
  printf("Scored and sorted c=1 poses in %.3f seconds\n", (get_time_ms() - t0) / 1000.0);  //dbug
  if (have_true_pose_)
    print_good_poses(S);

  return S;
}


void scope_round2(scope_samples_t *S, scope_model_data_t *model_data, scope_obs_data_t *obs_data, scope_params_t *params)
{
  double t0;
  int c,i;
  for (c = 1; c < params->num_correspondences; ++c) {

    t0 = get_time_ms();  //dbug

    for (i = 0; i < S->num_samples; ++i) {

      scope_sample_t sample;
      scope_sample_alloc(&sample, S->samples[i].nc + 1);
      scope_sample_copy(&sample, &S->samples[i]);

      // sample a new correspondence
      sample_correspondence_given_model_pose(&sample, model_data, obs_data, params);

      // sample a new model pose (and bingham distribution) given the correspondences
      sample_model_pose_given_correspondences(&sample, model_data, obs_data, params);

      // adjust model correspondences to maximize olf correspondence likelihood
      if (params->do_icp) {
	scope_sample_t new_sample;
	scope_sample_alloc(&new_sample, sample.nc);
	scope_sample_copy(&new_sample, &sample);

	double logp = compute_olf_correspondence_score(&new_sample, params);

	int k, icp_num_iter = 20;  //TODO: make this a param
	for (k = 0; k < icp_num_iter; k++) {

	  resample_model_correspondences(&new_sample, model_data, obs_data, params);
	  sample_model_pose_given_correspondences(&new_sample, model_data, obs_data, params);
	  double new_logp = compute_olf_correspondence_score(&new_sample, params);

	  if (new_logp > logp) {
	    scope_sample_copy(&sample, &new_sample);
	    logp = new_logp;
	  }
	  else  // stop ICP when model pose likelihood doesn't increase anymore
	    break;
	}
	scope_sample_free(&new_sample);
      }

      // only add correspondence when it increases the model placement score
      double score = model_placement_score(&sample, model_data, obs_data, params, 2);
      if (score > S->W[i]) {
	scope_sample_copy(&S->samples[i], &sample);
	S->W[i] = score;
      }

      scope_sample_free(&sample);
    }
    
    printf("Sampled c=%d poses in %.3f seconds\n", c+1, (get_time_ms() - t0) / 1000.0);  //dbug
    if (have_true_pose_)
      print_good_poses(S);
  }

  // score hypotheses
  //t0 = get_time_ms();
  //for (i = 0; i < S->num_samples; i++)
  //  S->W[i] = model_placement_score(&S->samples[i], model_data, obs_data, params, 2);
  sort_pose_samples(S);
  //printf("Scored and sorted c=%d poses in %.3f seconds\n", c, (get_time_ms() - t0) / 1000.0);  //dbug

  // remove low-weight samples
  S->num_samples = MIN(S->num_samples, params->num_samples);

  if (have_true_pose_)
    print_good_poses(S);
}


/*
 * Single Cluttered Object Pose Estimation (SCOPE)
 */
scope_samples_t *scope(olf_model_t *model, olf_obs_t *obs, scope_params_t *params, short have_true_pose, simple_pose_t *true_pose)
{
  int i;
  //const double MIN_SCORE = -100000.0;

  //dbug
  if (have_true_pose) {
    have_true_pose_ = 1;
    x_true_ = true_pose->X;
    q_true_ = true_pose->Q;
  }

  // get data
  scope_model_data_t *model_data = get_scope_model_data(model, params);
  scope_obs_data_t *obs_data = get_scope_obs_data(obs, params);

  double t0 = get_time_ms();  //dbug

  // step 1: sample initial poses given single correspondences
  scope_samples_t *S = scope_round1(model_data, obs_data, params);

  // step 2: align with BPA
  scope_round2(S, model_data, obs_data, params);

  printf("Ran scope in %.3f seconds\n", (get_time_ms() - t0) / 1000.0);  //dbug

  //dbug
  params->verbose = 1;
  params->num_validation_points = 0;
  for (i = 0; i < S->num_samples; i++)
    S->W[i] = model_placement_score(&S->samples[i], model_data, obs_data, params, 3);
  sort_pose_samples(S);

  // cluster poses
  //if (params->pose_clustering)
  //  remove_redundant_pose_samples(S, params);

  if (have_true_pose_)
    print_good_poses(S);

  // cleanup
  free_scope_model_data(model_data);
  free_scope_obs_data(obs_data);
  
  return S;
}

























  /*
  //----------------------------  BEGIN FINAL ALIGNMENTS / PRUNING -----------------------------//


  // params
  double w_sigma = 1.0;  //TODO: make this a parameter
  double w_sigma_pre_align = .1;
  int num_score_trials = 10;
  double w_sigma_multi = w_sigma_pre_align + w_sigma / sqrt((double)num_score_trials);


  //dbug
  t0 = get_time_ms();


  //dbug: score hypotheses and return
  olf_pose_samples_t *pose_samples;
  safe_calloc(pose_samples, 1, olf_pose_samples_t);
  pose_samples->X = X;
  pose_samples->Q = Q;
  pose_samples->W = W;
  pose_samples->n = num_samples;
  params->verbose = 1;
  pose_samples->num_scores = 10;
  pose_samples->scores = new_matrix2(pose_samples->n, pose_samples->num_scores);
  for (i = 0; i < pose_samples->n; i++) {
    W[i] = model_placement_score(X[i], Q[i], pcd_model, range_edges_model, model_xyz_index, &model_xyz_params, pcd_obs_bg, obs_range_image, obs_edge_image, obs_bg_xyzn, params, 2);
    double scores[10] = {xyz_score_, normal_score_, vis_score_, segment_score_, edge_score_, edge_vis_score_, edge_occ_score_, lab_scores_[0], lab_scores_[1], lab_scores_[2]};
    memcpy(pose_samples->scores[i], scores, pose_samples->num_scores*sizeof(double));
  }
  printf("Scored c=3 poses in %.3f seconds\n", (get_time_ms() - t0) / 1000.0);  //dbug
  goto CLEANUP;
  //



  // re-weight
  for (i = 0; i < num_samples; i++)
    W[i] = model_placement_score(X[i], Q[i], pcd_model, range_edges_model, model_xyz_index, &model_xyz_params, pcd_obs_bg, obs_range_image, NULL, obs_bg_xyzn, params, 2);
  sort_pose_samples(X, Q, W, C_obs, C_model, num_samples, params->num_correspondences);



  // cluster poses
  if (params->pose_clustering)
    num_samples = remove_redundant_pose_samples(X, Q, W, num_samples, params->x_cluster_thresh, params->q_cluster_thresh);

  //dbug
  if (have_true_pose)
    print_good_poses(true_pose, X, Q, num_samples);

  // remove low-weight samples
  num_samples = find_first_lt(W, W[0] - 2*w_sigma_multi, num_samples);
  num_samples = MIN(num_samples, 50);

  printf("Scored and sorted final poses in %.3f seconds\n", (get_time_ms() - t0) / 1000.0);
  
  //dbug
  if (have_true_pose)
    //print_good_poses_verbose(true_pose, X, Q, W, w_sigma_multi, num_samples);
    print_good_poses(true_pose, X, Q, num_samples);


  if (params->do_final_icp) {


    // align
    //t0 = get_time_ms();
    //for (i = 0; i < num_samples; i++)
    //  align_model_icp_dense(X[i], Q[i], pcd_model, pcd_obs, obs_range_image, obs_fg_range_image, params, 10, 100);
    //printf("Ran dense ICP on %d samples in %.3f seconds\n", num_samples, (get_time_ms() - t0) / 1000.0);

    // cluster
    //if (params->pose_clustering)
    //  num_samples = remove_redundant_pose_samples(X, Q, W, num_samples, params->x_cluster_thresh, params->q_cluster_thresh);

    // remove low-weight samples
    w_sigma_multi = w_sigma / sqrt((double)num_score_trials);
    //num_samples = find_first_lt(W, W[0] - 2*w_sigma_multi, num_samples);

    // align more
    //for (i = 0; i < num_samples; i++)
    //  align_model_icp_dense(X[i], Q[i], pcd_model, pcd_obs, obs_range_image, obs_fg_range_image, params, 10, 500);

    // align with gradients
    t0 = get_time_ms();
    for (i = 0; i < num_samples; i++)
      for (j = 0; j < 5; j++)
	align_model_gradient(X[i], Q[i], pcd_model, range_edges_model, obs_range_image, obs_edge_image, params, 2, 100);
    printf("Ran gradient alignment on %d samples in %.3f seconds\n", num_samples, (get_time_ms() - t0) / 1000.0);

    //dbug: align true pose more with gradients
    //for (j = 0; j < 10; j++)
    //  align_model_gradient(X[0], Q[0], pcd_model, range_edges_model, obs_range_image, obs_edge_image, params, 20, 500);

    // re-weight (round 3)
    params->num_validation_points *= 2;
    for (i = 0; i < num_samples; i++)
      W[i] = model_placement_score(X[i], Q[i], pcd_model, range_edges_model, model_xyz_index, &model_xyz_params, pcd_obs_bg, obs_range_image, obs_edge_image, obs_bg_xyzn, params, 3);
    sort_pose_samples(X, Q, W, C_obs, C_model, num_samples, params->num_correspondences);

    num_samples = MIN(num_samples, 20);  //dbug

    // align with gradients more
    t0 = get_time_ms();
    for (i = 0; i < num_samples; i++)
      for (j = 0; j < 5; j++)
	align_model_gradient(X[i], Q[i], pcd_model, range_edges_model, obs_range_image, obs_edge_image, params, 2, 100);
    printf("Ran gradient alignment on %d samples in %.3f seconds\n", num_samples, (get_time_ms() - t0) / 1000.0);

    // re-weight (round 3)
    params->num_validation_points *= 2;
    for (i = 0; i < num_samples; i++)
      W[i] = model_placement_score(X[i], Q[i], pcd_model, range_edges_model, model_xyz_index, &model_xyz_params, pcd_obs_bg, obs_range_image, obs_edge_image, obs_bg_xyzn, params, 3);
    sort_pose_samples(X, Q, W, C_obs, C_model, num_samples, params->num_correspondences);

    num_samples = MIN(num_samples, 10);  //dbug

    // align with gradients more
    t0 = get_time_ms();
    for (i = 0; i < num_samples; i++) 
      for (j = 0; j < 5; j++)
	align_model_gradient(X[i], Q[i], pcd_model, range_edges_model, obs_range_image, obs_edge_image, params, 20, 1000);
    printf("Ran gradient alignment on %d samples in %.3f seconds\n", num_samples, (get_time_ms() - t0) / 1000.0);

    // re-weight with full models
    t0 = get_time_ms();
    params->num_validation_points = 0;
    for (i = 0; i < num_samples; i++)
      W[i] = model_placement_score(X[i], Q[i], pcd_model, range_edges_model, model_xyz_index, &model_xyz_params, pcd_obs_bg, obs_range_image, obs_edge_image, obs_bg_xyzn, params, 3);
    sort_pose_samples(X, Q, W, C_obs, C_model, num_samples, params->num_correspondences);
    printf("Scored %d samples with full models in %.3f seconds\n", num_samples, (get_time_ms() - t0) / 1000.0);

    // remove low-weight samples
    //w_sigma_multi = w_sigma / sqrt((double)num_score_trials);
    //num_samples = find_first_lt(W, W[0] - 2*w_sigma_multi, num_samples);

    num_samples = MIN(num_samples, 10);  //dbug
  }

  //dbug
  printf("Ran scope() in %.3f seconds\n", (get_time_ms() - t0_scope) / 1000.0);  //dbug

  // pack return value
  //olf_pose_samples_t *pose_samples;
  safe_calloc(pose_samples, 1, olf_pose_samples_t);
  pose_samples->X = X;
  pose_samples->Q = Q;
  pose_samples->W = W;
  pose_samples->n = num_samples;

  //dbug
  //printf("Scored and sorted final poses in %.3f seconds\n", (get_time_ms() - t0) / 1000.0);
  if (have_true_pose)
    print_good_poses_verbose(true_pose, X, Q, W, w_sigma_multi, num_samples);

  //dbug
  params->verbose = 1;
  pose_samples->vis_probs = new_matrix2(num_samples, pcd_model->num_points);
  pose_samples->xyz_dists = new_matrix2(num_samples, pcd_model->num_points);
  pose_samples->normal_dists = new_matrix2(num_samples, pcd_model->num_points);
  pose_samples->range_edge_pixels = new_matrix2i(num_samples, pcd_model->num_points);
  pose_samples->range_edge_vis_prob = new_matrix2(num_samples, pcd_model->num_points);
  safe_calloc(pose_samples->num_range_edge_points, num_samples, int);
  pose_samples->occ_edge_pixels = new_matrix2i(num_samples, pcd_model->num_points);
  safe_calloc(pose_samples->num_occ_edge_points, num_samples, int);
  pose_samples->scores = new_matrix2(num_samples, 10);
  for (i = 0; i < num_samples; i++) {
    //printf("\nsample %d:\n", i);
    params->num_validation_points = 0;
    model_placement_score(X[i], Q[i], pcd_model, range_edges_model, model_xyz_index, &model_xyz_params, pcd_obs_bg, obs_range_image, obs_edge_image, obs_bg_xyzn, params, 3);
    memcpy(pose_samples->vis_probs[i], mps_vis_prob_, pcd_model->num_points * sizeof(double));
    memcpy(pose_samples->xyz_dists[i], mps_xyz_dists_, pcd_model->num_points * sizeof(double));
    memcpy(pose_samples->normal_dists[i], mps_normal_dists_, pcd_model->num_points * sizeof(double));
    memcpy(pose_samples->range_edge_pixels[i], range_edge_pixels_[0], 2*num_range_edge_points_ * sizeof(int));
    memcpy(pose_samples->range_edge_vis_prob[i], range_edge_vis_prob_, num_range_edge_points_ * sizeof(double));
    pose_samples->num_range_edge_points[i] = num_range_edge_points_;
    if (num_occ_edge_points_ > 0)
      memcpy(pose_samples->occ_edge_pixels[i], occ_edge_pixels_[0], 2*num_occ_edge_points_ * sizeof(int));
    pose_samples->num_occ_edge_points[i] = num_occ_edge_points_;

    // scores
    pose_samples->num_scores = 10;
    double scores[10] = {xyz_score_, normal_score_, vis_score_, segment_score_, edge_score_, edge_vis_score_, edge_occ_score_, lab_scores_[0], lab_scores_[1], lab_scores_[2]};
    memcpy(pose_samples->scores[i], scores, pose_samples->num_scores*sizeof(double));
  }
  */
















//-------------------------- DEPRECATED --------------------------//


/*
 * add balls model to a pcd
 *
static void pcd_add_balls(pcd_t *pcd)
{
  int i, s, b;
  int ch_balls = pcd_channel(pcd, "balls");
  int ch_segments = pcd_channel(pcd, "segments");

  if (ch_balls < 0)
    return;

  // create pcd_balls_t
  safe_calloc(pcd->balls, 1, pcd_balls_t);
  pcd_balls_t *B = pcd->balls;

  // get ball labels
  safe_calloc(B->ball_labels, pcd->num_points, int);
  for (i = 0; i < pcd->num_points; i++)
    B->ball_labels[i] = (int)(pcd->data[ch_balls][i]);

  // get segment labels
  safe_calloc(B->segment_labels, pcd->num_points, int);
  if (ch_segments >= 0)
    for (i = 0; i < pcd->num_points; i++)
      B->segment_labels[i] = (int)(pcd->data[ch_segments][i]);

  // get num segments
  B->num_segments = imax(B->segment_labels, pcd->num_points) + 1;

  ***
  // compute segment centers
  B->segment_centers = new_matrix2(B->num_segments, 3);
  int segment_cnts[B->num_segments];
  memset(segment_cnts, 0, B->num_segments * sizeof(int));
  for (i = 0; i < pcd->num_points; i++) {
    s = B->segment_labels[i];
    B->segment_centers[s][0] += pcd->points[0][i];
    B->segment_centers[s][1] += pcd->points[1][i];
    B->segment_centers[s][2] += pcd->points[2][i];
    segment_cnts[s]++;
  }
  for (s = 0; s < B->num_segments; s++)
    mult(B->segment_centers[s], B->segment_centers[s], 1.0/(double)segment_cnts[s], 3);

  // compute segment radii
  safe_calloc(B->segment_radii, B->num_segments, double);
  for (i = 0; i < pcd->num_points; i++) {
    s = B->segment_labels[i];
    double dx = pcd->points[0][i] - B->segment_centers[s][0];
    double dy = pcd->points[1][i] - B->segment_centers[s][1];
    double dz = pcd->points[2][i] - B->segment_centers[s][2];
    B->segment_radii[s] += dx*dx + dy*dy + dz*dz;
  }
  for (s = 0; s < B->num_segments; s++)
    B->segment_radii[s] = sqrt(B->segment_radii[s] / (double)segment_cnts[s]);

  // compute mean segment radius
  B->mean_segment_radius = sum(B->segment_radii, B->num_segments);
  B->mean_segment_radius /= (double)B->num_segments;
  ***

  // get num balls
  safe_calloc(B->num_balls, B->num_segments, int);
  for (i = 0; i < pcd->num_points; i++) {
    s = B->segment_labels[i];
    b = B->ball_labels[i];
    if (B->num_balls[s] < b+1)
      B->num_balls[s] = b+1;
  }

  // compute ball centers
  safe_calloc(B->ball_centers, B->num_segments, double**);
  for (s = 0; s < B->num_segments; s++)
    B->ball_centers[s] = new_matrix2(B->num_balls[s], 3);
  int max_balls = imax(B->num_balls, B->num_segments);
  int ball_cnts[B->num_segments][max_balls];
  memset(ball_cnts, 0, B->num_segments * max_balls * sizeof(int));
  for (i = 0; i < pcd->num_points; i++) {
    s = B->segment_labels[i];
    b = B->ball_labels[i];
    B->ball_centers[s][b][0] += pcd->points[0][i];
    B->ball_centers[s][b][1] += pcd->points[1][i];
    B->ball_centers[s][b][2] += pcd->points[2][i];
    ball_cnts[s][b]++;
  }
  for (s = 0; s < B->num_segments; s++)
    for (b = 0; b < B->num_balls[s]; b++)
      mult(B->ball_centers[s][b], B->ball_centers[s][b], 1.0/(double)ball_cnts[s][b], 3);

  // compute ball radii
  B->ball_radii = new_matrix2(B->num_segments, max_balls);
  for (i = 0; i < pcd->num_points; i++) {
    s = B->segment_labels[i];
    b = B->ball_labels[i];
    double dx = B->ball_centers[s][b][0] - pcd->points[0][i];
    double dy = B->ball_centers[s][b][1] - pcd->points[1][i];
    double dz = B->ball_centers[s][b][2] - pcd->points[2][i];
    B->ball_radii[s][b] += dx*dx + dy*dy + dz*dz;
  }
  for (s = 0; s < B->num_segments; s++)
    for (b = 0; b < B->num_balls[s]; b++)
      B->ball_radii[s][b] = sqrt(B->ball_radii[s][b] / (double)ball_cnts[s][b]);

  // compute mean ball radius
  int nballs = 0;
  for (s = 0; s < B->num_segments; s++) {
    for (b = 0; b < B->num_balls[s]; b++) {
      if (ball_cnts[s][b] > 0) {
	B->mean_ball_radius += B->ball_radii[s][b];
	nballs++;
      }
    }
  }
  B->mean_ball_radius /= (double)nballs;

  // compute segment centers & radii
  B->segment_centers = new_matrix2(B->num_segments, 3);
  safe_calloc(B->segment_radii, B->num_segments, double);
  for (s = 0; s < B->num_segments; s++) {
    // compute center
    nballs = 0;
    for (b = 0; b < B->num_balls[s]; b++) {
      if (ball_cnts[s][b] > 0) {
	nballs++;
	add(B->segment_centers[s], B->segment_centers[s], B->ball_centers[s][b], 3);
      }
    }
    mult(B->segment_centers[s], B->segment_centers[s], 1/(double)nballs, 3);
    // compute radius
    for (b = 0; b < B->num_balls[s]; b++) {
      if (ball_cnts[s][b] > 0) {
	double d = dist(B->segment_centers[s], B->ball_centers[s][b], 3);
	double r = B->ball_radii[s][b];
	if (B->segment_radii[s] < d + r)
	  B->segment_radii[s] = d + r;
      }
    }
  }

  // compute mean segment radius
  B->mean_segment_radius = sum(B->segment_radii, B->num_segments);
  B->mean_segment_radius /= (double)B->num_segments;
}
*/

/*
 * free pcd->balls
 *
static void pcd_free_balls(pcd_t *pcd)
{
  if (pcd->balls == NULL)
    return;

  pcd_balls_t *B = pcd->balls;
  int i;

  if (B->num_balls)
    free(B->num_balls);
  if (B->segment_labels)
    free(B->segment_labels);
  if (B->ball_labels)
    free(B->ball_labels);
  if (B->segment_centers)
    free_matrix2(B->segment_centers);
  if (B->segment_radii)
    free(B->segment_radii);
  if (B->ball_centers) {
    for (i = 0; i < B->num_segments; i++)
      free_matrix2(B->ball_centers[i]);
    free(B->ball_centers);
  }
  if (B->ball_radii)
    free_matrix2(B->ball_radii);

  free(pcd->balls);
}
*/


/*
 * check if a pcd has all the channels needed to biuld an OLF model
 *
static int pcd_has_olf_channels(pcd_t *pcd)
{
  return (pcd_channel(pcd, "x")>=0 && pcd_channel(pcd, "y")>=0 && pcd_channel(pcd, "z")>=0 &&
	  pcd_channel(pcd, "nx")>=0 && pcd_channel(pcd, "ny")>=0 && pcd_channel(pcd, "nz")>=0 &&
	  pcd_channel(pcd, "pcx")>=0 && pcd_channel(pcd, "pcy")>=0 && pcd_channel(pcd, "pcz")>=0 &&
	  pcd_channel(pcd, "pc1")>=0 && pcd_channel(pcd, "pc2")>=0 && pcd_channel(pcd, "cluster")>=0 &&
	  pcd_channel(pcd, "f1")>=0 && pcd_channel(pcd, "f33")>=0);
}
*/

/*
static void pcd_random_walk(int *I, pcd_t *pcd, int i0, int n, double sigma)
{
  //dbug
  double **X = new_matrix2(pcd->num_points, 3);
  transpose(X, pcd->points, 3, pcd->num_points);

  I[0] = i0;
  int cnt, i = i0;
  double x[3];
  for (cnt = 1; cnt < n; cnt++) {
    x[0] = normrand(pcd->points[0][i], sigma);
    x[1] = normrand(pcd->points[1][i], sigma);
    x[2] = normrand(pcd->points[2][i], sigma);

    i = kdtree_NN(pcd->points_kdtree, x);
    I[cnt] = i;

    fprintf(stderr, "random walk step = %f\n", dist(X[I[cnt]], X[I[cnt-1]], 3)); //dbug
  }
}
*/

/*
 * (fast, randomized) intersection of two pcds with a balls model --
 * computes a list of point indices that (approximately) intersect with the model balls,
 * returns the number of intersecting points
 *
static int pcd_intersect(int *idx, pcd_t *pcd, pcd_t *model, double *x, double *q)
{
  int i, s, b;
  int num_points = pcd->num_points;
  int num_model_balls = model->balls->num_balls[0];
  int num_segments = pcd->balls->num_segments;
  int *num_segment_balls = pcd->balls->num_balls;
  double model_radius = model->balls->segment_radii[0];
  double *model_ball_radii = model->balls->ball_radii[0];
  double **segment_centers = pcd->balls->segment_centers;
  double *segment_radii = pcd->balls->segment_radii;
  double ***segment_ball_centers = pcd->balls->ball_centers;
  double **segment_ball_radii = pcd->balls->ball_radii;

  // apply transform (x,q) to model
  double **R = new_matrix2(3,3);
  quaternion_to_rotation_matrix(R,q);
  double model_center[3];
  matrix_vec_mult(model_center, R, model->balls->segment_centers[0], 3, 3);
  add(model_center, model_center, x, 3);
  double **model_ball_centers = new_matrix2(num_model_balls, 3);
  for (b = 0; b < num_model_balls; b++) {
    matrix_vec_mult(model_ball_centers[b], R, model->balls->ball_centers[0][b], 3, 3);
    add(model_ball_centers[b], model_ball_centers[b], x, 3);
  }
  free_matrix2(R);

  // compute close segments
  int close_segments[num_segments];
  memset(close_segments, 0, num_segments * sizeof(int));
  for (s = 0; s < num_segments; s++) {
    //printf("break 1\n");
    if (dist(segment_centers[s], model_center, 3) < segment_radii[s] + model_radius) {  // segment intersects model
      for (b = 0; b < num_model_balls; b++) {
	//printf("break 2\n");
	if (dist(segment_centers[s], model_ball_centers[b], 3) < segment_radii[s] + model_ball_radii[b]) {  // segment intersects model ball
	  double p = 0;
	  for (i = 0; i < num_segment_balls[s]; i++) {
	    if (segment_ball_radii[s][i] > 0.0) {
	      //printf("break 3\n");
	      if (dist(segment_ball_centers[s][i], model_ball_centers[b], 3) < segment_ball_radii[s][i] + model_ball_radii[b])  // segment ball intersects model ball
		p += 1/(double)num_segment_balls[s];
	    }
	  }
	  if (frand() < p)
	    close_segments[s] = 1;
	}
      }
    }
  }
  free_matrix2(model_ball_centers);

  // compute point indices
  int n = 0;
  for (i = 0; i < num_points; i++) {
    s = pcd->balls->segment_labels[i];
    if (close_segments[s])
      idx[n++] = i;
  }

  return n;
}
*/


/*
 * loads an olf from fname.pcd and fname.bmx
 *
olf_t *load_olf(char *fname)
{
  int i;
  char f[1024];

  //dbug
  //double t;
  //t = get_time_ms();

  // load pcd
  sprintf(f, "%s.pcd", fname);
  pcd_t *pcd = load_pcd(f);
  if (pcd == NULL)
    return NULL;
  if (!pcd_has_olf_channels(pcd)) {
    fprintf(stderr, "Warning: pcd doesn't have olf channels!\n");
    pcd_free(pcd);
    free(pcd);
    return NULL;
  }

  //dbug
  //fprintf(stderr, "Loaded olf pcd in %f ms\n", get_time_ms() - t);
  //t = get_time_ms();  

  // load bmx
  sprintf(f, "%s.bmx", fname);
  int num_clusters;
  bingham_mix_t *bmx = load_bmx(f, &num_clusters);
  if (bmx == NULL) {
    pcd_free(pcd);
    free(pcd);
    return NULL;
  }

  //dbug
  //fprintf(stderr, "Loaded olf bmx in %f ms\n", get_time_ms() - t);
  //t = get_time_ms();  

  // create olf
  olf_t *olf;
  safe_calloc(olf, 1, olf_t);
  olf->pcd = pcd;
  olf->bmx = bmx;
  olf->num_clusters = num_clusters;

  // get shape descriptor length
  olf->shape_length = 33;

  // create temporary shape matrix
  double **S = new_matrix2(pcd->num_points, olf->shape_length);
  transpose(S, pcd->shapes, olf->shape_length, pcd->num_points);

  // get cluster weights
  safe_calloc(olf->cluster_weights, num_clusters, double);
  for (i = 0; i < pcd->num_points; i++) {
    int c = pcd->clusters[i];
    olf->cluster_weights[c]++;
  }
  mult(olf->cluster_weights, olf->cluster_weights, 1/(double)pcd->num_points, num_clusters);

  // get mean shapes
  olf->mean_shapes = new_matrix2(num_clusters, olf->shape_length);
  for (i = 0; i < pcd->num_points; i++) {
    int c = pcd->clusters[i];
    add(olf->mean_shapes[c], olf->mean_shapes[c], S[i], olf->shape_length);
  }
  for (i = 0; i < num_clusters; i++) {
    double cluster_size = olf->cluster_weights[i] * pcd->num_points;
    mult(olf->mean_shapes[i], olf->mean_shapes[i], 1/cluster_size, olf->shape_length);
  }

  // get shape variances
  safe_calloc(olf->shape_variances, num_clusters, double);
  for (i = 0; i < pcd->num_points; i++) {
    int c = pcd->clusters[i];
    olf->shape_variances[c] += dist2(S[i], olf->mean_shapes[c], olf->shape_length);
  }
  for (i = 0; i < num_clusters; i++) {
    double cluster_size = olf->cluster_weights[i] * pcd->num_points;
    olf->shape_variances[i] /= cluster_size;
  }

  free_matrix2(S);

  //dbug
  //fprintf(stderr, "Computed cluster shapes in %f ms\n", get_time_ms() - t);
  //t = get_time_ms();  

  // load hll models
  sprintf(f, "%s.hll", fname);
  int num_hll;
  olf->hll = load_hlls(f, &num_hll);

  if (olf->hll == NULL) {

    // create hll models
    safe_calloc(olf->hll, num_clusters, hll_t);
    int c;
    for (c = 0; c < num_clusters; c++) {
      int n = (int)round(olf->cluster_weights[c] * pcd->num_points);
      double **Q = new_matrix2(2*n, 4);
      double **X = new_matrix2(2*n, 3);
      int cnt=0;
      for (i = 0; i < pcd->num_points; i++) {
	if (pcd->clusters[i] == c) {
	  memcpy(Q[cnt], pcd->quaternions[0][i], 4*sizeof(double));
	  memcpy(Q[cnt+1], pcd->quaternions[1][i], 4*sizeof(double));
	  X[cnt][0] = X[cnt+1][0] = pcd->points[0][i];
	  X[cnt][1] = X[cnt+1][1] = pcd->points[1][i];
	  X[cnt][2] = X[cnt+1][2] = pcd->points[2][i];
	  cnt += 2;
	}
      }
      hll_new(&olf->hll[c], Q, X, 2*n, 4, 3);
      hll_cache(&olf->hll[c], Q, 2*n);
    }

    // save hll models
    save_hlls(f, olf->hll, num_clusters);
  }

  //dbug
  //fprintf(stderr, "Created HLL in %f ms\n", get_time_ms() - t);
  //t = get_time_ms();

  // set olf params (TODO--load this from a .olf file)
  olf->rot_symm = 0;
  olf->num_validators = 5;
  olf->lambda = 1; //.5;
  olf->pose_agg_x = .05;  // meters
  olf->pose_agg_q = .2;  // radians

  return olf;
}
*/

/*
 * frees the contents of an olf_t, but not the pointer itself
 *
void olf_free(olf_t *olf)
{
  int i;

  if (olf->pcd)
    pcd_free(olf->pcd);
  if (olf->bmx) {
    for (i = 0; i < olf->num_clusters; i++)
      bingham_mixture_free(&olf->bmx[i]);
    free(olf->bmx);
  }
  if (olf->cluster_weights)
    free(olf->cluster_weights);
  if (olf->mean_shapes)
    free_matrix2(olf->mean_shapes);
  if (olf->shape_variances)
    free(olf->shape_variances);
}
*/

/*
 * classify pcd points (add channel "cluster") using olf shapes
 *
void olf_classify_points(pcd_t *pcd, olf_t *olf)
{
  int ch = pcd_channel(pcd, "cluster");
  if (ch < 0) {
    ch = pcd_add_channel(pcd, "cluster");
    safe_malloc(pcd->clusters, pcd->num_points, int);
  }

  // create temporary shape matrix
  double **S = new_matrix2(pcd->num_points, olf->shape_length);
  transpose(S, pcd->shapes, olf->shape_length, pcd->num_points);

  int i, j;

  printf("shape stdev = [");
  double sigma = 0.0;
  for (i = 0; i < olf->num_clusters; i++) {
    sigma += sqrt(olf->shape_variances[i]);
    printf("%f ", sqrt(olf->shape_variances[i]));
  }
  printf("]\n");
  sigma /= (double) olf->num_clusters;

  double d, jmin=0;
  //double dmin;
  double p[olf->num_clusters];
  for (i = 0; i < pcd->num_points; i++) {
    //dmin = DBL_MAX;
    for (j = 0; j < olf->num_clusters; j++) {
      d = dist(S[i], olf->mean_shapes[j], olf->shape_length);
      p[j] = normpdf(d, 0, sigma/10.0);
      //if (d < dmin) {
      //  dmin = d;
      //  jmin = j;
      //}
    }
    mult(p, p, 1/sum(p, olf->num_clusters), olf->num_clusters);
    jmin = pmfrand(p, olf->num_clusters);
    pcd->clusters[i] = jmin;
    pcd->data[ch][i] = (double)jmin;
  }

  free_matrix2(S);
}
*/

/*
 * computes the pdf of pose (x,q) given n points from pcd w.r.t. olf,
 * assumes that pcd has channel "cluster" (i.e. points are already classified)
 *
double olf_pose_pdf(double *x, double *q, olf_t *olf, pcd_t *pcd, int *indices, int n)
{
  int i;

  //dbug
  //printf("x = %f, %f, %f\n", x[0], x[1], x[2]);
  //printf("q = %f, %f, %f, %f\n", q[0], q[1], q[2], q[3]);

  // multi-feature likelihood
  if (n > 1) {
    double logp = 0;
    for (i = 0; i < n; i++)
      logp += log(olf_pose_pdf(x, q, olf, pcd, &indices[i], 1));

    return olf->lambda * exp(olf->lambda*logp/(double)n);
  }

  i = indices[0];  // validation point index

  double *x_world_to_model = x;
  double *q_world_to_model = q;
  double q_model_to_world[4];  //q_inv[4];

  quaternion_inverse(q_model_to_world, q_world_to_model);  //(q_inv, q);

  double x_feature[3] = {pcd->points[0][i], pcd->points[1][i], pcd->points[2][i]};
  double *q_feature = (frand() < .5 ? pcd->quaternions[0][i] : pcd->quaternions[1][i]);
  //double *q_feature = pcd->quaternions[0][i]; //dbug


  *** dbug
  double x_dbug[3] = {1000,0,0};    // world to model
  double q_dbug[4] = {0,1,0,0};  // world to model
  double q_feature_dbug[4];
  quaternion_mult(q_feature_dbug, q_feature, q_dbug);
  q_feature = q_feature_dbug;
  double **R_dbug = new_matrix2(3,3);
  quaternion_to_rotation_matrix(R_dbug, q_dbug);
  double x_feature_dbug[3];
  matrix_vec_mult(x_feature_dbug, R_dbug, x_feature, 3, 3);
  add(x_feature, x_feature_dbug, x_dbug, 3);
  free_matrix2(R_dbug);
  ***

  // q2: rotation from model -> feature
  double q_model_to_feature[4];  //q2[4];
  double *q_model_to_feature_ptr[1];  // *q2_ptr[1];
  q_model_to_feature_ptr[0] = q_model_to_feature;  //q2_ptr[0] = q2;
  quaternion_mult(q_model_to_feature, q_feature, q_model_to_world);  //(q2, q_feature, q_inv);

  // x2: translation from model -> feature
  double xi[3];
  sub(xi, x_feature, x_world_to_model, 3);
  double **R_model_to_world = new_matrix2(3,3);  //R_inv
  quaternion_to_rotation_matrix(R_model_to_world, q_model_to_world);  //(R_inv, q_inv);
  double x_model_to_feature[3];   // x2 = R_inv*xi
  matrix_vec_mult(x_model_to_feature, R_model_to_world, xi, 3, 3);
  free_matrix2(R_model_to_world);

  // p(q2)
  int c = pcd->clusters[i];
  double p = bingham_mixture_pdf(q_model_to_feature, &olf->bmx[c]);

  // p(x2|q2)
  double x_mean[3];
  double *x_mean_ptr[1];
  x_mean_ptr[0] = x_mean;
  double **x_cov = new_matrix2(3,3);
  hll_sample(x_mean_ptr, &x_cov, q_model_to_feature_ptr, &olf->hll[c], 1);
  p *= mvnpdf(x_model_to_feature, x_mean, x_cov, 3);
  free_matrix2(x_cov);
  

  //dbug
  //printf("q_feature = (%f, %f, %f, %f)\n", q_feature[0], q_feature[1], q_feature[2], q_feature[3]);
  //printf("q_model_to_world = (%f, %f, %f, %f)\n", q_model_to_world[0], q_model_to_world[1], q_model_to_world[2], q_model_to_world[3]);

  //dbug
  //printf("x_model_to_feature = (%f, %f, %f)\n", x_model_to_feature[0], x_model_to_feature[1], x_model_to_feature[2]);
  //printf("q_model_to_feature = (%f, %f, %f, %f)\n", q_model_to_feature[0], q_model_to_feature[1], q_model_to_feature[2], q_model_to_feature[3]);

  return p;
}
*/

/*
 * samples n weighted poses (X,Q,W) using olf model "olf" and point cloud "pcd"
 *
olf_pose_samples_t *olf_pose_sample(olf_t *olf, pcd_t *pcd, int n)
{
  double epsilon = 1e-50;

  olf_pose_samples_t *poses = olf_pose_samples_new(n);

  double **X = poses->X;
  double **Q = poses->Q;
  double *W = poses->W;

  int num_validators = olf->num_validators;
  int npoints = pcd->num_points;

  double *q_feature, q_model_to_feature[4], q_feature_to_model[4];
  double x_mean[3], **x_cov = new_matrix2(3,3);
  double x_feature[3], x_model_to_feature[3], x_model_to_feature_rot[3];
  double **R = new_matrix2(3,3);
  int indices[num_validators];
  int close_points[npoints];

  // pointers
  double *q_model_to_feature_ptr[1], *x_mean_ptr[1];
  q_model_to_feature_ptr[0] = q_model_to_feature;
  x_mean_ptr[0] = x_mean;

  // proposal weights
  int i;
  int *cluster_indices[olf->num_clusters];
  int cluster_counts[olf->num_clusters];
  double proposal_weights[olf->num_clusters];
  if (olf->proposal_weights) {
    for (i = 0; i < olf->num_clusters; i++) {
      safe_malloc(cluster_indices[i], npoints, int);
      cluster_counts[i] = findeq(cluster_indices[i], pcd->clusters, i, npoints);
      proposal_weights[i] = cluster_counts[i] * olf->proposal_weights[i];
    }
    mult(proposal_weights, proposal_weights, 1.0/sum(proposal_weights, olf->num_clusters), olf->num_clusters);
  }

  // segment indices (for cluttered pcds)
  int **segment_indices = NULL;
  int *segment_cnts = NULL;
  if (olf->cluttered) {
    segment_indices = new_matrix2i(pcd->balls->num_segments, pcd->num_points);
    safe_calloc(segment_cnts, pcd->balls->num_segments, int);
    for (i = 0; i < pcd->num_points; i++) {
      int s = pcd->balls->segment_labels[i];
      segment_indices[s][ segment_cnts[s]++ ] = i;
    }
  }


  for (i = 0; i < n; i++) {
    // sample a proposal feature
    int j = 0;
    if (olf->proposal_weights) {
      int cluster = pmfrand(proposal_weights, olf->num_clusters);
      j = cluster_indices[cluster][ irand(cluster_counts[cluster]) ];
    }
    else if (olf->num_proposal_segments > 0) {
      //printf("break 1\n");
      //printf("olf->num_proposal_segments = %d\n", olf->num_proposal_segments);
      int s = olf->proposal_segments[ irand(olf->num_proposal_segments) ];
      //printf("s = %d\n", s);
      j = segment_indices[s][ irand(segment_cnts[s]) ];
      printf("proposal = %d\n", j);
    }
    else
      j = irand(npoints);

    q_feature = (frand() < .5 ? pcd->quaternions[0][j] : pcd->quaternions[1][j]);

    x_feature[0] = pcd->points[0][j];
    x_feature[1] = pcd->points[1][j];
    x_feature[2] = pcd->points[2][j];


    *** dbug
    double x_dbug[3] = {1000,0,0};    // world to model
    double q_dbug[4] = {0,1,0,0};  // world to model
    double q_feature_dbug[4];
    quaternion_mult(q_feature_dbug, q_feature, q_dbug);
    q_feature = q_feature_dbug;
    double **R_dbug = new_matrix2(3,3);
    quaternion_to_rotation_matrix(R_dbug, q_dbug);
    double x_feature_dbug[3];
    matrix_vec_mult(x_feature_dbug, R_dbug, x_feature, 3, 3);
    add(x_feature, x_feature_dbug, x_dbug, 3);
    free_matrix2(R_dbug);
    ***

    // sample model orientation
    int c = pcd->clusters[j];
    bingham_mixture_sample(q_model_to_feature_ptr, &olf->bmx[c], 1);
    quaternion_inverse(q_feature_to_model, q_model_to_feature);
    quaternion_mult(Q[i], q_feature_to_model, q_feature);

    // sample model position given orientation
    hll_sample(x_mean_ptr, &x_cov, q_model_to_feature_ptr, &olf->hll[c], 1);
    mvnrand(x_model_to_feature, x_mean, x_cov, 3);
    quaternion_to_rotation_matrix(R, Q[i]);
    matrix_vec_mult(x_model_to_feature_rot, R, x_model_to_feature, 3, 3);
    sub(X[i], x_feature, x_model_to_feature_rot, 3);

    // if cluttered, use smart validation (with segmentation/balls model)
    int k, valid = 1;
    if (olf->cluttered) {
      if (olf->num_proposal_segments > 0) { //dbug
	// sample validation points
	for (k = 0; k < num_validators; k++) {
	  int s = olf->proposal_segments[ irand(olf->num_proposal_segments) ];
	  indices[k] = segment_indices[s][ irand(segment_cnts[s]) ];
	  //printf("validation = %d\n", indices[k]);
	}
      }
      else {
	int num_close_points = pcd_intersect(close_points, pcd, olf->pcd, X[i], Q[i]);
	if (num_close_points == 0)
	  valid = 0;
	else {
	  printf("num_close_points = %d\n", num_close_points); //dbug
	  // sample validation points
	  for (k = 0; k < num_validators; k++)
	    indices[k] = close_points[ irand(num_close_points) ];
	}
      }
    }
    else {
      // sample validation points
      for (k = 0; k < num_validators; k++)
	indices[k] = irand(npoints);
      //randperm(indices, npoints, num_validators);
    }

    //dbug: test random walk
    //double sigma = 30;
    //int indices_walk[10*num_validators];
    //pcd_random_walk(indices_walk, pcd, j, 10*num_validators, sigma/10.0);
    //int k;
    //for (k = 0; k < num_validators; k++)
    //  indices[k] = indices_walk[10*k];

    ***dbug
    if (i == 0) {
      memcpy(X[i], x_dbug, 3*sizeof(double));
      memcpy(Q[i], q_dbug, 4*sizeof(double));
      indices[0] = 50;
      indices[1] = 100;
      indices[2] = 150;
      indices[3] = 200;
      indices[4] = 250;
      //printf("x = {%f, %f, %f}; q = {%f, %f, %f, %f};\n", X[i][0], X[i][1], X[i][2], Q[i][0], Q[i][1], Q[i][2], Q[i][3]);
      //printf("indices = {");
      //for (k = 0; k < num_validators; k++)
      //  printf("%d ", indices[k]);
      //printf("};\n");
    }
    ***

    // compute target density for the given pose
    if (!valid)
      W[i] = epsilon;
    else if (num_validators > 0)
      W[i] = olf_pose_pdf(X[i], Q[i], olf, pcd, indices, num_validators);
    else
      W[i] = 1.0;

    // dbug
    //if (i == 0) {
    //  printf("W[0] = %e\n", W[i]);
    //  exit(1);
    //}
  }

  // sort pose samples by weight
  int I[n];
  double W2[n];
  mult(W2, W, -1, n);
  sort_indices(W2, I, n);  // sort -W (i.e. descending W)

  for (i = 0; i < n; i++)
    W[i] = -W2[I[i]];
  reorder_rows(X, X, I, n, 3);
  reorder_rows(Q, Q, I, n, 4);

  // normalize W
  mult(W, W, 1.0/sum(W,n), n);

  free_matrix2(x_cov);
  free_matrix2(R);

  if (olf->cluttered) {
    free_matrix2i(segment_indices);
    free(segment_cnts);
  }

  return poses;
}
*/

/*
 * aggregate the weighted pose samples, (X,Q,W)
 *
olf_pose_samples_t *olf_aggregate_pose_samples(olf_pose_samples_t *poses, olf_t *olf)
{
  olf_pose_samples_t *agg_poses = olf_pose_samples_new(poses->n);

  double **R1 = new_matrix2(3,3);
  double **R2 = new_matrix2(3,3);
  double z1[3], z2[3], z[3];

  int i, j, cnt=0;
  for (i = 0; i < poses->n; i++) {
    
    if (olf->rot_symm) {
      quaternion_to_rotation_matrix(R1, poses->Q[i]);
      z1[0] = R1[0][2];
      z1[1] = R1[1][2];
      z1[2] = R1[2][2];
    }

    for (j = 0; j < cnt; j++) {
      double dx = dist(poses->X[i], agg_poses->X[j], 3);
      double dq = 0.0;
      if (olf->rot_symm) {
	quaternion_to_rotation_matrix(R2, agg_poses->Q[j]);
	z2[0] = R2[0][2];
	z2[1] = R2[1][2];
	z2[2] = R2[2][2];
	dq = acos(dot(z1, z2, 3));
      }
      else
	dq = acos(fabs(dot(poses->Q[i], agg_poses->Q[j], 4)));
      
      //fprintf(stderr, "dx = %3.0f, dq = %3.0f\n", dx, dq*(180.0/M_PI));

      //if (a*a*dx*dx + b*b*dq*dq < 1.0) {  // add pose i to cluster j
      if (dx < olf->pose_agg_x && dq < olf->pose_agg_q) {  // add pose i to cluster j
	double wtot = poses->W[i] + agg_poses->W[j];
	double w = poses->W[i] / wtot;
	wavg(agg_poses->X[j], poses->X[i], agg_poses->X[j], w, 3);
	if (olf->rot_symm) {
	  wavg(z, z1, z2, w, 3);
	  normalize(z, z, 3);
	  if (1 - z[2] < .00000001) {  // close to identity rotation
	    agg_poses->Q[j][0] = 1;
	    agg_poses->Q[j][1] = 0;
	    agg_poses->Q[j][2] = 0;
	    agg_poses->Q[j][3] = 0;
	  }
	  else {
	    double a = 1.0 / sqrt(1 - z[2]*z[2]);
	    double c = sqrt((1 + z[2])/2.0);
	    double s = sqrt((1 - z[2])/2.0);
	    agg_poses->Q[j][0] = c;
	    agg_poses->Q[j][1] = -s*a*z[1];
	    agg_poses->Q[j][2] = s*a*z[0];
	    agg_poses->Q[j][3] = 0;
	  }
	  
	}
	else {
	  wavg(agg_poses->Q[j], poses->Q[i], agg_poses->Q[j], w, 4);
	  normalize(agg_poses->Q[j], agg_poses->Q[j], 4);
	}
	agg_poses->W[j] = wtot;
	break;
      }
    }
    if (j == cnt) {  // add a new cluster
      memcpy(agg_poses->X[cnt], poses->X[i], 3*sizeof(double));
      memcpy(agg_poses->Q[cnt], poses->Q[i], 4*sizeof(double));
      agg_poses->W[cnt] = poses->W[i];
      cnt++;
    }
  }

  free_matrix2(R1);
  free_matrix2(R2);

  // sort pose samples by weight
  int n = cnt;
  int I[n];
  double W2[n];
  mult(W2, agg_poses->W, -1, n);
  sort_indices(W2, I, n);  // sort -W (i.e. descending W)

  for (i = 0; i < n; i++)
    agg_poses->W[i] = -W2[I[i]];
  reorder_rows(agg_poses->X, agg_poses->X, I, n, 3);
  reorder_rows(agg_poses->Q, agg_poses->Q, I, n, 4);

  agg_poses->n = n;

  return agg_poses;
}
*/

/*
 * create a new olf_pose_samples_t
 *
olf_pose_samples_t *olf_pose_samples_new(int n)
{
  olf_pose_samples_t *poses;
  safe_calloc(poses, 1, olf_pose_samples_t);
  poses->X = new_matrix2(n,3);
  poses->Q = new_matrix2(n,4);
  safe_calloc(poses->W, n, double);
  poses->n = n;

  return poses;
}
*/

/*
 * free pose samples
 *
void olf_pose_samples_free(olf_pose_samples_t *poses)
{
  if (poses->X)
    free_matrix2(poses->X);
  if (poses->Q)
    free_matrix2(poses->Q);
  if (poses->W)
    free(poses->W);
  free(poses);
}
*/


