
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "bingham/util.h"
#include "bingham/olf.h"




//---------------------------- STATIC HELPER FUNCTIONS ---------------------------//


/*
 * compute quaternions given normals and principal curvatures
 */
void compute_orientation_quaternions(double ***Q, double **N, double **PCS, int num_points)
{
  int i;
  double nx, ny, nz, pcx, pcy, pcz, pcx2, pcy2, pcz2;
  double **R = new_matrix2(3,3);

  for (i = 0; i < num_points; i++) {
    nx = N[0][i];
    ny = N[1][i];
    nz = N[2][i];
    pcx = PCS[0][i];
    pcy = PCS[1][i];
    pcz = PCS[2][i];
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
 * add balls model to a pcd
 */
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

  /*
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
  */

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


/*
 * free pcd->balls
 */
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


/*
 * add data pointers to a pcd
 */
static void pcd_add_data_pointers(pcd_t *pcd)
{
  int i;
  int ch_cluster = pcd_channel(pcd, "cluster");
  int ch_x = pcd_channel(pcd, "x");
  int ch_y = pcd_channel(pcd, "y");
  int ch_z = pcd_channel(pcd, "z");
  int ch_nx = pcd_channel(pcd, "nx");
  int ch_ny = pcd_channel(pcd, "ny");
  int ch_nz = pcd_channel(pcd, "nz");
  int ch_pcx = pcd_channel(pcd, "pcx");
  int ch_pcy = pcd_channel(pcd, "pcy");
  int ch_pcz = pcd_channel(pcd, "pcz");
  int ch_f1 = pcd_channel(pcd, "f1");
  int ch_f33 = pcd_channel(pcd, "f33");
  int ch_balls = pcd_channel(pcd, "balls");


  if (ch_cluster>=0) {
    //pcd->clusters = pcd->data[ch_cluster];
    safe_malloc(pcd->clusters, pcd->num_points, int);
    for (i = 0; i < pcd->num_points; i++)
      pcd->clusters[i] = (int)(pcd->data[ch_cluster][i]);
  }
  if (ch_x>=0 && ch_y>=0 && ch_z>=0) {
    safe_calloc(pcd->points, 3, double *);
    pcd->points[0] = pcd->data[ch_x];
    pcd->points[1] = pcd->data[ch_y];
    pcd->points[2] = pcd->data[ch_z];
  }
  if (ch_nx>=0 && ch_ny>=0 && ch_nz>=0) {
    safe_calloc(pcd->normals, 3, double *);
    pcd->normals[0] = pcd->data[ch_nx];
    pcd->normals[1] = pcd->data[ch_ny];
    pcd->normals[2] = pcd->data[ch_nz];
  }
  if (ch_pcx>=0 && ch_pcy>=0 && ch_pcz>=0) {
    safe_calloc(pcd->principal_curvatures, 3, double *);
    pcd->principal_curvatures[0] = pcd->data[ch_pcx];
    pcd->principal_curvatures[1] = pcd->data[ch_pcy];
    pcd->principal_curvatures[2] = pcd->data[ch_pcz];
  }
  if (ch_f1>=0 && ch_f33>=0) {
    safe_calloc(pcd->shapes, 33, double *);
    for (i = 0; i < 33; i++)
      pcd->shapes[i] = pcd->data[ch_f1 + i];
  }

  // add quaternion orientation features
  if (ch_nx>=0 && ch_ny>=0 && ch_nz>=0 && ch_pcx>=0 && ch_pcy>=0 && ch_pcz>=0) {
    pcd->quaternions[0] = new_matrix2(pcd->num_points, 4);
    pcd->quaternions[1] = new_matrix2(pcd->num_points, 4);
    compute_orientation_quaternions(pcd->quaternions, pcd->normals, pcd->principal_curvatures, pcd->num_points);
  }

  // add points kdtree
  double **X = new_matrix2(pcd->num_points, 3);
  transpose(X, pcd->points, 3, pcd->num_points);
  pcd->points_kdtree = kdtree(X, pcd->num_points, 3);
  free_matrix2(X);

  // add balls model
  if (ch_balls>=0)
    pcd_add_balls(pcd);
}


/*
 * free data pointers in a pcd
 */
static void pcd_free_data_pointers(pcd_t *pcd)
{
  if (pcd->points)
    free(pcd->points);
  if (pcd->normals)
    free(pcd->normals);
  if (pcd->principal_curvatures)
    free(pcd->principal_curvatures);
  if (pcd->shapes)
    free(pcd->shapes);
  if (pcd->clusters)
    free(pcd->clusters);

  if (pcd->quaternions[0])
    free_matrix2(pcd->quaternions[0]);
  if (pcd->quaternions[1])
    free_matrix2(pcd->quaternions[1]);

  if (pcd->points_kdtree)
    kdtree_free(pcd->points_kdtree);

  if (pcd->balls)
    pcd_free_balls(pcd);
}


/*
 * check if a pcd has all the channels needed to biuld an OLF model
 */
static int pcd_has_olf_channels(pcd_t *pcd)
{
  /*
  printf("x channel = %d\n", pcd_channel(pcd, "x"));
  printf("y channel = %d\n", pcd_channel(pcd, "y"));
  printf("z channel = %d\n", pcd_channel(pcd, "z"));
  printf("nx channel = %d\n", pcd_channel(pcd, "nx"));
  printf("ny channel = %d\n", pcd_channel(pcd, "ny"));
  printf("nz channel = %d\n", pcd_channel(pcd, "nz"));
  printf("pcx channel = %d\n", pcd_channel(pcd, "pcx"));
  printf("pcy channel = %d\n", pcd_channel(pcd, "pcy"));
  printf("pcz channel = %d\n", pcd_channel(pcd, "pcz"));
  printf("pc1 channel = %d\n", pcd_channel(pcd, "pc1"));
  printf("pc2 channel = %d\n", pcd_channel(pcd, "pc2"));
  printf("cluster channel = %d\n", pcd_channel(pcd, "cluster"));
  printf("f1 channel = %d\n", pcd_channel(pcd, "f1"));
  printf("f33 channel = %d\n", pcd_channel(pcd, "f33"));
  */

  return (pcd_channel(pcd, "x")>=0 && pcd_channel(pcd, "y")>=0 && pcd_channel(pcd, "z")>=0 &&
	  pcd_channel(pcd, "nx")>=0 && pcd_channel(pcd, "ny")>=0 && pcd_channel(pcd, "nz")>=0 &&
	  pcd_channel(pcd, "pcx")>=0 && pcd_channel(pcd, "pcy")>=0 && pcd_channel(pcd, "pcz")>=0 &&
	  pcd_channel(pcd, "pc1")>=0 && pcd_channel(pcd, "pc2")>=0 && pcd_channel(pcd, "cluster")>=0 &&
	  pcd_channel(pcd, "f1")>=0 && pcd_channel(pcd, "f33")>=0);
}


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


/*
 * (fast, randomized) intersection of two pcds with a balls model --
 * computes a list of point indices that (approximately) intersect with the model balls,
 * returns the number of intersecting points
 */
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



//---------------------------- EXTERNAL API ---------------------------//



/*
 * loads a pcd
 */
pcd_t *load_pcd(char *f_pcd)
{
  int i, j;
  FILE *f = fopen(f_pcd, "r");

  if (f == NULL) {
    fprintf(stderr, "Invalid filename: %s", f_pcd);
    return NULL;
  }

  pcd_t *pcd;
  safe_calloc(pcd, 1, pcd_t);

  char sbuf[1024], *s = sbuf;
  while (!feof(f)) {
    s = sbuf;
    if (fgets(s, 1024, f)) {

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
	if (wordcmp(s, "ascii", " \t\n")) {
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
	  if (fgets(s, 1024, f) == NULL)
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


/*
 * loads an olf from fname.pcd and fname.bmx
 */
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


/*
 * frees the contents of an olf_t, but not the pointer itself
 */
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


/*
 * classify pcd points (add channel "cluster") using olf shapes
 */
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

  double d, dmin, jmin=0;
  double p[olf->num_clusters];
  for (i = 0; i < pcd->num_points; i++) {
    dmin = DBL_MAX;
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


/*
 * computes the pdf of pose (x,q) given n points from pcd w.r.t. olf,
 * assumes that pcd has channel "cluster" (i.e. points are already classified)
 */
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


  /* dbug
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
  */

  // q2: rotation from model -> feature
  double q_model_to_feature[4];  //q2[4];
  double *q_model_to_feature_ptr[1];  //*q2_ptr[1];
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


/*
 * samples n weighted poses (X,Q,W) using olf model "olf" and point cloud "pcd"
 */
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
    segment_indices = new_matrix2(pcd->balls->num_segments, pcd->num_points);
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


    /* dbug
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
    */

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

    /*dbug
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
    */

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
    free_matrix2(segment_indices);
    free(segment_cnts);
  }

  return poses;
}


/*
 * aggregate the weighted pose samples, (X,Q,W)
 */
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


/*
 * create a new olf_pose_samples_t
 */
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


/*
 * free pose samples
 */
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
