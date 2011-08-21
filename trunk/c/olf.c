
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
  double q[4];
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


  if (ch_cluster>=0) {
    pcd->clusters = pcd->data[ch_cluster];
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
  if (pcd->quaternions[0])
    free_matrix2(pcd->quaternions[0]);
  if (pcd->quaternions[1])
    free_matrix2(pcd->quaternions[1]);
}


/*
 * check if a pcd has all the channels needed to biuld an OLF model
 */
static int pcd_has_olf_channels(pcd_t *pcd)
{
  return (pcd_channel(pcd, "x")>=0 && pcd_channel(pcd, "y")>=0 && pcd_channel(pcd, "z")>=0 &&
	  pcd_channel(pcd, "nx")>=0 && pcd_channel(pcd, "ny")>=0 && pcd_channel(pcd, "nz")>=0 &&
	  pcd_channel(pcd, "pcx")>=0 && pcd_channel(pcd, "pcy")>=0 && pcd_channel(pcd, "pcz")>=0 &&
	  pcd_channel(pcd, "pc1")>=0 && pcd_channel(pcd, "pc2")>=0 && pcd_channel(pcd, "cluster")>=0 &&
	  pcd_channel(pcd, "f1")>=0 && pcd_channel(pcd, "f33")>=0);
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
    fgets(s, 1024, f);
    
    if (!wordcmp(s, "COLUMNS", " \t\n") || !wordcmp(s, "FIELDS", " \t\n")) {
      s = sword(s, " \t", 1);
      pcd->channels = split(s, " \t", &pcd->num_channels);

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
    fprintf("Warning: channel %s already exists\n", channel);
    return;
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
 * loads an olf from fname.pcd and fname.olf
 */
olf_t *load_olf(char *fname)
{
  int i;
  char f[1024];

  // load pcd
  sprintf(f, "%s.pcd", fname);
  pcd_t *pcd = load_pcd(f);
  if (pcd == NULL)
    return NULL;
  if (!pcd_has_olf_channels(pcd)) {
    pcd_free(pcd);
    free(pcd);
    return NULL;
  }

  // load bmx
  sprintf(f, "%s.bmx", fname);
  int num_clusters;
  bingham_mix_t *bmx = load_bmx(f, &num_clusters);
  if (bmx == NULL) {
    pcd_free(pcd);
    free(pcd);
    return NULL;
  }

  // create olf
  olf_t *olf;
  safe_calloc(olf, 1, olf_t);
  olf->pcd = pcd;
  olf->bmx = bmx;
  olf->num_clusters = num_clusters;

  // create hll model
  safe_calloc(olf->hll, num_clusters, hll_t);

  // get shape descriptor length
  olf->shape_length = 33;

  // create temporary shape matrix
  double **S = new_matrix2(pcd->num_points, olf->shape_length);
  transpose(S, pcd->shapes, olf->shape_length, pcd->num_points);

  // get cluster weights
  safe_calloc(olf->cluster_weights, num_clusters, double);
  for (i = 0; i < pcd->num_points; i++) {
    int c = (int)(pcd->clusters[i]);
    olf->cluster_weights[c]++;
  }
  mult(olf->cluster_weights, olf->cluster_weights, 1/(double)pcd->num_points, num_clusters);

  // get mean shapes
  olf->mean_shapes = new_matrix2(num_clusters, olf->shape_length);
  for (i = 0; i < pcd->num_points; i++) {
    int c = (int)(pcd->clusters[i]);
    add(olf->mean_shapes[c], olf->mean_shapes[c], S[i], olf->shape_length);
  }
  for (i = 0; i < num_clusters; i++) {
    double cluster_size = olf->cluster_weights[i] * pcd->num_points;
    mult(olf->mean_shapes[i], olf->mean_shapes[i], 1/cluster_size, olf->shape_length);
  }

  // get shape variances
  safe_calloc(olf->shape_variances, num_clusters, double);
  for (i = 0; i < pcd->num_points; i++) {
    int c = (int)(pcd->clusters[i]);
    olf->shape_variances[c] += dist2(S[i], olf->mean_shapes[c], olf->shape_length);
  }
  for (i = 0; i < num_clusters; i++) {
    double cluster_size = olf->cluster_weights[i] * pcd->num_points;
    olf->shape_variances[i] /= cluster_size;
  }

  free_matrix2(S);

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
    pcd->clusters = pcd->data[ch];
  }

  // create temporary shape matrix
  double **S = new_matrix2(pcd->num_points, olf->shape_length);
  transpose(S, pcd->shapes, olf->shape_length, pcd->num_points);

  int i, j;
  double d, dmin, jmin;
  for (i = 0; i < pcd->num_points; i++) {
    dmin = DBL_MAX;
    for (j = 0; j < olf->num_clusters; j++) {
      d = dist2(S[i], olf->mean_shapes[j], olf->shape_length);
      if (d < dmin) {
	dmin = d;
	jmin = j;
      }
    }
    pcd->clusters[i] = jmin;
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
  const double lambda = .5;

  // multi-feature likelihood
  if (n > 1) {
    double logp = 0;
    for (i = 0; i < n; i++)
      logp += log(olf_pose_pdf(x, q, olf, pcd, &indices[i], 1));

    return lambda * exp(lambda*logp/(double)n);
  }

  i = indices[0];
  double q_inv[4];
  quaternion_inverse(q_inv, q);

  // q2: rotation from model -> feature
  double q2[4];
  if (frand() < .5)
    quaternion_mult(q2, pcd->quaternions[0][i], q_inv);
  else
    quaternion_mult(q2, pcd->quaternions[1][i], q_inv);

  // x2: translation from model -> feature
  double xi[3];
  xi[0] = pcd->points[0][i] - x[0];
  xi[1] = pcd->points[1][i] - x[1];
  xi[2] = pcd->points[2][i] - x[2];
  double **R_inv = new_matrix2(3,3);
  quaternion_to_rotation_matrix(R_inv, q_inv);
  double x2[3];   // x2 = R_inv*xi
  x2[0] = dot(R_inv[0], xi, 3);
  x2[1] = dot(R_inv[1], xi, 3);
  x2[2] = dot(R_inv[2], xi, 3);
  free_matrix2(R_inv);

  // p(q2)
  int c = (int)(pcd->clusters[i]);
  double p = bingham_mixture_pdf(q2, &olf->bmx[c]);

  // p(x2|q2)
  double x_mean[3];
  double **x_cov = new_matrix2(3,3);
  hll_sample(&x_mean, &x_cov, &q2, &olf->hll[c], 1);
  double z[3], **V = new_matrix2(3,3);
  eigen_symm(z, V, x_cov, 3);
  p *= mvnpdf_pcs(x2, x_mean, z, V, 3);
  free_matrix2(V);
  free_matrix2(x_cov);
  
  return p;
}


/*
 * samples n weighted poses (X,Q,W) using olf model "olf" and point cloud "pcd"
 */
void olf_pose_sample(double **X, double **Q, double *W, olf_t *olf, pcd_t *pcd, int n)
{
  int num_samples = 5;  // num validation points
  double lambda = .5;

  int npoints = pcd->num_points;

  /***

  double *q;
  double *f;
  int i;
  for (i = 0; i < n; i++) {

    // sample a point feature
    int j = irand(npoints);
    f = pcd->shapes[j];

    if (frand() < .5)
      q = pcd->quaternions[0][j];
    else
      q = pcd->quaternions[1][j];

    
    // sample a feature orientation
    int c = (int)(pcd->clusters[j]);



    // compute the model orientation posterior given the feature
    BMM = tofoo_posterior(tofoo, q, f);
    
    % sample an orientation from the proposal distribution
    r2 = bingham_mixture_sample(BMM.B, BMM.W, 1);
    p2 = bingham_mixture_pdf(r2, BMM.B, BMM.W);
    %r2_err = acos(abs(r2(1)^2 - r2(2)^2 - r2(3)^2 + r2(4)^2))'

    % sample from the proposal distribution of position given orientation
    xj = [pcd.X(j) pcd.Y(j) pcd.Z(j)];
    c = find(FCP(j,:));
    q2 = quaternion_mult(q, qinv(r2));
    [x_mean x_cov] = qksample_tofoo(q2,c,tofoo);
    x0 = mvnrnd(x_mean, x_cov);
    R = quaternionToRotationMatrix(r2);
    x2 = (xj' - R*x0')';
    p2 = p2*mvnpdf(x0, x_mean, x_cov);
    
    % compute target density for the given orientation 
    t2 = sope_cloud_pdf(x2, r2, tofoo, pcd, FCP, num_samples, lambda);
        
    XQ(i,:) = [x2 r2];
    W(i) = t2; %/p2;
    
    fprintf('.');
end
fprintf('\n');


[W I] = sort(W,'descend');
XQ = XQ(I,:);
W = W/sum(W);

  ***/
}
