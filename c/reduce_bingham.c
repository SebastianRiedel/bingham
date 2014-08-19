
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bingham.h"
#include "bingham/util.h"
#include "bingham/bingham_constants.h"
#include "bingham/hypersphere.h"

void segfault_test()
{
/**
allocated: True
[ components: 8 ]
weights: [ 0.729  0.081  0.081  0.081  0.009  0.009  0.009  0.001]
weights_sum: 1.0
allocated: True
d: 4
F: 0.000577650276242
Z: [-719.84105097 -719.82891014 -719.69424277]
V:
[[-0.00766872 -0.25514035  0.96544118  0.052611  ]
 [-0.00496416  0.96683722  0.25409415  0.02524502]
 [-0.03615472 -0.0111005  -0.05758552  0.99762394]]
allocated: True
d: 4
F: 0.00106138281911
Z: [-479.96585564 -479.96585564 -479.93171128]
V:
[[ -2.13925505e-03  -4.69582958e-02   9.95900123e-01  -7.72870417e-02]
 [  2.27962777e-02   8.54084419e-01  -5.93076203e-06  -5.19634616e-01]
 [ -4.01059868e-02   5.18008946e-01   9.02762278e-02   8.49651955e-01]]
allocated: True
d: 4
F: 0.00106149725317
Z: [-479.94097437 -479.94097437 -479.88194875]
V:
[[ -1.12924510e-02   9.29699265e-01  -5.40133835e-06   3.68146380e-01]
 [ -1.03767019e-02  -5.39903104e-02   9.89176567e-01   1.36040768e-01]
 [ -2.48703700e-02  -3.64339887e-01  -1.46580606e-01   9.19321509e-01]]
allocated: True
d: 4
F: 0.00106189394905
Z: [-479.85474172 -479.85474172 -479.70948343]
V:
[[ -4.57748375e-03   9.99807694e-01   8.99973059e-04  -1.90476147e-02]
 [ -9.13944744e-03   2.89012666e-11   9.98737983e-01   4.93853427e-02]
 [ -3.52983192e-02   1.88955700e-02  -4.96700295e-02   9.97962862e-01]]
allocated: True
d: 4
F: 0.00300487232934
Z: [-240. -240. -240.]
V:
[[-0.00491763  0.99978738  0.01824866 -0.0082467 ]
 [-0.0082467  -0.01824866  0.99978738  0.00491763]
 [-0.01824866  0.0082467  -0.00491763  0.99978738]]
allocated: True
d: 4
F: 0.00300487232934
Z: [-240. -240. -240.]
V:
[[ 0.00316284  0.99923579  0.03863772 -0.00499578]
 [-0.00499578 -0.03863772  0.99923579 -0.00316284]
 [-0.03863772  0.00499578  0.00316284  0.99923579]]
allocated: True
d: 4
F: 0.00300487232934
Z: [-240. -240. -240.]
V:
[[-0.00557504  0.99855927  0.05296982 -0.00651857]
 [-0.00651857 -0.05296982  0.99855927  0.00557504]
 [-0.05296982  0.00651857 -0.00557504  0.99855927]]
allocated: True
d: 4
F: 19.73921
Z: [ 0.  0.  0.]
V:
[[ 0.  1.  0.  0.]
 [ 0.  0.  1.  0.]
 [ 0.  0.  0.  1.]]

**/

  double Z[3] = {-719.84105097, -719.82891014, -719.69424277};
  double V[3][4] = {{-0.00766872, -0.25514035, 0.96544118, 0.052611  }, {-0.00496416, 0.96683722, 0.25409415, 0.02524502}, {-0.03615472, -0.0111005, -0.05758552, 0.99762394}};
  double *Vp[3] = {&V[0][0], &V[1][0], &V[2][0]};
  bingham_t B;
  bingham_new(&B, 4, Vp, Z);

  double Z2[3] = {-479.96585564, -479.96585564, -479.93171128};
  double V2[3][4] = {{-2.13925505e-03, -4.69582958e-02, 9.95900123e-01, -7.72870417e-02}, {2.27962777e-02, 8.54084419e-01, -5.93076203e-06, -5.19634616e-01}, {-4.01059868e-02, 5.18008946e-01, 9.02762278e-02, 8.49651955e-01}};
  double *Vp2[3] = {&V2[0][0], &V2[1][0], &V2[2][0]};
  bingham_t B2;
  bingham_new(&B2, 4, Vp2, Z2);

  double Z3[3] = {-479.94097437, -479.94097437, -479.88194875};
  double V3[3][4] = {{-1.12924510e-02, 9.29699265e-01, -5.40133835e-06, 3.68146380e-01}, {-1.03767019e-02, -5.39903104e-02, 9.89176567e-01, 1.36040768e-01}, {-2.48703700e-02, -3.64339887e-01, -1.46580606e-01, 9.19321509e-01}};
  double *Vp3[3] = {&V3[0][0], &V3[1][0], &V3[2][0]};
  bingham_t B3;
  bingham_new(&B3, 4, Vp3, Z3);  

  double Z4[3] = {-479.85474172, -479.85474172, -479.70948343};
  double V4[3][4] = {{-4.57748375e-03, 9.99807694e-01, 8.99973059e-04, -1.90476147e-02}, {-9.13944744e-03, 2.89012666e-11, 9.98737983e-01, 4.93853427e-02}, {-3.52983192e-02, 1.88955700e-02, -4.96700295e-02, 9.97962862e-01}};
  double *Vp4[3] = {&V4[0][0], &V4[1][0], &V4[2][0]};
  bingham_t B4;
  bingham_new(&B4, 4, Vp4, Z4);  

  double Z5[3] = {-240., -240., -240.};
  double V5[3][4] = {{-0.00491763, 0.99978738, 0.01824866, -0.0082467}, {-0.0082467, -0.01824866, 0.99978738, 0.00491763}, {-0.01824866, 0.0082467, -0.00491763, 0.99978738}};
  double *Vp5[3] = {&V5[0][0], &V5[1][0], &V5[2][0]};
  bingham_t B5;
  bingham_new(&B5, 4, Vp5, Z5);    

  double Z6[3] = {-240., -240., -240.};
  double V6[3][4] = {{0.00316284, 0.99923579, 0.03863772, -0.00499578}, {-0.00499578, -0.03863772, 0.99923579, -0.00316284}, {-0.03863772, 0.00499578, 0.00316284, 0.99923579}};
  double *Vp6[3] = {&V6[0][0], &V6[1][0], &V6[2][0]};
  bingham_t B6;
  bingham_new(&B6, 4, Vp6, Z6);    

  double Z7[3] = {-240., -240., -240.};
  double V7[3][4] = {{-0.00557504, 0.99855927, 0.05296982, -0.00651857}, {-0.00651857, -0.05296982, 0.99855927, 0.00557504}, {-0.05296982, 0.00651857, -0.00557504, 0.99855927}};
  double *Vp7[3] = {&V7[0][0], &V7[1][0], &V7[2][0]};
  bingham_t B7;
  bingham_new(&B7, 4, Vp7, Z7);

  double Z8[3] = {0., 0., 0.};
  double V8[3][4] = {{0,1,0,0}, {0,0,1,0}, {0,0,0,1}};
  double *Vp8[3] = {&V8[0][0], &V8[1][0], &V8[2][0]};
  bingham_t B8;
  bingham_new(&B8, 4, Vp8, Z8);

  bingham_mix_t bmm;
  bmm.n = 8;
  safe_malloc(bmm.w, 8, double);
  safe_malloc(bmm.B, 8, bingham_t);
  bingham_alloc(&bmm.B[0], 4);
  bingham_alloc(&bmm.B[1], 4);
  bingham_alloc(&bmm.B[2], 4);
  bingham_alloc(&bmm.B[3], 4);
  bingham_alloc(&bmm.B[4], 4);
  bingham_alloc(&bmm.B[5], 4);
  bingham_alloc(&bmm.B[6], 4);
  bingham_alloc(&bmm.B[7], 4);  
  bingham_copy(&bmm.B[0], &B);
  bingham_copy(&bmm.B[1], &B2);
  bingham_copy(&bmm.B[2], &B3);
  bingham_copy(&bmm.B[3], &B4);
  bingham_copy(&bmm.B[4], &B5);
  bingham_copy(&bmm.B[5], &B6);
  bingham_copy(&bmm.B[6], &B7);
  bingham_copy(&bmm.B[7], &B8);  

  bmm.w[0] = 0.729;
  bmm.w[1] = 0.081;
  bmm.w[2] = 0.081;
  bmm.w[3] = 0.081;  
  bmm.w[4] = 0.009;
  bmm.w[5] = 0.009;
  bmm.w[6] = 0.009;
  bmm.w[7] = 0.001;    

  bingham_mixture_reduce(&bmm, 5);
  printf("Done\n");
}

void small_test()
{
  double Z[3] = {-900, -900, 0};
  double V[3][4] = {{0,1,0,0}, {0,0,1,0}, {0,0,0,1}};
  double *Vp[3] = {&V[0][0], &V[1][0], &V[2][0]};
  bingham_t B;
  bingham_new(&B, 4, Vp, Z);

  double Z2[3] = {-900, 0, -900};
  double V2[3][4] = {{0,1,0,0}, {0,0,1,0}, {0,0,0,1}};
  double *Vp2[3] = {&V2[0][0], &V2[1][0], &V2[2][0]};
  bingham_t B2;
  bingham_new(&B2, 4, Vp2, Z2);  

  double Z3[3] = {-900, 0, -900};
  double V3[3][4] = {{0,1,0,0}, {0,0,1,0}, {0,0,0,1}};
  double *Vp3[3] = {&V3[0][0], &V3[1][0], &V3[2][0]};
  bingham_t B3;
  bingham_new(&B3, 4, Vp3, Z3);

  double Z4[3] = {-900, 0, -450};
  double V4[3][4] = {{0,1,0,0}, {0,0,1,0}, {0,0,0,1}};
  double *Vp4[3] = {&V4[0][0], &V4[1][0], &V4[2][0]};
  bingham_t B4;
  bingham_new(&B4, 4, Vp4, Z4);  

  bingham_mix_t bmm;
  bmm.n = 4;
  safe_malloc(bmm.w, 4, double);
  safe_malloc(bmm.B, 4, bingham_t);
  bingham_alloc(&bmm.B[0], 4);
  bingham_alloc(&bmm.B[1], 4);
  bingham_alloc(&bmm.B[2], 4);
  bingham_alloc(&bmm.B[3], 4);
  bingham_copy(&bmm.B[0], &B);
  bingham_copy(&bmm.B[1], &B2);
  bingham_copy(&bmm.B[2], &B3);
  bingham_copy(&bmm.B[3], &B4);
  bmm.w[0] = 0.4;
  bmm.w[1] = 0.2;
  bmm.w[2] = 0.2;
  bmm.w[3] = 0.2;

  print_bingham(&bmm.B[0]);
  print_bingham(&bmm.B[1]);
  print_bingham(&bmm.B[2]);
  print_bingham(&bmm.B[3]);

  bingham_mixture_reduce(&bmm, 2);
  printf("Done\n");
}

void random_test(unsigned int n_start_components, unsigned int n_reduced_components)
{
/*  bingham_t binghams[n_start_components];
  double weights[n_start_components];

  unsigned int i;
  for(i = 0; i < n_start_components; ++i)
  {
    double Z[3];
    Z[0] = irand(900) - 900;
    Z[1] = irand(900) - 900;
    Z[2] = irand(900) - 900;
    double V[3][4] = {{0,1,0,0}, {0,0,1,0}, {0,0,0,1}};
    double *Vp[3] = {&V[0][0], &V[1][0], &V[2][0]};

    bingham_t b;
    bingham_new(&b, 4, Vp, Z);

    // apply kind of random rotation
    double q[4];
    q[0] = frand() * 2 - 1;
    q[1] = frand() * 2 - 1;
    q[2] = frand() * 2 - 1;
    q[3] = frand() * 2 - 1;
    double q_norm[4];
    normalize(q_norm, q, 4);

    bingham_alloc(&binghams[i], 4);
    bingham_post_rotate_3d(&binghams[i], &b, q_norm);

    // random weight
    weights[i] = frand();

    bingham_free(&b);
  }

  double weights_norm[n_start_components];
  normalize(weights_norm, weights, n_start_components);

  bingham_mix_t bmm;
  bmm.n = n_start_components;
  safe_malloc(bmm.w, n_start_components, double);
  safe_malloc(bmm.B, n_start_components, bingham_t);
  for(i = 0; i < n_start_components; ++i)
  {
    bingham_alloc(&bmm.B[i], 4);
    bingham_copy(&bmm.B[i], &binghams[i]);
    bmm.w[i] = weights_norm[i];
  }*/

  bingham_mix_t bmm;
  bingham_mixture_new_random(&bmm, n_start_components, -450);
  bingham_mixture_reduce(&bmm, n_reduced_components);
  printf("Done\n");
}

int main(int argc, char *argv[])
{
  segfault_test();
  random_test(100,10);
  return 0;
}
