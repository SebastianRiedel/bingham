//------------------- Bingham sampling -------------------//


/*
 * Return a uniform sample (in X) over the sphere S2.
 */
static void uniform_sample_2d(double *X)
{
  double u1, u2, u3;
  double x1, x2, x3;
  double c1, c2, c3;

  while (1) {
    u1 = frand();
    u2 = frand();
    if (u1 + u2 <= 1)
      break;
  }
  u3 = 1 - u1 - u2;

  c1 = (frand() < .5 ? -1 : 1);
  c2 = (frand() < .5 ? -1 : 1);
  c3 = (frand() < .5 ? -1 : 1);

  x1 = c1*sqrt(u1);
  x2 = c2*sqrt(u2);
  x3 = c3*sqrt(u3);

  X[0] = x1;
  X[1] = x2;
  X[2] = x3;


  //dbug

  static double x1_avg=0, x2_avg=0, x3_avg=0, cnt=0;
  static double VXsq1_avg=0, VXsq2_avg=0, VXsq3_avg=0;
  double d1, d2, d3;

  x1_avg = (cnt*x1_avg + x1)/(cnt+1);
  x2_avg = (cnt*x2_avg + x2)/(cnt+1);
  x3_avg = (cnt*x3_avg + x3)/(cnt+1);

  d1 = x1;
  d2 = x2;
  d3 = x3;
  VXsq1_avg = (cnt*VXsq1_avg + d1*d1)/(cnt+1);
  VXsq2_avg = (cnt*VXsq2_avg + d2*d2)/(cnt+1);
  VXsq3_avg = (cnt*VXsq3_avg + d3*d3)/(cnt+1);

  cnt++;

  printf(" ***  x_avg = (%f, %f, %f)  ***  ", x1_avg, x2_avg, x3_avg);

  printf("VXsq_avg = (%f, %f, %f)  ***  ", VXsq1_avg, VXsq2_avg, VXsq3_avg);
}


/*
 * Return a sample (in X) from the bingham distribution (Z,V)
 * using truncated Gaussians.
 */
static void bingham_sample_2d(double *X, double Z[2], double V[][3], double sigma1, double sigma2)
{
  double u1, u2, x1, x2, x3; //, c3;

  while (1) {
    u1 = frand();
    u2 = frand();
    x1 = sigma1*sqrt(2)*erfinv((2*u1-1)*erf(1/(sigma1*sqrt(2))));
    x2 = sigma2*sqrt(2)*erfinv((2*u2-1)*erf(1/(sigma2*sqrt(2))));
    x3 = 2*frand() - 1;

    printf("(%f, %f, %f)\n", x1, x2, x3);

    //if (x1*x1 + x2*x2 + x3*x3 <= 1)
      break;
    //printf(".");
  }

  double d = sqrt(x1*x1+x2*x2+x3*x3);
  x1 /= d;
  x2 /= d;
  x3 /= d;

  // pick sign of x3 at random
  //c3 = (frand() < .5 ? -1 : 1);
  //x3 = c3*sqrt(1 - x1*x1 - x2*x2);

  printf(" ***  x = (%f, %f, %f)  ***  ", x1, x2, x3);

  X[0] = x1*V[0][0] + x2*V[1][0] + x3*V[2][0];
  X[1] = x1*V[0][1] + x2*V[1][1] + x3*V[2][1];
  X[2] = x1*V[0][2] + x2*V[1][2] + x3*V[2][2];


  //dbug

  static double x1_avg=0, x2_avg=0, x3_avg=0, cnt=0;
  static double VXsq1_avg=0, VXsq2_avg=0, VXsq3_avg=0;
  double d1, d2, d3;

  x1_avg = (cnt*x1_avg + x1)/(cnt+1);
  x2_avg = (cnt*x2_avg + x2)/(cnt+1);
  x3_avg = (cnt*x3_avg + x3)/(cnt+1);

  d1 = dot(V[0], X, 3);
  d2 = dot(V[1], X, 3);
  d3 = dot(V[2], X, 3);
  VXsq1_avg = (cnt*VXsq1_avg + d1*d1)/(cnt+1);
  VXsq2_avg = (cnt*VXsq2_avg + d2*d2)/(cnt+1);
  VXsq3_avg = (cnt*VXsq3_avg + d3*d3)/(cnt+1);

  cnt++;

  printf(" ***  x_avg = (%f, %f, %f)  ***  ", x1_avg, x2_avg, x3_avg);

  printf("VXsq_avg = (%f, %f, %f)  ***  ", VXsq1_avg, VXsq2_avg, VXsq3_avg);
}



/*
 * Return a sample (in X) from the bingham distribution (Z,V)
 * using truncation to simplex (method 1) from Kent 2004.
 */
static void bingham_sample_2d_kent(double *X, double Z[2], double V[][3])
{
  double u1, u2, s1, s2, s3, c1, c2, c3;

  while (1) {
    u1 = frand();
    u2 = frand();
    s1 = (1/Z[0])*log(1 - u1*(1 - exp(Z[0])));
    s2 = (1/Z[1])*log(1 - u2*(1 - exp(Z[1])));
    if (s1 + s2 <= 1)
      break;
    //printf(".");
  }

  s3 = 1 - s1 - s2;

  //printf(" ***  s = (%f, %f, %f)  ***  ", s1, s2, s3);

  // pick signs C at random, return X = V*C*sqrt(s)
  c1 = (frand() < .5 ? -1 : 1);
  c2 = (frand() < .5 ? -1 : 1);
  c3 = (frand() < .5 ? -1 : 1);

  X[0] = c1*sqrt(s1)*V[0][0] + c2*sqrt(s2)*V[1][0] + c3*sqrt(s3)*V[2][0];
  X[1] = c1*sqrt(s1)*V[0][1] + c2*sqrt(s2)*V[1][1] + c3*sqrt(s3)*V[2][1];
  X[2] = c1*sqrt(s1)*V[0][2] + c2*sqrt(s2)*V[1][2] + c3*sqrt(s3)*V[2][2];


  //dbug

  static double s1_avg=0, s2_avg=0, s3_avg=0, cnt=0;
  static double VXsq1_avg=0, VXsq2_avg=0, VXsq3_avg=0;
  double d1, d2, d3;

  s1_avg = (cnt*s1_avg + s1)/(cnt+1);
  s2_avg = (cnt*s2_avg + s2)/(cnt+1);
  s3_avg = (cnt*s3_avg + s3)/(cnt+1);

  d1 = dot(V[0], X, 3);
  d2 = dot(V[1], X, 3);
  d3 = dot(V[2], X, 3);
  VXsq1_avg = (cnt*VXsq1_avg + d1*d1)/(cnt+1);
  VXsq2_avg = (cnt*VXsq2_avg + d2*d2)/(cnt+1);
  VXsq3_avg = (cnt*VXsq3_avg + d3*d3)/(cnt+1);

  cnt++;

  printf(" ***  s_avg = (%f, %f, %f)  ***  ", s1_avg, s2_avg, s3_avg);

  printf("VXsq_avg = (%f, %f, %f)  ***  ", VXsq1_avg, VXsq2_avg, VXsq3_avg);
}


/*
 * Return a sample (in X) from the bingham distribution (Z,V)
 * using truncation to simplex (method 1) from Kent 2004.
 */
static void bingham_sample_3d(double *X, double Z[3], double V[][4])
{
  double u1, u2, u3, s1, s2, s3, s4, c1, c2, c3, c4;

  while (1) {
    u1 = frand();
    u2 = frand();
    u3 = frand();
    s1 = (1/Z[0])*log(1 - u1*(1 - exp(Z[0])));
    s2 = (1/Z[1])*log(1 - u2*(1 - exp(Z[1])));
    s3 = (1/Z[2])*log(1 - u3*(1 - exp(Z[2])));
    if (s1 + s2 + s3 <= 1)
      break;
    //printf(".");
  }

  s4 = 1 - s1 - s2 - s3;

  // pick signs C at random, return X = V*C*sqrt(s)
  c1 = (frand() < .5 ? -1 : 1);
  c2 = (frand() < .5 ? -1 : 1);
  c3 = (frand() < .5 ? -1 : 1);
  c4 = (frand() < .5 ? -1 : 1);

  X[0] = c1*sqrt(s1)*V[0][0] + c2*sqrt(s2)*V[1][0] + c3*sqrt(s3)*V[2][0] + c4*sqrt(s4)*V[3][0];
  X[1] = c1*sqrt(s1)*V[0][1] + c2*sqrt(s2)*V[1][1] + c3*sqrt(s3)*V[2][1] + c4*sqrt(s4)*V[3][1];
  X[2] = c1*sqrt(s1)*V[0][2] + c2*sqrt(s2)*V[1][2] + c3*sqrt(s3)*V[2][2] + c4*sqrt(s4)*V[3][2];
  X[3] = c1*sqrt(s1)*V[0][3] + c2*sqrt(s2)*V[1][3] + c3*sqrt(s3)*V[2][3] + c4*sqrt(s4)*V[3][3];
}




