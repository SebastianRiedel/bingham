
#include <stdio.h>
#include <stdlib.h>
#include <mpfr.h>


/***** hqnum -- wrappers for arbitrary precision "mpfr" library *****/

#define PRECISION 100  // bits
#define ROUNDING GMP_RNDN

typedef mpfr_t hqnum;
typedef mpfr_ptr hqnum_ptr;

#define hqinit(x) mpfr_init2(x, PRECISION)
#define hqget(x) mpfr_get_d(x, ROUNDING)                  // (double) x
#define hqset(x,y) mpfr_set_d(x, y, ROUNDING);            // x = y
#define hqfree(x) mpfr_clear(x)                           // free(x)
#define hqlog(x,y) mpfr_log(x, y, ROUNDING)               // x = log(y)
#define hqadd(x,y,z) mpfr_add(x, y, z, ROUNDING)          // x = y+z
#define hqsub(x,y,z) mpfr_sub(x, y, z, ROUNDING)          // x = y-z
#define hqmult(x,y,z) mpfr_mul(x, y, z, ROUNDING)         // x = y*z
#define hqmultsi(x,y,z) mpfr_mul_si(x, y, z, ROUNDING)    // x = y*z
#define hqmultui(x,y,z) mpfr_mul_ui(x, y, z, ROUNDING)    // x = y*z
#define hqexp(x,y) mpfr_exp(x, y, ROUNDING)               // x = exp(y)
#define hqlgamma(x,y) mpfr_lngamma(x, y, ROUNDING)        // x = lgamma(y)

int hqlogd(hqnum x, double y)                             // x = log(y)
{
  hqnum tmp;
  hqinit(tmp);
  hqset(tmp, y);
  int retval = hqlog(x, tmp);
  hqfree(tmp);
  return retval;
}

int hqlgammad(hqnum x, double y)                          // x = lgamma(y)
{
  hqnum tmp;
  hqinit(tmp);
  hqset(tmp, y);
  int retval = hqlgamma(x, tmp);
  hqfree(tmp);
  return retval;
}

hqnum_ptr hqalloc()
{
  hqnum tmp;  // use this to determine size to allocate
  hqnum_ptr x = (hqnum_ptr)calloc(1, sizeof(*tmp));
  if (!x) {
    printf("Error: bad alloc...exiting.\n");
    exit(-1);
  }
  return x;
}

#define MAXFACT 10000
hqnum_ptr hqlfact(unsigned int x)                         // lfact(x)
{
  static hqnum_ptr logf[MAXFACT];
  static int first = 1;
  int i;

  if (first) {
    first = 0;
    logf[0] = hqalloc();
    hqinit(logf[0]);
    hqset(logf[0], 0.0);
    for (i = 1; i < MAXFACT; i++) {        // logf[i] = log(i) + logf[i-1];
      logf[i] = hqalloc();
      hqinit(logf[i]);
      hqlogd(logf[i], i);
      hqadd(logf[i], logf[i], logf[i-1]);
    }
  }

  //printf("exp(hqlfact(%u)) = %.0f\n", x, exp(hqget(logf[x])));

  return logf[x];  
}

/********************************************************************/


//-------------------------- 2D ----------------------------//

double bingham_F_2d(double z1, double z2, int iter)
{
  int i, j;
  double log_z1_approx = log(fabs(z1));
  double log_z2_approx = log(fabs(z2));
  hqnum F, f, g, x, log_z1, log_z2;
  hqinit(F);
  hqinit(f);
  hqinit(g);
  hqinit(x);
  hqinit(log_z1);
  hqinit(log_z2);

  hqset(F, 0.0);
  hqlogd(log_z1, fabs(z1));
  hqlogd(log_z2, fabs(z2));

  //double g;
  //double log_z1 = log(fabs(z1));
  //double log_z2 = log(fabs(z2));

  for (i = 0; i < iter; i++) {
    for (j = 0; j < iter; j++) {

      //g = gamma(i+.5) + gamma(j+.5) - gamma(i+j+1.5) +
      //  i*log_z1 + j*log_z2 - lfact(i) - lfact(j);
      //if (exp(g) < 1e-8)
      //  break;
      //f = pow(-1, i+j)*exp(g);
      //F = F + f;

      double g_approx = gamma(i+.5) + gamma(j+.5) - gamma(i+j+1.5) +
	i*log_z1_approx + j*log_z2_approx - lfact(i) - lfact(j);

      if ((i > fabs(z1) || j > fabs(z2)) && g_approx < -20)  // exp(g_approx) < 2e-9
        break;

      hqset(x, 0.0);            // x = 0
      hqlgammad(g, i+.5);       // g = lgamma(i+.5)
      hqadd(x, x, g);           // x += g
      hqlgammad(g, j+.5);       // g = lgamma(j+.5)
      hqadd(x, x, g);           // x += g
      hqlgammad(g, i+j+1.5);    // g = lgamma(i+j+1.5)
      hqsub(x, x, g);           // x -= g
      hqmultui(g, log_z1, i);   // g = i*log_z1
      hqadd(x, x, g);           // x += g
      hqmultui(g, log_z2, j);   // g = j*log_z2
      hqadd(x, x, g);           // x += g
      hqsub(x, x, hqlfact(i));  // x -= lfact(i)
      hqsub(x, x, hqlfact(j));  // x -= lfact(j)

      //if ((i > fabs(z1) || j > fabs(z2)) && exp(hqget(x)) < 1e-8)
      //  break;

      hqexp(f, x);                    // f = exp(x)
      hqmultsi(f, f, (i+j)%2 ? -1 : 1);   // f *= pow(-1, i+j)

      hqadd(F, F, f);                 // F += f

      //printf("i=%d, j=%d, F = %f\n", i, j, hqget(F));
      //printf("i=%d, j=%d, k=%d, i+j+k=%d, pow(-1, i+j+k)=%f\n", i, j, k, i+j+k, pow(-1, i+j+k));
    }
  }

  double retval = hqget(F);

  hqfree(F);
  hqfree(f);
  hqfree(g);
  hqfree(x);
  hqfree(log_z1);
  hqfree(log_z2);

  return retval;
}

double bingham_dF1_2d(double z1, double z2, int iter)
{
  int i, j;
  double log_z1_approx = log(fabs(z1));
  double log_z2_approx = log(fabs(z2));
  hqnum F, f, g, x, log_z1, log_z2;
  hqinit(F);
  hqinit(f);
  hqinit(g);
  hqinit(x);
  hqinit(log_z1);
  hqinit(log_z2);

  hqset(F, 0.0);
  hqlogd(log_z1, fabs(z1));
  hqlogd(log_z2, fabs(z2));

  for (i = 1; i < iter; i++) {
    for (j = 0; j < iter; j++) {

      double g_approx = gamma(i+.5) + gamma(j+.5) - gamma(i+j+1.5) +
	(i-1)*log_z1_approx + j*log_z2_approx - lfact(i-1) - lfact(j);

      if ((i > fabs(z1) || j > fabs(z2)) && g_approx < -20)  // exp(g_approx) < 2e-9
        break;

      hqset(x, 0.0);              // x = 0
      hqlgammad(g, i+.5);         // g = lgamma(i+.5)
      hqadd(x, x, g);             // x += g
      hqlgammad(g, j+.5);         // g = lgamma(j+.5)
      hqadd(x, x, g);             // x += g
      hqlgammad(g, i+j+1.5);      // g = lgamma(i+j+1.5)
      hqsub(x, x, g);             // x -= g
      hqmultui(g, log_z1, i-1);   // g = (i-1)*log_z1
      hqadd(x, x, g);             // x += g
      hqmultui(g, log_z2, j);     // g = j*log_z2
      hqadd(x, x, g);             // x += g
      hqsub(x, x, hqlfact(i-1));  // x -= lfact(i-1)
      hqsub(x, x, hqlfact(j));    // x -= lfact(j)

      //if ((i > fabs(z1) || j > fabs(z2)) && exp(hqget(x)) < 1e-8)
      //  break;

      hqexp(f, x);                          // f = exp(x)
      hqmultsi(f, f, (i+j-1)%2 ? -1 : 1);   // f *= pow(-1, i+j-1)

      hqadd(F, F, f);                 // F += f

      //printf("i=%d, j=%d, F = %f\n", i, j, hqget(F));
      //printf("i=%d, j=%d, k=%d, i+j+k=%d, pow(-1, i+j+k)=%f\n", i, j, k, i+j+k, pow(-1, i+j+k));
    }
  }

  double retval = hqget(F);

  hqfree(F);
  hqfree(f);
  hqfree(g);
  hqfree(x);
  hqfree(log_z1);
  hqfree(log_z2);

  return retval;
}

double bingham_dF2_2d(double z1, double z2, int iter)
{
  int i, j;
  double log_z1_approx = log(fabs(z1));
  double log_z2_approx = log(fabs(z2));
  hqnum F, f, g, x, log_z1, log_z2;
  hqinit(F);
  hqinit(f);
  hqinit(g);
  hqinit(x);
  hqinit(log_z1);
  hqinit(log_z2);

  hqset(F, 0.0);
  hqlogd(log_z1, fabs(z1));
  hqlogd(log_z2, fabs(z2));

  for (i = 0; i < iter; i++) {
    for (j = 1; j < iter; j++) {

      double g_approx = gamma(i+.5) + gamma(j+.5) - gamma(i+j+1.5) +
	i*log_z1_approx + (j-1)*log_z2_approx - lfact(i) - lfact(j-1);

      if ((i > fabs(z1) || j > fabs(z2)) && g_approx < -20)  // exp(g_approx) < 2e-9
        break;

      hqset(x, 0.0);              // x = 0
      hqlgammad(g, i+.5);         // g = lgamma(i+.5)
      hqadd(x, x, g);             // x += g
      hqlgammad(g, j+.5);         // g = lgamma(j+.5)
      hqadd(x, x, g);             // x += g
      hqlgammad(g, i+j+1.5);      // g = lgamma(i+j+1.5)
      hqsub(x, x, g);             // x -= g
      hqmultui(g, log_z1, i);     // g = i*log_z1
      hqadd(x, x, g);             // x += g
      hqmultui(g, log_z2, j-1);   // g = (j-1)*log_z2
      hqadd(x, x, g);             // x += g
      hqsub(x, x, hqlfact(i));    // x -= lfact(i)
      hqsub(x, x, hqlfact(j-1));  // x -= lfact(j-1)

      //if ((i > fabs(z1) || j > fabs(z2)) && exp(hqget(x)) < 1e-8)
      //  break;

      hqexp(f, x);                          // f = exp(x)
      hqmultsi(f, f, (i+j-1)%2 ? -1 : 1);   // f *= pow(-1, i+j-1)

      hqadd(F, F, f);                 // F += f

      //printf("i=%d, j=%d, F = %f\n", i, j, hqget(F));
      //printf("i=%d, j=%d, k=%d, i+j+k=%d, pow(-1, i+j+k)=%f\n", i, j, k, i+j+k, pow(-1, i+j+k));
    }
  }

  double retval = hqget(F);

  hqfree(F);
  hqfree(f);
  hqfree(g);
  hqfree(x);
  hqfree(log_z1);
  hqfree(log_z2);

  return retval;
}

