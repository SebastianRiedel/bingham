
#include <stdlib.h>
#include <math.h>


// fixed96 -- fixed point number of the form "a.b"
typedef struct {
  long long a;
  unsigned int b;
} fixed96;

double fixed96_to_double(fixed96 f)
{
  return f.a + f.b/pow(2,32);
}

fixed96 new_fixed96(double x)
{
  fixed96 f;

  if (x == 0.0) {
    f.a = 0;
    f.b = 0;
  }
  else {
    long long s = (x < 0 ? -1 : 1);
    double y = fabs(x);
    f.a = s*(long long)y;
    //printf("y - f.a = %f\n", y - (long long)y);
    f.b = (unsigned int)((y - (long long)y)*pow(2,32));
    if (f.b != 0 && s < 0) {  // make f.b > 0
      f.b = (unsigned int)(0x100000000ll - (long long)f.b);
      f.a = f.a - 1;
    }
  }

  double err = fixed96_to_double(f) - x;
  if (fabs(err) > .000001)
    printf("new_fixed96(%f) = (%lld, %f)  -->  err = %f\n", x, f.a, f.b/pow(2,32), err);

  return f;
}

fixed96 add96(fixed96 x, fixed96 y)
{
  fixed96 z;

  if (x.a == 0 && x.b == 0) {
    z = y;
    return z;
  }

  if (y.a == 0 && y.b == 0) {
    z = x;
    return z;
  }

  // compute z.a
  z.a = x.a + y.a;

  // compute z.b
  //int sx = (x.a < 0 ? -1 : 1);
  //int sy = (y.a < 0 ? -1 : 1);
  //long long b = (sx*(long long)x.b) + (sy*(long long)y.b);
  long long b = ((long long)x.b) + ((long long)y.b);

  if (b > 0xFFFFFFFFll) {
    b = b - 0x100000000ll;
    z.a = z.a + 1;
  }
  else if (b < -0xFFFFFFFFll) {
    b = b + 0x100000000ll;
    z.a = z.a - 1;
  }

  // now normalize z.b to be positive
  if (b < 0) {
    z.a = z.a - 1;
    z.b = (unsigned int)(b + 0x100000000ll);
  }
  else
    z.b = (unsigned int)b;

  static double tot_err = 0.0;
  double err = fixed96_to_double(z) - (fixed96_to_double(x) + fixed96_to_double(y));
  tot_err += err;
  if (fabs(err) > .000001)
    printf("(%lld, %f) + (%lld, %f) = (%lld, %f)  ***  err = %f, tot_err = %f\n", x.a, x.b/pow(2,32),
	   y.a, y.b/pow(2,32), z.a, z.b/pow(2,32), err, tot_err);

  return z;
}
