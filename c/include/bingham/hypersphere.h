
#ifndef BINGHAM_HYPERSPHERE_H
#define BINGHAM_HYPERSPHERE_H

#include "bingham/tetramesh.h"
#include "bingham/octetramesh.h"


/** Tools for creating a finite element representation of a hypersphere **/


void hypersphere_init();
tetramesh_t *tessellate_S3(int n);






//------------ DEPRECATED ------------//

// Create a tesselation of the 3-sphere (in R4) with at least n cells.
//tetramesh_t *hypersphere_tessellation_tetra(int n);
//octetramesh_t *hypersphere_tessellation_octetra(int n);

// Multi-resolution tessellation of the 3-sphere based on approximating a scalar function f:S3->R to a given resolution.
//octetramesh_t *hypersphere_tessellation_octetra_mres(double(*f)(double *, void *), void *fdata, double resolution);





#endif
