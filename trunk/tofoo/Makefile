IDIR=.#include
ODIR=.
LDIR=.#lib

INSTALL_DIR=/usr

CC=g++ #gcc
LINK=ar rcs
#CFLAGS=-Wall -I$(IDIR) -O3 #-pg
CFLAGS = -Ieigen -I/home/jglov/bingham/c/include

#CYGWIN=1

#ifdef CYGWIN
#CFLAGS += -mno-cygwin -mwindows -I/usr/include
#endif

#LFLAGS=-lm -lgsl -lgslcblas #-lduma
LFLAGS=-L/home/jglov/bingham/c -lbingham -lgsl -lgslcblas

#ifdef CYGWIN
#LFLAGS=-static -L/home/jglov/bingham -lbingham
#else
#LFLAGS=-L/home/jglov/bingham -lbingham
#endif


#_DEPS = bingham.h bingham/bingham_constants.h bingham/bingham_constant_tables.h \
#	bingham/util.h bingham/tetramesh.h bingham/octetramesh.h bingham/hypersphere.h
#DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))
DEPS = CommonIORoutines.h StringTokenizer.h pointcloud.h ANN.h
DEPS += eigen/Eigen/Core


#ifdef CYGWIN
#LIBS = $(LDIR)/bingham.dll
#TARGETS = $(LIBS)
#else
#LIBS = libbingham.a
#TARGETS = $(LIBS) test_bingham test_util fit_bingham cluster_bingham
#endif

TARGETS = merge_pcd downsample_pcd pfh_cluster bingham_fit_clusters tofoo



all: $(TARGETS)

$(ODIR)/%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

merge_pcd: merge_pcd.o pointcloud.o CommonIORoutines.o StringTokenizer.o
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

downsample_pcd: downsample_pcd.o pointcloud.o CommonIORoutines.o StringTokenizer.o
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

pfh_cluster: pfh_cluster.o pointcloud.o CommonIORoutines.o StringTokenizer.o
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

bingham_fit_clusters: bingham_fit_clusters.o pointcloud.o CommonIORoutines.o StringTokenizer.o
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

tofoo: tofoo.o pointcloud.o CommonIORoutines.o StringTokenizer.o
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)


.PHONY: clean #install

clean:
	rm -f $(TARGETS) $(ODIR)/*.o *.a *~ core *.exe

#install:
#	if [ ! -d $(INSTALL_DIR)/include/bingham ] ; then mkdir $(INSTALL_DIR)/include/bingham ; fi
#	cp $(DEPS) $(INSTALL_DIR)/include/bingham/
#	mv $(INSTALL_DIR)/include/bingham/bingham.h $(INSTALL_DIR)/include/
#	cp $(LIBS) $(INSTALL_DIR)/lib/
