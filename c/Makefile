IDIR=include
ODIR=.
LDIR=lib

INSTALL_DIR=/usr

CC=gcc
CPP=g++
NVCC=nvcc
#MEX=mex
MEX="C:/Program Files/MATLAB/R2009a/bin/mex.bat"
LINK=ar rcs
CFLAGS=-Wall -I$(IDIR) -O3 -msse2
#CFLAGS=-Wall -I$(IDIR) -O3 -msse2 -pg
#CFLAGS=-Wall -I$(IDIR) -O0 -g -pg

CUDAFLAGS=-O3 -lcuda -lcudart -lcurand -arch=compute_20
#CUDAFLAGS=-O3 -lcuda -lcudart -lcurand -arch=compute_20 -lineinfo
#CUDAFLAGS=-g -G -lcuda -lcudart -lcurand -arch=compute_20

#CYGWIN=1
#WINDOWS=1


ifdef WINDOWS
CFLAGS += -static -mno-cygwin -I"C:\Program Files\GnuWin32\include" -DHAVE_WINDOWS #-mwindows -I/usr/include
#CFLAGS = -I$(IDIR) -I"C:\Program*\GnuWin32\include" -DHAVE_WINDOWS
endif

LFLAGS=-lm #-llapacke -llapack -lblas -lgfortran #-lflann #-lduma

_DEPS = bingham.h bingham/bingham_constants.h bingham/bingham_constant_tables.h \
	bingham/util.h bingham/tetramesh.h bingham/octetramesh.h bingham/hypersphere.h bingham/hll.h bingham/olf.h bingham/cuda_wrapper.h #bingham/gauss_mix.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

ifdef CYGWIN
LIBS += libbingham.so.1.0.1 #$(LDIR)/bingham.dll
else
LIBS = libbingham.a  #libolf.a
endif
ifndef WINDOWS
PROGRAMS = fit_bingham cluster_bingham bingham_lookup bingham_sample tessellate_S3 test_util  #scope mope olf_pose_sample gauss_mix test_olf test_bingham test_hll
endif

TARGETS = $(LIBS) $(PROGRAMS)


all: $(TARGETS)

#ifdef WINDOWS
#$(ODIR)/%.o: %.c $(DEPS)
#	$(MEX) -c $< $(CFLAGS)
#else
$(ODIR)/%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
$(ODIR)/%.o: %.cpp $(DEPS)
	$(CPP) -c -o $@ $< $(CFLAGS)
#$(ODIR)/%.o: %.cu $(DEPS)
#	$(NVCC) -c -o $@ $< $(CUDAFLAGS)
#endif

libolf.a: olf.o olf_cuda.o bingham.o bingham_constants.o tetramesh.o octetramesh.o hypersphere.o util.o
	$(LINK) $@ $^

libbingham.a: bingham.o bingham_constants.o tetramesh.o octetramesh.o hypersphere.o util.o #olf.o olf_cuda.o
	$(LINK) $@ $^

libbingham.so.1.0.1: bingham.o bingham_constants.o tetramesh.o octetramesh.o hypersphere.o util.o #olf.o olf_cuda.o
	$(CC) -shared -Wl,-soname,libbingham.so.1 -o $@ $^ $(LFLAGS)

$(LDIR)/bingham.dll: bingham.o bingham_constants.o tetramesh.o octetramesh.o hypersphere.o util.o #olf.o olf_cuda.o
	$(CC) -shared -o $@ $^ $(LFLAGS)

test_bingham: test_bingham.o libbingham.a
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

fit_bingham: fit_bingham.o libbingham.a
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

bingham_lookup: bingham_lookup.o libbingham.a
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

bingham_sample: bingham_sample.o libbingham.a
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

cluster_bingham: cluster_bingham.o libbingham.a
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

tessellate_S3: tessellate_S3.o libbingham.a
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

test_util: test_util.o util.o
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

test_hll: test_hll.o hll.o libbingham.a
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

#test_olf: test_olf.o olf.o libbingham.a
#	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

#olf_pose_sample: olf_pose_sample.o olf.o libbingham.a
#	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

scope: scope.o olf.o olf_cuda.o libbingham.a
	$(CPP) -o $@ $^ $(CFLAGS) $(LFLAGS) -L/usr/local/cuda/lib64 -lcuda -lcudart -lcurand

mope: mope.o olf.o olf_cuda.o libbingham.a
	$(CPP) -o $@ $^ $(CFLAGS) $(LFLAGS) -L/usr/local/cuda/lib64 -lcuda -lcudart -lcurand

#gauss_mix: gauss_mix.o libbingham.a
#	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

.PHONY: clean install doc

clean:
	rm -f $(TARGETS) $(ODIR)/*.o *.a *~ core *.exe $(IDIR)/*~ $(IDIR)/*/*~

install:
	if [ ! -d $(INSTALL_DIR)/include/bingham ] ; then mkdir $(INSTALL_DIR)/include/bingham ; fi
	cp $(DEPS) $(INSTALL_DIR)/include/bingham/
	mv $(INSTALL_DIR)/include/bingham/bingham.h $(INSTALL_DIR)/include/
	cp $(LIBS) $(INSTALL_DIR)/lib/
	cp $(PROGRAMS) $(INSTALL_DIR)/bin/

doc:
	doxygen Doxyfile
