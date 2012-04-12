IDIR=include
ODIR=.
LDIR=lib

INSTALL_DIR=/usr

CC=gcc
#MEX=mex
MEX="C:/Program Files/MATLAB/R2009a/bin/mex.bat"
LINK=ar rcs
CFLAGS=-Wall -I$(IDIR) -O3 -msse2 #-pg

#CYGWIN=1
#WINDOWS=1


ifdef WINDOWS
CFLAGS += -static -mno-cygwin -I"C:\Program Files\GnuWin32\include" -DHAVE_WINDOWS #-mwindows -I/usr/include
#CFLAGS = -I$(IDIR) -I"C:\Program*\GnuWin32\include" -DHAVE_WINDOWS
endif

LFLAGS=-lm #-lduma

_DEPS = bingham.h bingham/bingham_constants.h bingham/bingham_constant_tables.h \
	bingham/util.h bingham/tetramesh.h bingham/octetramesh.h bingham/hypersphere.h bingham/hll.h bingham/olf.h bingham/gauss_mix.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

ifdef CYGWIN
LIBS += libbingham.so.1.0.1 #$(LDIR)/bingham.dll
else
LIBS = libbingham.a
endif
ifdef WINDOWS
TARGETS = $(LIBS)
else
TARGETS = $(LIBS) test_bingham test_hll test_olf test_util fit_bingham cluster_bingham bingham_lookup bingham_sample olf_pose_sample gauss_mix
endif



all: $(TARGETS)

#ifdef WINDOWS
#$(ODIR)/%.o: %.c $(DEPS)
#	$(MEX) -c $< $(CFLAGS)
#else
$(ODIR)/%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
#endif

libbingham.a: bingham.o bingham_constants.o hll.o olf.o gauss_mix.o tetramesh.o octetramesh.o hypersphere.o util.o
	$(LINK) $@ $^

libbingham.so.1.0.1: bingham.o bingham_constants.o hll.o olf.o gauss_mix.o tetramesh.o octetramesh.o hypersphere.o util.o
	$(CC) -shared -Wl,-soname,libbingham.so.1 -o $@ $^ $(LFLAGS)

$(LDIR)/bingham.dll: bingham.o bingham_constants.o hll.o olf.o gauss_mix.o tetramesh.o octetramesh.o hypersphere.o util.o
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

test_util: test_util.o util.o
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

test_hll: test_hll.o libbingham.a
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

test_olf: test_olf.o libbingham.a
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

olf_pose_sample: olf_pose_sample.o libbingham.a
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

gauss_mix: gauss_mix.o libbingham.a
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

.PHONY: clean install doc

clean:
	rm -f $(TARGETS) $(ODIR)/*.o *.a *~ core *.exe $(IDIR)/*~ $(IDIR)/*/*~
	rm -Rf doc

install:
	if [ ! -d $(INSTALL_DIR)/include/bingham ] ; then mkdir $(INSTALL_DIR)/include/bingham ; fi
	cp $(DEPS) $(INSTALL_DIR)/include/bingham/
	mv $(INSTALL_DIR)/include/bingham/bingham.h $(INSTALL_DIR)/include/
	cp $(LIBS) $(INSTALL_DIR)/lib/

doc:
	doxygen Doxyfile
