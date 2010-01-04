IDIR=include
ODIR=.
LDIR =lib

INSTALL_DIR=/usr

CC=gcc
LINK=ar rcs
CFLAGS=-Wall -I$(IDIR) -O3 #-pg

#CYGWIN=1

ifdef CYGWIN
CFLAGS += -mno-cygwin -mwindows -I/usr/include
endif

LFLAGS=-lm -lgsl -lgslcblas #-lduma

_DEPS = bingham.h bingham/bingham_constants.h bingham/bingham_constant_tables.h \
	bingham/util.h bingham/tetramesh.h bingham/octetramesh.h bingham/hypersphere.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

ifdef CYGWIN
LIBS = $(LDIR)/bingham.dll
TARGETS = $(LIBS)
else
LIBS = libbingham.a
TARGETS = $(LIBS) test_bingham test_util fit_bingham cluster_bingham
endif



all: $(TARGETS)

$(ODIR)/%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

libbingham.a: bingham.o bingham_constants.o tetramesh.o octetramesh.o hypersphere.o util.o
	$(LINK) $@ $^

$(LDIR)/bingham.dll: bingham.o bingham_constants.o tetramesh.o octetramesh.o hypersphere.o util.o
	$(CC) -shared -o $@ $^ $(LFLAGS)

test_bingham: test_bingham.o libbingham.a
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

fit_bingham: fit_bingham.o libbingham.a
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

cluster_bingham: cluster_bingham.o libbingham.a
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

test_util: test_util.o util.o
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

.PHONY: clean install

clean:
	rm -f $(TARGETS) $(ODIR)/*.o *.a *~ core *.exe $(IDIR)/*~ $(IDIR)/*/*~

install:
	if [ ! -d $(INSTALL_DIR)/include/bingham ] ; then mkdir $(INSTALL_DIR)/include/bingham ; fi
	cp $(DEPS) $(INSTALL_DIR)/include/bingham/
	mv $(INSTALL_DIR)/include/bingham/bingham.h $(INSTALL_DIR)/include/
	cp $(LIBS) $(INSTALL_DIR)/lib/
