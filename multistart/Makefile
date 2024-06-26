CC=gcc-11
CFLAGS=-Wall -O3 -fopenmp
#CFLAGS+=-DDEBUG
LDLIBS=-lm

# Build all versions
all: multistart_mds_seq multistart_mds_omp

# Sequential version
multistart_mds_seq: multistart_mds_seq.c torczon.c Makefile
	$(CC) $(CFLAGS) -o multistart_mds_seq multistart_mds_seq.c torczon.c $(LDLIBS)

# OpenMP version
multistart_mds_omp: multistart_mds_omp.c torczon.c Makefile
	$(CC) $(CFLAGS) -o multistart_mds_omp multistart_mds_omp.c torczon.c $(LDLIBS)


clean:
	rm -f multistart_mds_seq multistart_mds_omp
