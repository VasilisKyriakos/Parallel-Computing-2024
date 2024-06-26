CC=gcc-11
CFLAGS=-Wall -O3 -fopenmp
#CFLAGS+=-DDEBUG
LDLIBS=-lm

#TODO: add the following cases: multistart_mds_omp multistart_mds_omp_tasks multistart_mds_mpi

# Build all versions
all: multistart_mds_seq multistart_mds_omp multistart_mds_omp_tasks

# Sequential version
multistart_mds_seq: multistart_mds_seq.c torczon.c Makefile
	$(CC) $(CFLAGS) -o multistart_mds_seq multistart_mds_seq.c torczon.c $(LDLIBS)

# OpenMP version
multistart_mds_omp: multistart_mds_omp.c torczon.c Makefile
	$(CC) $(CFLAGS) -o multistart_mds_omp multistart_mds_omp.c torczon.c $(LDLIBS)


# OpenMP Tasks version
multistart_mds_omp_tasks: multistart_mds_omp_tasks.c torczon_omp_tasks.c Makefile
	$(CC) $(CFLAGS) -o multistart_mds_omp_tasks multistart_mds_omp_tasks.c torczon_omp_tasks.c $(LDLIBS)

clean:
	rm -f multistart_mds_seq multistart_mds_omp multistart_mds_omp_tasks
