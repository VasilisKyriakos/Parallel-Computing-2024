CC=gcc-11
CFLAGS=-Wall -O3 -fopenmp
LDLIBS=-lm

MPIFLAGS =-Wall -O3
MPICC=mpicc

#CFLAGS+=-DDEBUG

#TODO: add the following cases: multistart_mds_omp multistart_mds_omp_tasks multistart_mds_mpi

# Build all versions
all: multistart_mds_seq multistart_mds_omp multistart_mds_omp_tasks multistart_mds_mpi

# Sequential version
multistart_mds_seq: multistart_mds_seq.c torczon.c json_writer.c Makefile
	$(CC) $(CFLAGS) -o multistart_mds_seq multistart_mds_seq.c torczon.c json_writer.c $(LDLIBS)

# OpenMP version
multistart_mds_omp: multistart_mds_omp.c torczon.c json_writer.c Makefile
	$(CC) $(CFLAGS) -o multistart_mds_omp multistart_mds_omp.c torczon.c json_writer.c $(LDLIBS)


# OpenMP Tasks version
multistart_mds_omp_tasks: multistart_mds_omp_tasks.c torczon_temp.c json_writer.c Makefile
	$(CC) $(CFLAGS) -o multistart_mds_omp_tasks multistart_mds_omp_tasks.c torczon_temp.c json_writer.c $(LDLIBS)


# MPI version
multistart_mds_mpi: multistart_mds_mpi.c torczon.c json_writer.c Makefile
	$(MPICC) $(MPIFLAGS) -o multistart_mds_mpi multistart_mds_mpi.c torczon.c json_writer.c $(LDLIBS)


clean:
	rm -f multistart_mds_seq multistart_mds_omp multistart_mds_omp_tasks multistart_mds_mpi


#pip install pandas matplotlib seaborn scipy
