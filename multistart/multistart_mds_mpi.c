#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>

#define MAXVARS (250)   /* max # of variables */
#define EPSMIN (1E-6)  /* ending value of stepsize */

/* prototype of local optimization routine, code available in torczon.c */
extern void mds(double *startpoint, double *endpoint, int n, double *val, double eps, int maxfevals, int maxiter,
                double mu, double theta, double delta, int *ni, int *nf, double *xl, double *xr, int *term);

/* global variables */
unsigned long local_funevals = 0;

/* Rosenbrock classic parabolic valley ("banana") function */
double f(double *x, int n) {
    double fv;
    int i;

    local_funevals++;
    fv = 0.0;
    for (i = 0; i < n - 1; i++)   /* rosenbrock */
        fv = fv + 100.0 * pow((x[i + 1] - x[i] * x[i]), 2) + pow((x[i] - 1.0), 2);
    usleep(10);  /* do not remove, introduces some artificial work */

    return fv;
}

double get_wtime(void) {

    return MPI_Wtime();

}

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* problem parameters */
    int nvars = 4;      /* number of variables (problem dimension) */
    int ntrials = 64;   /* number of trials */
    double lower[MAXVARS], upper[MAXVARS]; /* lower and upper bounds */

    /* mds parameters */
    double eps = EPSMIN;
    int maxfevals = 10000;
    int maxiter = 10000;
    double mu = 1.0;
    double theta = 0.25;
    double delta = 0.25;

    double startpt[MAXVARS], endpt[MAXVARS];    /* initial and final point of mds */
    double fx;  /* function value at the final point of mds */
    int nt, nf; /* number of iterations and function evaluations used by mds */

    /* information about the best point found by multistart */
    double best_pt[MAXVARS];
    double best_fx = 1e10;
    int best_trial = -1;
    int best_nt = -1;
    int best_nf = -1;

    /* local variables */
    int trial, i;
    double t0, t1;

    /* initialization of lower and upper bounds of search space */
    for (i = 0; i < MAXVARS; i++) lower[i] = -2.0; /* lower bound: -2.0 */
    for (i = 0; i < MAXVARS; i++) upper[i] = +2.0; /* upper bound: +2.0 */

    t0 = get_wtime();
    long tseed =1;

    unsigned short randBuffer[3];
    randBuffer[0] = 0;
    randBuffer[1] = 0;
    randBuffer[2] = tseed + rank;
    
    for (trial = rank; trial < ntrials; trial += size) {
        /* starting guess for rosenbrock test function, search space in [-2, 2) */
        for (i = 0; i < nvars; i++) {
            startpt[i] = lower[i] + (upper[i] - lower[i]) * erand48(randBuffer);
        }

        int term = -1;
        mds(startpt, endpt, nvars, &fx, eps, maxfevals, maxiter, mu, theta, delta, &nt, &nf, lower, upper, &term);

        if (fx < best_fx) {
            best_trial = trial;
            best_nt = nt;
            best_nf = nf;
            best_fx = fx;
            for (i = 0; i < nvars; i++)
                best_pt[i] = endpt[i];
        }
    }

    /* Gather results from all processes */
    struct {
        double fx;
        int rank;
    } local_result, global_result;

    local_result.fx = best_fx;
    local_result.rank = rank;

    MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

    if (rank == global_result.rank) {
        MPI_Bcast(&best_trial, 1, MPI_INT, global_result.rank, MPI_COMM_WORLD);
        MPI_Bcast(&best_nt, 1, MPI_INT, global_result.rank, MPI_COMM_WORLD);
        MPI_Bcast(&best_nf, 1, MPI_INT, global_result.rank, MPI_COMM_WORLD);
        MPI_Bcast(best_pt, MAXVARS, MPI_DOUBLE, global_result.rank, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(&best_trial, 1, MPI_INT, global_result.rank, MPI_COMM_WORLD);
        MPI_Bcast(&best_nt, 1, MPI_INT, global_result.rank, MPI_COMM_WORLD);
        MPI_Bcast(&best_nf, 1, MPI_INT, global_result.rank, MPI_COMM_WORLD);
        MPI_Bcast(best_pt, MAXVARS, MPI_DOUBLE, global_result.rank, MPI_COMM_WORLD);
    }

    /* Reduce the function evaluations across all processes */
    unsigned long global_funevals;
    MPI_Reduce(&local_funevals, &global_funevals, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    t1 = get_wtime();

    if (rank == 0) {
        printf("\n\nFINAL RESULTS (MPI):\n");
        printf("Elapsed time = %.3lf s\n", t1 - t0);
        printf("Total number of trials = %d\n", ntrials);
        printf("Total number of function evaluations = %ld\n", global_funevals);
        printf("Best result at trial %d used %d iterations, %d function calls and returned\n", best_trial, best_nt, best_nf);
        for (i = 0; i < nvars; i++) {
            printf("x[%3d] = %15.7le \n", i, best_pt[i]);
        }
        printf("f(x) = %15.7le\n", best_fx);
    }

    MPI_Finalize();
    return 0;
}
