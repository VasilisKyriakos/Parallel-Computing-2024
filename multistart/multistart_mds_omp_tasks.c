#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>

#define MAXVARS     (250)   /* max # of variables         */
#define EPSMIN      (1E-6)  /* ending value of stepsize  */

/* prototype of local optimization routine, code available in torczon.c */
extern void mds(double *startpoint, double *endpoint, int n, double *val, double eps, int maxfevals, int maxiter,
         double mu, double theta, double delta, int *ni, int *nf, double *xl, double *xr, int *term);


extern void write_results_to_json(const char* filename, double elapsed_time, int ntrials, unsigned long funevals, 
                           int best_trial, int best_nt, int best_nf, double* best_pt, int nvars, double best_fx);

/* global variables */
unsigned long funevals = 0;
  

/* Rosenbrock classic parabolic valley ("banana") function */
double f(double *x, int n)
{
    double fv;
    int i;

    #pragma omp atomic
    funevals++;

    fv = 0.0;

    for (i = 0; i < n - 1; i++){   /* rosenbrock */
        fv = fv + 100.0 * pow((x[i + 1] - x[i] * x[i]), 2) + pow((x[i] - 1.0), 2);
    }
    
    #pragma omp task untied 
    usleep(10);  /* do not remove, introduces some artificial work */

    return fv;
}

int main(int argc, char *argv[])
{

    /* parse number of threads */
    int num_threads = atoi(argv[1]);
 
    omp_set_num_threads(num_threads);

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

    t0 = omp_get_wtime();

    long tseed = 1;

    #pragma omp parallel
    {
        #pragma omp single 
        {
    
            unsigned short randBuffer[3];
            
            //int thread_id = omp_get_thread_num();

            randBuffer[0] = 0;
            randBuffer[1] = 0;
            randBuffer[2] = tseed + ntrials + 1;  // Ensure unique seed for each thread
            
            //printf("\n\nStart Trials ...");

            for (trial = 0; trial < ntrials; trial++) {

                //printf("Trial : %d\n",trial);


                //printf("\n\nThread num %d || Rand Buffer: %d",omp_get_thread_num(),randBuffer[2]);

                /* starting guess for rosenbrock test function, search space in [-2, 2) */
                for (i = 0; i < nvars; i++) {
                    startpt[i] = lower[i] + (upper[i] - lower[i]) * erand48(randBuffer);
                    //rintf("Start pt [%d] : %f \n",i+1,startpt[i]);
                }         

                //printf("\n");  

        
                int term = -1;

                mds(startpt, endpt, nvars, &fx, eps, maxfevals, maxiter, mu, theta, delta, &nt, &nf, lower, upper, &term);

                
                /* keep the best solution */
                if (fx < best_fx) {
                    best_trial = trial;
                    best_nt = nt;
                    best_nf = nf;
                    best_fx = fx;
                    for (i = 0; i < nvars; i++)
                        best_pt[i] = endpt[i];
                }
                
                
            }
            
        }
    }

    t1 = omp_get_wtime();

    printf("\n\nFINAL RESULTS (OPENMP TASKS):\n");
    printf("Elapsed time = %.3lf s\n", t1 - t0);
    printf("Total number of trials = %d\n", ntrials);
    printf("Total number of function evaluations = %ld\n", funevals);
    printf("Best result at trial %d used %d iterations, %d function calls and returned\n", best_trial, best_nt, best_nf);
    for (i = 0; i < nvars; i++) {
        printf("x[%3d] = %15.7le \n", i, best_pt[i]);
    }
    printf("f(x) = %15.7le\n", best_fx);
    
    write_results_to_json("results_openmp_tasks.json", t1 - t0, ntrials, funevals, best_trial, best_nt, best_nf, best_pt, nvars, best_fx);

    return 0;
}
