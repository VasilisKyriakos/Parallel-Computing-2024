#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h> // Include OpenMP for parallelism

extern double f(double *x, int n);

void initialize_simplex(double *u, int n, double *point, double delta) {
    int i, j;

    // Initialize simplex
    for (j = 0; j < n; j++)
        u[j] = point[j];
    for (i = 1; i < n + 1; i++) {
        for (j = 0; j < n; j++) {
            if (i - 1 == j)
                u[i * n + j] = point[j] + delta;
            else
                u[i * n + j] = point[j];
        }
    }
}

int minimum_simplex(double *fu, int n) {
    int i, imin = 0;
    double min = fu[0];

    for (i = 1; i < n + 1; i++) {
        if (min > fu[i]) {
            min = fu[i];
            imin = i;
        }
    }
    return imin;
}

double simplex_size(double *u, int n) {
    int i, j;
    double *mesos = (double *)malloc(n * sizeof(double));
    double dist, max_dist = -1;

    for (j = 0; j < n; j++) {
        mesos[j] = 0.0;
        for (i = 0; i < n + 1; i++) {
            mesos[j] += u[i * n + j];
        }
        mesos[j] /= (n + 1);
    }

    for (i = 0; i < n + 1; i++) {
        dist = 0.0;
        for (j = 0; j < n; j++) {
            dist += (mesos[j] - u[i * n + j]) * (mesos[j] - u[i * n + j]);
        }
        dist = sqrt(dist);
        if (dist > max_dist) {
            max_dist = dist;
        }
    }
    free(mesos);
    return max_dist;
}

void swap_simplex(double *u, double *fu, int n, int from, int to) {
    int j;
    double tmp, ftmp;

    for (j = 0; j < n; j++) {
        tmp = u[from * n + j];
        u[from * n + j] = u[to * n + j];
        u[to * n + j] = tmp;
    }

    ftmp = fu[from];
    fu[from] = fu[to];
    fu[to] = ftmp;
}

void assign_simplex(double *s1, double *fs1, double *s2, double *fs2, int n) {
    int i, j;
    for (i = 1; i < n + 1; i++) {
        for (j = 0; j < n; j++) {
            s1[i * n + j] = s2[i * n + j];
        }
        fs1[i] = fs2[i];
    }
}

int inbounds_simplex(double *s, int n, double *xl, double *xr) {
    int i, j;
    for (i = 0; i < n + 1; i++) {
        for (j = 0; j < n; j++) {
            if (s[i * n + j] > xr[j] || s[i * n + j] < xl[j])
                return 0;
        }
    }
    return 1;
}

void mds(double *point, double *endpoint, int n, double *val, double eps, int maxfevals, int maxiter, double mu,
         double theta, double delta, int *nit, int *nf, double *xl, double *xr, int *term) {
    
    int i, j, k, found_better, iter, kec, terminate;
    int out_of_bounds;
    int local_nf = 0;

    double *u = (double *)malloc(n * (n + 1) * sizeof(double));
    double *r = (double *)malloc(n * (n + 1) * sizeof(double));
    double *ec = (double *)malloc(n * (n + 1) * sizeof(double));
    double *fu = (double *)malloc((n + 1) * sizeof(double));
    double *fr = (double *)malloc((n + 1) * sizeof(double));
    double *fec = (double *)malloc((n + 1) * sizeof(double));

    iter = 0;
    *term = 0;
    *nf = 0;

    initialize_simplex(u, n, point, delta);

    #pragma omp parallel for private(i) reduction(+:local_nf)
    for (i = 0; i < n + 1; i++) {
        fu[i] = f(&u[i * n], n);
        local_nf++;
    }

    k = minimum_simplex(fu, n);
    swap_simplex(u, fu, n, k, 0);

    *val = fu[0];
    terminate = 0;
    iter = 0;

    while (terminate == 0 && iter < maxiter) {

        k = minimum_simplex(fu, n);
        swap_simplex(u, fu, n, k, 0);

        found_better = 0;
        while (found_better == 0) {

            if (local_nf > maxfevals) {
                *term = 1;
                terminate = 1;
                break;
            }

            if (simplex_size(u, n) < eps) {
                *term = 2;
                terminate = 1;
                break;
            }

            fr[0] = fu[0];

            found_better = 1;
            for (i = 1; i < n + 1; i++) {
                for (j = 0; j < n; j++) {
                    r[i * n + j] = u[0 * n + j] - (u[i * n + j] - u[0 * n + j]);
                    if (r[i * n + j] > xr[j] || r[i * n + j] < xl[j]) {
                        found_better = 0;
                        break;
                    }
                }
                if (found_better == 0)
                    break;
            }

            if (found_better == 1) {
                #pragma omp parallel for private(i, j) reduction(+:local_nf)
                for (i = 1; i < n + 1; i++) {
                    for (j = 0; j < n; j++) {
                        r[i * n + j] = u[0 * n + j] - (u[i * n + j] - u[0 * n + j]);
                    }
                    fr[i] = f(&r[i * n], n);
                    local_nf++;
                }

                found_better = 0;
                k = minimum_simplex(fr, n);
                if (fr[k] < fu[0])
                    found_better = 1;
            }

            if (found_better == 1) {
                out_of_bounds = 0;
                for (i = 1; i < n + 1; i++) {
                    for (j = 0; j < n; j++) {
                        ec[i * n + j] = u[0 * n + j] - mu * ((u[i * n + j] - u[0 * n + j]));
                        if (ec[i * n + j] > xr[j] || ec[i * n + j] < xl[j]) {
                            out_of_bounds = 1;
                            break;
                        }
                    }
                    if (out_of_bounds == 1)
                        break;
                }

                if (out_of_bounds == 0) {
                    fec[0] = fu[0];
                    #pragma omp parallel for private(i, j) reduction(+:local_nf)
                    for (i = 1; i < n + 1; i++) {
                        for (j = 0; j < n; j++) {
                            ec[i * n + j] = u[0 * n + j] - mu * ((u[i * n + j] - u[0 * n + j]));
                        }
                        fec[i] = f(&ec[i * n], n);
                        local_nf++;
                    }

                    kec = minimum_simplex(fec, n);
                    if (fec[kec] < fr[k]) {
                        assign_simplex(u, fu, ec, fec, n);
                    } else {
                        assign_simplex(u, fu, r, fr, n);
                    }
                } else {
                    assign_simplex(u, fu, r, fr, n);
                }
            } else {
                fec[0] = fu[0];
                #pragma omp parallel for private(i, j) reduction(+:local_nf)
                for (i = 1; i < n + 1; i++) {
                    for (j = 0; j < n; j++) {
                        ec[i * n + j] = u[0 * n + j] + theta * ((u[i * n + j] - u[0 * n + j]));
                    }
                    fec[i] = f(&ec[i * n], n);
                    local_nf++;
                }

                kec = minimum_simplex(fec, n);
                if (fec[kec] < fu[0]) {
                    found_better = 1;
                }
                assign_simplex(u, fu, ec, fec, n);
            }
        }
        iter++;
        if (iter == maxiter)
            *term = 3;
    }

    k = minimum_simplex(fu, n);
    swap_simplex(u, fu, n, k, 0);
    for (i = 0; i < n; i++)
        endpoint[i] = u[i];
    *val = fu[0];
    *nit = iter;
    *nf = local_nf;

    free(u);
    free(r);
    free(ec);
    free(fu);
    free(fr);
    free(fec);
}
