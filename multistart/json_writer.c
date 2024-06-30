#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void write_results_to_json(const char* filename, double elapsed_time, int ntrials, unsigned long funevals, 
                           int best_trial, int best_nt, int best_nf, double* best_pt, int nvars, double best_fx) {
    FILE *file = fopen(filename, "r+");
    if (file) {
        fseek(file, 0, SEEK_END);
        long file_size = ftell(file);
        if (file_size == 0) {
            // File is empty, write the opening bracket for JSON array
            fprintf(file, "[\n");
        } else {
            fseek(file, -2, SEEK_END); // Go back 2 characters to overwrite the trailing newline and comma
            fprintf(file, ",\n");
        }

        // Write the new JSON object
        fprintf(file, "  {\n");
        fprintf(file, "    \"elapsed_time\": %.3lf,\n", elapsed_time);
        fprintf(file, "    \"ntrials\": %d,\n", ntrials);
        fprintf(file, "    \"funevals\": %ld,\n", funevals);
        fprintf(file, "    \"best_trial\": %d,\n", best_trial);
        fprintf(file, "    \"best_nt\": %d,\n", best_nt);
        fprintf(file, "    \"best_nf\": %d,\n", best_nf);
        fprintf(file, "    \"best_pt\": [\n");
        for (int i = 0; i < nvars; i++) {
            fprintf(file, "      %.7le", best_pt[i]);
            if (i < nvars - 1) {
                fprintf(file, ",");
            }
            fprintf(file, "\n");
        }
        fprintf(file, "    ],\n");
        fprintf(file, "    \"best_fx\": %.7le\n", best_fx);
        fprintf(file, "  }\n");
        fprintf(file, "]\n");

        fclose(file);
    } else {
        fprintf(stderr, "Error opening file for writing\n");
    }
}
