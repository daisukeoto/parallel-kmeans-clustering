#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <sys/types.h>

// #include "kmeans_util.c"
// #include "kmeans.h"
// #define MAX_FEATURES 1000
// #define MAX_DATA 1000000

typedef struct {
    ssize_t ndata;      // Count of data
    ssize_t dim;        // Dimension of features for data
    float *features;   // Pointers to individual features
    int *assigns;       // Cluster to which data is assigned
    int *labels;        // Label for data if available
    int nlabels;        // Max value of labels +1, number 0, 1, ..., nlabel0
} KMData;               // Data set to be clustered

typedef struct {
    ssize_t nclust;     // Number of clusters, the "k" in kmeans
    ssize_t dim;        // Dimension of features for data
    float *features;   // Indexing for individual cluster center features
    float *counts;      // Number of data in each cluster
} KMClust;              // Cluster information


// Load a data set from the named file
KMData *kmdata_load(char *datafile) {

    //allocate memory for data
    KMData *data = malloc(sizeof(KMData));

    data->ndata = 0;
    data->dim = 0;
    ssize_t tot_tokens = 0;
    filestats(datafile, &tot_tokens, &data->ndata);

    //allocate space for features and labels
    data->features = malloc(((tot_tokens) - (2 * data->ndata)) * sizeof(float));
    data->labels = malloc(data->ndata * sizeof(float));

    //open file to read 
    FILE *fp = fopen(datafile, "r");

    int line = 0;
    int idx = 0;

    //loop through line to plot the tokens into features
    while (line != data->ndata) {

        int label;

        int ret = fscanf(fp, "%d", &label);
        data->labels[line] = label;

        char buff[10];
        ret = fscanf(fp, "%s", buff);
        float token = 0;
        for (int i = 0; i < 784; i++) {
            int ret = fscanf(fp, "%f ", &token);
            data->features[idx] = token;
            idx++;
        }
        line++;
    }

    //set the dimension
    data->dim = 784;

    //finding the max
    int max = data->labels[0];
    for (int i = 0; i < data->ndata; i++){
        if (data->labels[i] > max){
            max = data->labels[i];
        }
    }

    //set nlabel
    int nlabel = max + 1;
    data->nlabels = nlabel;

    //close file
    fclose(fp);

    return data;
}

// Allocate space for clusters in an object
KMClust *kmclust_new(ssize_t nclust, ssize_t dim) {

    //allocate space for clust
    KMClust *clust = malloc(sizeof(KMClust));

    clust->nclust = nclust;
    clust->dim = dim;
    clust->features = malloc(nclust * sizeof(float) * dim);
    clust->counts = malloc(nclust * sizeof(float));

    //setting features 
    for (int i = 0; i < nclust; i++) {
        for (int j = 0; j < dim; j++) {
            clust->features[i * dim + j] = 0.0;
        }
        clust->counts[i] = 0.0;
    }

    return clust;
}

// Save clust centers in the PGM (portable gray map) image format
void save_pgm_files(KMClust *clust, char *savedir) {

    int dim_root = (int)sqrt(clust->dim);

    if (clust->dim % dim_root == 0) {
        printf("Saving cluster centers to %s/cent_0000.pgm ...\n", savedir);

        //finding the max in features
        float maxfeat = clust->features[0];
        for (int i = 0; i < clust->nclust * clust->dim; i++) {
            if (clust->features[i] > maxfeat){
                maxfeat = clust->features[i];
            }
        }

        //writing and saving the pgm files for each cluster
        for (int i = 0; i < clust->nclust; i++) {
            char filename[sizeof(savedir) + 128];
            sprintf(filename, "%s/cent_%04d.pgm", savedir, i);
            FILE *fp = fopen(filename, "w");
            fprintf(fp, "P2\n");
            fprintf(fp, "%d %d\n", dim_root, dim_root);
            fprintf(fp, "%.0f\n", maxfeat);
            for (int j = 0; j < clust->dim; j++) {
                if (j > 0 && (j % dim_root) == 0) {
                    fprintf(fp, "\n");
                }
                fprintf(fp, "%3.0f ", clust->features[i * clust->dim + j]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

// THE MAIN ALGORITHM
int main(int argc, char **argv) {

    if (argc < 3) {
        printf("usage: kmeans_serial.c <datafile> <nclust> [savedir] [maxiter]\n");
        return -1;
    }

    char *datafile = argv[1];
    ssize_t nclust = atoi(argv[2]);
    char *savedir = ".";
    int MAXITER = 100; // Bounds the iterations

    if (argc > 3) { // Create save directory if specified
        savedir = argv[3];
        mkdir(savedir, 0777);
    }

    if (argc > 4) {
        MAXITER = atoi(argv[4]);
    }

    printf("datafile: %s\n", datafile);
    printf("nclust: %ld\n", nclust);
    printf("savedir: %s\n", savedir);

    KMData *data = kmdata_load(datafile); // Read in the data file, allocate cluster space

    KMClust *clust = kmclust_new(nclust, data->dim);

    printf("ndata: %ld\n",data->ndata);
    printf("dim: %ld\n\n",data->dim);

    data->assigns = malloc(data->ndata * sizeof(int)); // Random, regular initial cluster assignment

    for (int i = 0; i < data->ndata; i++) {
        int c = i % clust->nclust;
        data->assigns[i] = c;
    }

    //set the counts for each cluster 
    for (int i = 0; i < clust->nclust; i++) {
        int icount = floor(data->ndata / clust->nclust);
        int extra = 0;
        if (i < (data->ndata % clust->nclust)) {
            extra = 1; // Extras in earlier clusters
        }
        clust->counts[i] = icount + extra;
    }


    int curiter = 1; // Current Iteration
    int nchanges = data->ndata; // Check for changes in cluster assignment; 0 is converged
    printf("==CLUSTERING: MAXITER %d==\n", MAXITER);
    printf("ITER NCHANGE CLUST_COUNTS\n");

    while (nchanges > 0 && curiter <= MAXITER) { // Loop until convergence

        // DETERMINE NEW CLUSTER CENTERS

        for (int i = 0; i < clust->nclust; i++) { // Reset cluster centers to 0.0
            for (int j = 0; j < clust->dim; j++) {
                clust->features[i * clust->dim + j] = 0.0;
            }
        }
        for (int i = 0; i < data->ndata; i++) { // Sum up data in each cluster
            int c = data->assigns[i];
                for (int j = 0; j < clust->dim; j++) {
                    clust->features[c * clust->dim + j] += data->features[i * clust->dim + j];
                }
        }
       
        for (int i = 0; i < clust->nclust; i++) { // Divide by ndatas of data to get mean of cluster center
            if (clust->counts[i] > 0) {
                for (int j = 0; j < clust->dim; j++) {
                    clust->features[i * clust->dim + j] =
                    clust->features[i * clust->dim + j] / clust->counts[i];
                }
            }
        }

        // DETERMINE NEW CLUSTER ASSIGNMENTS FOR EACH DATA
        for (int i = 0; i < clust->nclust; i++) { // Reset cluster counts to 0
            clust->counts[i] = 0;
        }
        
        nchanges = 0;
        for (int i = 0; i < data->ndata; i++) { // Iterate over all data
            int best;
            float best_distsq = INFINITY;
            for (int j = 0; j < clust->nclust; j++) { // Compare data to each cluster and assign to closest
                float distsq = 0.0;
                for (int k = 0; k < clust->dim; k++) { // Calculate squared distance to each data dimension
                    float diff = data->features[i * clust->dim + k] - clust->features[j * clust->dim + k];
                    distsq += diff * diff;
                }
                if (distsq < best_distsq) { // If closer to this cluster, than current best
                    best = j;
                    best_distsq = distsq;
                }
            }
            clust->counts[best] += 1;
            if (best != data->assigns[i]) { // Assigning data to a different cluster?
                nchanges += 1;
                data->assigns[i] = best;
            }
        }
        // Print iteration information at the end of the iter
        printf("%3d: %5d | ", curiter, nchanges);
        for (int c = 0; c < nclust; c++) {
            printf(" %4.0f ", clust->counts[c]);
        }
        printf("\n");
        curiter += 1;
    }
    // Loop has converged
    if (curiter > MAXITER) {
        printf("WARNING: maximum iteration %d exceeded, may not have conveged", MAXITER);
    } 
    else {
        printf("CONVERGED: after %d iterations\n", curiter);
    }
    printf("\n");

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CLEANUP + OUTPUT

    // CONFUSION MATRIX

    //allocate memory for matrix
    ssize_t **confusion = malloc(data->nlabels * sizeof(ssize_t));
    for (int i = 0; i < data->nlabels; i++) {
        confusion[i] = malloc(nclust * sizeof(ssize_t));
        for(int j = 0; j < nclust; j++){
            confusion[i][j] = 0;
        }
    }
    for (int i = 0; i < data->ndata; i++) { // Count which labels in which clusters
        confusion[data->labels[i]][data->assigns[i]] += 1;
        // printf("%d: %d %d\n",i+1, data->labels[i], data->assigns[i]);
    }
    printf("==CONFUSION MATRIX + COUNTS==\n");
    printf("LABEL \\ CLUST\n");
    printf(" ");
    for (int j = 0; j < nclust; j++) { // Confusion matrix header
        printf(" %4d", j);
    }
    printf(" TOT\n");
    for (int i = 0; i < data->nlabels; i++) { // Each row of confusion matrix
        printf("%2d:", i);
        int tot = 0;
        for (int j = 0; j < nclust; j++) {
            printf(" %4ld", confusion[i][j]);
            tot += confusion[i][j];
        }
        printf(" %4d\n", tot);
    }
    printf("TOT"); // Final total row of confusion matrix
    int tot = 0;
    for (int c = 0; c < nclust; c++) {
        printf(" %4.0f", clust->counts[c]);
        tot += clust->counts[c];
    }
    printf(" %4d\n", tot);
    printf("\n");
    // LABEL FILE OUTPUT
    char outfile[sizeof(savedir) + 128];
    sprintf(outfile, "%s/labels.txt", savedir);
    printf("Saving cluster labels to file %s\n", outfile);
    FILE *fp = fopen(outfile, "w");
    for (int i = 0; i < data->ndata; i++) {
        fprintf(fp, "%2d %2d\n", data->labels[i], data->assigns[i]);
    }
    // SAVE PGM FILES CONDITIONALLY
    save_pgm_files(clust, savedir);

    fclose(fp);


    //free allocated memory
    for (int i = 0; i < data->nlabels; i++) {
        free(confusion[i]);
    }
    free(confusion);

    free(data->features);

    free(data->labels);

    free(data->assigns);

    free(clust->features);

    free(clust->counts);

    free(data);
    free(clust);
}