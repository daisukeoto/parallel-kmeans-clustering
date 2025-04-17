#include "kmeans_util.h"
#include <assert.h>
#define CHECK(expr) (assert(cudaSuccess==(expr)))
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
    // KMData *data = malloc(sizeof(KMData));

    // temp = static_cast<mynode *> ( malloc(sizeof(mynode)) );
    KMData *data = static_cast<KMData *> (malloc(sizeof(KMData)));

    data->ndata = 0;
    data->dim = 0;
    ssize_t tot_tokens = 0;
    filestats(datafile, &tot_tokens, &data->ndata);

    //allocate space for features and labels
    data->features = static_cast<float*> (malloc(((tot_tokens) - (2 * data->ndata)) * sizeof(float)));
    data->labels = static_cast<int*> (malloc(data->ndata * sizeof(float)));

    // data->features = malloc(((tot_tokens) - (2 * data->ndata)) * sizeof(float));
    // data->labels = malloc(data->ndata * sizeof(float));

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
    KMClust *clust = static_cast<KMClust *> (malloc(sizeof(KMClust)));
    // KMClust *clust = malloc(sizeof(KMClust));


    clust->nclust = nclust;
    clust->dim = dim;
    clust->features = static_cast<float*> (malloc(nclust * sizeof(float) * dim));
    clust->counts = static_cast<float*>(malloc(nclust * sizeof(float)));
    // clust->features = malloc(nclust * sizeof(float) * dim);
    // clust->counts = malloc(nclust * sizeof(float));

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

__global__ void calc_feats(ssize_t nclust, ssize_t ndata, ssize_t data_dim, int* data_assigns, float* data_feats, float *clust_feats, float* clust_counts){

    for (int i = 0; i < ndata; i++) { // Sum up data in each cluster
        int c = data_assigns[i];  
        if( c == blockIdx.x){     //if the assigned cluster has same number of blockIdx, then calculate the features. 
            clust_feats[c * data_dim + threadIdx.x] += data_feats[i * data_dim + threadIdx.x];  //each thread executes this
        }
    }

    if(clust_counts[blockIdx.x] > 0){
        clust_feats[blockIdx.x * data_dim + threadIdx.x] = clust_feats[blockIdx.x * data_dim + threadIdx.x] / clust_counts[blockIdx.x];
    }

}

__global__ void assign_clusts(ssize_t nclust, ssize_t data_dim, float* data_feats, float* clust_feats, int* data_assigns, float* clust_counts, int* nchanges, int curiter){

    int best;
    float best_distsq = INFINITY;

    int index = threadIdx.x + blockIdx.x * blockDim.x;  //calculate which data the thread is checking

    for (int j = 0; j < nclust; j++) { // Compare data to each cluster and assign to closest
        float distsq = 0.0;
        for (int k = 0; k < data_dim; k++) { // Calculate squared distance to each data dimension
            float diff = data_feats[index * data_dim + k] - clust_feats[j * data_dim + k];
            distsq += diff * diff;
        }
        if (distsq < best_distsq) { // If closer to this cluster, than current best
            best = j;
            best_distsq = distsq;
        }
    }

    atomicAdd(&clust_counts[best], 1);  //using atomic here because we don't want all threads accessing this data at once

    if (best != data_assigns[index]) { // Assigning data to a different cluster?
        atomicAdd(nchanges, 1);  //using atomic here because we don't want all threads accessing this data at once
        data_assigns[index] = best;
    }   

    __syncthreads();

    //after all threads are done, print out the matrix
    if(blockIdx.x == 0 && threadIdx.x == 0){
        printf("%3d: %5d | ", curiter, *nchanges);
        for (int c = 0; c < nclust; c++) {
            printf(" %4.0f ", clust_counts[c]);
        }
        printf("\n");
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

    data->assigns = static_cast<int*> (malloc(data->ndata * sizeof(int))); // Random, regular initial cluster assignment

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


    //setting variables used in GPU
    float *dev_data_feats;  //gpu version of data->features
    CHECK(cudaMalloc((void**) &dev_data_feats,   data->ndata * data->dim * sizeof(float)));
    CHECK(cudaMemcpy(dev_data_feats, data->features, data->ndata * data->dim * sizeof(float), cudaMemcpyHostToDevice));

    int *dev_data_assigns;  //gpu version of data->assigns
    CHECK(cudaMalloc((void**) &dev_data_assigns, data->ndata * sizeof(int)));
    cudaMemcpy(dev_data_assigns, data->assigns, data->ndata*sizeof(int), cudaMemcpyHostToDevice);

    float *dev_clust_feats;  //gpu version of clust->features
    CHECK(cudaMalloc((void**) &dev_clust_feats, clust->nclust * data->dim * sizeof(float)));
    cudaMemset(dev_clust_feats, 0.0, clust->nclust * data->dim * sizeof(float));

    float *dev_clust_counts;  //gpu version of data->features
    CHECK(cudaMalloc((void**) &dev_clust_counts,   clust->nclust * sizeof(float)));
    cudaMemcpy(dev_clust_counts, clust->counts, clust->nclust * sizeof(float), cudaMemcpyHostToDevice);

    int *dev_nchanges;  //gpu version of nchanges 
    cudaMalloc((void **)&dev_nchanges, sizeof(int));


    //setting how many blocks and threads using for calculating centers
    int nblocks_calc = clust->nclust;
    int nthreads_calc = 784;

    // determine how many blocks and threads needed for assigning data to clusters. 
    int nblocks_assign = 1; 
    int nthreads_assign = data->ndata;
    //If more than 1000 threads needed, add blocks
    if(nthreads_assign > 1000){
        nblocks_assign = nthreads_assign / 1000;
        nthreads_assign = 1000;
    }


    while (nchanges > 0 && curiter <= MAXITER) { // Loop until convergence

        // DETERMINE NEW CLUSTER CENTERS

        // resets the clust feats to zero
        cudaMemset(dev_clust_feats, 0.0, clust->nclust * data->dim * sizeof(float));

        //launch kerner for calc_feats
        calc_feats<<<nblocks_calc, nthreads_calc>>>(clust->nclust, data->ndata, data->dim, dev_data_assigns, dev_data_feats, dev_clust_feats, dev_clust_counts);

        // DETERMINE NEW CLUSTER ASSIGNMENTS FOR EACH DATA
        for (int i = 0; i < clust->nclust; i++) { // Reset cluster counts to 0
            clust->counts[i] = 0;
        }

        //resets the cluster count to zero
        cudaMemset(dev_clust_counts, 0.0, clust->nclust * sizeof(float));

        nchanges = 0;
        cudaMemset(dev_nchanges, 0, sizeof(int));

        assign_clusts<<<nblocks_assign, nthreads_assign>>>(clust->nclust, data->dim, dev_data_feats, dev_clust_feats, dev_data_assigns, dev_clust_counts, dev_nchanges, curiter);

        // copying cluster nchanges to the host because it is needed for checking if loop have converged
        cudaMemcpy(&nchanges, dev_nchanges, sizeof(int), cudaMemcpyDeviceToHost);
        
        curiter += 1;
    }

    //copying the results out to the host
    cudaMemcpy(clust->counts, dev_clust_counts, clust->nclust * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(data->assigns, dev_data_assigns, data->ndata * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(clust->features, dev_clust_feats, clust->nclust * data->dim * sizeof(float), cudaMemcpyDeviceToHost);

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
    ssize_t **confusion = static_cast<ssize_t**> (malloc(data->nlabels * sizeof(ssize_t)));
    // ssize_t **confusion = malloc(data->nlabels * sizeof(ssize_t));

    for (int i = 0; i < data->nlabels; i++) {
        confusion[i] = static_cast<ssize_t*> (malloc(nclust * sizeof(ssize_t)));
        // confusion[i] = malloc(nclust * sizeof(ssize_t));
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

    //free cuda allocated memody
    cudaFree(dev_data_feats);

    cudaFree(dev_data_assigns);

    cudaFree(dev_clust_feats);

    cudaFree(dev_clust_counts);

    cudaFree(dev_nchanges);

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