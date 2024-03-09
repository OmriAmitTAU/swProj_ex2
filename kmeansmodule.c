#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define INFINITY1 (__builtin_inf())

double **vectors;
double **centroids;
double **new_centroids;
int *centroid_size;

double distance(double *v, double *u, int d);
int init_vector_list(int d, int n);
int init_centroids(int k, int d);
int find_closest_centroid_index(double *curr, int k, int d);
void assign_to_new_centroid(int closest_centroid_idx, int x, int d);
void calculate_mean(int i, int d);
void reset_new_centroids(int i, int d);
int k_means(int k, int n, int d, int max_iter, double epsilon);
void free_memory(int k, int d);

static PyObject* fit(PyObject *self,PyObject *args) {
    int k, n, d, max_iter, eps;
    int i, j, result;
    PyObject *vectors_py;
    PyObject *centroids_py;
    PyObject *single_centroid;
    PyObject *c_output;
    
    if(!PyArg_ParseTuple(args, "iiiifOO", &k, &n, &d, &max_iter, &eps, &vectors_py, &centroids_py)) {
        printf("An Error Has Occurred");
        return NULL;
    }

    if (init_vector_list(d, n) == 1 || init_centroids(k, d) == 1) {
        printf("An Error Has Occurred\n");
        return NULL;
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < d; j++) {
            vectors[i][j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(vectors_py, i),j));
        }
    }

    for (i = 0; i < k; i++) {
        for (j = 0; j < d; j++) {
            centroids[i][j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(centroids_py, i),j));
        }
    }

    result = k_means(k, n, d, max_iter, eps);

    if (result == 1) {
        free_memory(k, d);
        printf("An Error Has Occurred");
        return NULL;
    }

    c_output = PyList_New(k);
    
    for (i = 0; i < k; i++) {
        single_centroid = PyList_New(d);
        for (j = 0; j < d; j++) {
            double value = centroids[i][j];
            PyObject *py_float = PyFloat_FromDouble(value);
            PyList_SetItem(single_centroid, j, py_float);
        }
        PyList_SetItem(c_output, i, single_centroid);
    }

    free_memory(k, d);
    return c_output;
}

static PyMethodDef kmeans_pp_methods[] = {
    {"fit",
    (PyCFunction) fit,
    METH_VARARGS,
    PyDoc_STR("Fit function for K-means clustering")},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    "K-means clustering implementation",
    -1,
    kmeans_pp_methods
};

PyMODINIT_FUNC PyInit_mykmeanssp(void) {
    return PyModule_Create(&moduledef);
}

double distance(double *v, double *u, int d)
{
    int i;
    double add = 0.0;
    for (i = 0; i < d; i++) {
        add += pow((v[i] - u[i]), 2);
    }
    return sqrt(add);
}

int init_vector_list(int d, int n)
{
    int x;
    vectors = (double **)malloc((n * sizeof(double *)));
    if (vectors == NULL) {
        return 1;
    }

    for (x = 0; x < n; x++) {
        vectors[x] = (double *)malloc((d * sizeof(double *)));
        if (vectors[x] == NULL) {
            return 1;
        }
    }
    return 0;
}

int init_centroids(int k, int d)
{   
    int i;
    centroids = (double **)malloc(k * sizeof(double *));
    new_centroids = (double **)malloc(k * sizeof(double *));
    centroid_size = (int *)malloc(k * sizeof(int));

    if (centroids == NULL || new_centroids == NULL || centroid_size == NULL) {
        return 1;
    }

    for (i = 0; i < k; i++) {
        centroids[i] = (double *)malloc(d * sizeof(double));
        new_centroids[i] = (double *)malloc(d * sizeof(double));
        centroid_size[i] = 0;
        if (centroids[i] == NULL || new_centroids[i] == NULL) {
            return 1;
        }
    }
    return 0;
}

int find_closest_centroid_index(double *curr, int k, int d)
{
    int i;
    double closest_d = INFINITY1;
    int closest_centroid_index = -1;

    for (i = 0; i < k; i++) {
        double dis = distance(curr, centroids[i], d);
        if (dis < closest_d) {
            closest_d = dis;
            closest_centroid_index = i;
        }
    }
    return closest_centroid_index;
}

void assign_to_new_centroid(int closest_centroid_idx, int i, int d)
{
    int j;
    for (j = 0; j < d; j++) {
        // summing with previously assigend vectors
        new_centroids[closest_centroid_idx][j] += vectors[i][j];
    }
    centroid_size[closest_centroid_idx]++;
}

void calculate_mean(int i, int d)
{
    int j;
    for (j = 0; j < d; j++) {
        // new_centroids[i][j] == sum of all vectors closest to this centroid
        double mean = (new_centroids[i][j]) / (centroid_size[i]);
        new_centroids[i][j] = mean;
    }
}
void reset_new_centroids(int i, int d)
{
    int x;
    for (x = 0; x < d; x++) {
        centroids[i][x] = new_centroids[i][x];
        new_centroids[i][x] = 0.0;
    }
    centroid_size[i] = 0;
}

void free_memory(int k, int d)
{
    int i;
    for (i = 0; i < d; i++) {
        free(vectors[i]);
        
        if (i < k) {
            free(centroids[i]);
            free(new_centroids[i]);
        }
    }
    free(vectors);
    free(centroids);
    free(new_centroids);
}

int k_means(int k, int n, int d, int max_iter, double epsilon)
{
    int i, j, iter = 0;
    double delta_miu = INFINITY1;
    double curr_miu;

    while (delta_miu >= epsilon && iter < max_iter)
    {
        delta_miu = distance(centroids[0], new_centroids[0], d);
        for (i = 0; i < n; i++) {
            int curr_idx = find_closest_centroid_index(vectors[i], k, d);
            assign_to_new_centroid(curr_idx, i, d);
        }

        for (j = 0; j < k; j++) {
            calculate_mean(j, d);
            curr_miu = distance(centroids[j], new_centroids[j], d);
            delta_miu = (curr_miu > delta_miu) ? curr_miu : delta_miu;
            reset_new_centroids(j, d);
        }
        iter++;
    }
    return 0;
}
