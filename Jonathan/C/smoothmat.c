#include<Python.h>
#include<stdlib.h>
#include<errno.h>
#include<stdio.h>

#define RET_MALLOC_ERROR 1
#define X1 1
#define X2 (-1.f/8.f)
#define X3 (-1.f/8.f)
#define X4 0

void malloc_error_detail(int errorcode)
{/*{{{*/
    switch(errorcode){
        case ENOMEM: fprintf(stderr, "Out of memory.\n"); break;
        default: fprintf(stderr, "Unknown malloc error.\n"); break;
    }
    exit(RET_MALLOC_ERROR);
    return; /* haha, joking */
}/*}}}*/
void *own_malloc(int size)
{/*{{{*/
    int errsv = 0;
    void *ptr;
    if ( NULL == ( ptr = malloc(size) ) ) {
        errsv = errno;
        malloc_error_detail(errsv);
    }
    return ptr;
}/*}}}*/
static PyObject *
smoothing_matrix(PyObject *self, PyObject *args)
{/*{{{*/
    const int height, width;
    const int size = height * width;
    int i,j;
    float** X;
    int i_col, i_row, j_col, j_row;
    X = own_malloc(sizeof(float*) * size);
    for ( i = 0 ; i < height ; i++ ) {
        X[i] = own_malloc(sizeof(float) * size);
    }
    for ( i = 0 ; i < size ; i++ ) {
        X[i][i] = X1;
        for ( j = i + 1 ; j < size ; j++ ) {
            i_row = i / width;
            i_col = i % width;
            j_row = j / width;
            j_col = j % width;
            if ( i_row == j_row && abs(i_col - j_col) == 1 ) {
                X[i][j] = X2;
            } else if ( abs(i_row - j_row) == 1 && i_col == j_col ) {
                X[i][j] = X2;
            } else if ( abs(i_row - j_row) == 1 && abs(i_col - j_col) == 1 ) {
                X[i][j] = X3;
            } else {
                X[i][j] = X4;
            }
            X[j][i] = X[i][j];
        }
    }
    return 0;
}/*}}}*/
static PyMethodDef SmoothMatMethods[] = {
    {"smoothMat", smoothing_matrix, METH_VARARGS,
     "Create the smoothing matrix"},
    {NULL, NULL, 0, NULL} /* Sentinel */
}
static struct PyModuleDef SmoothMatModule = {
    PyModuleDef_HEAD_INIT,
    "smoothMat", /* Module name */
    NULL, /* Documentation */
    -1, /* size of per-interpreter state of the module,
           or -1 if the module keeps state in global variables. */
    SmoothMatMethods
};
PyMODINIT_FUNC
PyInit_smoothMat(void)
{
    return PyModule_Create(&SmoothMatModule);
}
