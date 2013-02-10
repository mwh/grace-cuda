#include "gracelib.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda.h>


#include <builtin_types.h>

Object cuda_module;

ClassData CudaFloatArray;

struct CudaFloatArray {
    OBJECT_HEADER;
    int size;
    float data[];
};

static void errcheck(CUresult error) {
    if (error == CUDA_SUCCESS)
        return;
    fprintf(stderr, "some kind of cuda error: %i\n", error);
    exit(1);
}

Object cuda_FloatArray_at(Object self, int nparts, int *argcv,
        Object *argv, int flags) {
    int n = integerfromAny(argv[0]);
    struct CudaFloatArray *a = (struct CudaFloatArray *)self;
    return alloc_Float64(a->data[n-1]);
}
Object cuda_FloatArray_at_put(Object self, int nparts, int *argcv,
        Object *argv, int flags) {
    int n = integerfromAny(argv[0]);
    struct CudaFloatArray *a = (struct CudaFloatArray *)self;
    float f = (float)*((double *)argv[1]->data);
    a->data[n-1] = f;
    return alloc_none();
}
Object alloc_CudaFloatArray(int n) {
    if (!CudaFloatArray) {
        CudaFloatArray = alloc_class("CudaFloatArray", 2);
        add_Method(CudaFloatArray, "at", &cuda_FloatArray_at);
        add_Method(CudaFloatArray, "at()put", &cuda_FloatArray_at_put);
    }
    Object o = alloc_obj(sizeof(struct CudaFloatArray) - sizeof(struct Object)
            + sizeof(float) * n,
            CudaFloatArray);
    struct CudaFloatArray *a = (struct CudaFloatArray *)o;
    a->size = n;
    for (int i = 0; i < n; i++) {
        a->data[i] = 0.0f;
    }
    return o;
}
Object cuda_over_do(Object self, int nparts, int *argcv,
        Object *argv, int flags) {
    CUresult error;
    cuInit(0);
    int deviceCount = 0;
    error = cuDeviceGetCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "no devices found\n");
        exit(1);
    }
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction cuFunc;
    error = cuDeviceGet(&cuDevice, 0);
    error = cuCtxCreate(&cuContext, 0, cuDevice);
    CUdeviceptr d_A;
    CUdeviceptr d_B;
    CUdeviceptr d_res;
    error = cuModuleLoad(&cuModule, grcstring(argv[argcv[0]]));
    if (error != CUDA_SUCCESS) {
        fprintf(stderr, "some kind of cuda error: %i %s\n", error,
                grcstring(argv[argcv[0]]));
        exit(1);
    }
    struct CudaFloatArray *a = (struct CudaFloatArray *)argv[0];
    struct CudaFloatArray *b = (struct CudaFloatArray *)argv[1];
    int size = a->size;
    struct CudaFloatArray *r =
        (struct CudaFloatArray *)(alloc_CudaFloatArray(size));
    int fsize = sizeof(float) * size;
    errcheck(cuMemAlloc(&d_A, fsize));
    errcheck(cuMemcpyHtoD(d_A, &a->data, fsize));
    errcheck(cuMemAlloc(&d_B, fsize));
    errcheck(cuMemcpyHtoD(d_B, &b->data, fsize));
    errcheck(cuMemAlloc(&d_res, fsize));
    errcheck(cuMemcpyHtoD(d_res, &r->data, fsize));

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    void *args[] = { &d_res, &d_A, &d_B, &size };
    char name[256];
    strcpy(name, "block");
    strcat(name, grcstring(argv[argcv[0]]) + strlen("_cuda/"));
    for (int i=0; name[i] != 0; i++)
        if (name[i] == '.') {
            name[i] = 0;
            break;
        }
    error = cuModuleGetFunction(&cuFunc, cuModule, name);
    if (error != CUDA_SUCCESS) {
        fprintf(stderr, "some kind of cuda error in getting func: %i %s\n", error,
                name);
        exit(1);
    }
    error = cuLaunchKernel(cuFunc, blocksPerGrid, 1, 1,
        threadsPerBlock, 1, 1,
        0,
        NULL, args, NULL);
    if (error != CUDA_SUCCESS) {
        fprintf(stderr, "some kind of cuda error in running: %i %s\n", error,
                grcstring(argv[argcv[0]]));
        exit(1);
    }
    errcheck(cuMemcpyDtoH(&r->data, d_res, fsize));
    cuMemFree(d_A);
    cuMemFree(d_B);
    cuMemFree(d_res);
    return (Object)r;
}
Object cuda_floatArray(Object self, int nparts, int *argcv,
        Object *argv, int flags) {
    int n = integerfromAny(argv[0]);
    return alloc_CudaFloatArray(n);
}
Object module_cuda_init() {
    if (cuda_module != NULL)
        return cuda_module;
    ClassData c = alloc_class("Module<cuda>", 13);
    add_Method(c, "over()do", &cuda_over_do);
    add_Method(c, "floatArray", &cuda_floatArray);
    Object o = alloc_newobj(0, c);
    cuda_module = o;
    gc_root(o);
    return o;
}
