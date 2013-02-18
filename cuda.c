#include "gracelib.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda.h>


#include <builtin_types.h>

// Expand paths given on command line to string constants, or use
// the default if none were given.
#define str(s) #s
#define xstr(s) str(s)
#ifndef CUDA_BIN_DIR
#define GRACE_CUDA_BIN_DIR "/opt/cuda/bin"
#else
#define GRACE_CUDA_BIN_DIR xstr(CUDA_BIN_DIR)
#endif
#ifndef CUDA_INCLUDE_DIR
#define GRACE_CUDA_INCLUDE_DIR "/opt/cuda/include"
#else
#define GRACE_CUDA_INCLUDE_DIR xstr(CUDA_INCLUDE_DIR)
#endif

Object cuda_module;
Object CudaError;
Object ErrorObject; // From gracelib
Object alloc_Exception(char *name, Object parent);

ClassData CudaFloatArray;

struct CudaFloatArray {
    OBJECT_HEADER;
    int size;
    float data[];
};

static void raiseError(char *errstr) {
    Object a = alloc_String(errstr);
    int i = 1;
    callmethod(CudaError, "raise", 1, &i, &a);
}
static void errcheck(CUresult error) {
    if (error == CUDA_SUCCESS)
        return;
    char buf[255];
    sprintf(buf, "CUDA error code %i\n", error);
    raiseError(buf);
}

Object cuda_FloatArray_at(Object self, int nparts, int *argcv,
        Object *argv, int flags) {
    int n = integerfromAny(argv[0]);
    struct CudaFloatArray *a = (struct CudaFloatArray *)self;
    return alloc_Float64(a->data[n]);
}
Object cuda_FloatArray_at_put(Object self, int nparts, int *argcv,
        Object *argv, int flags) {
    int n = integerfromAny(argv[0]);
    struct CudaFloatArray *a = (struct CudaFloatArray *)self;
    float f = (float)*((double *)argv[1]->data);
    a->data[n] = f;
    return alloc_none();
}
Object alloc_CudaFloatArray(int n) {
    if (!CudaFloatArray) {
        CudaFloatArray = alloc_class("CudaFloatArray", 4);
        add_Method(CudaFloatArray, "at", &cuda_FloatArray_at);
        add_Method(CudaFloatArray, "at()put", &cuda_FloatArray_at_put);
        add_Method(CudaFloatArray, "[]", &cuda_FloatArray_at);
        add_Method(CudaFloatArray, "[]:=", &cuda_FloatArray_at_put);
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
Object cuda_using_do_blockWidth_blockHeight_gridWidth_gridHeight(Object self, int nparts, int *argcv,
        Object *argv, int flags) {
    CUresult error;
    cuInit(0);
    int deviceCount = 0;
    error = cuDeviceGetCount(&deviceCount);
    if (deviceCount == 0) {
        raiseError("No CUDA devices found");
    }
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction cuFunc;
    error = cuDeviceGet(&cuDevice, 0);
    error = cuCtxCreate(&cuContext, 0, cuDevice);
    // do through gridWidth only have one argument each
    int argOffset = argcv[0] + 1;
    int blockDimX = integerfromAny(argv[argOffset++]);
    int blockDimY = integerfromAny(argv[argOffset++]);
    int gridDimX = integerfromAny(argv[argOffset++]);
    int gridDimY = integerfromAny(argv[argOffset++]);

    char *tmp = grcstring(argv[argcv[0]]);
    char argStr[strlen(tmp) + 1];
    strcpy(argStr, tmp);
    char *tmp2 = strtok(argStr, " ");
    char blockname[128];
    strcpy(blockname, tmp2);
    errcheck(cuModuleLoad(&cuModule, blockname));
    CUdeviceptr dps[argcv[0]];
    float floats[argcv[0]];
    void *args[argcv[0]];
    int ints[argcv[0]];
    argStr[strlen(blockname)] = ' ';
    strtok(argStr, " ");
    for (int i=0; i<argcv[0]; i++) {
        char *argType = strtok(NULL, " ");
        if (argType[0] == 'f' && argType[1] == '*') {
            struct CudaFloatArray *a = (struct CudaFloatArray *)argv[i];
            errcheck(cuMemAlloc(&dps[i], a->size * sizeof(float)));
            errcheck(cuMemcpyHtoD(dps[i], &a->data, a->size * sizeof(float)));
            args[i] = &dps[i];
        } else if (argType[0] == 'f') {
            floats[i] = (float)*((double *)(argv[i]->data));
            args[i] = &floats[i];
        } else if (argType[0] == 'i') {
            ints[i] = integerfromAny(argv[i]);
            args[i] = &ints[i];
        } else {
            // Fail
            char buf[256];
            sprintf(buf, "CUDA argument cannot be coerced. This shouldn't happen. Argument string: %s\n", argType);
            raiseError(buf);
        }
    }
    char name[256];
    strcpy(name, "block");
    strcat(name, blockname + strlen("_cuda/"));
    for (int i=0; name[i] != 0; i++)
        if (name[i] == '.') {
            name[i] = 0;
            break;
        }
    errcheck(cuModuleGetFunction(&cuFunc, cuModule, name));
    errcheck(cuLaunchKernel(cuFunc, gridDimX, gridDimY, 1,
        blockDimX, blockDimY, 1,
        0,
        NULL, args, NULL));
    for (int i=0; i<argcv[0]; i++) {
        struct CudaFloatArray *a = (struct CudaFloatArray *)argv[i];
        errcheck(cuMemcpyDtoH(&a->data, dps[i], a->size * sizeof(float)));
        cuMemFree(dps[i]);
    }
    return alloc_none();
}
Object cuda_using_times_do(Object self, int nparts, int *argcv,
        Object *argv, int flags) {
    CUresult error;
    cuInit(0);
    int deviceCount = 0;
    error = cuDeviceGetCount(&deviceCount);
    if (deviceCount == 0) {
        raiseError("No CUDA devices found");
    }
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction cuFunc;
    error = cuDeviceGet(&cuDevice, 0);
    error = cuCtxCreate(&cuContext, 0, cuDevice);
    // We will infer a suitable gridDimX that includes at least
    // one thread for each of times.
    int times = integerfromAny(argv[argcv[0]]);
    int blockDimX = 256;
    int blockDimY = 1;
    int gridDimX = (times + blockDimX - 1) / blockDimX;
    int gridDimY = 1;

    char *tmp = grcstring(argv[argcv[0] + 1]);
    char argStr[strlen(tmp) + 1];
    strcpy(argStr, tmp);
    char *tmp2 = strtok(argStr, " ");
    char blockname[128];
    strcpy(blockname, tmp2);
    errcheck(cuModuleLoad(&cuModule, blockname));
    CUdeviceptr dps[argcv[0]];
    float floats[argcv[0]];
    void *args[argcv[0] + 1];
    args[argcv[0]] = &times;
    int ints[argcv[0]];
    argStr[strlen(blockname)] = ' ';
    strtok(argStr, " ");
    int size = 0;
    for (int i=0; i<argcv[0]; i++) {
        char *argType = strtok(NULL, " ");
        if (argType[0] == 'f' && argType[1] == '*') {
            struct CudaFloatArray *a = (struct CudaFloatArray *)argv[i];
            errcheck(cuMemAlloc(&dps[i], a->size * sizeof(float)));
            errcheck(cuMemcpyHtoD(dps[i], &a->data, a->size * sizeof(float)));
            if (a->size > size)
                size = a->size;
            args[i] = &dps[i];
        } else if (argType[0] == 'f') {
            floats[i] = (float)*((double *)(argv[i]->data));
            args[i] = &floats[i];
        } else if (argType[0] == 'i') {
            ints[i] = integerfromAny(argv[i]);
            args[i] = &ints[i];
        } else {
            // Fail
            char buf[256];
            sprintf(buf, "CUDA argument cannot be coerced. This shouldn't happen. Argument string: %s\n", argType);
            raiseError(buf);
        }
    }
    gridDimX = (size + blockDimX - 1) / blockDimX;
    char name[256];
    strcpy(name, "block");
    strcat(name, blockname + strlen("_cuda/"));
    for (int i=0; name[i] != 0; i++)
        if (name[i] == '.') {
            name[i] = 0;
            break;
        }
    errcheck(cuModuleGetFunction(&cuFunc, cuModule, name));
    errcheck(cuLaunchKernel(cuFunc, gridDimX, gridDimY, 1,
        blockDimX, blockDimY, 1,
        0,
        NULL, args, NULL));
    for (int i=0; i<argcv[0]; i++) {
        struct CudaFloatArray *a = (struct CudaFloatArray *)argv[i];
        errcheck(cuMemcpyDtoH(&a->data, dps[i], a->size * sizeof(float)));
        cuMemFree(dps[i]);
    }
    return alloc_none();
}
Object cuda_using_do(Object self, int nparts, int *argcv,
        Object *argv, int flags) {
    CUresult error;
    cuInit(0);
    int deviceCount = 0;
    error = cuDeviceGetCount(&deviceCount);
    if (deviceCount == 0) {
        raiseError("No CUDA devices found");
    }
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction cuFunc;
    error = cuDeviceGet(&cuDevice, 0);
    error = cuCtxCreate(&cuContext, 0, cuDevice);
    // We will infer a suitable size that includes one thread for
    // each element of the largest floatArray passed in.
    int blockDimX = 256;
    int blockDimY = 1;
    int gridDimX = 1;
    int gridDimY = 1;

    char *tmp = grcstring(argv[argcv[0]]);
    char argStr[strlen(tmp) + 1];
    strcpy(argStr, tmp);
    char *tmp2 = strtok(argStr, " ");
    char blockname[128];
    strcpy(blockname, tmp2);
    errcheck(cuModuleLoad(&cuModule, blockname));
    CUdeviceptr dps[argcv[0]];
    float floats[argcv[0]];
    void *args[argcv[0]];
    int ints[argcv[0]];
    argStr[strlen(blockname)] = ' ';
    strtok(argStr, " ");
    int size = 0;
    for (int i=0; i<argcv[0]; i++) {
        char *argType = strtok(NULL, " ");
        if (argType[0] == 'f' && argType[1] == '*') {
            struct CudaFloatArray *a = (struct CudaFloatArray *)argv[i];
            errcheck(cuMemAlloc(&dps[i], a->size * sizeof(float)));
            errcheck(cuMemcpyHtoD(dps[i], &a->data, a->size * sizeof(float)));
            if (a->size > size)
                size = a->size;
            args[i] = &dps[i];
        } else if (argType[0] == 'f') {
            floats[i] = (float)*((double *)(argv[i]->data));
            args[i] = &floats[i];
        } else if (argType[0] == 'i') {
            ints[i] = integerfromAny(argv[i]);
            args[i] = &ints[i];
        } else {
            // Fail
            char buf[256];
            sprintf(buf, "CUDA argument cannot be coerced. This shouldn't happen. Argument string: %s\n", argType);
            raiseError(buf);
        }
    }
    gridDimX = (size + blockDimX - 1) / blockDimX;
    char name[256];
    strcpy(name, "block");
    strcat(name, blockname + strlen("_cuda/"));
    for (int i=0; name[i] != 0; i++)
        if (name[i] == '.') {
            name[i] = 0;
            break;
        }
    errcheck(cuModuleGetFunction(&cuFunc, cuModule, name));
    errcheck(cuLaunchKernel(cuFunc, gridDimX, gridDimY, 1,
        blockDimX, blockDimY, 1,
        0,
        NULL, args, NULL));
    for (int i=0; i<argcv[0]; i++) {
        struct CudaFloatArray *a = (struct CudaFloatArray *)argv[i];
        errcheck(cuMemcpyDtoH(&a->data, dps[i], a->size * sizeof(float)));
        cuMemFree(dps[i]);
    }
    return alloc_none();
}
Object cuda_over_map(Object self, int nparts, int *argcv,
        Object *argv, int flags) {
    CUresult error;
    cuInit(0);
    int deviceCount = 0;
    error = cuDeviceGetCount(&deviceCount);
    if (deviceCount == 0) {
        raiseError("No CUDA devices found");
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
    errcheck(cuModuleLoad(&cuModule, grcstring(argv[argcv[0]])));
    CUdeviceptr dps[argcv[0]];
    void *args[argcv[0]+2];
    int size = INT_MAX;
    for (int i=0; i<argcv[0]; i++) {
        struct CudaFloatArray *a = (struct CudaFloatArray *)argv[i];
        if (a->size < size)
            size = a->size;
        errcheck(cuMemAlloc(&dps[i], size * sizeof(float)));
        errcheck(cuMemcpyHtoD(dps[i], &a->data, size * sizeof(float)));
        args[i+1] = &dps[i];
    }
    struct CudaFloatArray *r =
        (struct CudaFloatArray *)(alloc_CudaFloatArray(size));
    int fsize = sizeof(float) * size;
    errcheck(cuMemAlloc(&d_res, fsize));
    errcheck(cuMemcpyHtoD(d_res, &r->data, fsize));
    args[0] = &d_res;
    args[argcv[0]+1] = &size;

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    char name[256];
    strcpy(name, "block");
    strcat(name, grcstring(argv[argcv[0]]) + strlen("_cuda/"));
    for (int i=0; name[i] != 0; i++)
        if (name[i] == '.') {
            name[i] = 0;
            break;
        }
    errcheck(cuModuleGetFunction(&cuFunc, cuModule, name));
    errcheck(cuLaunchKernel(cuFunc, blocksPerGrid, 1, 1,
        threadsPerBlock, 1, 1,
        0,
        NULL, args, NULL));
    errcheck(cuMemcpyDtoH(&r->data, d_res, fsize));
    cuMemFree(d_res);
    for (int i=0; i<argcv[0]; i++)
        cuMemFree(dps[i]);
    return (Object)r;
}
Object cuda_floatArray(Object self, int nparts, int *argcv,
        Object *argv, int flags) {
    int n = integerfromAny(argv[0]);
    return alloc_CudaFloatArray(n);
}
Object cuda_deviceName(Object self, int nparts, int *argcv,
        Object *argv, int flags) {
    cuInit(0);
    int deviceCount = 0;
    cuDeviceGetCount(&deviceCount);
    if (deviceCount == 0) {
        raiseError("No CUDA devices found");
    }
    CUdevice cuDevice;
    cuDeviceGet(&cuDevice, 0);
    char name[100];
    cuDeviceGetName(name, 100, cuDevice);
    return alloc_String(name);
}
Object cuda_computeCapability(Object self, int nparts, int *argcv,
        Object *argv, int flags) {
    cuInit(0);
    int deviceCount = 0;
    cuDeviceGetCount(&deviceCount);
    if (deviceCount == 0) {
        raiseError("No CUDA devices found");
    }
    CUdevice cuDevice;
    int major, minor;
    cuDeviceComputeCapability(&major, &minor, cuDevice);
    return alloc_Float64(major + minor / 10.0);
}
int coreMultiplicand(int major, int minor) {
    if (major == 1)
        return 8;
    if (major == 3)
        return 192;
    if (major == 2 && minor == 0)
        return 32;
    if (major == 2 && minor == 1)
        return 48;
    return 192;
}
Object cuda_cores(Object self, int nparts, int *argcv,
        Object *argv, int flags) {
    cuInit(0);
    int deviceCount = 0;
    cuDeviceGetCount(&deviceCount);
    if (deviceCount == 0) {
        raiseError("No CUDA devices found");
    }
    CUdevice cuDevice;
    int mpcount;
    cuDeviceGetAttribute(&mpcount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            cuDevice);
    int major, minor;
    cuDeviceComputeCapability(&major, &minor, cuDevice);
    mpcount *= coreMultiplicand(major, minor);
    return alloc_Float64(mpcount);
}
Object cuda_bindir(Object self, int nparts, int *argcv,
        Object *argv, int flags) {
    return alloc_String(GRACE_CUDA_BIN_DIR);
}
Object cuda_includedir(Object self, int nparts, int *argcv,
        Object *argv, int flags) {
    return alloc_String(GRACE_CUDA_INCLUDE_DIR);
}
Object module_cuda_init() {
    if (cuda_module != NULL)
        return cuda_module;
    CudaError = alloc_Exception("CudaError", ErrorObject);
    gc_root(CudaError);
    ClassData c = alloc_class("Module<cuda>", 13);
    add_Method(c, "over()map", &cuda_over_map);
    add_Method(c, "floatArray", &cuda_floatArray);
    add_Method(c, "deviceName", &cuda_deviceName);
    add_Method(c, "computeCapability", &cuda_computeCapability);
    add_Method(c, "cores", &cuda_cores);
    add_Method(c, "using()do()blockWidth()blockHeight()gridWidth()gridHeight",
        &cuda_using_do_blockWidth_blockHeight_gridWidth_gridHeight);
    add_Method(c, "using()do", &cuda_using_do);
    add_Method(c, "using()times()do", &cuda_using_times_do);
    add_Method(c, "bindir", &cuda_bindir);
    add_Method(c, "includedir", &cuda_includedir);
    Object o = alloc_newobj(0, c);
    cuda_module = o;
    gc_root(o);
    return o;
}
