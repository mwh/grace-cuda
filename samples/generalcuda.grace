import "cuda" as cuda
import "sys" as sys
// This sample uses the using()do method to run a general
// CUDA computation using three arrays of numbers, computing both the
// product and the quotient of the input numbers (a1), as well as e^x.
// The block passed to that method can use many ordinary numeric
// operations, and looks similar to the C that would be written
// otherwise. using()do infers how many times to run by the size of
// the largest input array, which is passed in as the last argument.

def elements = 1000000
print "{sys.elapsed}: Starting population of {elements} elements..."
var a1 := cuda.floatArray(elements)
for (0..(elements-1)) do {i->
    if ((i % 1000000) == 0) then {
        print " {sys.elapsed}: At {i}"
    }
    a1.at(i)put(i)
}
var a2 := cuda.floatArray(elements)
var a3 := cuda.floatArray(elements)

print "{sys.elapsed}: Starting CUDA..."
cuda.using(a1, a2, a3, 2.5)do {x : floatArray, y : floatArray, z : floatArray,
n : float, size : int ->
    var i : int := blockDim.x * blockIdx.x + threadIdx.x
    if (i < size) then {
        z[i] := expf(x[i])
        y[i] := x[i] / n
        x[i] := x[i] * n
    }
}
print "{sys.elapsed}: Done"

for (0..14) do {i->
    print "{i}: {a1.at(i)} {a2.at(i)} {a3.at(i)}"
}
print "{elements-1}: {a1.at(elements-1)} {a2.at(elements-1)} {a3.at(elements-1)}"
