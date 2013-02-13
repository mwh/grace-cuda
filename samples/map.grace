import "cuda" as cuda
// This sample uses CUDA to perform a "map" operation over multiple
// arrays of numbers, computing their elementwise products.

def size = 1000000

print "Starting population..."

var x := cuda.floatArray(size)
var y := cuda.floatArray(size)
var z := cuda.floatArray(size)

for (0..(size-1)) do {i->
    x.at(i)put(i)
    y.at(i)put(i)
    z.at(i)put(i)
}

print "Starting CUDA..."

def res = cuda.over(x, y, z) map {a, b, c->
    a * b * c
}

print "1000^3: {res.at 1000}"
print "93397^3: {res.at 93397}"
