import "cuda" as cuda
import "sys" as sys
// This sample performs matrix multiplication. It includes two
// implementations: one using CUDA and one in Grace, with the same
// simple algorithm and almost the same code (modulo types).
// Alter the three numbers below to adjust the size of the matrices.

def m1w = 20
def m1h = 30
def m2w = 40
def m2h = m1w
var doPrinting := false
if ((m1w * m2w * m1h) <= 512) then {
    // With larger matrices you probably don't want to print them
    // out, but checking correctness for smaller ones is useful.
    // This test is a simple heuristic for when to print, but
    // you can set the doPrinting variable how you wish.
    doPrinting := true
}
var m1 := cuda.floatArray(m1w * m1h)
var m2 := cuda.floatArray(m2w * m2h)
var m3 := cuda.floatArray(m1h * m2w)
var m4 := cuda.floatArray(m1h * m2w)
for (0..(m1w * m1h-1)) do {i->
    m1.at(i)put(i+1)
}
for (0..(m2w * m2h)) do {i->
    m2.at(i)put(i+1)
}
method printMatrix(label, m, h, w) {
    if (!doPrinting) then {
        return m
    }
    print "{label}:"
    for (0..(h-1)) do {y->
        var row := ""
        for (0..(w-1)) do {x->
            row := row ++ " " ++ (m.at(y*w+x))
        }
        print(row)
    }
}
printMatrix("M1", m1, m1h, m1w)
printMatrix("M2", m2, m2h, m2w)

print "{sys.elapsed}: Starting CUDA..."
def startCuda = sys.elapsed
cuda.over(m1, m2, m3)numbers(m1h, m1w, m2w)do {a, b, c, n, m, p, size ->
    var index : int := blockDim.x * blockIdx.x + threadIdx.x
    def ma : int = m - 1
    def p2 : int = p
    if (index < size) then {
        def i : int = index / p2
        def j : int = index % p2
        var val : float := 0
        for (0..ma) do {k->
            def ai : int = k + i * m
            def bi : int = j + k * m
            val := val + a[ai] * b[bi]
        }
        c[index] := val
    }
}size(m1h * m2w)
def cudaTime = sys.elapsed - startCuda
print "{sys.elapsed}: Done"

printMatrix("M3", m3, m1h, m2w)

method baseMult(a, b, c, n, m, p) {
    def size = n * p
    for (0..(size-1)) do {index->
        def ma = m - 1
        def p2 = p
        if (index < size) then {
            def i = (index / p2).truncate
            def j = index % p2
            var val := 0
            for (0..ma) do {k->
                def ai = k + i * m
                def bi = j + k * m
                val := val + a.at(ai) * b.at(bi)
            }
            c.at(index)put(val)
        }
    }
}
print "{sys.elapsed}: Starting Grace..."
def startGrace = sys.elapsed
baseMult(m1, m2, m4, m1h, m1w, m2w)
def graceTime = sys.elapsed - startGrace
print "{sys.elapsed}: Done."
printMatrix("M4", m4, m1h, m2w)

print "CUDA multiplication time:  {cudaTime}"
print "Grace multiplication time: {graceTime}"
