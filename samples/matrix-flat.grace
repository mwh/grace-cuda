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
    // Matrices are stored across-then-down in the arrays.
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
// This block is syntactically Grace code, but is translated by the
// compiler plugin to CUDA C code (whence the "int" and "float" types).
// It is executed in parallel lockstep across the CUDA cores size times,
// with m1, m2, and m3 available as arrays a, b, and c, and m1h, m1w,
// and m2w floats called n, m, and p. It reads from m1 and m2 and saves
// the results into m3. Each core calculates one cell of the matrix
// at once using the definition of matrix multiplication.
cuda.using(m1, m2, m3, m1h, m1w, m2w)times(m1h * m2w)
    do {a : floatArray, b : floatArray, c : floatArray, n : int, m : int,
        p : int, size : int ->
    var index : int := blockDim.x * blockIdx.x + threadIdx.x
    // The code from here on, with the types removed, is exactly
    // the same as in the Grace implementation below with one
    // exception. The Grace version requires that the
    // initialisation of i be truncated to an integer explicitly,
    // because all numbers may be fractional. Other than those
    // changes, the body of the loop in graceMatrixMultiply is
    // identical to here.
    def ma : int = m - 1
    def p2 : int = p
    if (index < size) then {
        // This block calculates:
        //     C(i,j) = Σ(0, m-1, A(i,k)B(k,j))
        // a, b, and c are all arrays flattening the matrices
        // across-then-down.
        // First find the (i,j) coordinates from the index
        // of the destination in the flat output array.
        def i : int = index / p2
        def j : int = index % p2
        var val : float := 0
        for (0..ma) do {k->
            // Linearise the matrix coordinates to find the
            // right locations in the flat input arrays.
            def ai : int = k + i * m
            def bi : int = j + k * p
            val := val + a[ai] * b[bi]
        }
        c[index] := val
    }
}
def cudaTime = sys.elapsed - startCuda
print "{sys.elapsed}: Done"

printMatrix("M3", m3, m1h, m2w)

method graceMatrixMultiply(a, b, c, n, m, p) {
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
                def bi = j + k * p
                val := val + a[ai] * b[bi]
            }
            c[index] := val
        }
    }
}
print "{sys.elapsed}: Starting Grace..."
def startGrace = sys.elapsed
graceMatrixMultiply(m1, m2, m4, m1h, m1w, m2w)
def graceTime = sys.elapsed - startGrace
print "{sys.elapsed}: Done."
printMatrix("M4", m4, m1h, m2w)

print "CUDA multiplication time:  {cudaTime}"
print "Grace multiplication time: {graceTime}"
