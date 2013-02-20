import "cuda" as cuda
import "sys" as sys
// This sample performs matrix multiplication, taking control of the
// CUDA block size itself and using the inferred-type do()... form.
// It includes two implementations: one using CUDA and one in Grace,
// with the same simple algorithm.
// Alter the three numbers below to adjust the size of the matrices.
def m = 100
def n = 100
def p = 100
var doPrinting := false
if ((m * p * n) <= 512) then {
    // With larger matrices you probably don't want to print them
    // out, but checking correctness for smaller ones is useful.
    // This test is a simple heuristic for when to print, but
    // you can set the doPrinting variable how you wish.
    doPrinting := true
}
print("Device: {cuda.deviceName} with capability {cuda.computeCapability}"
    ++ " and {cuda.cores} cores")
var a := cuda.floatArray(m * n)
var b := cuda.floatArray(p * m)
var c := cuda.floatArray(n * p)
var d := cuda.floatArray(n * p)
for (0..(m * n-1)) do {i->
    a.at(i)put(i+1)
}
for (0..(p * m)) do {i->
    b.at(i)put(i+1)
}
method printMatrix(label, mat, h, w) {
    if (!doPrinting) then {
        return m
    }
    // Matrices are stored across-then-down in the arrays.
    print "{label}:"
    for (0..(h-1)) do {y->
        var row := ""
        for (0..(w-1)) do {x->
            row := row ++ " " ++ (mat.at(y*w+x))
        }
        print(row)
    }
}
printMatrix("a", a, n, m)
printMatrix("b", b, m, p)

def bs = 16
def gw = (p + bs - 1) / bs
def gh = (n + bs - 1) / bs
print "{sys.elapsed}: Starting CUDA..."
def startCuda = sys.elapsed
// This block is syntactically Grace code, but is translated by the
// compiler plugin to CUDA C code, inferring the types of captured
// variables. It is executed in parallel lockstep across the CUDA cores.
// It reads from a and b and saves the results into c. Each core
// calculates one cell of the matrix at once using the definition of
// matrix multiplication.
cuda.do {
    def i = blockDim.y * blockIdx.y + threadIdx.y
    def j = blockDim.x * blockIdx.x + threadIdx.x
    def index = i * p + j
    // The code from here on, with the types removed, is exactly
    // the same as in the Grace implementation below in
    // graceMatrixMultiply.
    def ma = m - 1
    if ((i < n) && (j < p)) then {
        // This block calculates:
        //     C(i,j) = Σ(0, m-1, A(i,k)B(k,j))
        // a, b, and c are all arrays flattening the matrices
        // across-then-down.
        var val := 0
        for (0..ma) do {k->
            // Linearise the matrix coordinates to find the
            // right locations in the flat input arrays.
            def ai = k + i * m
            def bi = j + k * p
            val := val + a[ai] * b[bi]
        }
        c[index] := val
    }
} blockWidth(bs) blockHeight(bs) gridWidth(gw) gridHeight(gh)
def cudaTime = sys.elapsed - startCuda
print "{sys.elapsed}: Done in {cudaTime}s."

printMatrix("c", c, n, p)

method graceMatrixMultiply {
    def size = n * p
    for (0..(n-1)) do {i->
        for (0..(p-1)) do {j->
            // This loop body is the same as the CUDA code above
            def index = i * p + j
            def ma = m - 1
            if ((i < n) && (j < p)) then {
                var val := 0
                for (0..ma) do {k->
                    def ai = k + i * m
                    def bi = j + k * p
                    val := val + a[ai] * b[bi]
                }
                d[index] := val
            }
        }
    }
}
print "{sys.elapsed}: Starting Grace..."
def startGrace = sys.elapsed
graceMatrixMultiply
def graceTime = sys.elapsed - startGrace
print "{sys.elapsed}: Done in {graceTime}s."
printMatrix("d", d, n, p)

print "CUDA multiplication time:  {cudaTime}"
print "Grace multiplication time: {graceTime}"
