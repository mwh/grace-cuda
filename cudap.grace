import "io" as io

def CudaError = Error.refine "CudaError"

method replaceNode(node) {
    if (node.kind == "call") then {
        if (node.value.kind != "member") then {
            return node
        }
        if (node.value.in.value != "cuda") then {
            return node
        }
        if (node.value.value != "over()map") then {
            return node
        }
        if (node.with[2].args[1].kind != "block") then {
            return node
        }
        node.with[2].args[1] := compileBlock(node.with[2].args[1])
    }
    return node
}

method compileBlock(block) {
    var str := ""
    var res := "0"
    var last := ""
    for (block.body) do {n->
        str := str ++ last ++ "\n"
        res := compileNode(n) ++ ";\n"
        last := res
    }
    str := str ++ "__res[__i] = {res};" 
    str := str ++ "}"
    def id = str.hashcode
    var header := "extern \"C\" __global__ void block{id}(float *__res"
    var init := ""
    for (block.params) do {p->
        header := header ++ ", const float *_arg_{p.value}"
        init := init ++ "const float {p.value} = _arg_{p.value}[__i];\n"
    }
    header := header ++ ", int N) \{\n"
    header := header ++ "int __i = blockDim.x * blockIdx.x + threadIdx.x;\n"
    header := header ++ "if (__i < N) \{"
    def fp = io.open("_cuda/{str.hashcode}.cu", "w")
    fp.write(header)
    fp.write(init)
    fp.write(str)
    fp.write("}")
    fp.close
    io.error.write("Compiling CUDA...\n")
    io.system("/opt/cuda/bin/nvcc -m64 -I/opt/cuda/include -I. -I.. " 
        ++ "-I../../common/inc -o _cuda/{id}.ptx -ptx _cuda/{id}.cu")
    return object {
        def kind is public, readable = "string"
        var value is public, readable, writable := "_cuda/{id}.ptx"
        var register is public, readable, writable := ""
        def line is public, readable = block.line
        method accept(visitor) is public {
            visitor.visitString(self)
        }
    }
}

method compileOp(node) {
    return "{compileNode(node.left)} {node.value} {compileNode(node.right)}"
}
method compileNode(node) {
    match(node.kind)
        case { "identifier" -> return node.value }
        case { "op" -> compileOp(node) }
        case { _ ->
            CudaError.raise "Cannot compile {node.kind}:{node.value} to CUDA."}
}
