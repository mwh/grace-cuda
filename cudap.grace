import "io" as io
import "mgcollections" as collections
import "ast" as ast
import "cuda" as cuda
import "util" as util

def CudaError = Error.refine "CudaError"

method replaceNode(node) {
    if (node.kind == "call") then {
        if (node.value.kind != "member") then {
            return node
        }
        if (node.value.in.value != "cuda") then {
            return node
        }
        if (node.value.value == "over()map") then {
            return overMap(node)
        }
        if (node.value.value ==
            "using()do()blockWidth()blockHeight()gridWidth()gridHeight") then {
            return basicDo(node)
        }
        if (node.value.value == "using()times()do") then {
            return usingTimesDo(node)
        }
        if (node.value.value == "using()do") then {
            return usingDo(node)
        }
        if (node.value.value == "do") then {
            return plainDo(node)
        }
        if (node.value.value ==
            "do()blockWidth()blockHeight()gridWidth()gridHeight") then {
            return plainDo(node)
        }
    }
    return node
}

def callVisitor = object {
    inherits ast.baseVisitor
    method visitCall(o) -> Boolean {
        replaceNode(o)
        false
    }
}
method processAST(values) {
    for (values) do {v->
        v.accept(callVisitor)
    }
}

method nvcc(id) {
    if (!io.system("{cuda.bindir}/nvcc -m64 -I{cuda.includedir} "
        ++ "-o _cuda/{id}.ptx -ptx _cuda/{id}.cu")) then {
        CudaError.raise("NVCC returned an error when compiling CUDA code")
    }
}
method basicDo(node) {
    node.with[2].args[1] := compileBasicBlock(node.with[2].args[1], node)
    return node
}
method usingTimesDo(node) {
    node.with[3].args[1] := compileBasicBlock(node.with[3].args[1], node)
    return node
}
method usingDo(node) {
    node.with[2].args[1] := compileBasicBlock(node.with[2].args[1], node)
    return node
}
method overMap(node) {
    if (node.with[2].args[1].kind != "block") then {
        return node
    }
    node.with[2].args[1] := compileMapBlock(node.with[2].args[1])
    return node
}
method plainDo(node) {
    def info = compileInferredBlock(node.with[1].args[1], node)
    node.with[1].args[1] := info.id
    for (info.arguments) do {a->
        node.with[1].args.push(ast.identifierNode.new(a, false))
    }
    return node
}

method escapeident(s) {
    var ns := ""
    for (s) do { c ->
        def o = c.ord
        if (((o >= 65) && (o <= 90))
            || ((o >= 97) && (o <= 122))
            || ((o >= 48) && (o <= 57))
            || (o == 95)) then {
            ns := ns ++ c
        } else {
            ns := ns ++ "_{o}_"
        }
    }
    ns
}

method compile(header, init, body, extra, block) {
    def id = escapeident(util.modname) ++ "__{block.line}"
    if (util.extensions.contains("cudapverbose")) then {
        print "CUDA code id: {id}"
        print "{header}{init}{body}"
    }
    def fp = io.open("_cuda/{id}.cu", "w")
    fp.write("extern \"C\" __global__ void block{id}")
    fp.write(header)
    fp.write(init)
    fp.write(body)
    fp.write("}")
    fp.close
    nvcc(id)
    var replacement := "_cuda/{id}.ptx"
    if (extra.size > 0) then {
        replacement := replacement ++ " " ++ extra
    }
    return object {
        def kind is public, readable = "string"
        var value is public, readable, writable := replacement
        var register is public, readable, writable := ""
        def line is public, readable = block.line
        method accept(visitor) is public {
            visitor.visitString(self)
        }
        method map(blk)before(blkBefore)after(blkAfter) {
            self
        }
    }
}
method compileMapBlock(block) {
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
    var header := "(float *__res"
    var init := ""
    for (block.params) do {p->
        header := header ++ ", const float *_arg_{p.value}"
        init := init ++ "const float {p.value} = _arg_{p.value}[__i];\n"
    }
    header := header ++ ", int N) \{\n"
    header := header ++ "int __i = blockDim.x * blockIdx.x + threadIdx.x;\n"
    header := header ++ "if (__i < N) \{"
    compile(header, init, str, "", block)
}

method compileBasicBlock(block, node) {
    var str := ""
    for (block.body) do {n->
        str := str ++ compileNode(n) ++ ";\n"
    }
    var header := "("
    var init := ""
    var pIndex := 1
    var typeStr := ""
    for (block.params) do {p->
        if (p.dtype.value == "float") then {
            header := header ++ "const float {p.value}, "
            typeStr := typeStr ++ "f "
        }
        if (p.dtype.value == "int") then {
            header := header ++ "const int {p.value}, "
            typeStr := typeStr ++ "i "
        }
        if (p.dtype.value == "floatArray") then {
            header := header ++ "float *{p.value}, "
            typeStr := typeStr ++ "f* "
        }
    }
    header := header.substringFrom(1)to(header.size - 2)
    header := header ++ ") \{\n"
    // Here we encode the parameter types given in the block into the
    // string we replace the block with, along with the name of the
    // ptx file. The runtime library understands how to unpack this
    // string and handle the arguments accordingly.
    compile(header, init, str, typeStr, block)
}
method compileInferredBlock(block, node) {
    var str := ""
    def usedIdentifiers = collections.set.new
    def declaredIdentifiers = collections.set.new
    def builtInIdentifiers = collections.set.new("expf", "for()do", "Dynamic")
    def identifierTypes = collections.map.new
    def markType = { n, t ->
        if (n.kind == "identifier") then {
            identifierTypes.put(n.value, t)
        }
    }
    def guessType = { n ->
        match (n.kind)
            case { "member" -> "int" }
            case { "identifier" ->
                    if (identifierTypes.contains(n.value)) then {
                        identifierTypes.get(n.value)
                    }
                }
            case { "op" ->
                    def lt = guessType.apply(n.left)
                    def rt = guessType.apply(n.right)
                    if ((lt == "int") && (rt == "int")) then {
                        "int"
                    } else {
                        "float"
                    }
                }
            case { _ -> "float"}
    }
    def visitor = object {
        inherits ast.baseVisitor
        method visitIdentifier(n) {
            usedIdentifiers.add(n.value)
        }
        method visitVarDec(n) {
            declaredIdentifiers.add(n.name.value)
            def t = guessType.apply(n.value)
            identifierTypes.put(n.name.value, t)
            true
        }
        method visitDefDec(n) {
            declaredIdentifiers.add(n.name.value)
            def t = guessType.apply(n.value)
            identifierTypes.put(n.name.value, t)
            true
        }
        method visitMember(n) {
            false
        }
        method visitIndex(n) {
            markType.apply(n.value, "floatArray")
            if (n.index.kind == "identifier") then {
                markType.apply(n.index, "int")
            }
            if (n.index.kind == "op") then {
                markType.apply(n.index.left, "int")
                markType.apply(n.index.right, "int")
            }
            return true
        }
        method visitBlock(n) {
            for (n.params) do {p->
                declaredIdentifiers.add(p.value)
            }
            for (n.body) do {s->
                s.accept(visitor)
            }
            return true
        }
        method visitOp(n) {
            if (n.value == "%") then {
                markType.apply(n.left, "int")
                markType.apply(n.right, "int")
            }
            if (n.value == "..") then {
                markType.apply(n.left, "int")
                markType.apply(n.right, "int")
            }
            return true
        }
        method visitBind(n) {
            n.dest.accept(visitor)
            n.value.accept(visitor)
            if (n.dest.kind == "index") then {
                markType.apply(n.value, "float")
            }
            return false
        }
    }
    for (block.body) do {n->
        n.accept(visitor)
    }
    for (usedIdentifiers - identifierTypes) do {dt->
        identifierTypes.put(dt, "float")
    }
    def visitor2 = object {
        inherits ast.baseVisitor
        method visitVarDec(n) {
            if (n.dtype.value == "Dynamic") then {
                n.dtype := ast.identifierNode.new(
                    identifierTypes.get(n.name.value), false)
            }
        }
        method visitDefDec(n) {
            if (n.dtype.value == "Dynamic") then {
                n.dtype := ast.identifierNode.new(
                    identifierTypes.get(n.name.value), false)
            }
        }
    }
    for (block.body) do {n->
        n.accept(visitor2)
        str := str ++ compileNode(n) ++ ";\n"
    }
    var args := usedIdentifiers - declaredIdentifiers - builtInIdentifiers
    var header := "("
    var init := ""
    var pIndex := 1
    var typeStr := ""
    for (args) do {arg->
        def pt = identifierTypes.get(arg)
        if (pt == "float") then {
            header := header ++ "const float {arg}, "
            typeStr := typeStr ++ "f "
        }
        if (pt == "int") then {
            header := header ++ "const int {arg}, "
            typeStr := typeStr ++ "i "
        }
        if (pt == "floatArray") then {
            header := header ++ "float *{arg}, "
            typeStr := typeStr ++ "f* "
        }
    }
    header := header.substringFrom(1)to(header.size - 2)
    header := header ++ ") \{\n"
    // Here we provide both the "id" string returned from the compile
    // method (also encoding types), and the arguments that will need to
    // be passed in (i.e., captured variables).
    object {
        def id is readable = compile(header, init, str, typeStr, block)
        def arguments is readable = args
    }
}
method compileNum(node) {
    return "{node.value}"
}
method compileOp(node) {
    return "{compileNode(node.left)} {node.value} {compileNode(node.right)}"
}
method compileMember(node) {
    return "{compileNode(node.in)}.{node.value}"
}
method compileIndex(node) {
    return "{compileNode(node.value)}[{compileNode(node.index)}]"
}
method compileBind(node) {
    return "{compileNode(node.dest)} = {compileNode(node.value)}"
}
method compileVarDec(node) {
    return "{node.dtype.value} {compileNode(node.name)} = {compileNode(node.value)}"
}
method compileDefDec(node) {
    return "const {node.dtype.value} {compileNode(node.name)} = {compileNode(node.value)}"
}
method compileFor(node) {
    def over = node.with[1].args[1]
    if (over.value != "..") then {
        CudaError.raise "Can only write CUDA for loop over numbers x..y."
    }
    def vn = node.with[2].args[1].params[1].value
    var r := "  for (int {vn}={compileNode(over.left)}; {vn}<={compileNode(over.right)}; {vn}++) \{\n"
    for (node.with[2].args[1].body) do {n->
        r := r ++ "    {compileNode(n)};\n"
    }
    r := r ++ "  \}"
    return r
}
method compileCall(node) {
    if (node.value.value == "for()do") then {
        return compileFor(node)
    }
    var r := "{compileNode(node.value)}("
    for (node.with[1].args) do {a->
        r := r ++ compileNode(a) ++ ", "
    }
    r := r.substringFrom(1)to(r.size-2) ++ ")"
    return r
}
method compileIf(node) {
    var r := "  if ({compileNode(node.value)}) \{\n"
    for (node.thenblock) do {n->
        r := r ++ "    {compileNode(n)};\n"
    }
    r := r ++ "  \}"
    return r
}
method compileNode(node) {
    match(node.kind)
        case { "identifier" -> return node.value }
        case { "op" -> compileOp(node) }
        case { "num" -> compileNum(node) }
        case { "member" -> compileMember(node) }
        case { "index" -> compileIndex(node) }
        case { "bind" -> compileBind(node) }
        case { "if" -> compileIf(node) }
        case { "vardec" -> compileVarDec(node) }
        case { "defdec" -> compileDefDec(node) }
        case { "call" -> compileCall(node) }
        case { _ ->
            CudaError.raise "Cannot compile {node.kind}:{node.value} to CUDA."}
}
