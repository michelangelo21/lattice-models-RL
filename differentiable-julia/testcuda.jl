using CUDA
using BenchmarkTools
using Zygote

function foo(x)
    xleft = circshift(x, (0, 1))
    xup = circshift(x, (-1, 0))
    E = -sum(
        cos.(x - xleft) + cos.(x - xup))
    return E
end

function fooitr(x, previdx)
    ni, nj = size(x)
    -sum(
        cos(x[i, j] - x[i, previdx[j]]) +
        cos(x[i, j] - x[previdx[i], j])
        for i in ni, j in nj
    )
end

function foo(x, xleft, xup)
    E = -sum(
        cos.(x - xleft) + cos.(x - xup))
    return E
end

function foo_vec(x)
    xleft = circshift(x, (0, 1, 0))
    xup = circshift(x, (-1, 0, 0))
    E = -sum(
        cos.(x - xleft) + cos.(x - xup),
        dims=(1, 2))
    return E
end

L = 32

tmp = rand(Float32, L, L)

cutmp = cu(tmp)

previdx = circshift(1:size(tmp)[1], -1)

tmpleft = circshift(tmp, (0, 1))
tmpup = circshift(tmp, (-1, 0))

cutmpleft = cu(tmpleft)
cutmpup = circshift(cutmp, (-1, 0))

foo(tmp, tmpleft, tmpup)
foo(tmp)

dst = rand(L, L)
cudst = cu(rand(L, L))

@benchmark circshift($tmp, $(0, 1))
@benchmark circshift($cutmp, $(0, 1))

@benchmark circshift!($dst, $tmp, $(0, 1))
@benchmark CUDA.@sync circshift!($cudst, $cutmp, $(0, 1))

@benchmark foo($tmp)
@benchmark foo($cutmp)

@benchmark foo($tmp, $tmpleft, $tmpup)
@benchmark foo($cutmp, $cutmpleft, $cutmpup)

tv = rand(Float32, 16, 16, 3 * 1024)
cutv = cu(tv)

@benchmark foo_vec($tv)
@benchmark foo_vec($cutv)

# i MB of memory
cutvs = [cu(rand(Float32, 32, 32, i * 256)) for i in 1:16]

@benchmark foo_vec($cutvs[1]) # 1 MB is slow
@benchmark foo_vec($cutvs[2]) # more than 2MB is very fast

@profview for _ in 1:1024
    foo_vec(cutvs[2])
end

@benchmark gradient($foo, $tmp)
@benchmark gradient($foo, $cutmp)

@benchmark withgradient($foo, $tmp)
@benchmark withgradient($foo, $cutmp)

@benchmark gradient(x -> sum(foo_vec(x)), $tv)
@benchmark gradient(x -> x |> foo_vec |> sum, $cutv)

foo_vec(tv)
@benchmark [foo(s) for s in eachslice($tv, dims=(3))]
@benchmark [foo(s) for s in eachslice($cutv, dims=(3))]

