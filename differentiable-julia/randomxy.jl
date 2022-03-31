using Zygote
using Flux
using ProgressLogging
using Plots
plotlyjs()

state2lattice(state) = π .* state

mutable struct lattice{N,T}
    # side_length::Int
    θ::Matrix{T}
    A::Array{T,3}
    previdx::NTuple{N,Int64}
    # neighbours
end

function lattice(side_length::Integer, A; T=Float64)
    θ = 2π * rand(T, side_length, side_length)
    @assert size(A)[1:2] == size(θ) && size(A)[3] == 2
    previdx = Tuple(circshift(1:side_length, 1))
    nextidx = circshift(1:side_length, -1)
    # neighbours = [
    #     CartesianIndex.((
    #         (i[1], previdx[i[2]]),
    #         # (i[1], nextidx[i[2]]),
    #         (previdx[i[1]], i[2]),
    #         # (nextidx[i[1]], i[2]),
    #     ))
    #     for i in CartesianIndices(θ)
    # ]
    return lattice{side_length,T}(
        θ,
        A,
        previdx
    )
end

# function energy_old(lattice)
#     θpi = 2π .* lattice.θ #! 2π
#     side_length = size(lattice.θ, 1)
#     # previdx = mod1.((1:side_length) .- 1, side_length)
#     return -sum(
#         cos(θpi[i, j] - θpi[lattice.previdx[i], j] + lattice.A[i, j, 1]) +
#         cos(θpi[i, j] - θpi[i, lattice.previdx[i]] + lattice.A[i, j, 2])
#         for i in 1:side_length, j in 1:side_length
#     )
# end

function energy(lattice)
    # lattice.θ = 2π .* lattice.θ 
    θup = circshift(lattice.θ, (1, 0))
    θleft = circshift(lattice.θ, (0, 1))
    # return -sum(
    #     @. cos(lattice.θ - θup + @view lattice.A[:, :, 1]) +
    #        cos(lattice.θ - θleft + @view lattice.A[:, :, 2])
    # )
    return -sum(cos, lattice.θ - θup + @view lattice.A[:, :, 1]) -
           sum(cos, lattice.θ - θleft + @view lattice.A[:, :, 2])
end

function energy_cat(lattice)
    θup = circshift(lattice.θ, (1, 0))
    θleft = circshift(lattice.θ, (0, 1))
    return -sum(cos,
        cat(lattice.θ - θup, lattice.θ - θleft, dims=3) + lattice.A
    )
    # return -sum(cos,
    #     cat(lattice.θ - circshift(lattice.θ, (1, 0)), lattice.θ - circshift(lattice.θ, (0, 1)), dims=3) + lattice.A
    # )
end


function find_ground_state(side_length, A; steps=100, η=0.001, T=Float64)
    lat = lattice(side_length, A; T)
    energies = [energy(lat)]

    # loss(θ) = 
    ps = params(lat.θ)

    opt = Descent(η)

    for i in 1:steps
        # ∇ = gradient(energy, state)[1]
        # E, grads = withgradient(() -> energy(lat), lat.θ)
        E, grads = withgradient(() -> energy_cat(lat), ps)
        # θ̄ = -grads[ps[1]] * η
        # # maximum(abs.(δθ)) > 1 && @warn δθ
        # # lat.θ = (lat.θ .- ∇θ .* 0.01) .% 1.0
        # lat.θ .+= θ̄
        Flux.update!(opt, ps, grads)
        # ? rem2pi
        # lat.θ .= mod.(lat.θ, 2.0 * π) #! negative angles are not allowed
        lat.θ .= mod2pi.(lat.θ)


        push!(energies, E)

        # @info i energy(state)
    end

    return lat, energies
end

side_length = 8
A = 4π * rand(side_length, side_length, 2) .- 2π
steps = 20_000

# plt = plot()
# results = []
@benchmark ground_state, energies = find_ground_state(side_length, A;
    steps=steps,
    η=0.001,
    T=Float32
)

results = []
@simd for _ in 1:100
    res = find_ground_state(side_length, A;
        steps=steps,
        η=0.001,
        T=Float32
    )
    push!(results, res)
end
plot(0:steps, last.(results))
ylims!(-105, -90)

-623.9576

heatmap(results[9][1].θ)
heatmap(results[38][1].θ)
heatmap(results[13][1].θ)
heatmap(results[17][1].θ)
minimum(last.(results))

# plot!(plt, energies)
push!(results, (ground_state, energies))m
# end
display(plt)
savefig(plt, "randomxyzoom.png")
xlims!(plt, 1000, 5000)
ylims!(plt, -155, -140)

plot(diff(energies))
heatmap(ground_state.θ)

minimum(energies)



@time find_ground_state(20, 2000)
@profview find_ground_state(20, 2000)
@trace find_ground_state(20, 2000)

side_length = 5
A = [Tuple(rand(2)) for i in 1:side_length, j in 1:side_length]
lat = lattice(side_length, A)
@trace energy(lat)
@benchmark energy(lat)




minenergy =
    plot(energies .- prevfloat(floor(minenergy)), yscale=:log10)

rem2pi
mod2pi