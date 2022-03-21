using Zygote
using ProgressLogging
using Plots
plotlyjs()

state2lattice(state) = π .* state

mutable struct lattice{N}
    # side_length::Int
    θ::Matrix{Float64}
    A::Matrix{Tuple{Float64,Float64}}
    previdx::NTuple{N,Int64}
    # neighbours
end

function lattice(side_length::Integer, A; T=Float64)
    θ = rand(T, side_length, side_length)
    @assert size(A) == size(θ)
    previdx = Tuple(circshift(1:side_length, -1))
    nextidx = circshift(1:side_length, 1)
    # neighbours = [
    #     CartesianIndex.((
    #         (i[1], previdx[i[2]]),
    #         # (i[1], nextidx[i[2]]),
    #         (previdx[i[1]], i[2]),
    #         # (nextidx[i[1]], i[2]),
    #     ))
    #     for i in CartesianIndices(θ)
    # ]
    return lattice(
        θ,
        A,
        previdx
    )
end

function energy(lattice)
    θpi = π .* lattice.θ
    side_length = size(lattice.θ, 1)
    # previdx = mod1.((1:side_length) .- 1, side_length)
    E = -sum(
        cos(θpi[i, j] - θpi[lattice.previdx[i], j] + lattice.A[i, j][1]) +
        cos(θpi[i, j] - θpi[i, lattice.previdx[i]] + lattice.A[i, j][2])
        for i in 1:side_length
        for j in 1:side_length
    )
    return E
end


function find_ground_state(side_length, steps=100)
    A = [Tuple(rand(2)) for i in 1:side_length, j in 1:side_length]

    lat = lattice(side_length, A)
    energies = [energy(lat)]

    @progress for i in 1:steps
        # ∇ = gradient(energy, state)[1]
        E, grads = withgradient(energy, lat)
        δθ = -grads[1].θ * 0.01

        # lat.θ = (lat.θ .- ∇θ .* 0.01) .% 1.0
        lat.θ .+= δθ
        lat.θ .%= 1.0


        push!(energies, E)

        # @info i energy(state)
    end

    return lat, energies
end

ground_state, energies = find_ground_state(20, 2000)

@time find_ground_state(20, 2000)
@profview find_ground_state(20, 2000)
@trace find_ground_state(20, 2000)

side_length = 5
A = [Tuple(rand(2)) for i in 1:side_length, j in 1:side_length]
lat = lattice(side_length, A)
@trace energy(lat)
@benchmark energy(lat)



jheatmap(ground_state)

plot(energies)
minenergy = minimum(energies)
plot(energies .- prevfloat(floor(minenergy)), yscale=:log10)
