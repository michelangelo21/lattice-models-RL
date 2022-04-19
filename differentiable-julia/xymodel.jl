using Zygote
using ProgressLogging
using Plots
plotlyjs()

@inline state2lattice(state) = π .* state

struct lattice{T<:Real,I<:NTuple}
    side_length::Integer
    θ::Matrix{T}
    previdx::I |
    nextidx::I
    neighbours
end

function lattice(side_length::Integer; T=Float64)
    θ = rand(T, side_length, side_length)
    previdx = Tuple(mod1.((1:side_length) .- 1, side_length))
    nextidx = Tuple(mod1.((1:side_length) .+ 1, side_length))
    neighbours(i) = CartesianIndex.((
        (i[1], previdx[i[2]]),
        (i[1], nextidx[i[2]]),
        (previdx[i[1]], i[2]),
        (nextidx[i[1]], i[2]),
    ))
    return neighbours
end


function energy(state)
    lattice = state2lattice(state)
    side_length = size(lattice, 1)
    # previdx = mod1.((1:side_length) .- 1, side_length)
    previdx(i) = mod1(i - 1, side_length)
    E = -sum(
        cos(lattice[i, j] - lattice[previdx(i), j]) +
        cos(lattice[i, j] - lattice[i, previdx(j)])
        for i in 1:side_length
        for j in 1:side_length
    )
    return E
end

side_length = 20
previdx = circshift(1:side_length, -1)
const NEIGHBOURS = [
    CartesianIndex.((
        (i[1], previdx[i[2]]),
        # (i[1], nextidx[i[2]]),
        (previdx[i[1]], i[2]),
        # (nextidx[i[1]], i[2]),
    ))
    for i in CartesianIndices(ground_state)
]

function energy_cartesian(state)
    lattice = state2lattice(state)
    E = -sum(
        cos(lattice[i] - lattice[j])
        for i in eachindex(lattice)
        for j in NEIGHBOURS[i]
    )
    return E
end


function find_ground_state(side_length, steps=100)
    state = rand(side_length, side_length) # random initial state [-1,1]

    energies = [energy_cartesian(state)]

    @progress for i in 1:steps
        E, grads = withgradient(energy_cartesian, state)
        δstate = -grads[1] * 0.01
        state = mod.(state .+ δstate, 1.0)
        push!(energies, E)

        # @info i energy(state)
    end

    return state, energies
end

ground_state, energies = find_ground_state(20, 2000)
@time find_ground_state(20, 2000)
@profview find_ground_state(20, 2000)
@trace find_ground_state(20, 2000)

heatmap(ground_state)

plot(energies)
minenergy = minimum(energies)
plot(energies .- prevfloat(floor(minenergy)), yscale=:log10)
