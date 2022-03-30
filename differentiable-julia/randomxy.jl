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
    θpi = 2π .* lattice.θ #! 2π
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


function find_ground_state(side_length, A; steps=100, η=0.001)
    lat = lattice(side_length, A)
    energies = [energy(lat)]

    @progress for i in 1:steps
        # ∇ = gradient(energy, state)[1]
        E, grads = withgradient(energy, lat)
        δθ = -grads[1].θ * η
        # maximum(abs.(δθ)) > 1 && @warn δθ
        # lat.θ = (lat.θ .- ∇θ .* 0.01) .% 1.0
        lat.θ .+= δθ
        # ? rem2pi
        lat.θ = mod.(lat.θ, 1.0) #! negative angles are not allowed


        push!(energies, E)

        # @info i energy(state)
    end

    return lat, energies
end

side_length = 10
A = [Tuple(4π * rand(2) .- 2π) for i in 1:side_length, j in 1:side_length]

plt = plot()
results = []
for _ in 1:80
    ground_state, energies = find_ground_state(side_length, A; steps=5000, η=0.001)
    plot!(plt, energies)
    push!(results, (ground_state, energies))
end
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