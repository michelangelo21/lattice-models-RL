using Zygote
using ProgressLogging
using Plots
plotlyjs()

state2lattice(state) = π .* state

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


function find_ground_state(side_length, steps = 100)
    state = rand(side_length, side_length) # random initial state [-1,1]

    energies = [energy(state)]

    @progress for i in 1:steps
        ∇ = gradient(energy, state)[1]
        state = (state .- ∇ .* 0.01) .% 1
        push!(energies, energy(state))

        # @info i energy(state)
    end

    return state, energies
end

ground_state, energies = find_ground_state(20, 2000)

heatmap(ground_state)

plot(energies)
minenergy = minimum(energies)
plot(energies .- prevfloat(floor(minenergy)), yscale = :log10)
