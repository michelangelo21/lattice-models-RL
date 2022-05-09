using Flux
using Zygote
using ProgressLogging



function energy_cat(θ) #todo add A
    θup = circshift(θ, (1, 0))
    θleft = circshift(θ, (0, 1))
    E = -sum(cos,
        cat(θ - θup, θ - θleft, dims=3),
        dims=1:3
    )
    return E / length(θ)
end

# energy_cat(θ) = energy_cat(θ, zeros((size(θ)..., 2)))

L = 8 # side_length
model = Chain(
    Dense(1 => 128, tanh),
    Dense(128 => 1024, tanh),
    Dense(1024 => L^2),
    x -> reshape(x, (L, L, 1, size(x)[end])),
)




# loss(E_in, E_out) = Flux.Losses.mse(E_out, E_in)
function loss(x, y; L=L)
    θs = model(x)
    ŷs = energy_cat(θs) |> Flux.flatten
    return Flux.Losses.mse(ŷs, y)
end

x_train = transpose(4 * rand(10_000) .- 2) |> collect
y_train = x_train
loss(x_train, y_train)

data = [(x_train, y_train)]
# loss(x_train, y_train)
ps = Flux.params(model)
opt = ADAM(0.1)
@progress for epoch in 1:100
    x_train = transpose(4 * rand(10_000) .- 2) |> collect
    y_train = x_train
    data = [(x_train, y_train)]

    @info loss(x_train, y_train)

    Flux.train!(loss, ps, data, opt)
end

x_test = transpose(4 * rand(10_000) .- 2) |> collect
y_test = x_test
loss(x_train, y_train)
loss(x_test, y_test)

θ = model([0.0 -1.0])
energy_cat(θ)

θs = model([-1.0 0.0 3.0])
θup = circshift(θs, (1, 0))
θleft = circshift(θs, (0, 1))

energy_cat(θs)

tmp = reshape(1:24, (4, 3, 2))
circshift(tmp, (0, 1))

cat(θ - θup, θ - θleft, dims=4)
cat(reshape(1:4, (2, 2)), reshape(-20:-17, (2, 2)), dims=4)

opt = Descent()

