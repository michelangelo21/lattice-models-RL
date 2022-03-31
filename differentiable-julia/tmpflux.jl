using Flux

W = rand(2, 5)
b = rand(2)

predict(x) = (W * x) .+ b
loss(x, y) = sum((predict(x) .- y) .^ 2)

x, y = rand(5), rand(2)
l = loss(x, y)

θ = params(W, b)
grads = Flux.gradient(() -> loss(x, y), θ)
θ̄ = gradient(loss, x, y)

Zygote.Grads