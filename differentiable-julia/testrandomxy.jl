using BenchmarkTools

side_length = 3
T = Float32
A = 4π * rand(T, side_length, side_length, 2) .- 2π


lat = lattice(side_length, A; T)
energy(lat)
energy_circshift(lat)

θpi = 2π .* lat.θ #! 2π
θpi = reshape((1:9) .^ 2, (3, 3))
θleft = circshift(θpi, (0, 1))
θup = circshift(θpi, (1, 0))
(θpi - θleft)
θpi[2, 1]
θleft[2, 2]
[(θpi[i, j] - θpi[i, lat.previdx[j]]) for i in 1:side_length, j in 1:side_length]
lat.previdx[2]
θpi[2, 2] - θpi[2, 1]


ps = params(lat.θ)
E, grads = withgradient(() -> energy(lat), ps)
grads[lat.θ]
ps[1]
grads[ps[1]]


W = rand(2, 5);
b = rand(2);
Params([W, b])[1]
ps[1]


ps = params(lat.θ)
@benchmark energy($lat)
@benchmark withgradient(() -> energy($lat), $ps)

@benchmark energy_cat($lat)
@benchmark withgradient(() -> energy_cat($lat), $ps)

@benchmark withgradient($energy, $lat)
@benchmark withgradient($energy_cat, $lat)

using Random

xoshiro = Random.Xoshiro()
mersenne = Random.MersenneTwister()
device = Random.RandomDevice()

@benchmark rand(100, 100)
@benchmark rand($xoshiro, 100, 100)
@benchmark rand($mersenne, 100, 100)
@benchmark rand($device, 100, 100)

tmp = rand(3, 3)

@benchmark rand(Float64, size($tmp))
@benchmark rand(Float64, $(3, 3))
size(tmp)
Type

t = 1:1000 |> collect
s = CosAnneal(λ0=5e-2, λ1=0e-5, period=500)
s = Loop(Exp(λ=8e-2, γ=0.9), 100)
plot(t, s.(t);)