using SmoothMixtures
using SmoothMixtures.CUDA
using Test
using Random
using FiniteDifferences
using Flux


function prepare_data(;remove_const=false)
    xtrain, ytrain = MLDatasets.MNIST(:train)[:]
    xtest, ytest = MLDatasets.MNIST(:test)[:]
    xtrain = map(x -> x > 0.5, reshape(xtrain, :, size(xtrain,3)))
    xtest = map(x -> x > 0.5, reshape(xtest, :, size(xtest,3)))
    if remove_const
        free_indices = findall(size(xtrain, 2) .> vec(sum(xtrain, dims = 2)) .> 0)
        xtrain = xtrain[free_indices,:]
        xtest = xtest[free_indices,:]
    end
    ytrain = ytrain .+ 1
    ytest = ytest .+ 1
    ((xtrain, ytrain), (xtest, ytest))
end


train, test = prepare_data(;remove_const = false)
x, y = train
x = map(x -> x + 1, x)

n_categories = 2
n_dimension = size(x, 1)
n_components = Int(2^14)
n_observations = 256

ffnn = Chain(Dense(64,128,leakyrelu),
	Dense(128, 256, leakyrelu),
	Dense(256, 512, leakyrelu),
	Dense(512, n_categories * n_dimension, leakyrelu),
	Base.Fix2(reshape, (n_categories, n_dimension, :)),
	x -> logsoftmax(x, dims = 1),
	)
cu_ffnn = Flux.gpu(ffnn)

centers = randn(Float32, 64, n_components)
cu_ffnn(cu_centers)

cu_x = CuArray(x);

rule = Optimisers.Adam()  # use the Adam optimiser with its default settings
state_tree = Optimisers.setup(rule, cu_ffnn);  # initialise this optimiser's momentum etc.
for i in 1:100
	step = 0 
	fval = 0f0
	for xᵢ in DataLoader(cu_x, ; batchsize = n_observations)
		cu_centers = CUDA.rand(64, n_components)
		fᵢ, ∇cu_ffnn = Flux.Zygote.withgradient(cu_ffnn -> -sumlogsumexp_logprob_fused(cu_ffnn(cu_centers), xᵢ), cu_ffnn)
		state_tree, cu_ffnn = Optimisers.update(state_tree, cu_ffnn, ∇cu_ffnn[1]);
		step += 1
		fval += fᵢ
	end
	fval = -fval / step
	println(i,": ", fval, " bpp: ", - fval / n_dimension / log(2))
end

# in the second stage, we finetune the centers
cu_centers = CUDA.rand(64, n_components)
rule = Optimisers.Adam()  # use the Adam optimiser with its default settings
state_tree = Optimisers.setup(rule, cu_centers);  # initialise this optimiser's momentum etc.
for i in 1:10
	step = 0 
	fval = 0f0
	for xᵢ in DataLoader(cu_x, ; batchsize = n_observations)
		fᵢ, ∇cu_centers = Flux.Zygote.withgradient(cu_centers -> -sumlogsumexp_logprob_fused(cu_ffnn(cu_centers), xᵢ), cu_centers)
		state_tree, cu_centers = Optimisers.update(state_tree, cu_centers, ∇cu_centers[1]);
		step += 1
		fval += fᵢ
	end
	fval = -fval / step
	println(i,": ", fval, " bpp: ", - fval / n_dimension / log(2))
end

# Let's wrtie the conditional sampler
ii = sample(1:size(x,1), 200, replace = false)
xₛ = x[:,3]
xᵢᵢ = xₛ[ii]

centers = deserialize("mnist_centers.jls")
c = logsoftmax(centers[:,ii,:], dims = 1)
pzx = softmax(vec(sum(xᵢᵢ .* c[1,:,:]  .+ (1 .- xᵢᵢ) .* c[2, :,:], dims = 1)))
j = argmax(pzx)

_c = softmax(centers, dims = 1)[2,:,:]
heatmap(reverse(reshape(_c[:,j], 28, 28), dims = 2); colormap = :plasma)
heatmap(reverse(reshape(x[:,3], 28, 28), dims = 2); colormap = :plasma)


centers = deserialize("mnist_centers.jls")
function sampler(centers, ii)
	c = logsoftmax(centers[:,ii,:], dims = 1)
	pzx = softmax(vec(sum(xᵢᵢ .* c[1,:,:]  .+ (1 .- xᵢᵢ) .* c[2, :,:], dims = 1)))
end




n = ceil(Int, sqrt(size(c, 2)))
img = c[:,1]
f = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98),
    size = (16000, 16000))

ga = f[1, 1] = GridLayout()
for i in 1:n 
	for j in 1:n 
		k = 48 + (i - 1) * n + j
		ax = Axis(ga[i, j])
		img = c[:,k]
		heatmap!(ax, reverse(reshape(img, 28, 28), dims = 2); colormap = :plasma)
	end
end

