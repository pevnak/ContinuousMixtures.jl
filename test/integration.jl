
# compare CPU version to GPU version
n_categories = 10
n_dimension = 1_600
n_components = Int(2^14)
n_observations = 250

ffnn = Chain(Dense(64,128,leakyrelu),
	Dense(128, 256, leakyrelu),
	Dense(256, 512, leakyrelu),
	Dense(512, 1024, leakyrelu),
	Dense(1024, n_categories * n_dimension, leakyrelu),
	Base.Fix2(reshape, (n_categories, n_dimension, :)),
	x -> logsoftmax(x, dims = 1),
	)

cu_ffnn = Flux.gpu(ffnn)

centers = randn(Float32, 64, n_components)
cu_centers = CuArray(centers)
cu_ffnn(cu_centers)

logits = randn(Float32, n_categories, n_dimension, n_components);
x = rand(UInt8(1):UInt8(n_categories), n_dimension, n_observations);

cu_logits = CuArray(logits)
cu_x = CuArray(x)

Flux.Zygote.withgradient(cu_centers -> sumlogsumexp_logprob_fused(cu_ffnn(cu_centers), cu_x), cu_centers)

cu_x = CuArray(rand(UInt8(1):UInt8(n_categories), n_dimension, n_observations));
@benchmark CUDA.@sync Flux.Zygote.withgradient(cu_centers -> sumlogsumexp_logprob_fused(cu_ffnn(cu_centers), cu_x), cu_centers)

Flux.Zygote.withgradient(cu_logits -> sumlogsumexp_logprob_fused(cu_logits, cu_x), cu_logits)




cu_logprobs, cu_mx = logprob(cu_logits, cu_x)
lkl, cu_sumexp = sumlogsumexp(cu_logprobs, cu_mx)
∇logits = ∇logprob_fused(1, cu_logits, cu_x, cu_mx, cu_logprobs, cu_sumexp)

@benchmark CUDA.@sync Flux.Zygote.withgradient(cu_logits -> sumlogsumexp_logprob_fused(cu_logits, cu_x), cu_logits)
@benchmark CUDA.@sync Flux.Zygote.withgradient(cu_logits -> sumlogsumexp_logprob(cu_logits, cu_x), cu_logits)


Flux.Zygote.withgradient(cu_logits -> sumlogsumexp_logprob_fused(cu_logits, cu_x), cu_logits)
CUDA.@profile Flux.Zygote.withgradient(cu_logits -> sumlogsumexp_logprob_fused(cu_logits, cu_x), cu_logits)


@benchmark CUDA.@sync logprob(cu_logits, cu_x)
@benchmark CUDA.@sync sumlogsumexp(cu_logprobs, cu_mx)
@benchmark CUDA.@sync ∇logprob_fused(1, cu_logits, cu_x, cu_mx, cu_logprobs, cu_sumexp)



@benchmark CUDA.@sync logprob(cu_logits, cu_x)
@benchmark CUDA.@sync sumlogsumexp_logprob_fused(cu_logits, cu_x)
@benchmark CUDA.@sync sumlogsumexp_logprob(cu_logits, cu_x)

@benchmark CUDA.@sync Flux.Zygote.withgradient(cu_logits -> sumlogsumexp_logprob_fused(cu_logits, cu_x), cu_logits)
@benchmark CUDA.@sync Flux.Zygote.withgradient(cu_logits -> sumlogsumexp_logprob(cu_logits, cu_x), cu_logits)

@benchmark CUDA.@sync begin
    cu_logprobs, cu_mx = logprob(cu_logits, cu_x)
    lkl, cu_sumexp = sumlogsumexp(cu_logprobs, cu_mx)
	∇logits = ∇logprob_fused(1, cu_logits, cu_x, cu_mx, cu_logprobs, cu_sumexp)
end

cu_logprobs, cu_mx = logprob(cu_logits, cu_x)
lkl, cu_sumexp = sumlogsumexp(cu_logprobs, cu_mx)
∇logits = ∇logprob_fused(1, cu_logits, cu_x, cu_mx, cu_logprobs, cu_sumexp)

@benchmark CUDA.@sync logprob(cu_logits, cu_x)
@benchmark CUDA.@sync sumlogsumexp(cu_logprobs, cu_mx)
@benchmark CUDA.@sync ∇logprob_fused(1, cu_logits, cu_x, cu_mx, cu_logprobs, cu_sumexp)
