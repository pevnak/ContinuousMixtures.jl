include("cpu.jl")
include("cuda.jl")

"""
    sumlogsumexp_logprob(logits, x)

    compute likelihood of `x` given `logits,` 
    where logits defines a mixture of categorical distributions with uniform weight.
"""
function sumlogsumexp_logprob(logits, x)
    log_probs, max_ = logprob(logits, x)
    sum(max_' .+ log.(sum(exp.(log_probs .- max_'); dims = 1)))
end

function ChainRulesCore.rrule(::typeof(sumlogsumexp_logprob), logits, x)
    # The gradient is `softmax`, but both compute `tmp` so it's worth saving.
    log_probs, max_ = logprob(logits, x)
    max_ = max_'

    # this should be fused to the second kernel
    tmp = exp.(log_probs .- max_)
    sum_tmp = sum(tmp; dims = 1)
    lkl = sum(max_ .+ log.(sum_tmp))

    function sumlogsumexp_pullback(dy)
    	∇logprobs = unthunk(dy) .* tmp ./ sum_tmp
    	∇logits = ∇logprob(∇logprobs, logits, x)
    	(NoTangent(), ∇logits, NoTangent())
    end
    return lkl, sumlogsumexp_pullback
end

function sumlogsumexp_logprob_fused(logits, x)
	log_probs, max_ = logprob(logits, x)
	lkl, sumexp = sumlogsumexp(log_probs, max_)
	return(lkl)
end

function ChainRulesCore.rrule(::typeof(sumlogsumexp_logprob_fused), logits, x)
    # The gradient is `softmax`, but both compute `tmp` so it's worth saving.
    log_probs, max_ = logprob(logits, x)
    lkl, sumexp = sumlogsumexp(log_probs, max_)
    function sumlogsumexp_pullback(dy)
    	∇logits = ∇logprob_fused(dy, logits, x, max_, log_probs, sumexp)
    	(NoTangent(), ∇logits, NoTangent())
    end
    return lkl, sumlogsumexp_pullback
end
