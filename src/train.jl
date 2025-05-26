"""
    train_mixture_model(data::AbstractArray; kwargs...)
    train_mixture_model(data::AbstractArray, model; kwargs...)

Train a smooth mixture model on the provided data.

# Arguments
- `data::AbstractArray`: Input data array with dimensions [features, samples]

# Keyword Arguments
## Model configuration
- `n_categories::Int=2`: Number of categories for each dimension
- `n_components::Int=2^10`: Number of mixture components
- `hidden_dims::Union{NTuple{<:Any,Int}, Vector{Int}}=(128, 256, 512)`: Dimensions of hidden layers
- `activation=leakyrelu`: Activation function for hidden layers
- `encoder_dim::Int=64`: Dimension of the latent space for latents

## Training parameters
- `batchsize::Int=256`: Batch size for training
- `max_epochs=100`: Maximum number of training epochs
- `learning_rate=1f-3`: Learning rate for optimizer
- `finetune_latents::Bool=true`: Whether to finetune latents after training
- `finetune_epochs::Int=10`: Number of epochs for finetuning latents

## Device and miscellaneous
- `device=gpu`: Device to use for training (gpu or cpu)
- `verbose=true`: Whether to print progress information
- `checkpoint_path=nothing`: Path to save checkpoints

# Returns
- `model`: Trained model
- `zᵢ`: Optimized latents, each column corresponds to one latent vector
- `stats`: Dictionary with training statistics
"""
function train_mixture_model(data::AbstractArray; n_categories::Int=maximum(data), hidden_dims=[128, 256, 512], activation=leakyrelu, encoder_dim, kwargs...)
    n_dimension = size(data_processed, 1)
    # Create model
    verbose && println("Creating model...")
    model = create_model(n_dimension, n_categories, hidden_dims, activation, encoder_dim)
    train_mixture_model(model, data; n_categories, encoder_dim, kwargs...)
end

function train_mixture_model(model, data::AbstractArray; encoder_dim, n_categories::Int=maximum(data), n_components::Int=2^10, batchsize::Int=256, max_epochs=100, learning_rate=1e-3, finetune_latents::Bool=true, finetune_epochs::Int=10, device=gpu, verbose=true, checkpoint_path=nothing)
    model = device(model)
    # Process data
    verbose && println("Processing data...")
    data_processed = preprocess_data(data, n_categories)
    n_dimension = size(data_processed, 1)
    data_device = device(data_processed)

    optimizer = Optimisers.Adam(learning_rate)
    opt_state = Optimisers.setup(optimizer, model)
    stats = Dict("train_losses" => Float32[], "bits_per_dim" => Float32[], "best_loss" => Inf32)

    verbose && println("Starting model training for $max_epochs epochs...")
    best_model = deepcopy(model)
    for epoch in 1:max_epochs
        epoch_loss, steps = 0.0, 0

        # Train on batches
        for xᵢ in DataLoader(data_device; batchsize=batchsize)
            # Generate random latents for this batch
            zᵢ = device(randn(Float32, encoder_dim, n_components))
            # Compute loss and gradients
            fᵢ, grads = Flux.Zygote.withgradient(model -> -sumlogsumexp_logprob_fused(model(zᵢ), xᵢ), model)
            # Update model parameters
            opt_state, model = Optimisers.update(opt_state, model, grads[1])
            epoch_loss += fᵢ
            steps += 1
        end

        # Calculate average loss
        avg_loss = epoch_loss / steps
        bits_per_dim = avg_loss / n_dimension / log(2)

        # Store statistics
        push!(stats["train_losses"], avg_loss)
        push!(stats["bits_per_dim"], bits_per_dim)

        # Update best model if needed
        if avg_loss < minimum(stats["best_loss"])
            stats["best_loss"] = avg_loss
            best_model = deepcopy(model)

            # Save checkpoint if path is provided
            !isnothing(checkpoint_path) && serialize(checkpoint_path, best_model)
        end

        # Print progress
        verbose && println("Epoch $epoch: loss = $(round(avg_loss, digits=4)), bits_per_dim = $(round(bits_per_dim, digits=4))")
    end

    # Use the best model found during training
    model = best_model

    # Finetune latents if requested
    if finetune_latents
        verbose && println("Finetuning latents for $finetune_epochs epochs...")
        zᵢ = device(randn(Float32, encoder_dim, n_components))
        zᵢ, finetune_stats = finetune_mixture_latents(model, zᵢ, data_device; batchsize, max_epochs=finetune_epochs, learning_rate, verbose)
        # Add finetuning stats to main stats
        stats["finetune_losses"] = finetune_stats["losses"]
        stats["finetune_bits_per_dim"] = finetune_stats["bits_per_dim"]
    else
        # Just create some latents but don't optimize them
        zᵢ = device(randn(Float32, encoder_dim, n_components))
    end

    verbose && println("Training complete!")
    return model, zᵢ, stats
end

"""
    create_model(input_dim, n_categories, hidden_dims, activation, encoder_dim)

Create a mixture model with the specified architecture.

# Arguments

## Model configuration
- `n_categories::Int=2`: Number of categories for each dimension
- `n_components::Int=2^10`: Number of mixture components
- `hidden_dims::Union{NTuple{<:Any,Int}, Vector{Int}}=(128, 256, 512)`: Dimensions of hidden layers
- `activation=leakyrelu`: Activation function for hidden layers
- `encoder_dim::Int=64`: Dimension of the latent space for centers

"""
function create_model(input_dim::Int, n_categories::Int=2, hidden_dims::Union{Vector{Int},NTuple{<:Any,Int}}=(128, 256, 512), activation=leakyrelu, encoder_dim::Int=64)
    layers = []
    # First layer
    push!(layers, Dense(encoder_dim, hidden_dims[1], activation))

    # Hidden layers
    for i in 1:(length(hidden_dims)-1)
        push!(layers, Dense(hidden_dims[i], hidden_dims[i+1], activation))
    end

    # Output layer
    push!(layers, Dense(hidden_dims[end], n_categories * input_dim, activation))

    # Final model
    ffnn = Chain(
        layers...,
        Base.Fix2(reshape, (n_categories, input_dim, :)),
        x -> logsoftmax(x, dims=1)
    )
    return ffnn
end

"""
    preprocess_data(data, n_categories)

    Thest that the data are in the proper format.
"""
function preprocess_data(data::AbstractArray{Bool}, n_categories::Int)
    n_categories != 2 && error("with binary data, use `n_categories` = 2")
    return (Int32.(data) .+ 1)
end

function preprocess_data(data::AbstractArray{<:Integer}, n_categories::Int)
    if all(∈((0, 1)), data)
        n_categories != 2 && error("with binary data, use `n_categories` = 2")
        return (Int32.(data) .+ 1)
    end

    # If the data is already categorical (1 to n_categories)
    if all(x -> 1 <= x <= n_categories, data)
        return (Int32.(data))
    end

    # Otherwise, discretize the data
    error("Unsupported data format. Please convert your data to integers in range 1:n_categories")
end

"""
    finetune_mixture_latents(model, zᵢ, data; kwargs...)

Finetune the latents `zᵢ` of the mixture model which define the components
"""
function finetune_mixture_latents(model, zᵢ, data; batchsize::Int=256, max_epochs::Int=10, learning_rate=1.0f-3, verbose=true)
    optimizer = Optimisers.Adam(learning_rate)
    opt_state = Optimisers.setup(optimizer, zᵢ)

    stats = Dict("losses" => Float32[], "bits_per_dim" => Float32[])
    n_dimension = size(data, 1)
    for epoch in 1:max_epochs
        epoch_loss, steps = 0.0, 0

        # Train on batches
        for batch in DataLoader(data; batchsize=batchsize)
            # Compute loss and gradients
            fᵢ, grads = Flux.Zygote.withgradient(zᵢ -> -sumlogsumexp_logprob_fused(model(zᵢ), batch), zᵢ)

            # Update zᵢ
            opt_state, zᵢ = Optimisers.update(opt_state, zᵢ, grads[1])
            epoch_loss += fᵢ
            steps += 1
        end

        # Calculate average loss
        avg_loss = epoch_loss / steps
        bits_per_dim = avg_loss / n_dimension / log(2)

        # Store statistics
        push!(stats["losses"], avg_loss)
        push!(stats["bits_per_dim"], bits_per_dim)

        # Print progress
        verbose && println("Finetune Epoch $epoch: loss = $(round(avg_loss, digits=4)), bits_per_dim = $(round(bits_per_dim, digits=4))")
    end

    return zᵢ, stats
end
