#                                                                      ==/\/\/\/\/\/\/\/\/\/\/\/\==
#                                                                          Digital__Hana__Arts®
#                                                                      ==/\/\/\/\/\/\/\/\/\/\/\/\==
#__________________________________________________________
# nft_ml_gen2.jl  –  NFT styles + PCA + GAN (Julia 1.9.4)
# 
#  License: Proprietary – All rights reserved.
#__________________________________________________________
#==/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\====
# Digital__Hana__Arts®
#==\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/====


using Pkg
# Uncomment to install missing packages:
# Pkg.add(["Images", "ImageIO", "FileIO", "ImageFiltering",
#          "ImageContrastAdjustment", "ImageTransformations",
#          "ImageEdgeDetection", "Colors", "Statistics",
#          "LinearAlgebra", "Random", "MultivariateStats",
#          "Flux", "BSON", "Zygote"])

using Images
using FileIO
using Colors
using ImageFiltering
using ImageTransformations
using ImageEdgeDetection
using Statistics
using LinearAlgebra
using Random
using MultivariateStats
using Flux
using BSON
using Zygote

const INPUT_DIR   = "input"
const OUTPUT_DIR  = "output"
const ML_OUTPUT   = "ml_output"
const MODEL_DIR   = "saved_models"

isdir(OUTPUT_DIR) || mkdir(OUTPUT_DIR)
isdir(ML_OUTPUT)  || mkdir(ML_OUTPUT)
isdir(MODEL_DIR)  || mkdir(MODEL_DIR)

# ===========================================================================
# 1. Safe image saving (clamp to [0,1], convert to N0f8 per channel)
# ===========================================================================
function safe_save(path, img)
    img_rgb = RGB.(img)
    img_clamped = map(c -> RGB(
        clamp(red(c), 0.0, 1.0),
        clamp(green(c), 0.0, 1.0),
        clamp(blue(c), 0.0, 1.0)
    ), img_rgb)
    img_8bit = RGB{N0f8}.(
        N0f8.(red.(img_clamped)),
        N0f8.(green.(img_clamped)),
        N0f8.(blue.(img_clamped))
    )
    save(path, img_8bit)
end

# ===========================================================================
# 2. NFT style functions (unchanged)
# ===========================================================================
function clamp01(img)
    clamp.(img, 0, 1)
end

function cyberpunk(img)
    img = RGB.(img)
    R = red.(img).^0.7
    G = green.(img).^1.4 .* 0.6
    B = blue.(img).^0.6 .* 1.3
    out = RGB.(clamp.(R,0,1), clamp.(G,0,1), clamp.(B,0,1))
    edges = detect_edges(Gray.(img), Canny())
    for i in eachindex(out)
        if edges[i] > 0.2
            out[i] = RGB(0,1,1)
        end
    end
    out
end

function vaporwave(img)
    RGB.(
        clamp.(red.(img).+0.25,0,1),
        clamp.(green.(img).*0.8,0,1),
        clamp.(blue.(img).+0.35,0,1)
    )
end

function glitch(img)
    out = copy(img)
    h,w = size(out)
    Random.seed!(1234)
    for k in 1:40
        row = rand(20:h-20)
        shift = rand(-50:50)
        out[row:end,:] = circshift(out[row:end,:],(0,shift))
    end
    out
end

function pixel(img)
    h,w = size(img)
    small = imresize(img,(div(h,12),div(w,12)))
    imresize(small,(h,w))
end

function comic(img)
    edges = detect_edges(Gray.(img),Canny())
    blur = imfilter(img,Kernel.gaussian(1.5))
    out = copy(blur)
    for i in eachindex(edges)
        if edges[i]>0.2
            out[i]=RGB(0,0,0)
        end
    end
    out
end

function oil(img)
    imfilter(img,Kernel.gaussian(5))
end

function hologram(img)
    RGB.(
        red.(img).*0.5,
        green.(img).*1.4,
        blue.(img).*1.5
    )
end

function gold(img)
    g = Gray.(img)
    RGB.(
        clamp.(g .*1.4 .+0.3,0,1),
        clamp.(g .*1.1 .+0.2,0,1),
        clamp.(g .*0.2,0,1)
    )
end

function pop(img)
    quant(x)=round(x*4)/4
    RGB.(
        quant.(red.(img)),
        quant.(green.(img)),
        quant.(blue.(img))
    )
end

function abstract(img)
    blur = imfilter(img,Kernel.gaussian(8))
    RGB.(
        sin.(red.(blur).*π).^2,
        cos.(green.(blur).*π).^2,
        sin.(blue.(blur).*2π).^2
    )
end

function save_style(img, name, style)
    outfile = joinpath(OUTPUT_DIR, "$(name)_$(style).png")
    safe_save(outfile, img)
end

function process_image(file)
    println("Processing ", file)
    img = load(joinpath(INPUT_DIR, file))
    name = splitext(file)[1]
    save_style(cyberpunk(img), name, "cyberpunk")
    save_style(vaporwave(img), name, "vaporwave")
    save_style(glitch(img),    name, "glitch")
    save_style(pixel(img),     name, "pixel")
    save_style(comic(img),     name, "comic")
    save_style(oil(img),       name, "oil")
    save_style(hologram(img),  name, "hologram")
    save_style(gold(img),      name, "gold")
    save_style(pop(img),       name, "pop")
    save_style(abstract(img),  name, "abstract")
end

# ===========================================================================
# 3. Generate NFT styles
# ===========================================================================
style_files = filter(f -> occursin(r"\.(jpg|jpeg|png|bmp|tif|tiff)$"i, f), readdir(INPUT_DIR))
if !isempty(style_files)
    println("\n--- Generating NFT styles ---")
    for file in style_files
        process_image(file)
    end
    println("Styles saved to ", OUTPUT_DIR)
else
    println("No input images found – skipping style generation.")
end

# ===========================================================================
# 4. PCA + per‑photo variations (models saved)
# ===========================================================================
const IMG_SIZE   = (128, 128)
const UPSCALE    = true

function get_output_size()
    files = filter(f -> occursin(r"\.(jpg|jpeg|png|bmp|tif|tiff)$"i, f), readdir(INPUT_DIR))
    max_w, max_h = 0, 0
    for f in files
        img = load(joinpath(INPUT_DIR, f))
        h, w = size(img)[1:2]
        max_w = max(max_w, w)
        max_h = max(max_h, h)
    end
    if max_w == 0
        return (1024, 1024)
    end
    return (max_h, max_w)
end

const OUTPUT_SIZE = get_output_size()
println("Upscaling ML outputs to: $OUTPUT_SIZE")

function load_image_robust(path, sz = IMG_SIZE)
    img = load(path)
    img = RGB.(img)
    img = imresize(img, sz)
    return Float32.(channelview(img))
end

flatten_rgb(arr) = vec(arr)

println("\nLoading images for PCA...")
images_data = Vector{Float32}[]
image_names = String[]

for f in style_files
    full = joinpath(INPUT_DIR, f)
    try
        arr = load_image_robust(full)
        push!(images_data, flatten_rgb(arr))
        push!(image_names, "input/$f")
    catch e
        println("WARNING: skipping corrupt input image $f ($(e))")
    end
end

output_files = filter(f -> occursin(r"\.(jpg|jpeg|png|bmp|tif|tiff)$"i, f), readdir(OUTPUT_DIR))
for f in output_files
    full = joinpath(OUTPUT_DIR, f)
    try
        arr = load_image_robust(full)
        push!(images_data, flatten_rgb(arr))
        push!(image_names, "output/$f")
    catch e
        println("WARNING: skipping corrupt output image $f ($(e))")
    end
end

if isempty(images_data)
    error("No usable images found – aborting.")
end

X = hcat(images_data...)'
n_features = size(X, 2)
println("Loaded $(size(X,1)) valid images, each with $n_features features.")

# PCA
mean_vec = mean(X, dims=1)
Xc = X .- mean_vec
model_pca = fit(PCA, Xc'; maxoutdim=50)
println("PCA fitted. Latent dims: $(size(model_pca.proj,2))")

latent_all = transform(model_pca, Xc')'
n_input = length(style_files)
latent_input = latent_all[1:n_input, :]
global_std = vec(std(latent_all, dims=1))

# Save PCA model
BSON.bson(joinpath(MODEL_DIR, "pca_model.bson"),
          Dict("model" => model_pca,
               "mean_vec" => vec(mean_vec),
               "img_size" => IMG_SIZE,
               "explained_var" => principalvars(model_pca) ./ tvar(model_pca) * 100))
println("PCA model saved to saved_models/pca_model.bson")

# Generate per‑photo PCA variations
variants_per_photo = max(2, ceil(Int, 10 / n_input))
println("Generating $variants_per_photo PCA variations per input photo...")
Random.seed!(42)
for (idx, fname) in enumerate(style_files)
    base_name = splitext(fname)[1]
    base_latent = vec(latent_input[idx, :])
    for v in 1:variants_per_photo
        noise = 0.3 * global_std .* randn(Float32, length(base_latent))
        new_latent = base_latent + noise
        new_features = reconstruct(model_pca, new_latent) .+ mean_vec'
        img_vec = new_features[:]
        img_arr = reshape(img_vec, 3, IMG_SIZE[1], IMG_SIZE[2])
        img_arr = clamp.(img_arr, 0.0f0, 1.0f0)
        img = colorview(RGB, img_arr)
        if UPSCALE && OUTPUT_SIZE != IMG_SIZE
            img = imresize(img, OUTPUT_SIZE)
        end
        outpath = joinpath(ML_OUTPUT, "ml_$(base_name)_$(v).png")
        safe_save(outpath, img)
        println("Saved $outpath")
    end
end

if n_input == 0
    println("No input images – generating 10 global PCA synthetic images.")
    latent_mean = vec(mean(latent_all, dims=1))
    global_std = vec(std(latent_all, dims=1))
    novel_latent = randn(Float32, size(latent_all,2), 10) .* global_std .+ latent_mean
    novel_features = reconstruct(model_pca, novel_latent) .+ mean_vec'
    for i in 1:10
        img_vec = novel_features[:, i]
        img_arr = reshape(img_vec, 3, IMG_SIZE[1], IMG_SIZE[2])
        img_arr = clamp.(img_arr, 0.0f0, 1.0f0)
        img = colorview(RGB, img_arr)
        if UPSCALE && OUTPUT_SIZE != IMG_SIZE
            img = imresize(img, OUTPUT_SIZE)
        end
        outpath = joinpath(ML_OUTPUT, "ml_synthetic_$(i).png")
        safe_save(outpath, img)
        println("Saved $outpath")
    end
end

# ===========================================================================
# 5. GAN training (DCGAN) – corrected for CPU, Flux layout
# ===========================================================================
println("\n===== TRAINING GAN (CPU, this may take several minutes) =====")

const LATENT_DIM = 100
const GAN_EPOCHS = 30
const BATCH_SIZE = 8
const LR_G = 0.0002
const LR_D = 0.0002

# Prepare GAN dataset
gan_data = Array{Float32}(undef, 3, IMG_SIZE[1], IMG_SIZE[2], length(images_data))
for (i, vec) in enumerate(images_data)
    gan_data[:,:,:,i] = reshape(vec, 3, IMG_SIZE[1], IMG_SIZE[2])
end
gan_data = 2.0f0 .* gan_data .- 1.0f0
# Flux expects (W, H, C, N) → permute from (C, H, W, N)
gan_data = permutedims(gan_data, (3, 2, 1, 4))   # (128,128,3,N)

# Generator
function create_generator(latent_dim)
    return Chain(
        Dense(latent_dim, 8*8*256, relu; init=Flux.glorot_uniform),
        x -> reshape(x, 8, 8, 256, :),
        # 8→16
        ConvTranspose((4,4), 256=>128, relu; stride=2, pad=1, init=Flux.glorot_uniform),
        BatchNorm(128),
        # 16→32
        ConvTranspose((4,4), 128=>64, relu; stride=2, pad=1, init=Flux.glorot_uniform),
        BatchNorm(64),
        # 32→64
        ConvTranspose((4,4), 64=>32, relu; stride=2, pad=1, init=Flux.glorot_uniform),
        BatchNorm(32),
        # 64→128
        ConvTranspose((4,4), 32=>3, tanh; stride=2, pad=1, init=Flux.glorot_uniform),
    )
end

# Discriminator – fixed leakyrelu via anonymous function
function create_discriminator()
    myleaky = x -> leakyrelu(x, 0.2f0)
    return Chain(
        # 128→64
        Conv((4,4), 3=>32, myleaky; stride=2, pad=1, init=Flux.glorot_uniform),
        Dropout(0.3),
        # 64→32
        Conv((4,4), 32=>64, myleaky; stride=2, pad=1, init=Flux.glorot_uniform),
        BatchNorm(64),
        Dropout(0.3),
        # 32→16
        Conv((4,4), 64=>128, myleaky; stride=2, pad=1, init=Flux.glorot_uniform),
        BatchNorm(128),
        Dropout(0.3),
        # 16→8
        Conv((4,4), 128=>256, myleaky; stride=2, pad=1, init=Flux.glorot_uniform),
        BatchNorm(256),
        Dropout(0.3),
        # 8→4
        Conv((4,4), 256=>512, myleaky; stride=2, pad=1, init=Flux.glorot_uniform),
        BatchNorm(512),
        Dropout(0.3),
        Flux.flatten,
        Dense(512*4*4, 1, sigmoid)
    )
end

generator = create_generator(LATENT_DIM)
discriminator = create_discriminator()

opt_g = Adam(LR_G, (0.5, 0.999))
opt_d = Adam(LR_D, (0.5, 0.999))

loss_d(x_real, x_fake) = -mean(log.(discriminator(x_real) .+ 1e-8)) -
                          mean(log.(1 .- discriminator(x_fake) .+ 1e-8))
loss_g(x_fake) = -mean(log.(discriminator(x_fake) .+ 1e-8))

function train_gan!(gen, disc, data, epochs, batch_size, opt_g, opt_d)
    params_g = Flux.params(gen)
    params_d = Flux.params(disc)
    n_samples = size(data, 4)
    steps_per_epoch = max(1, div(n_samples, batch_size))

    for epoch in 1:epochs
        perm = randperm(n_samples)
        data_shuffled = data[:,:,:,perm]

        for i in 1:steps_per_epoch
            idx = (i-1)*batch_size+1 : min(i*batch_size, n_samples)
            real_batch = data_shuffled[:,:,:,idx]

            noise = randn(Float32, LATENT_DIM, length(idx))
            fake_batch = gen(noise)

            # Discriminator update
            grad_d = gradient(params_d) do
                loss_d(real_batch, fake_batch)
            end
            Flux.update!(opt_d, params_d, grad_d)

            # Generator update
            noise = randn(Float32, LATENT_DIM, batch_size)
            grad_g = gradient(params_g) do
                loss_g(gen(noise))
            end
            Flux.update!(opt_g, params_g, grad_g)
        end

        if epoch % 5 == 0 || epoch == epochs
            noise_fixed = randn(Float32, LATENT_DIM, 1)
            fake_img = gen(noise_fixed)
            println("Epoch $epoch/$epochs  – mean fake pixel $(mean(fake_img))")
        end
    end
end

println("Starting GAN training...")
@time train_gan!(generator, discriminator, gan_data, GAN_EPOCHS, BATCH_SIZE, opt_g, opt_d)

# Save models
BSON.bson(joinpath(MODEL_DIR, "gan_generator.bson"), Dict("generator" => generator))
BSON.bson(joinpath(MODEL_DIR, "gan_discriminator.bson"), Dict("discriminator" => discriminator))
println("GAN models saved to saved_models/")

# ===========================================================================
# 6. GAN‑based novel images – named after input photos
# ===========================================================================
const N_GAN_IMAGES = 10

println("\nGenerating novel images with GAN (named after input photos)...")

if n_input > 0
    gan_per_photo = ceil(Int, N_GAN_IMAGES / n_input)
    total_gan = n_input * gan_per_photo
    println("Generating $gan_per_photo GAN images per input photo ($total_gan total)")

    for (photo_idx, fname) in enumerate(style_files)
        base_name = splitext(fname)[1]
        for v in 1:gan_per_photo
            Random.seed!(photo_idx * 1000 + v)
            single_noise = randn(Float32, LATENT_DIM, 1)
            single_img = generator(single_noise)          # (128,128,3,1)
            img_arr = (single_img[:,:,:,1] .+ 1.0f0) ./ 2.0f0
            img_arr = permutedims(img_arr, (3, 1, 2))    # (3,128,128) for colorview
            img_arr = clamp.(img_arr, 0.0f0, 1.0f0)
            img = colorview(RGB, img_arr)
            if UPSCALE && OUTPUT_SIZE != IMG_SIZE
                img = imresize(img, OUTPUT_SIZE)
            end
            outpath = joinpath(ML_OUTPUT, "ml_gan_$(base_name)_$(v).png")
            safe_save(outpath, img)
            println("Saved $outpath")
        end
    end
else
    println("No input images – generating global GAN images.")
    for i in 1:N_GAN_IMAGES
        Random.seed!(i)
        single_noise = randn(Float32, LATENT_DIM, 1)
        single_img = generator(single_noise)
        img_arr = (single_img[:,:,:,1] .+ 1.0f0) ./ 2.0f0
        img_arr = permutedims(img_arr, (3, 1, 2))
        img_arr = clamp.(img_arr, 0.0f0, 1.0f0)
        img = colorview(RGB, img_arr)
        if UPSCALE && OUTPUT_SIZE != IMG_SIZE
            img = imresize(img, OUTPUT_SIZE)
        end
        outpath = joinpath(ML_OUTPUT, "ml_gan_$(i).png")
        safe_save(outpath, img)
        println("Saved $outpath")
    end
end

println("\n===== All done! =====")
println("Output folders:")
println("  - NFT styles  : $OUTPUT_DIR")
println("  - ML images   : $ML_OUTPUT")
println("  - Saved models: $MODEL_DIR")
