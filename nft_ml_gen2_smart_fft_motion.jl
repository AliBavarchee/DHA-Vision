#                                                                      ==/\/\/\/\/\/\/\/\/\/\/\/\==
#                                                                          Digital__Hana__Arts®
#                                                                      ==/\/\/\/\/\/\/\/\/\/\/\/\==
# =================================================
#  nft_ml_gen2.jl  –  NFT styles + PCA + GAN
#  Language: Julia 1.9.4
# 
#  Company: Digital__Hana__Arts®
#  Author:  Ali Deragschan, Ali Bavarcci
# 
#  License: Proprietary – All rights reserved.
# ==================================================
# ==================================================
# nft_ml_gen2_public_demo.jl
# Julia 1.9.4
# ==================================================
# ==================================================
# PUBLIC DEMO VERSION
#
# This file illustrates the structure of the DHA‑Vision v3 pipeline. The actual
# mathematical‑art generators, duotone extraction, FFT enhancement, fluid
# motion warp, morphisms, composites, and most classical styles are NOT included.
#
#
# Cold Open includes:
#   • 3 classical NFT styles (cyberpunk, vaporwave, glitch)
#   • PCA latent‑space variations
#   • DCGAN training / retraining logic
#   • Input‑change detection (smart retrain)
#   • Directory handling and safe image saving
#
# Full Version has:
#   • 7 classical styles (pixel, comic, oil, hologram, gold, pop, abstract)
#   • All image‑conditioned mathematical generators
#   • IFS, Markov texture, random walk
#   • Chirikov, Hénon, Logistic map generators
#   • Duotone extraction, FFT enhancement, motion warp
#   • Domain_warp_gray, morph_gray, postmodern_composite_gray
# =============================================================================

using Pkg
# Pkg.add(["Images", "ImageIO", "FileIO", "ImageFiltering",
#          "ImageTransformations", "ImageEdgeDetection", "Colors",
#          "Statistics", "LinearAlgebra", "Random", "MultivariateStats",
#          "Flux", "BSON", "Zygote"])

using Images, FileIO, Colors, ImageFiltering, ImageTransformations, ImageEdgeDetection
using Statistics, LinearAlgebra, Random, MultivariateStats, Flux, BSON, Zygote


# 0. Configuration

const INPUT_DIR       = "input"
const OUTPUT_DIR      = "output"
const ML_OUTPUT       = "ml_output"
const MODEL_DIR       = "saved_models"
const FRACT_OUTPUT    = "fract_output"

const IMG_SIZE        = (128, 128)
const FRACT_SIZE      = (1024, 1024)
const UPSCALE         = true

const PCA_COMPONENTS  = 50
const PCA_VARIANTS    = 3
const PCA_NOISE_SCALE = 0.30

const LATENT_DIM      = 100
const GAN_EPOCHS      = 30
const BATCH_SIZE      = 8
const LR_G            = 0.0002
const LR_D            = 0.0002
const N_GAN_IMAGES    = 10

# Constants kept for reference (unused by stubs)
const FRACTAL_ITERATIONS = 80
const JULIA_ESCAPE       = 16.0
const FRACTAL_SEED       = 20260807
const IMG_C_STRENGTH     = 0.38
const IMG_Z_STRENGTH     = 0.22
const EDGE_STRENGTH      = 0.35
const CHAOS_STRENGTH     = 0.18
const CHIRIKOV_N_POINTS  = 300_000
const HENON_N_POINTS     = 250_000
const LOGISTIC_RES       = 1024
const LOGISTIC_ITERS     = 1500
const FFT_BOOST          = 1.8
const FFT_NOISE_AMP      = 0.05
const MOTION_STRENGTH    = 0.12


# 1. Directories

for d in (OUTPUT_DIR, ML_OUTPUT, MODEL_DIR, FRACT_OUTPUT)
    isdir(d) || mkpath(d)
end

# 2. Utilities

function image_files(dir)
    isdir(dir) || return String[]
    filter(f -> occursin(r"\.(jpg|jpeg|png|bmp|tif|tiff)$"i, f), readdir(dir))
end

function safe_save(path, img)
    rgb = RGB.(img)
    clamped = map(rgb) do c
        RGB(
            clamp(Float64(red(c)),   0.0, 1.0),
            clamp(Float64(green(c)), 0.0, 1.0),
            clamp(Float64(blue(c)),  0.0, 1.0)
        )
    end
    save(path, RGB{N0f8}.(clamped))
end

function luminance(img)
    Float64.(Gray.(img))
end

function image_hash_seed(img; base_seed=FRACTAL_SEED)
    small = imresize(RGB.(img), (32, 32))
    s = 0.0
    k = 1
    for p in small
        s += (0.299*red(p) + 0.587*green(p) + 0.114*blue(p)) * k
        k += 1
    end
    Int(mod(abs(round(Int, s * 1_000_000)) + base_seed, typemax(Int)))
end

# 3. Input‑change detection

function input_hash()
    files = sort(image_files(INPUT_DIR))
    if isempty(files)
        return ""
    end
    s = IOBuffer()
    for f in files
        path = joinpath(INPUT_DIR, f)
        sz = filesize(path)
        mt = mtime(path)
        write(s, f, sz, mt)
    end
    string(hash(take!(s)))
end

const HASH_FILE = joinpath(MODEL_DIR, "input_hash.txt")

function training_needed()
    current = input_hash()
    if !isfile(HASH_FILE)
        return true, current
    end
    previous = read(HASH_FILE, String)
    if current == previous
        return false, previous
    else
        return true, current
    end
end

# 4. Classical NFT styles

# --- Demo styles ---

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
        clamp.(red.(img) .+ 0.25, 0, 1),
        clamp.(green.(img) .* 0.8, 0, 1),
        clamp.(blue.(img) .+ 0.35, 0, 1)
    )
end

function glitch(img)
    out = copy(img)
    h, w = size(out)
    Random.seed!(1234)
    if h > 40
        for _ in 1:40
            row = rand(20:(h - 20))
            shift = rand(-50:50)
            out[row:end, :] = circshift(out[row:end, :], (0, shift))
        end
    end
    out
end

# --- Stubs for the remaining styles (return black image) ---

function pixel(img)
    h, w = size(img)
    fill(RGB(0.0,0.0,0.0), h, w)  # proprietary pixel style omitted
end

function comic(img)
    h, w = size(img)
    fill(RGB(0.0,0.0,0.0), h, w)  # proprietary comic style omitted
end

function oil(img)
    h, w = size(img)
    fill(RGB(0.0,0.0,0.0), h, w)  # proprietary oil style omitted
end

function hologram(img)
    h, w = size(img)
    fill(RGB(0.0,0.0,0.0), h, w)  # proprietary hologram style omitted
end

function gold(img)
    h, w = size(img)
    fill(RGB(0.0,0.0,0.0), h, w)  # proprietary gold style omitted
end

function pop(img)
    h, w = size(img)
    fill(RGB(0.0,0.0,0.0), h, w)  # proprietary pop style omitted
end

function abstract(img)
    h, w = size(img)
    fill(RGB(0.0,0.0,0.0), h, w)  # proprietary abstract style omitted
end

function save_style(img, name, style)
    safe_save(joinpath(OUTPUT_DIR, "$(name)_$(style).png"), img)
end

function process_image(file)
    println("Processing $file")
    img = RGB.(load(joinpath(INPUT_DIR, file)))
    name = splitext(file)[1]
    save_style(cyberpunk(img), name, "cyberpunk")
    save_style(vaporwave(img), name, "vaporwave")
    save_style(glitch(img),    name, "glitch")
    # The remaining styles will be saved as black images
    save_style(pixel(img),    name, "pixel")
    save_style(comic(img),    name, "comic")
    save_style(oil(img),      name, "oil")
    save_style(hologram(img), name, "hologram")
    save_style(gold(img),     name, "gold")
    save_style(pop(img),      name, "pop")
    save_style(abstract(img), name, "abstract")
end

# 5. Generate classical styles

style_files = image_files(INPUT_DIR)
if !isempty(style_files)
    println("\n===== Generating classical NFT styles (3 demo + 7 stubs) =====")
    for file in style_files
        process_image(file)
    end
else
    println("No input images found – skipping classical styles.")
end


# 6. Determine if PCA / GAN training needed

need_train, new_hash = training_needed()
if need_train
    println("\nInput images changed → retraining PCA & GAN.")
else
    println("\nInput images unchanged → loading saved PCA & GAN models.")
end

# 7. PCA (train or load) – fully functional

function get_output_size()
    files = image_files(INPUT_DIR)
    max_w, max_h = 0, 0
    for f in files
        try
            img = load(joinpath(INPUT_DIR, f))
            h, w = size(img)[1:2]
            max_w = max(max_w, w)
            max_h = max(max_h, h)
        catch e
            println("WARNING: could not inspect $f: $e")
        end
    end
    max_w == 0 ? (1024, 1024) : (max_h, max_w)
end

const OUTPUT_SIZE = get_output_size()

function load_image_robust(path, sz=IMG_SIZE)
    img = RGB.(load(path))
    img = imresize(img, sz)
    Float32.(channelview(img))
end

flatten_rgb(arr) = vec(arr)

if need_train
    println("\nLoading images for PCA...")
    images_data = Vector{Float32}[]
    for f in style_files
        try
            arr = load_image_robust(joinpath(INPUT_DIR, f))
            push!(images_data, flatten_rgb(arr))
        catch e
            println("WARNING: skipping $f ($e)")
        end
    end
    output_files = image_files(OUTPUT_DIR)
    for f in output_files
        try
            arr = load_image_robust(joinpath(OUTPUT_DIR, f))
            push!(images_data, flatten_rgb(arr))
        catch e
            println("WARNING: skipping $f ($e)")
        end
    end
    if isempty(images_data)
        error("No usable images for PCA. Put images into input/.")
    end
    X = hcat(images_data...)'
    n_features = size(X,2)
    n_samples = size(X,1)
    pca_dim = min(PCA_COMPONENTS, n_samples, n_features)
    mean_vec = mean(X, dims=1)
    Xc = X .- mean_vec
    model_pca = fit(PCA, Xc'; maxoutdim=pca_dim)
    latent_all = transform(model_pca, Xc')'
    n_input = length(style_files)
    latent_input = n_input > 0 ? latent_all[1:n_input, :] : zeros(Float64,0,size(latent_all,2))
    global_std = vec(std(latent_all, dims=1))
    global_std[global_std .== 0] .= 1e-6
    explained = principalvars(model_pca) ./ tvar(model_pca) .* 100
    BSON.bson(joinpath(MODEL_DIR, "pca_model.bson"),
              Dict("model"=>model_pca, "mean_vec"=>vec(mean_vec),
                   "img_size"=>IMG_SIZE, "explained_var"=>explained))
    println("PCA model saved.")

    variants_per_photo = n_input > 0 ? PCA_VARIANTS : 0
    if n_input > 0
        println("Generating $variants_per_photo PCA variations per input...")
        Random.seed!(42)
        for (idx, fname) in enumerate(style_files)
            base = splitext(fname)[1]
            base_latent = vec(latent_input[idx,:])
            for v in 1:variants_per_photo
                noise = Float64(PCA_NOISE_SCALE) .* global_std .* randn(length(base_latent))
                new_latent = base_latent + noise
                new_feat = reconstruct(model_pca, new_latent) .+ vec(mean_vec)
                arr = reshape(Float32.(new_feat), 3, IMG_SIZE[1], IMG_SIZE[2])
                arr = clamp.(arr, 0f0, 1f0)
                img = colorview(RGB, arr)
                if UPSCALE && OUTPUT_SIZE != IMG_SIZE
                    img = imresize(img, OUTPUT_SIZE)
                end
                outpath = joinpath(ML_OUTPUT, "ml_pca_$(base)_$(v).png")
                safe_save(outpath, img)
                println("Saved $outpath")
            end
        end
    end
else
    if !isfile(joinpath(MODEL_DIR, "pca_model.bson"))
        error("No saved PCA model found – retrain required.")
    end
    pca_dict = BSON.load(joinpath(MODEL_DIR, "pca_model.bson"))
    model_pca = pca_dict[:model]
    mean_vec = pca_dict[:mean_vec]
    println("PCA model loaded.")
end

# 8. DCGAN (train or load)

if need_train
    println("\n===== Preparing GAN =====")
    images_data = Vector{Float32}[]
    for f in style_files
        push!(images_data, flatten_rgb(load_image_robust(joinpath(INPUT_DIR, f))))
    end
    for f in output_files
        push!(images_data, flatten_rgb(load_image_robust(joinpath(OUTPUT_DIR, f))))
    end
    gan_data = Array{Float32}(undef, 3, IMG_SIZE[1], IMG_SIZE[2], length(images_data))
    for (i, vec) in enumerate(images_data)
        gan_data[:,:,:,i] = reshape(vec, 3, IMG_SIZE[1], IMG_SIZE[2])
    end
    gan_data = 2.0f0 .* gan_data .- 1.0f0
    gan_data = permutedims(gan_data, (3,2,1,4))

    function create_generator(latent_dim)
        Chain(
            Dense(latent_dim, 8*8*256, relu; init=Flux.glorot_uniform),
            x -> reshape(x, 8, 8, 256, :),
            ConvTranspose((4,4), 256=>128, relu; stride=2, pad=1, init=Flux.glorot_uniform),
            BatchNorm(128),
            ConvTranspose((4,4), 128=>64, relu; stride=2, pad=1, init=Flux.glorot_uniform),
            BatchNorm(64),
            ConvTranspose((4,4), 64=>32, relu; stride=2, pad=1, init=Flux.glorot_uniform),
            BatchNorm(32),
            ConvTranspose((4,4), 32=>3, tanh; stride=2, pad=1, init=Flux.glorot_uniform)
        )
    end

    function create_discriminator()
        myleaky(x) = leakyrelu(x, 0.2f0)
        Chain(
            Conv((4,4), 3=>32, myleaky; stride=2, pad=1, init=Flux.glorot_uniform),
            Dropout(0.3),
            Conv((4,4), 32=>64, myleaky; stride=2, pad=1, init=Flux.glorot_uniform),
            BatchNorm(64), Dropout(0.3),
            Conv((4,4), 64=>128, myleaky; stride=2, pad=1, init=Flux.glorot_uniform),
            BatchNorm(128), Dropout(0.3),
            Conv((4,4), 128=>256, myleaky; stride=2, pad=1, init=Flux.glorot_uniform),
            BatchNorm(256), Dropout(0.3),
            Conv((4,4), 256=>512, myleaky; stride=2, pad=1, init=Flux.glorot_uniform),
            BatchNorm(512), Dropout(0.3),
            Flux.flatten,
            Dense(512*4*4, 1, sigmoid)
        )
    end

    generator = create_generator(LATENT_DIM)
    discriminator = create_discriminator()
    opt_g = Adam(LR_G, (0.5, 0.999))
    opt_d = Adam(LR_D, (0.5, 0.999))

    function loss_d(x_real, x_fake)
        -mean(log.(discriminator(x_real) .+ 1e-8)) -
         mean(log.(1 .- discriminator(x_fake) .+ 1e-8))
    end

    function loss_g(x_fake)
        -mean(log.(discriminator(x_fake) .+ 1e-8))
    end

    function train_gan!(gen, disc, data, epochs, batch_size, opt_g, opt_d)
        params_g = Flux.params(gen)
        params_d = Flux.params(disc)
        n_samples = size(data, 4)
        steps_per_epoch = max(1, cld(n_samples, batch_size))
        for epoch in 1:epochs
            perm = randperm(n_samples)
            data_shuffled = data[:,:,:,perm]
            for i in 1:steps_per_epoch
                idx = (i-1)*batch_size+1 : min(i*batch_size, n_samples)
                real = data_shuffled[:,:,:,idx]
                cur = length(idx)
                noise = randn(Float32, LATENT_DIM, cur)
                fake = gen(noise)
                grad_d = gradient(params_d) do
                    loss_d(real, Zygote.dropgrad(fake))
                end
                Flux.update!(opt_d, params_d, grad_d)
                noise_g = randn(Float32, LATENT_DIM, cur)
                grad_g = gradient(params_g) do
                    loss_g(gen(noise_g))
                end
                Flux.update!(opt_g, params_g, grad_g)
            end
            if epoch % 5 == 0 || epoch == epochs
                println("Epoch $epoch/$epochs")
            end
        end
    end

    println("Training GAN...")
    Random.seed!(12345)
    @time train_gan!(generator, discriminator, gan_data, GAN_EPOCHS, BATCH_SIZE, opt_g, opt_d)
    BSON.bson(joinpath(MODEL_DIR, "gan_generator.bson"), Dict("generator" => generator))
    BSON.bson(joinpath(MODEL_DIR, "gan_discriminator.bson"), Dict("discriminator" => discriminator))

    println("\nGenerating GAN images...")
    n_input = length(style_files)
    if n_input > 0
        gan_per_photo = max(1, ceil(Int, N_GAN_IMAGES / n_input))
        for (photo_idx, fname) in enumerate(style_files)
            base = splitext(fname)[1]
            for v in 1:gan_per_photo
                Random.seed!(photo_idx * 1000 + v)
                noise = randn(Float32, LATENT_DIM, 1)
                gen_img = generator(noise)
                arr = (gen_img[:,:,:,1] .+ 1.0f0) ./ 2.0f0
                arr = permutedims(arr, (3,1,2))
                arr = clamp.(arr, 0f0,1f0)
                img = colorview(RGB, arr)
                if UPSCALE && OUTPUT_SIZE != IMG_SIZE
                    img = imresize(img, OUTPUT_SIZE)
                end
                safe_save(joinpath(ML_OUTPUT, "ml_gan_$(base)_$(v).png"), img)
            end
        end
    end
    write(HASH_FILE, new_hash)
else
    if isfile(joinpath(MODEL_DIR, "gan_generator.bson"))
        gen_dict = BSON.load(joinpath(MODEL_DIR, "gan_generator.bson"))
        generator = gen_dict[:generator]
    end
    println("GAN model loaded. Skipping GAN training and generation.")
end

# 9. PROPRIETARY STUBS

function extract_duotone(img)
    # Proprietary k‑means duotone extraction omitted.
    RGB(0.0, 0.0, 0.0), RGB(1.0, 1.0, 1.0)
end

function duotone(gray_img, dark, light)
    # Proprietary duotone blend omitted.
    out = Array{RGB{Float64}}(undef, size(gray_img))
    for i in eachindex(gray_img)
        v = clamp(gray_img[i], 0.0, 1.0)
        out[i] = RGB(v, v, v)
    end
    out
end

function fft_enhance(img::Matrix{Float64}; boost=FFT_BOOST, noise_amp=FFT_NOISE_AMP)
    # Proprietary FFT texture enhancement omitted.
    img
end

function motion_warp(gray_img; strength=MOTION_STRENGTH, seed=0)
    # Proprietary fluid motion warp omitted.
    gray_img
end

# --- Mathematical generators (all return black images) ---

function image_julia(img; out_size=FRACT_SIZE, maxiter=FRACTAL_ITERATIONS, c0=-0.745+0.113im)
    zeros(Float64, out_size)
end

function image_burning_ship(img; out_size=FRACT_SIZE, maxiter=FRACTAL_ITERATIONS)
    zeros(Float64, out_size)
end

function image_orbit_trap(img; out_size=FRACT_SIZE, maxiter=FRACTAL_ITERATIONS)
    zeros(Float64, out_size)
end

function ifs_image_conditioned(img; n_iter=500_000, out_size=FRACT_SIZE)
    fill(RGB(0.0,0.0,0.0), out_size)
end

function markov_image(img; n_states=64)
    fill(RGB(0.0,0.0,0.0), FRACT_SIZE)
end

function randomwalk_image(img; n_steps=250_000, step_size=0.003)
    fill(RGB(0.0,0.0,0.0), FRACT_SIZE)
end

function image_chirikov(img; out_size=FRACT_SIZE, n_points=CHIRIKOV_N_POINTS)
    zeros(Float64, out_size)
end

function image_henon(img; out_size=FRACT_SIZE, n_points=HENON_N_POINTS)
    zeros(Float64, out_size)
end

function image_logistic_bifurcation(img; out_size=FRACT_SIZE, res=LOGISTIC_RES, iters=LOGISTIC_ITERS)
    zeros(Float64, out_size)
end

function domain_warp_gray(gray_img, field_img; strength=0.15)
    zeros(Float64, size(gray_img))
end

function morph_gray(img1, img2, alpha=0.5)
    zeros(Float64, size(img1))
end

function postmodern_composite_gray(gray_list, field_img; alpha=0.6)
    zeros(Float64, size(first(gray_list)))
end

# 10. Mathematical art loop – stub version

println("\n===== IMAGE-CONDITIONED MATHEMATICAL ART (demo stubs) =====")
ml_files = filter(f -> startswith(f, "ml_pca_"), image_files(ML_OUTPUT))

if isempty(ml_files)
    println("No PCA images found in $ML_OUTPUT – skipping.")
else
    for f in ml_files
        println("\nGenerating stub outputs for $f")
        img = RGB.(load(joinpath(ML_OUTPUT, f)))
        base = splitext(f)[1]
        img_proc = imresize(img, (128,128))

        dark, light = extract_duotone(img_proc)

        function process_gray(gray)
            warped = motion_warp(gray)
            enhanced = fft_enhance(warped)
            duotone(enhanced, dark, light)
        end

        function process_rgb_to_duo(rgb_img)
            g = Float64.(Gray.(rgb_img))
            process_gray(g)
        end

        # Base generators
        safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_julia.png"),        process_gray(image_julia(img_proc)))
        safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_burning_ship.png"), process_gray(image_burning_ship(img_proc)))
        safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_orbit_trap.png"),   process_gray(image_orbit_trap(img_proc)))
        safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_ifs.png"),          process_rgb_to_duo(ifs_image_conditioned(img_proc)))
        safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_markov.png"),       process_rgb_to_duo(markov_image(img_proc)))
        safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_randomwalk.png"),   process_rgb_to_duo(randomwalk_image(img_proc)))
        safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_chirikov.png"),     process_gray(image_chirikov(img_proc)))
        safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_henon.png"),        process_gray(image_henon(img_proc)))
        safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_logistic.png"),     process_gray(image_logistic_bifurcation(img_proc)))

        # Morphisms (black)
        for step in 1:4
            a = (step-1)/3
            safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_morph_js_$(step).png"),
                      process_gray(morph_gray(image_julia(img_proc), image_burning_ship(img_proc), a)))
            safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_morph_jc_$(step).png"),
                      process_gray(morph_gray(image_julia(img_proc), image_chirikov(img_proc), a)))
            safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_morph_ch_$(step).png"),
                      process_gray(morph_gray(image_chirikov(img_proc), image_henon(img_proc), a)))
        end

        # Domain warps
        safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_julia_warped.png"),
                  process_gray(domain_warp_gray(image_julia(img_proc), img_proc)))
        safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_orbit_warped.png"),
                  process_gray(domain_warp_gray(image_orbit_trap(img_proc), img_proc)))
        safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_chirikov_warped.png"),
                  process_gray(domain_warp_gray(image_chirikov(img_proc), img_proc)))

        # Composites
        comp1 = postmodern_composite_gray([image_julia(img_proc), Float64.(Gray.(ifs_image_conditioned(img_proc))), image_chirikov(img_proc)], img_proc)
        safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_composite_jic.png"), process_gray(comp1))
        comp2 = postmodern_composite_gray([image_burning_ship(img_proc), image_henon(img_proc), image_logistic_bifurcation(img_proc)], img_proc)
        safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_composite_shl.png"), process_gray(comp2))

        println("  ✓ Completed $base (stub)")
    end
end

println("""
===============================================================================
GENERATIVE ART PIPELINE (demo) COMPLETE
===============================================================================
Input images:              $INPUT_DIR
Classical NFT styles:      $OUTPUT_DIR  (3 demo + 7 stubs)
PCA / GAN images:          $ML_OUTPUT
Saved models:              $MODEL_DIR
Post‑modern art (stubs):   $FRACT_OUTPUT
===============================================================================
""")