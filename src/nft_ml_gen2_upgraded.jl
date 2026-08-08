# =============================================================================
# nft_ml_gen2_upgraded.jl
# Julia 1.9.4
#
# Image-conditioned NFT / generative-art pipeline
#
# Pipeline:
#   input/
#      │
#      ├── classic image styles
#      ├── PCA variations
#      ├── DCGAN variations
#      │
#      └── image-conditioned mathematical art
#             ├── Julia fractal
#             ├── Burning Ship fractal
#             ├── Orbit-trap fractal
#             ├── image-conditioned IFS
#             ├── Markov texture
#             └── gradient random walk
#
# The upgraded fractal engine uses the source image as a mathematical
# parameter field rather than only as a colour lookup.
#
# Tested design target:
#   Julia 1.9.4
#
# =============================================================================

using Pkg

# Uncomment once if packages are missing:
# Pkg.add([
#     "Images", "ImageIO", "FileIO", "ImageFiltering",
#     "ImageTransformations", "ImageEdgeDetection", "Colors",
#     "Statistics", "LinearAlgebra", "Random", "MultivariateStats",
#     "Flux", "BSON", "Zygote"
# ])

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

# =============================================================================
# 0. Configuration
# =============================================================================

const INPUT_DIR       = "input"
const OUTPUT_DIR      = "output"
const ML_OUTPUT       = "ml_output"
const MODEL_DIR       = "saved_models"
const FRACT_OUTPUT    = "fract_output"

const IMG_SIZE        = (128, 128)
const FRACT_SIZE      = (512, 512)
const UPSCALE         = true

# PCA
const PCA_COMPONENTS  = 50
const PCA_VARIANTS    = 3
const PCA_NOISE_SCALE = 0.30

# GAN
const LATENT_DIM      = 100
const GAN_EPOCHS      = 30
const BATCH_SIZE      = 8
const LR_G            = 0.0002
const LR_D            = 0.0002
const N_GAN_IMAGES    = 10

# Image-conditioned fractal engine
const FRACTAL_ITERATIONS = 180
const JULIA_ESCAPE       = 16.0
const FRACTAL_SEED       = 20260807

# Strength of image → mathematical parameter coupling.
const IMG_C_STRENGTH     = 0.38
const IMG_Z_STRENGTH     = 0.22
const EDGE_STRENGTH      = 0.35
const CHAOS_STRENGTH     = 0.18

# =============================================================================
# 1. Directories
# =============================================================================

for d in (OUTPUT_DIR, ML_OUTPUT, MODEL_DIR, FRACT_OUTPUT)
    isdir(d) || mkpath(d)
end

# =============================================================================
# 2. Utilities
# =============================================================================

function image_files(dir)
    isdir(dir) || return String[]
    filter(
        f -> occursin(r"\.(jpg|jpeg|png|bmp|tif|tiff)$"i, f),
        readdir(dir)
    )
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

clamp01(x) = clamp.(x, 0, 1)

function luminance(img)
    Float64.(Gray.(img))
end

function image_hash_seed(img; base_seed=FRACTAL_SEED)
    # Deterministic seed derived from low-resolution image statistics.
    small = imresize(RGB.(img), (32, 32))

    s = 0.0
    k = 1
    for p in small
        s += (
            0.299 * Float64(red(p)) +
            0.587 * Float64(green(p)) +
            0.114 * Float64(blue(p))
        ) * k
        k += 1
    end

    return Int(mod(abs(round(Int, s * 1_000_000)) + base_seed, typemax(Int)))
end

function smooth01(x)
    x = clamp(x, 0.0, 1.0)
    x * x * (3.0 - 2.0 * x)
end

# =============================================================================
# 3. Classical NFT styles
# =============================================================================

function cyberpunk(img)
    img = RGB.(img)
    R = red.(img).^0.7
    G = green.(img).^1.4 .* 0.6
    B = blue.(img).^0.6 .* 1.3

    out = RGB.(clamp.(R,0,1), clamp.(G,0,1), clamp.(B,0,1))
    edges = detect_edges(Gray.(img), Canny())

    for i in eachindex(out)
        if edges[i] > 0.2
            out[i] = RGB(0, 1, 1)
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

function pixel(img)
    h, w = size(img)
    small = imresize(img, (max(1, div(h, 12)), max(1, div(w, 12))))
    imresize(small, (h, w))
end

function comic(img)
    edges = detect_edges(Gray.(img), Canny())
    blur = imfilter(img, Kernel.gaussian(1.5))
    out = copy(blur)

    for i in eachindex(edges)
        if edges[i] > 0.2
            out[i] = RGB(0, 0, 0)
        end
    end

    out
end

oil(img) = imfilter(img, Kernel.gaussian(5))

function hologram(img)
    RGB.(
        clamp.(red.(img) .* 0.5, 0, 1),
        clamp.(green.(img) .* 1.4, 0, 1),
        clamp.(blue.(img) .* 1.5, 0, 1)
    )
end

function gold(img)
    g = Gray.(img)
    RGB.(
        clamp.(g .* 1.4 .+ 0.3, 0, 1),
        clamp.(g .* 1.1 .+ 0.2, 0, 1),
        clamp.(g .* 0.2, 0, 1)
    )
end

function pop(img)
    quant(x) = round(x * 4) / 4
    RGB.(
        quant.(red.(img)),
        quant.(green.(img)),
        quant.(blue.(img))
    )
end

function abstract(img)
    blur = imfilter(img, Kernel.gaussian(8))
    RGB.(
        sin.(red.(blur) .* π).^2,
        cos.(green.(blur) .* π).^2,
        sin.(blue.(blur) .* 2π).^2
    )
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
    save_style(pixel(img),    name, "pixel")
    save_style(comic(img),    name, "comic")
    save_style(oil(img),      name, "oil")
    save_style(hologram(img), name, "hologram")
    save_style(gold(img),     name, "gold")
    save_style(pop(img),      name, "pop")
    save_style(abstract(img), name, "abstract")
end

# =============================================================================
# 4. Classical styles
# =============================================================================

style_files = image_files(INPUT_DIR)

if !isempty(style_files)
    println("\n===== Generating classical NFT styles =====")
    for file in style_files
        process_image(file)
    end
else
    println("No input images found – skipping classical styles.")
end

# =============================================================================
# 5. PCA variations
# =============================================================================

function get_output_size()
    files = image_files(INPUT_DIR)

    max_w = 0
    max_h = 0

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

println("\n===== Loading images for PCA =====")

images_data = Vector{Float32}[]
image_names = String[]

for f in style_files
    try
        arr = load_image_robust(joinpath(INPUT_DIR, f))
        push!(images_data, flatten_rgb(arr))
        push!(image_names, "input/$f")
    catch e
        println("WARNING: skipping input image $f ($e)")
    end
end

output_files = image_files(OUTPUT_DIR)

for f in output_files
    try
        arr = load_image_robust(joinpath(OUTPUT_DIR, f))
        push!(images_data, flatten_rgb(arr))
        push!(image_names, "output/$f")
    catch e
        println("WARNING: skipping output image $f ($e)")
    end
end

if isempty(images_data)
    error("No usable images found. Put at least one image into input/.")
end

X = hcat(images_data...)'
n_features = size(X, 2)
n_samples = size(X, 1)

println("Loaded $n_samples images, each with $n_features features.")

# PCA cannot have more components than observations.
pca_dim = min(PCA_COMPONENTS, n_samples, n_features)

mean_vec = mean(X, dims=1)
Xc = X .- mean_vec

model_pca = fit(PCA, Xc'; maxoutdim=pca_dim)
latent_all = transform(model_pca, Xc')'

println("PCA fitted. Latent dimensions: $(size(model_pca.proj, 2))")

n_input = length(style_files)
latent_input = n_input > 0 ? latent_all[1:n_input, :] : zeros(Float64, 0, size(latent_all,2))

global_std = vec(std(latent_all, dims=1))
global_std[global_std .== 0] .= 1e-6

explained = principalvars(model_pca) ./ tvar(model_pca) .* 100

BSON.bson(
    joinpath(MODEL_DIR, "pca_model.bson"),
    Dict(
        "model" => model_pca,
        "mean_vec" => vec(mean_vec),
        "img_size" => IMG_SIZE,
        "explained_var" => explained
    )
)

println("PCA model saved.")

variants_per_photo = n_input > 0 ? PCA_VARIANTS : 0

if n_input > 0
    println("Generating $variants_per_photo PCA variations per input image...")
    Random.seed!(42)

    for (idx, fname) in enumerate(style_files)
        base_name = splitext(fname)[1]
        base_latent = vec(latent_input[idx, :])

        for v in 1:variants_per_photo
            noise = Float64(PCA_NOISE_SCALE) .* global_std .* randn(length(base_latent))
            new_latent = base_latent + noise

            new_features = reconstruct(model_pca, new_latent) .+ vec(mean_vec)
            img_arr = reshape(Float32.(new_features), 3, IMG_SIZE[1], IMG_SIZE[2])
            img_arr = clamp.(img_arr, 0.0f0, 1.0f0)

            img = colorview(RGB, img_arr)

            if UPSCALE && OUTPUT_SIZE != IMG_SIZE
                img = imresize(img, OUTPUT_SIZE)
            end

            outpath = joinpath(ML_OUTPUT, "ml_pca_$(base_name)_$(v).png")
            safe_save(outpath, img)
            println("Saved $outpath")
        end
    end
end

# =============================================================================
# 6. DCGAN
# =============================================================================

println("\n===== Preparing GAN =====")

gan_data = Array{Float32}(undef, 3, IMG_SIZE[1], IMG_SIZE[2], n_samples)

for (i, vec) in enumerate(images_data)
    gan_data[:, :, :, i] = reshape(vec, 3, IMG_SIZE[1], IMG_SIZE[2])
end

gan_data = 2.0f0 .* gan_data .- 1.0f0
gan_data = permutedims(gan_data, (3, 2, 1, 4))

function create_generator(latent_dim)
    Chain(
        Dense(latent_dim, 8 * 8 * 256, relu; init=Flux.glorot_uniform),
        x -> reshape(x, 8, 8, 256, :),
        ConvTranspose((4,4), 256=>128, relu;
            stride=2, pad=1, init=Flux.glorot_uniform),
        BatchNorm(128),
        ConvTranspose((4,4), 128=>64, relu;
            stride=2, pad=1, init=Flux.glorot_uniform),
        BatchNorm(64),
        ConvTranspose((4,4), 64=>32, relu;
            stride=2, pad=1, init=Flux.glorot_uniform),
        BatchNorm(32),
        ConvTranspose((4,4), 32=>3, tanh;
            stride=2, pad=1, init=Flux.glorot_uniform)
    )
end

function create_discriminator()
    myleaky(x) = leakyrelu(x, 0.2f0)

    Chain(
        Conv((4,4), 3=>32, myleaky;
            stride=2, pad=1, init=Flux.glorot_uniform),
        Dropout(0.3),

        Conv((4,4), 32=>64, myleaky;
            stride=2, pad=1, init=Flux.glorot_uniform),
        BatchNorm(64),
        Dropout(0.3),

        Conv((4,4), 64=>128, myleaky;
            stride=2, pad=1, init=Flux.glorot_uniform),
        BatchNorm(128),
        Dropout(0.3),

        Conv((4,4), 128=>256, myleaky;
            stride=2, pad=1, init=Flux.glorot_uniform),
        BatchNorm(256),
        Dropout(0.3),

        Conv((4,4), 256=>512, myleaky;
            stride=2, pad=1, init=Flux.glorot_uniform),
        BatchNorm(512),
        Dropout(0.3),

        Flux.flatten,
        Dense(512 * 4 * 4, 1, sigmoid)
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
        data_shuffled = data[:, :, :, perm]

        for i in 1:steps_per_epoch
            idx = (i-1)*batch_size + 1 : min(i*batch_size, n_samples)
            real_batch = data_shuffled[:, :, :, idx]

            current_batch = length(idx)

            noise = randn(Float32, LATENT_DIM, current_batch)
            fake_batch = gen(noise)

            grad_d = gradient(params_d) do
                loss_d(real_batch, fake_batch)
            end

            Flux.update!(opt_d, params_d, grad_d)

            noise_g = randn(Float32, LATENT_DIM, current_batch)

            grad_g = gradient(params_g) do
                loss_g(gen(noise_g))
            end

            Flux.update!(opt_g, params_g, grad_g)
        end

        if epoch % 5 == 0 || epoch == epochs
            fixed_noise = randn(Float32, LATENT_DIM, 1)
            fake_img = gen(fixed_noise)
            println("Epoch $epoch/$epochs – mean fake pixel = $(mean(fake_img))")
        end
    end
end

println("===== Training GAN =====")
Random.seed!(12345)
@time train_gan!(
    generator,
    discriminator,
    gan_data,
    GAN_EPOCHS,
    BATCH_SIZE,
    opt_g,
    opt_d
)

BSON.bson(
    joinpath(MODEL_DIR, "gan_generator.bson"),
    Dict("generator" => generator)
)

BSON.bson(
    joinpath(MODEL_DIR, "gan_discriminator.bson"),
    Dict("discriminator" => discriminator)
)

println("GAN models saved.")

# =============================================================================
# 7. GAN image generation
# =============================================================================

println("\n===== Generating GAN images =====")

if n_input > 0
    gan_per_photo = max(1, ceil(Int, N_GAN_IMAGES / n_input))

    for (photo_idx, fname) in enumerate(style_files)
        base_name = splitext(fname)[1]

        for v in 1:gan_per_photo
            Random.seed!(photo_idx * 1000 + v)

            noise = randn(Float32, LATENT_DIM, 1)
            generated = generator(noise)

            arr = (generated[:, :, :, 1] .+ 1.0f0) ./ 2.0f0
            arr = permutedims(arr, (3, 1, 2))
            arr = clamp.(arr, 0.0f0, 1.0f0)

            img = colorview(RGB, arr)

            if UPSCALE && OUTPUT_SIZE != IMG_SIZE
                img = imresize(img, OUTPUT_SIZE)
            end

            outpath = joinpath(
                ML_OUTPUT,
                "ml_gan_$(base_name)_$(v).png"
            )

            safe_save(outpath, img)
            println("Saved $outpath")
        end
    end
else
    for i in 1:N_GAN_IMAGES
        Random.seed!(i)

        noise = randn(Float32, LATENT_DIM, 1)
        generated = generator(noise)

        arr = (generated[:, :, :, 1] .+ 1.0f0) ./ 2.0f0
        arr = permutedims(arr, (3, 1, 2))
        arr = clamp.(arr, 0.0f0, 1.0f0)

        img = colorview(RGB, arr)

        if UPSCALE && OUTPUT_SIZE != IMG_SIZE
            img = imresize(img, OUTPUT_SIZE)
        end

        safe_save(joinpath(ML_OUTPUT, "ml_gan_$i.png"), img)
    end
end

# =============================================================================
# 8. Image-conditioned fractal engine
# =============================================================================

"""
Build a low-frequency image parameter field.

Returns:
    gray      luminance field
    edge      normalized edge field
    contrast  local-contrast field
"""
function build_image_fields(img)
    img = RGB.(img)

    gray = Float64.(Gray.(img))
    blur = imfilter(gray, Kernel.gaussian(3))

    contrast = abs.(gray .- blur)
    cmax = maximum(contrast)
    contrast = cmax > 0 ? contrast ./ cmax : zeros(size(contrast))

    edges = detect_edges(Gray.(img), Canny())
    edge = Float64.(edges)
    emax = maximum(edge)
    edge = emax > 0 ? edge ./ emax : zeros(size(edge))

    gray, edge, contrast
end

function sample_field(field, x, y)
    h, w = size(field)

    ix = clamp(round(Int, x * (w - 1)) + 1, 1, w)
    iy = clamp(round(Int, y * (h - 1)) + 1, 1, h)

    Float64(field[iy, ix])
end

"""
Smooth artistic palette.

The palette is deliberately continuous and restrained rather than using
random colours at every pixel.
"""
function palette_map(t, r, g, b)
    t = clamp(t, 0.0, 1.0)

    # Image-derived hue.
    hue = mod(
        0.58 * t +
        0.27 * r +
        0.15 * b,
        1.0
    )

    saturation = clamp(
        0.38 +
        0.42 * t +
        0.18 * g,
        0.0,
        1.0
    )

    value = clamp(
        0.18 +
        0.82 * t,
        0.0,
        1.0
    )

    HSV(360 * hue, saturation, value)
end

"""
Convert escape iteration to a smooth continuous value.
"""
function smooth_escape(iter, zabs, maxiter)
    if iter >= maxiter
        return 0.0
    end

    log_zn = log(max(zabs, 1.000001))
    nu = log(max(log_zn / log(2), 1e-12)) / log(2)

    value = (iter + 1 - nu) / maxiter
    clamp(value, 0.0, 1.0)
end

"""
Image-conditioned Julia fractal.

The source image controls:
    - real component of c
    - imaginary component of c
    - initial z perturbation
    - iteration weighting
    - final colour

This is the main upgraded generator.
"""
function image_julia(img;
    out_size=FRACT_SIZE,
    maxiter=FRACTAL_ITERATIONS,
    c0=-0.745 + 0.113im
)
    src = imresize(RGB.(img), (128, 128))
    gray, edge, contrast = build_image_fields(src)

    oh, ow = out_size
    canvas = Array{RGB{Float64},2}(undef, oh, ow)

    rng = MersenneTwister(image_hash_seed(src))

    # Small image-dependent global perturbation.
    global_phase = 2π * rand(rng)

    for py in 1:oh
        y = (py - 1) / (oh - 1)

        # Complex plane.
        cy = 2.2 * (y - 0.5)

        for px in 1:ow
            x = (px - 1) / (ow - 1)
            cx = 2.2 * (x - 0.5)

            p = RGB(
                sample_field(red.(src), x, y),
                sample_field(green.(src), x, y),
                sample_field(blue.(src), x, y)
            )

            r = Float64(red(p))
            g = Float64(green(p))
            b = Float64(blue(p))

            L = sample_field(gray, x, y)
            E = sample_field(edge, x, y)
            C = sample_field(contrast, x, y)

            # Image-conditioned Julia parameter.
            c = c0 +
                IMG_C_STRENGTH * ((r - 0.5) + (g - 0.5)im) +
                EDGE_STRENGTH * E *
                (0.18 * cos(global_phase) + 0.18im * sin(global_phase))

            # Image-conditioned initial condition.
            z = complex(cx, cy)

            z += IMG_Z_STRENGTH *
                 complex(
                     (r - 0.5) * 0.35,
                     (b - 0.5) * 0.35
                 )

            escaped = false
            iter_used = maxiter
            zabs = abs(z)

            for n in 1:maxiter
                # Mild image-controlled nonlinear perturbation.
                phase = CHAOS_STRENGTH * E
                z = z^2 + c

                if phase > 0
                    z += phase * complex(
                        cos(n * 0.031 + L * 4π),
                        sin(n * 0.037 + C * 4π)
                    ) * 0.01
                end

                zabs = abs(z)

                if zabs > JULIA_ESCAPE
                    escaped = true
                    iter_used = n
                    break
                end
            end

            if escaped
                t = smooth_escape(iter_used, zabs, maxiter)

                # Edge field subtly sharpens the composition.
                t = clamp(
                    t * (0.82 + 0.30 * E) +
                    0.08 * C,
                    0.0,
                    1.0
                )

                canvas[py, px] = palette_map(t, r, g, b)
            else
                # Dark, clean interior instead of noisy random colour.
                canvas[py, px] = RGB(
                    0.015 + 0.08 * r,
                    0.018 + 0.08 * g,
                    0.025 + 0.10 * b
                )
            end
        end
    end

    canvas
end

"""
Image-conditioned Burning Ship fractal.

The absolute-value operation creates a sharper, architectural aesthetic.
"""
function image_burning_ship(img;
    out_size=FRACT_SIZE,
    maxiter=FRACTAL_ITERATIONS
)
    src = imresize(RGB.(img), (128, 128))
    gray, edge, contrast = build_image_fields(src)

    oh, ow = out_size
    canvas = Array{RGB{Float64},2}(undef, oh, ow)

    cbase = -0.52 + 0.01im

    for py in 1:oh
        y = (py - 1) / (oh - 1)
        cy = 2.0 * (y - 0.55)

        for px in 1:ow
            x = (px - 1) / (ow - 1)
            cx = 2.8 * (x - 0.52)

            r = sample_field(red.(src), x, y)
            g = sample_field(green.(src), x, y)
            b = sample_field(blue.(src), x, y)

            E = sample_field(edge, x, y)
            C = sample_field(contrast, x, y)

            c = cbase +
                IMG_C_STRENGTH * ((r - 0.5) + (b - 0.5)im)

            z = complex(cx, cy)

            escaped = false
            iter_used = maxiter
            zabs = abs(z)

            for n in 1:maxiter
                z = complex(abs(real(z)), abs(imag(z)))
                z = z^2 + c

                # Tiny image-dependent twist.
                z += 0.008 * E * complex(
                    sin(n * 0.021),
                    cos(n * 0.017)
                )

                zabs = abs(z)

                if zabs > JULIA_ESCAPE
                    escaped = true
                    iter_used = n
                    break
                end
            end

            if escaped
                t = smooth_escape(iter_used, zabs, maxiter)
                t = clamp(t + 0.12 * C + 0.10 * E, 0, 1)
                canvas[py, px] = palette_map(t, r, g, b)
            else
                canvas[py, px] = RGB(
                    0.01 + 0.04*r,
                    0.01 + 0.04*g,
                    0.015 + 0.05*b
                )
            end
        end
    end

    canvas
end

"""
Orbit-trap Julia generator.

Instead of relying only on escape time, it measures the closest approach
of the orbit to a circle and a pair of axes. This creates clean modern
structures suitable for abstract NFT artwork.
"""
function image_orbit_trap(img;
    out_size=FRACT_SIZE,
    maxiter=FRACTAL_ITERATIONS
)
    src = imresize(RGB.(img), (128, 128))
    gray, edge, contrast = build_image_fields(src)

    oh, ow = out_size
    canvas = Array{RGB{Float64},2}(undef, oh, ow)

    for py in 1:oh
        y = (py - 1) / (oh - 1)
        cy = 2.4 * (y - 0.5)

        for px in 1:ow
            x = (px - 1) / (ow - 1)
            cx = 2.4 * (x - 0.5)

            r = sample_field(red.(src), x, y)
            g = sample_field(green.(src), x, y)
            b = sample_field(blue.(src), x, y)

            E = sample_field(edge, x, y)
            C = sample_field(contrast, x, y)
            L = sample_field(gray, x, y)

            c = (-0.72 + 0.19im) +
                0.28 * IMG_C_STRENGTH *
                ((r - 0.5) + (g - 0.5)im)

            z = complex(cx, cy)

            min_circle = Inf
            min_axis = Inf

            escaped = false

            for n in 1:maxiter
                z = z^2 + c

                radius = abs(z)
                min_circle = min(min_circle, abs(radius - (0.55 + 0.3*L)))

                min_axis = min(
                    min_axis,
                    min(abs(real(z)), abs(imag(z)))
                )

                if radius > JULIA_ESCAPE
                    escaped = true
                    break
                end
            end

            trap = exp(-7.0 * min_circle) * 0.72 +
                   exp(-22.0 * min_axis) * 0.28

            trap = clamp(trap, 0.0, 1.0)

            # Image controls the final contrast.
            t = clamp(
                trap^(0.65 + 0.35*(1-L)) *
                (0.72 + 0.45*E) +
                0.08*C,
                0,
                1
            )

            canvas[py, px] = palette_map(t, r, g, b)
        end
    end

    canvas
end

# =============================================================================
# 9. Image-conditioned IFS / chaos game
# =============================================================================

function weighted_transform(probs, rng)
    r = rand(rng)
    cumulative = 0.0

    for i in eachindex(probs)
        cumulative += probs[i]
        if r <= cumulative
            return i
        end
    end

    length(probs)
end

"""
Image-conditioned Barnsley fern.

Unlike the original version, the source image modifies transformation
probabilities and colour sampling.
"""
function ifs_image_conditioned(img;
    n_iter=500_000,
    out_size=FRACT_SIZE
)
    src = imresize(RGB.(img), (128, 128))
    gray, edge, contrast = build_image_fields(src)

    transforms = [
        (0.0,   0.0,   0.0,  0.16, 0.0, 0.0),
        (0.85,  0.04, -0.04, 0.85, 0.0, 1.6),
        (0.20, -0.26, 0.23,  0.22, 0.0, 1.6),
        (-0.15, 0.28, 0.26, 0.24, 0.0, 0.44)
    ]

    base_probs = [0.01, 0.85, 0.07, 0.07]

    xmin, xmax = -2.1820, 2.6558
    ymin, ymax = 0.0, 9.9983

    oh, ow = out_size
    canvas = fill(RGB(0.008, 0.01, 0.018), oh, ow)

    rng = MersenneTwister(image_hash_seed(src, base_seed=FRACTAL_SEED + 91))

    x = 0.0
    y = 0.0

    for _ in 1:n_iter
        u0 = clamp((x - xmin) / (xmax - xmin), 0, 1)
        v0 = clamp((y - ymin) / (ymax - ymin), 0, 1)

        L = sample_field(gray, u0, v0)
        E = sample_field(edge, u0, v0)

        probs = copy(base_probs)

        # Brightness and edge strength influence branch selection.
        probs[1] *= 0.55 + 1.3*(1-L)
        probs[2] *= 0.70 + 1.0*L
        probs[3] *= 0.75 + 1.5*E
        probs[4] *= 1.05 + 1.2*(1-E)

        probs ./= sum(probs)

        k = weighted_transform(probs, rng)

        a,b,c,d,e,f = transforms[k]

        x_new = a*x + b*y + e
        y_new = c*x + d*y + f

        x, y = x_new, y_new

        u = (x - xmin) / (xmax - xmin)
        v = (y - ymin) / (ymax - ymin)

        if 0 <= u <= 1 && 0 <= v <= 1
            ix = clamp(round(Int, u*(ow-1))+1, 1, ow)
            iy = clamp(round(Int, (1-v)*(oh-1))+1, 1, oh)

            r = sample_field(red.(src), u, v)
            g = sample_field(green.(src), u, v)
            bcol = sample_field(blue.(src), u, v)

            alpha = 0.25 + 0.75*E

            old = canvas[iy, ix]

            newcol = RGB(
                clamp((1-alpha)*red(old)   + alpha*r,    0, 1),
                clamp((1-alpha)*green(old) + alpha*g,    0, 1),
                clamp((1-alpha)*blue(old)  + alpha*bcol, 0, 1)
            )

            canvas[iy, ix] = newcol
        end
    end

    canvas
end

# =============================================================================
# 10. Markov texture
# =============================================================================

function sample_categorical(probs, rng=Random.default_rng())
    r = rand(rng)
    cumulative = 0.0

    for (i, p) in enumerate(probs)
        cumulative += p
        if r <= cumulative
            return i
        end
    end

    length(probs)
end

function markov_image(img; n_states=64)
    src = RGB.(img)

    n_per_channel = max(2, round(Int, cbrt(n_states)))
    n_states = n_per_channel^3

    h, w = size(src)
    indices = zeros(Int, h, w)

    for i in 1:h, j in 1:w
        r = red(src[i,j])
        g = green(src[i,j])
        b = blue(src[i,j])

        ri = min(floor(Int, r*n_per_channel)+1, n_per_channel)
        gi = min(floor(Int, g*n_per_channel)+1, n_per_channel)
        bi = min(floor(Int, b*n_per_channel)+1, n_per_channel)

        indices[i,j] =
            (ri-1)*n_per_channel^2 +
            (gi-1)*n_per_channel +
            bi
    end

    trans_count = zeros(Int, n_states, n_states)

    for i in 1:h, j in 1:w
        cur = indices[i,j]

        if j < w
            nxt = indices[i,j+1]
        elseif i < h
            nxt = indices[i+1,1]
        else
            continue
        end

        trans_count[cur,nxt] += 1
    end

    trans_prob = zeros(Float64, n_states, n_states)

    for s in 1:n_states
        row_sum = sum(trans_count[s,:])

        if row_sum > 0
            trans_prob[s,:] = trans_count[s,:] ./ row_sum
        else
            trans_prob[s,:] .= 1/n_states
        end
    end

    rng = MersenneTwister(image_hash_seed(src, base_seed=7777))

    oh, ow = FRACT_SIZE
    seq_len = oh*ow

    state = rand(rng, 1:n_states)
    seq = Vector{Int}(undef, seq_len)
    seq[1] = state

    for k in 2:seq_len
        state = sample_categorical(trans_prob[state,:], rng)
        seq[k] = state
    end

    out = Array{RGB{Float64},2}(undef, oh, ow)

    for idx in eachindex(seq)
        s = seq[idx]

        ri = div(s-1, n_per_channel^2)+1
        rem = (s-1) % (n_per_channel^2)
        gi = div(rem, n_per_channel)+1
        bi = rem % n_per_channel + 1

        r = (ri-1)/(n_per_channel-1)
        g = (gi-1)/(n_per_channel-1)
        b = (bi-1)/(n_per_channel-1)

        i = div(idx-1, ow)+1
        j = mod(idx-1, ow)+1

        out[i,j] = RGB(r,g,b)
    end

    out
end

# =============================================================================
# 11. Image-guided random walk
# =============================================================================

function randomwalk_image(img;
    n_steps=250_000,
    step_size=0.003
)
    src = RGB.(img)

    gray = Gray.(src)
    grad_x, grad_y = imgradients(gray, KernelFactors.sobel)

    h_in, w_in = size(src)
    oh, ow = FRACT_SIZE

    canvas = fill(RGB(0.01,0.012,0.02), oh, ow)

    rng = MersenneTwister(image_hash_seed(src, base_seed=9911))

    x = rand(rng)
    y = rand(rng)

    for _ in 1:n_steps
        ix = clamp(round(Int, x*(w_in-1))+1, 1, w_in)
        iy = clamp(round(Int, y*(h_in-1))+1, 1, h_in)

        dx = Float64(grad_x[iy,ix])
        dy = Float64(grad_y[iy,ix])

        if abs(dx) + abs(dy) < 1e-8
            angle = 2π*rand(rng)
        else
            angle = atan(dy,dx) + 0.45*randn(rng)
        end

        # Image brightness modifies particle step length.
        brightness = Float64(gray[iy,ix])
        local_step = step_size * (0.45 + 1.35*brightness)

        x = mod(x + local_step*cos(angle), 1.0)
        y = mod(y + local_step*sin(angle), 1.0)

        ox = clamp(round(Int, x*(ow-1))+1, 1, ow)
        oy = clamp(round(Int, y*(oh-1))+1, 1, oh)

        col = src[iy,ix]
        old = canvas[oy,ox]

        α = 0.28 + 0.60*brightness

        canvas[oy,ox] = RGB(
            clamp((1-α)*red(old)   + α*red(col),   0,1),
            clamp((1-α)*green(old) + α*green(col), 0,1),
            clamp((1-α)*blue(old)  + α*blue(col),  0,1)
        )
    end

    canvas
end

# =============================================================================
# 12. Artistic post-processing
# =============================================================================

"""
Subtle modern finishing pass.

Keeps the mathematical structure intact while improving contrast and
smoothing harsh pixel-level transitions.
"""
function artistic_finish(img; contrast=1.15, gamma=0.92)
    x = RGB.(img)

    out = map(x) do p
        r = clamp(((Float64(red(p))   - 0.5) * contrast + 0.5), 0, 1)^gamma
        g = clamp(((Float64(green(p)) - 0.5) * contrast + 0.5), 0, 1)^gamma
        b = clamp(((Float64(blue(p))  - 0.5) * contrast + 0.5), 0, 1)^gamma

        RGB(r,g,b)
    end

    # Very light smoothing avoids harsh single-pixel artifacts.
    imfilter(out, Kernel.gaussian(0.35))
end

# =============================================================================
# 13. Generate mathematical art from ML images
# =============================================================================

println("\n===== IMAGE-CONDITIONED MATHEMATICAL ART =====")

ml_files = image_files(ML_OUTPUT)

if isempty(ml_files)
    println("No images found in $ML_OUTPUT – skipping mathematical art.")
else
    for f in ml_files
        println("\nGenerating mathematical art from $f")

        img = RGB.(load(joinpath(ML_OUTPUT, f)))
        base = splitext(f)[1]

        # Uniform working image.
        img_proc = imresize(img, (128,128))

        # ---------------------------------------------------------------------
        # Julia
        # ---------------------------------------------------------------------
        println("  → Julia")
        julia = image_julia(img_proc)
        julia = artistic_finish(julia)

        safe_save(
            joinpath(FRACT_OUTPUT, "fract_$(base)_julia.png"),
            julia
        )

        # ---------------------------------------------------------------------
        # Burning Ship
        # ---------------------------------------------------------------------
        println("  → Burning Ship")
        ship = image_burning_ship(img_proc)
        ship = artistic_finish(ship)

        safe_save(
            joinpath(FRACT_OUTPUT, "fract_$(base)_burning_ship.png"),
            ship
        )

        # ---------------------------------------------------------------------
        # Orbit trap
        # ---------------------------------------------------------------------
        println("  → Orbit Trap")
        orbit = image_orbit_trap(img_proc)
        orbit = artistic_finish(orbit)

        safe_save(
            joinpath(FRACT_OUTPUT, "fract_$(base)_orbit_trap.png"),
            orbit
        )

        # ---------------------------------------------------------------------
        # Image-conditioned IFS
        # ---------------------------------------------------------------------
        println("  → Image-conditioned IFS")
        ifsg = ifs_image_conditioned(img_proc)
        ifsg = artistic_finish(ifsg)

        safe_save(
            joinpath(FRACT_OUTPUT, "fract_$(base)_ifs.png"),
            ifsg
        )

        # ---------------------------------------------------------------------
        # Markov
        # ---------------------------------------------------------------------
        println("  → Markov texture")
        markov = markov_image(img_proc)
        markov = artistic_finish(markov, contrast=1.05, gamma=0.95)

        safe_save(
            joinpath(FRACT_OUTPUT, "fract_$(base)_markov.png"),
            markov
        )

        # ---------------------------------------------------------------------
        # Random walk
        # ---------------------------------------------------------------------
        println("  → Gradient random walk")
        walk = randomwalk_image(img_proc)
        walk = artistic_finish(walk, contrast=1.08, gamma=0.94)

        safe_save(
            joinpath(FRACT_OUTPUT, "fract_$(base)_randomwalk.png"),
            walk
        )

        println("  ✓ Completed $base")
    end

    println("\nMathematical art saved to $FRACT_OUTPUT")
end

# =============================================================================
# 14. Summary
# =============================================================================

println("""

===============================================================================
GENERATIVE ART PIPELINE COMPLETE
===============================================================================

Input images:
    $INPUT_DIR

Classical NFT styles:
    $OUTPUT_DIR

PCA / GAN images:
    $ML_OUTPUT

Saved models:
    $MODEL_DIR

Image-conditioned mathematical art:
    $FRACT_OUTPUT

Fractal generators:
    • Image-conditioned Julia
    • Image-conditioned Burning Ship
    • Image-conditioned Orbit Trap
    • Image-conditioned IFS / Chaos Game
    • Image-conditioned Markov texture
    • Gradient-guided random walk

The source image is used as:
    • fractal parameter field
    • initial-condition modulation
    • edge / contrast field
    • deterministic random seed
    • branch-selection probability
    • colour information

===============================================================================
""")
