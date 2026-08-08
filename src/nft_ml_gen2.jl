# =============================================================================
# nft_ml_gen2_postmodern_fft.jl
# Julia 1.9.4
#
# Post‑modern, high‑resolution mathematical art with duotone + FFT enhancement.
# =============================================================================

using Pkg
# Uncomment once if packages are missing:
# Pkg.add(["Images", "ImageIO", "FileIO", "ImageFiltering",
#          "ImageTransformations", "ImageEdgeDetection", "Colors",
#          "Statistics", "LinearAlgebra", "Random", "MultivariateStats",
#          "Flux", "BSON", "Zygote", "FFTW"])

using Images, FileIO, Colors, ImageFiltering, ImageTransformations, ImageEdgeDetection
using Statistics, LinearAlgebra, Random, MultivariateStats, Flux, BSON, Zygote
using FFTW

# =============================================================================
# 0. Configuration
# =============================================================================

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
    filter(f -> occursin(r"\.(jpg|jpeg|png|bmp|tif|tiff)$"i, f), readdir(dir))
end

function safe_save(path, img)
    rgb = RGB.(img)
    clamped = map(rgb) do c
        RGB(clamp(Float64(red(c)),0,1), clamp(Float64(green(c)),0,1), clamp(Float64(blue(c)),0,1))
    end
    save(path, RGB{N0f8}.(clamped))
end

clamp01(x) = clamp.(x, 0, 1)

function luminance(img)
    Float64.(Gray.(img))
end

function image_hash_seed(img; base_seed=FRACTAL_SEED)
    small = imresize(RGB.(img), (32,32))
    s = 0.0; k = 1
    for p in small
        s += (0.299*red(p) + 0.587*green(p) + 0.114*blue(p)) * k
        k += 1
    end
    Int(mod(abs(round(Int, s*1_000_000)) + base_seed, typemax(Int)))
end

# =============================================================================
# 3. Classical NFT styles (unchanged)
# =============================================================================
# (all 10 style functions – cyberpunk, vaporwave, glitch, pixel, comic, oil,
#  hologram, gold, pop, abstract – are identical to previous versions)
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
    max_w = 0; max_h = 0
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
    Dict("model" => model_pca, "mean_vec" => vec(mean_vec),
         "img_size" => IMG_SIZE, "explained_var" => explained)
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
        BatchNorm(64),
        Dropout(0.3),
        Conv((4,4), 64=>128, myleaky; stride=2, pad=1, init=Flux.glorot_uniform),
        BatchNorm(128),
        Dropout(0.3),
        Conv((4,4), 128=>256, myleaky; stride=2, pad=1, init=Flux.glorot_uniform),
        BatchNorm(256),
        Dropout(0.3),
        Conv((4,4), 256=>512, myleaky; stride=2, pad=1, init=Flux.glorot_uniform),
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
                loss_d(real_batch, Zygote.dropgrad(fake_batch))
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
@time train_gan!(generator, discriminator, gan_data, GAN_EPOCHS, BATCH_SIZE, opt_g, opt_d)
BSON.bson(joinpath(MODEL_DIR, "gan_generator.bson"), Dict("generator" => generator))
BSON.bson(joinpath(MODEL_DIR, "gan_discriminator.bson"), Dict("discriminator" => discriminator))
println("GAN models saved.")
# =============================================================================
# 7. GAN image generation (unchanged)
# =============================================================================
# ...

# =============================================================================
# 8. FFT enhancement
# =============================================================================

"""
Apply a Fourier‑domain contrast/texture boost to a grayscale image.
Returns a new Float64 matrix with enhanced edges and subtle texture.
"""
function fft_enhance(img::Matrix{Float64}; boost=1.8, noise_amp=0.05)
    # 2‑D FFT
    F = fft(img)
    mag = abs.(F)
    phase = angle.(F)

    h, w = size(img)
    # create a high‑pass radius map (normalised frequency)
    cy, cx = h÷2, w÷2
    max_freq = sqrt(cx^2 + cy^2)
    radius = [(sqrt((y-cy)^2 + (x-cx)^2) / max_freq) for y in 1:h, x in 1:w]

    # boost high frequencies (radius > 0.15), leave low frequencies as is
    mask = @. (radius > 0.15) * (boost - 1.0) + 1.0
    # add slight random modulation to phase for texture
    rng = MersenneTwister(123)
    phase_mod = phase .+ noise_amp .* randn(rng, h, w) .* radius

    new_F = mag .* mask .* exp.(1im .* phase_mod)
    out = real(ifft(new_F))

    # normalise to [0,1]
    mn, mx = minimum(out), maximum(out)
    if mx > mn
        out = (out .- mn) ./ (mx - mn)
    else
        out .= 0.5
    end
    clamp!(out, 0.0, 1.0)
    out
end

# =============================================================================
# 9. Duotone palette (corrected)
# =============================================================================

function extract_duotone(img)
    small = imresize(RGB.(img), (32,32))
    pixels = [RGB{Float64}(p) for p in small]
    Random.seed!(42)
    centroids = [pixels[rand(1:end)], pixels[rand(1:end)]]
    for _ in 1:10
        a1 = RGB{Float64}[]; a2 = RGB{Float64}[]
        for p in pixels
            d1 = abs(red(p)-red(centroids[1])) + abs(green(p)-green(centroids[1])) + abs(blue(p)-blue(centroids[1]))
            d2 = abs(red(p)-red(centroids[2])) + abs(green(p)-green(centroids[2])) + abs(blue(p)-blue(centroids[2]))
            if d1 < d2 push!(a1,p) else push!(a2,p) end
        end
        if !isempty(a1) centroids[1] = sum(a1)/length(a1) end
        if !isempty(a2) centroids[2] = sum(a2)/length(a2) end
    end
    c1,c2 = centroids[1], centroids[2]
    if luminance(c1) > luminance(c2)
        c1,c2 = c2,c1
    end
    RGB{Float64}(c1), RGB{Float64}(c2)
end

function duotone(gray_img, dark, light)
    h,w = size(gray_img)
    out = Array{RGB{Float64}}(undef, h,w)
    for i in 1:h, j in 1:w
        t = clamp(gray_img[i,j],0,1)
        t = t^0.75
        out[i,j] = RGB(dark.r*(1-t)+light.r*t, dark.g*(1-t)+light.g*t, dark.b*(1-t)+light.b*t)
    end
    out
end

function add_grain(img, intensity=0.015)
    noise = rand(Float64, size(img)) .* intensity
    clamp.(img .+ noise, 0,1)
end

# =============================================================================
# 10. Image-conditioned fractal generators (grayscale) – unchanged
# =============================================================================

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

function minimal_grayscale(iter, zabs, maxiter)
    if iter >= maxiter
        return 0.0
    end
    log_zn = log(max(zabs, 1.000001))
    nu = log(log_zn / log(2)) / log(2)
    val = (iter + 1 - nu) / maxiter
    val = clamp(val, 0.0, 1.0)
    val^0.6
end

function image_julia(img; out_size=FRACT_SIZE, maxiter=FRACTAL_ITERATIONS, c0=-0.745 + 0.113im)
    src = imresize(RGB.(img), (128, 128))
    gray, edge, contrast = build_image_fields(src)
    oh, ow = out_size
    canvas = Array{Float64}(undef, oh, ow)
    rng = MersenneTwister(image_hash_seed(src))
    global_phase = 2π * rand(rng)
    for py in 1:oh
        y = (py - 1) / (oh - 1)
        cy = 2.2 * (y - 0.5)
        for px in 1:ow
            x = (px - 1) / (ow - 1)
            cx = 2.2 * (x - 0.5)
            r = sample_field(red.(src), x, y)
            g = sample_field(green.(src), x, y)
            b = sample_field(blue.(src), x, y)
            L = sample_field(gray, x, y)
            E = sample_field(edge, x, y)
            C = sample_field(contrast, x, y)
            c = c0 + IMG_C_STRENGTH * ((r-0.5) + (g-0.5)im) +
                EDGE_STRENGTH * E * (0.18*cos(global_phase) + 0.18im*sin(global_phase))
            z = complex(cx, cy) + IMG_Z_STRENGTH * complex((r-0.5)*0.35, (b-0.5)*0.35)
            escaped = false; iter_used = maxiter; zabs = abs(z)
            for n in 1:maxiter
                phase = CHAOS_STRENGTH * E
                z = z^2 + c
                if phase > 0
                    z += phase * complex(cos(n*0.031 + L*4π), sin(n*0.037 + C*4π)) * 0.01
                end
                zabs = abs(z)
                if zabs > JULIA_ESCAPE
                    escaped = true; iter_used = n; break
                end
            end
            if escaped
                t = minimal_grayscale(iter_used, zabs, maxiter)
                t = clamp(t * (0.82 + 0.3*E) + 0.08*C, 0.0, 1.0)
                canvas[py, px] = t
            else
                canvas[py, px] = 0.02
            end
        end
    end
    canvas
end

function image_burning_ship(img; out_size=FRACT_SIZE, maxiter=FRACTAL_ITERATIONS)
    src = imresize(RGB.(img), (128, 128))
    gray, edge, contrast = build_image_fields(src)
    oh, ow = out_size
    canvas = Array{Float64}(undef, oh, ow)
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
            c = cbase + IMG_C_STRENGTH * ((r-0.5) + (b-0.5)im)
            z = complex(cx, cy)
            escaped = false; iter_used = maxiter; zabs = abs(z)
            for n in 1:maxiter
                z = complex(abs(real(z)), abs(imag(z)))
                z = z^2 + c
                z += 0.008 * E * complex(sin(n*0.021), cos(n*0.017))
                zabs = abs(z)
                if zabs > JULIA_ESCAPE
                    escaped = true; iter_used = n; break
                end
            end
            if escaped
                t = minimal_grayscale(iter_used, zabs, maxiter)
                t = clamp(t + 0.12*C + 0.1*E, 0, 1)
                canvas[py, px] = t
            else
                canvas[py, px] = 0.01
            end
        end
    end
    canvas
end

function image_orbit_trap(img; out_size=FRACT_SIZE, maxiter=FRACTAL_ITERATIONS)
    src = imresize(RGB.(img), (128, 128))
    gray, edge, contrast = build_image_fields(src)
    oh, ow = out_size
    canvas = Array{Float64}(undef, oh, ow)
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
            c = (-0.72 + 0.19im) + 0.28*IMG_C_STRENGTH*((r-0.5)+(g-0.5)im)
            z = complex(cx, cy)
            min_circle = Inf; min_axis = Inf
            for n in 1:maxiter
                z = z^2 + c
                radius = abs(z)
                min_circle = min(min_circle, abs(radius - (0.55 + 0.3*L)))
                min_axis = min(min_axis, min(abs(real(z)), abs(imag(z))))
                if radius > JULIA_ESCAPE break end
            end
            trap = exp(-7.0*min_circle)*0.72 + exp(-22.0*min_axis)*0.28
            trap = clamp(trap, 0.0, 1.0)
            t = clamp(trap^(0.65+0.35*(1-L)) * (0.72+0.45*E) + 0.08*C, 0, 1)
            canvas[py, px] = t
        end
    end
    canvas
end

# =============================================================================
# =============================================================================
# 10. IFS, Markov, Random walk (return RGB, we'll grayscale later)
# =============================================================================

function weighted_transform(probs, rng)
    r = rand(rng)
    cumulative = 0.0
    for i in eachindex(probs)
        cumulative += probs[i]
        if r <= cumulative return i end
    end
    length(probs)
end

function ifs_image_conditioned(img; n_iter=500_000, out_size=FRACT_SIZE)
    src = imresize(RGB.(img), (128, 128))
    gray, edge, contrast = build_image_fields(src)
    transforms = [(0.0,0.0,0.0,0.16,0.0,0.0), (0.85,0.04,-0.04,0.85,0.0,1.6),
                  (0.20,-0.26,0.23,0.22,0.0,1.6), (-0.15,0.28,0.26,0.24,0.0,0.44)]
    base_probs = [0.01, 0.85, 0.07, 0.07]
    xmin, xmax = -2.1820, 2.6558; ymin, ymax = 0.0, 9.9983
    oh, ow = out_size; canvas = fill(RGB(0.008,0.01,0.018), oh, ow)
    rng = MersenneTwister(image_hash_seed(src, base_seed=FRACTAL_SEED+91))
    x, y = 0.0, 0.0
    for _ in 1:n_iter
        u0 = clamp((x - xmin)/(xmax - xmin), 0, 1)
        v0 = clamp((y - ymin)/(ymax - ymin), 0, 1)
        L = sample_field(gray, u0, v0); E = sample_field(edge, u0, v0)
        probs = copy(base_probs)
        probs[1] *= 0.55 + 1.3*(1-L); probs[2] *= 0.70 + 1.0*L
        probs[3] *= 0.75 + 1.5*E; probs[4] *= 1.05 + 1.2*(1-E)
        probs ./= sum(probs)
        k = weighted_transform(probs, rng)
        a,b,c,d,e,f = transforms[k]
        x_new = a*x + b*y + e; y_new = c*x + d*y + f
        x, y = x_new, y_new
        u = (x - xmin)/(xmax - xmin); v = (y - ymin)/(ymax - ymin)
        if 0 <= u <= 1 && 0 <= v <= 1
            ix = clamp(round(Int, u*(ow-1))+1, 1, ow)
            iy = clamp(round(Int, (1-v)*(oh-1))+1, 1, oh)
            r = sample_field(red.(src), u, v)
            g = sample_field(green.(src), u, v)
            bcol = sample_field(blue.(src), u, v)
            alpha = 0.25 + 0.75*E
            old = canvas[iy, ix]
            newcol = RGB(clamp((1-alpha)*red(old)+alpha*r,0,1),
                         clamp((1-alpha)*green(old)+alpha*g,0,1),
                         clamp((1-alpha)*blue(old)+alpha*bcol,0,1))
            canvas[iy, ix] = newcol
        end
    end
    canvas
end

function sample_categorical(probs, rng=Random.default_rng())
    r = rand(rng); cumulative = 0.0
    for (i,p) in enumerate(probs)
        cumulative += p
        if r <= cumulative return i end
    end
    length(probs)
end

function markov_image(img; n_states=64)
    src = RGB.(img)
    n_per_channel = max(2, round(Int, cbrt(n_states)))
    n_states = n_per_channel^3
    h, w = size(src); indices = zeros(Int, h, w)
    for i in 1:h, j in 1:w
        r = red(src[i,j]); g = green(src[i,j]); b = blue(src[i,j])
        ri = min(floor(Int, r*n_per_channel)+1, n_per_channel)
        gi = min(floor(Int, g*n_per_channel)+1, n_per_channel)
        bi = min(floor(Int, b*n_per_channel)+1, n_per_channel)
        indices[i,j] = (ri-1)*n_per_channel^2 + (gi-1)*n_per_channel + bi
    end
    trans_count = zeros(Int, n_states, n_states)
    for i in 1:h, j in 1:w
        cur = indices[i,j]
        if j < w nxt = indices[i,j+1]
        elseif i < h nxt = indices[i+1,1]
        else continue end
        trans_count[cur,nxt] += 1
    end
    trans_prob = zeros(Float64, n_states, n_states)
    for s in 1:n_states
        row_sum = sum(trans_count[s,:])
        if row_sum > 0 trans_prob[s,:] = trans_count[s,:] ./ row_sum
        else trans_prob[s,:] .= 1/n_states end
    end
    rng = MersenneTwister(image_hash_seed(src, base_seed=7777))
    oh, ow = FRACT_SIZE; seq_len = oh*ow
    state = rand(rng, 1:n_states); seq = Vector{Int}(undef, seq_len); seq[1] = state
    for k in 2:seq_len
        state = sample_categorical(trans_prob[state,:], rng)
        seq[k] = state
    end
    out = Array{RGB{Float64},2}(undef, oh, ow)
    for idx in eachindex(seq)
        s = seq[idx]
        ri = div(s-1, n_per_channel^2)+1; rem = (s-1) % (n_per_channel^2)
        gi = div(rem, n_per_channel)+1; bi = rem % n_per_channel + 1
        r = (ri-1)/(n_per_channel-1); g = (gi-1)/(n_per_channel-1); b = (bi-1)/(n_per_channel-1)
        i = div(idx-1, ow)+1; j = mod(idx-1, ow)+1
        out[i,j] = RGB(r,g,b)
    end
    out
end

function randomwalk_image(img; n_steps=250_000, step_size=0.003)
    src = RGB.(img)
    gray = Gray.(src)
    grad_x, grad_y = imgradients(gray, KernelFactors.sobel)
    h_in, w_in = size(src); oh, ow = FRACT_SIZE
    canvas = fill(RGB(0.01,0.012,0.02), oh, ow)
    rng = MersenneTwister(image_hash_seed(src, base_seed=9911))
    x, y = rand(rng), rand(rng)
    for _ in 1:n_steps
        ix = clamp(round(Int, x*(w_in-1))+1, 1, w_in)
        iy = clamp(round(Int, y*(h_in-1))+1, 1, h_in)
        dx = Float64(grad_x[iy,ix]); dy = Float64(grad_y[iy,ix])
        if abs(dx)+abs(dy) < 1e-8 angle = 2π*rand(rng)
        else angle = atan(dy,dx) + 0.45*randn(rng) end
        brightness = Float64(gray[iy,ix])
        local_step = step_size * (0.45 + 1.35*brightness)
        x = mod(x + local_step*cos(angle), 1.0)
        y = mod(y + local_step*sin(angle), 1.0)
        ox = clamp(round(Int, x*(ow-1))+1, 1, ow)
        oy = clamp(round(Int, y*(oh-1))+1, 1, oh)
        col = src[iy,ix]; old = canvas[oy,ox]
        α = 0.28 + 0.60*brightness
        canvas[oy,ox] = RGB(clamp((1-α)*red(old)+α*red(col),0,1),
                            clamp((1-α)*green(old)+α*green(col),0,1),
                            clamp((1-α)*blue(old)+α*blue(col),0,1))
    end
    canvas
end

# =============================================================================
# 11. Dynamical‑system map generators (return grayscale density)
# =============================================================================

function image_chirikov(img; out_size=FRACT_SIZE, n_points=CHIRIKOV_N_POINTS)
    src = imresize(RGB.(img), (128,128))
    gray, edge, contrast = build_image_fields(src)
    oh, ow = out_size
    density = zeros(Float64, oh, ow)
    rng = MersenneTwister(image_hash_seed(src, base_seed=777))
    θ = rand(rng)*2π; p = rand(rng)*2π - π
    for _ in 1:n_points
        u = θ/(2π); v = (p+π)/(2π)
        K = 1.0 + 0.8*sample_field(gray, u, v) + 0.3*sample_field(edge, u, v)
        p = p + K*sin(θ)
        θ = θ + p
        p = mod(p+π, 2π)-π
        θ = mod(θ, 2π)
        u_c = clamp(θ/(2π),0,1); v_c = clamp((p+π)/(2π),0,1)
        ix = clamp(round(Int, u_c*(ow-1))+1, 1, ow)
        iy = clamp(round(Int, v_c*(oh-1))+1, 1, oh)
        density[iy, ix] += 0.003
    end
    maxd = maximum(density); density = density ./ (maxd + 1e-6)
    density
end

function image_henon(img; out_size=FRACT_SIZE, n_points=HENON_N_POINTS)
    src = imresize(RGB.(img), (128,128))
    gray, edge, contrast = build_image_fields(src)
    oh, ow = out_size
    density = zeros(Float64, oh, ow)
    rng = MersenneTwister(image_hash_seed(src, base_seed=111))
    x = rand(rng)*0.5; y = rand(rng)*0.5
    for _ in 1:n_points
        u = clamp((x+1.5)/3.0, 0, 1)
        v = clamp((y+0.4)/1.2, 0, 1)
        L = sample_field(gray, u, v)
        E = sample_field(edge, u, v)
        a_mod = 1.4 - 0.3*L + 0.15*E
        b_mod = 0.3 - 0.1*L
        x_new = 1 - a_mod*x^2 + y
        y_new = b_mod * x
        x, y = x_new, y_new
        u_c = clamp((x+1.5)/3.0, 0, 1)
        v_c = clamp((y+0.4)/1.2, 0, 1)
        ix = clamp(round(Int, u_c*(ow-1))+1, 1, ow)
        iy = clamp(round(Int, v_c*(oh-1))+1, 1, oh)
        density[iy, ix] += 0.004
    end
    maxd = maximum(density); density = density ./ (maxd + 1e-6)
    density
end

function image_logistic_bifurcation(img; out_size=FRACT_SIZE, res=LOGISTIC_RES, iters=LOGISTIC_ITERS)
    src = imresize(RGB.(img), (128,128))
    oh, ow = out_size
    density = zeros(Float64, oh, ow)
    for ix in 1:ow
        u = (ix-1)/(ow-1)
        r_base = 2.8 + 1.2*u
        r_mod = r_base + 0.1*(sample_field(red.(src), u, 0.5) - 0.5)
        r_mod = clamp(r_mod, 2.5, 4.0)
        x = 0.5
        for _ in 1:300 x = r_mod*x*(1-x) end
        for _ in 1:iters
            x = r_mod*x*(1-x)
            v = clamp(x, 0, 1)
            iy = clamp(round(Int, (1-v)*(oh-1))+1, 1, oh)
            density[iy, ix] += 0.005
        end
    end
    maxd = maximum(density); density = density ./ (maxd + 1e-6)
    density
end

# =============================================================================
# 12. Domain‑warp for grayscale images
# =============================================================================

function domain_warp_gray(gray_img, field_img; strength=0.15)
    gray = Float64.(Gray.(field_img))
    edges = detect_edges(Gray.(field_img), Canny())
    edge = Float64.(edges)
    emax = maximum(edge); emax == 0 && (emax = 1.0)
    edge ./= emax
    gx, gy = imgradients(gray, KernelFactors.sobel)
    mag = sqrt.(gx.^2 + gy.^2)
    mag[mag .== 0] .= 1.0
    dx = -gy ./ mag
    dy =  gx ./ mag
    oh, ow = size(gray_img)
    h_f, w_f = size(field_img)
    canvas = similar(gray_img)
    for py in 1:oh, px in 1:ow
        u = (px-1)/(ow-1); v = (py-1)/(oh-1)
        fx = clamp(round(Int, u*(w_f-1))+1, 1, w_f)
        fy = clamp(round(Int, v*(h_f-1))+1, 1, h_f)
        amp = strength * (0.6 + 0.4*edge[fy,fx])
        du = amp * dx[fy,fx]; dv = amp * dy[fy,fx]
        u_warp = clamp(u+du, 0, 1); v_warp = clamp(v+dv, 0, 1)
        sx = u_warp*(ow-1)+1; sy = v_warp*(oh-1)+1
        ix_low = floor(Int, sx); ix_high = min(ix_low+1, ow)
        iy_low = floor(Int, sy); iy_high = min(iy_low+1, oh)
        wx = sx - ix_low; wy = sy - iy_low
        v11 = gray_img[iy_low, ix_low]; v21 = gray_img[iy_low, ix_high]
        v12 = gray_img[iy_high, ix_low]; v22 = gray_img[iy_high, ix_high]
        top = (1-wx)*v11 + wx*v21
        bot = (1-wx)*v12 + wx*v22
        canvas[py,px] = (1-wy)*top + wy*bot
    end
    canvas
end

# =============================================================================
# 13. Post‑modern morphisms (for grayscale)
# =============================================================================

function morph_gray(img1, img2, alpha=0.5)
    a = clamp(alpha, 0.0, 1.0)
    (1-a)*img1 + a*img2
end

function postmodern_composite_gray(gray_list, field_img; alpha=0.6)
    oh, ow = size(first(gray_list))
    gray = Float64.(Gray.(field_img))
    edges = detect_edges(Gray.(field_img), Canny())
    edge = Float64.(edges)
    emax = maximum(edge); emax == 0 && (emax = 1.0)
    mask = edge ./ emax
    lum_mask = 1.0 .- gray
    combined_mask = 0.5*lum_mask .+ 0.5*mask
    mask_big = imresize(combined_mask, (oh, ow))
    canvas = zeros(Float64, oh, ow)
    for (i, gimg) in enumerate(gray_list)
        shifted_mask = clamp.(mask_big .+ 0.15*i, 0.0, 1.0)
        a = alpha .* shifted_mask
        canvas = (1 .- a).*canvas + a.*gimg
    end
    clamp.(canvas, 0.0, 1.0)
end
# =============================================================================

# =============================================================================
# 14. Main mathematical art loop (with FFT enhancement)
# =============================================================================

println("\n===== IMAGE-CONDITIONED MATHEMATICAL ART (post‑modern + FFT) =====")
ml_files = image_files(ML_OUTPUT)
if isempty(ml_files)
    println("No images found in $ML_OUTPUT – skipping mathematical art.")
else
    for f in ml_files
        println("\nGenerating from $f")
        img = RGB.(load(joinpath(ML_OUTPUT, f)))
        base = splitext(f)[1]
        img_proc = imresize(img, (128,128))

        dark, light = extract_duotone(img_proc)

        # Helper: grayscale -> FFT -> duotone
        function process_gray(gray)
            enhanced = fft_enhance(gray, boost=1.8, noise_amp=0.05)
            duotone(add_grain(enhanced, 0.015), dark, light)
        end

        # Helper: RGB image -> grayscale -> FFT -> duotone
        function process_rgb_to_duo(rgb_img)
            g = Float64.(Gray.(rgb_img))
            process_gray(g)
        end

        # --- Base generators ---
        println("  → Julia")
        julia_gray = image_julia(img_proc)
        safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_julia.png"), process_gray(julia_gray))

        println("  → Burning Ship")
        ship_gray = image_burning_ship(img_proc)
        safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_burning_ship.png"), process_gray(ship_gray))

        println("  → Orbit Trap")
        orbit_gray = image_orbit_trap(img_proc)
        safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_orbit_trap.png"), process_gray(orbit_gray))

        println("  → IFS")
        ifs_rgb = ifs_image_conditioned(img_proc)
        safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_ifs.png"), process_rgb_to_duo(ifs_rgb))

        println("  → Markov")
        markov_rgb = markov_image(img_proc)
        safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_markov.png"), process_rgb_to_duo(markov_rgb))

        println("  → Random walk")
        walk_rgb = randomwalk_image(img_proc)
        safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_randomwalk.png"), process_rgb_to_duo(walk_rgb))

        println("  → Chirikov map")
        chirikov_gray = image_chirikov(img_proc)
        safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_chirikov.png"), process_gray(chirikov_gray))

        println("  → Hénon map")
        henon_gray = image_henon(img_proc)
        safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_henon.png"), process_gray(henon_gray))

        println("  → Logistic bifurcation")
        logistic_gray = image_logistic_bifurcation(img_proc)
        safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_logistic.png"), process_gray(logistic_gray))

        # --- Morphisms (now also FFT‑enhanced) ---
        println("  → Morphisms")
        for step in 1:4
            alpha = (step-1)/3
            safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_morph_js_$(step).png"),
                      process_gray(morph_gray(julia_gray, ship_gray, alpha)))
            safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_morph_jc_$(step).png"),
                      process_gray(morph_gray(julia_gray, chirikov_gray, alpha)))
            safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_morph_ch_$(step).png"),
                      process_gray(morph_gray(chirikov_gray, henon_gray, alpha)))
        end

        # Domain warps
        safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_julia_warped.png"),
                  process_gray(domain_warp_gray(julia_gray, img_proc, strength=0.12)))
        safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_orbit_warped.png"),
                  process_gray(domain_warp_gray(orbit_gray, img_proc, strength=0.18)))
        safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_chirikov_warped.png"),
                  process_gray(domain_warp_gray(chirikov_gray, img_proc, strength=0.14)))

        # Composites
        comp1 = postmodern_composite_gray([julia_gray, Float64.(Gray.(ifs_rgb)), chirikov_gray], img_proc, alpha=0.55)
        safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_composite_jic.png"), process_gray(comp1))

        comp2 = postmodern_composite_gray([ship_gray, henon_gray, logistic_gray], img_proc, alpha=0.5)
        safe_save(joinpath(FRACT_OUTPUT, "fract_$(base)_composite_shl.png"), process_gray(comp2))

        println("  ✓ Completed $base")
    end
    println("\nMathematical art saved to $FRACT_OUTPUT")
end

# =============================================================================
# 15. Summary
# =============================================================================

println("""
===============================================================================
GENERATIVE ART PIPELINE COMPLETE (FFT‑enhanced)
===============================================================================
Input images:              $INPUT_DIR
Classical NFT styles:      $OUTPUT_DIR
PCA / GAN images:          $ML_OUTPUT
Saved models:              $MODEL_DIR
Post‑modern art:           $FRACT_OUTPUT   (all with FFT contrast/texture)
===============================================================================
""")