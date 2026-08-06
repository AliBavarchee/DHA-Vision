# =============================================================================
# generate_from_models.jl
# Load saved models and:
#   - generate GAN images
#   - create PCA variations of a photo
#   - classify an image (real/fake) with the discriminator
# =============================================================================

using Images, FileIO, ImageTransformations, Colors, Statistics, LinearAlgebra
using Random, MultivariateStats, Flux, BSON

const MODEL_DIR   = "saved_models"
const OUTPUT_DIR  = "generated_images"
const IMG_SIZE    = (128, 128)      # must match training size
const UPSCALE_TO  = nothing         # e.g., (1024, 1024) or nothing to keep 128x128

isdir(OUTPUT_DIR) || mkdir(OUTPUT_DIR)

# ---------------------------------------------------------------------------
# 1.  Load all models
# ---------------------------------------------------------------------------
println("Loading models from $MODEL_DIR ...")

# --- GAN generator ---
gan_gen_data = BSON.load(joinpath(MODEL_DIR, "gan_generator.bson"))
generator    = gan_gen_data["generator"]

# --- GAN discriminator ---
gan_disc_data = BSON.load(joinpath(MODEL_DIR, "gan_discriminator.bson"))
discriminator = gan_disc_data["discriminator"]

# --- PCA ---
pca_data   = BSON.load(joinpath(MODEL_DIR, "pca_model.bson"))
pca_model  = pca_data["model"]
img_size   = pca_data["img_size"]

println("All models loaded successfully.")

# ---------------------------------------------------------------------------
# 2.  Helper functions
# ---------------------------------------------------------------------------
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

function array_to_img(arr)
    img_arr = clamp.(arr, 0.0f0, 1.0f0)
    img = colorview(RGB, img_arr)
    if UPSCALE_TO !== nothing
        img = imresize(img, UPSCALE_TO)
    end
    return img
end

# Preprocess image for discriminator (resize, normalize to [-1,1])
function preprocess_for_disc(image_path)
    img = load(image_path)
    img = RGB.(img)
    img = imresize(img, IMG_SIZE)
    arr = Float32.(channelview(img))          # (3,128,128)
    arr = permutedims(arr, (3,2,1))          # to (W,H,C) => (128,128,3)
    arr = 2.0f0 .* arr .- 1.0f0              # [0,1] → [-1,1]
    return reshape(arr, 128,128,3,1)         # add batch dim
end

# ---------------------------------------------------------------------------
# 3.  GAN generation
# ---------------------------------------------------------------------------
function generate_gan_images(n::Int; prefix="gan")
    println("\nGenerating $n GAN images...")
    Random.seed!(rand(UInt))
    for i in 1:n
        noise = randn(Float32, 100, 1)
        gen_img = generator(noise)                      # (128,128,3,1)
        img_arr = (gen_img[:,:,:,1] .+ 1.0f0) ./ 2.0f0
        img_arr = permutedims(img_arr, (3, 1, 2))       # (3,128,128)
        img = array_to_img(img_arr)
        outpath = joinpath(OUTPUT_DIR, "$(prefix)_$(i).png")
        safe_save(outpath, img)
        println("  Saved $outpath")
    end
end

# ---------------------------------------------------------------------------
# 4.  PCA based generation (variations of a given image)
# ---------------------------------------------------------------------------
function load_and_preprocess_pca(image_path)
    img = load(image_path)
    img = RGB.(img)
    img = imresize(img, img_size)
    return Float32.(channelview(img))   # (3,128,128)
end

function project_to_pca(img_arr)
    flat = vec(img_arr)
    centered = flat .- pca_model.mean
    return pca_model.proj' * centered
end

function reconstruct_from_pca(latent_vec)
    features = pca_model.proj * latent_vec .+ pca_model.mean
    return reshape(features, 3, img_size[1], img_size[2])
end

function pca_variations(image_path, n::Int; prefix="pca")
    println("\nCreating $n PCA variations of $image_path ...")
    img_arr = load_and_preprocess_pca(image_path)
    base_latent = project_to_pca(img_arr)
    # noise scale based on training data variance (or fixed small value)
    latent_std = 0.2f0 .* std(base_latent)
    if iszero(latent_std)
        latent_std = 0.1f0
    end

    base_name = splitext(basename(image_path))[1]
    Random.seed!(1234)
    for i in 1:n
        noise = latent_std .* randn(Float32, length(base_latent))
        new_latent = base_latent .+ noise
        new_arr = reconstruct_from_pca(new_latent)
        img = array_to_img(new_arr)
        outpath = joinpath(OUTPUT_DIR, "$(prefix)_$(base_name)_$(i).png")
        safe_save(outpath, img)
        println("  Saved $outpath")
    end
end

# ---------------------------------------------------------------------------
# 5.  Discriminator classification
# ---------------------------------------------------------------------------
function classify_image(image_path)
    println("\nClassifying $image_path ...")
    tensor = preprocess_for_disc(image_path)
    score = discriminator(tensor)[1]        # scalar in [0,1]
    println("  Real probability: $(round(score, digits=4))")
    if score > 0.5
        println("  → Looks like a REAL image (according to GAN)")
    else
        println("  → Looks FAKE / generated")
    end
    return score
end

# ---------------------------------------------------------------------------
# 6.  Command‑line interface
# ---------------------------------------------------------------------------
function main()
    if length(ARGS) == 0
        println("Usage:")
        println("  Generate GAN images:   julia generate_from_models.jl --gan 10")
        println("  PCA variations:        julia generate_from_models.jl --pca path/to/image.jpg 5")
        println("  Classify an image:     julia generate_from_models.jl --classify path/to/image.png")
        println("  Combine options:       julia generate_from_models.jl --gan 5 --pca cat.jpg 3 --classify dog.jpg")
        return
    end

    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--gan"
            n = parse(Int, ARGS[i+1])
            generate_gan_images(n)
            i += 2
        elseif ARGS[i] == "--pca"
            path = ARGS[i+1]
            n = parse(Int, ARGS[i+2])
            pca_variations(path, n)
            i += 3
        elseif ARGS[i] == "--classify"
            path = ARGS[i+1]
            classify_image(path)
            i += 2
        else
            println("Unknown argument: ", ARGS[i])
            i += 1
        end
    end
end

main()