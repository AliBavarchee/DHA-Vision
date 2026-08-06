#                                                                      ==/\/\/\/\/\/\/\/\/\/\/\/\/\/\==
#                                                                           Digital__Hana__Arts®
#                                                                      ==/\/\/\/\/\/\/\/\/\/\/\/\/\/\==
# =============================================================================
#  Project: DHA Vision – NFT & ML Art Generation Suite
#  Language: Julia 1.9.4
# 
#  Company: Digital__Hana__Arts®
#  Author:  Ale Deragschan
# 
#  Description:
#    This software automatically generates NFT-style digital artworks,
#    performs PCA/DCGAN machine learning on the created images, and
#    provides a local web interface for upload, generation, classification,
#    and model management.
# 
#  License: Proprietary – All rights reserved.
# =============================================================================

using HTTP, Sockets, Images, FileIO, ImageTransformations, Colors, Base64
using Random, Dates

const PORT = 8080
const APP_DIR = @__DIR__
const BG_IMAGE = "DHA_Vision.png"
const LOGO     = "DHA_Vision_logo.png"

# =============================================================================
# HTML + CSS + JavaScript (single page app)
# =============================================================================
const HTML_PAGE = raw"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DHA Vision</title>
    <style>
        body {
            margin: 0; padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: url('/bg') no-repeat center center fixed;
            background-size: cover;
            color: #fff;
            min-height: 100vh;
        }
        .overlay {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.6);
            z-index: 0;
        }
        .container {
            position: relative; z-index: 1;
            max-width: 800px; margin: auto;
            padding: 40px 20px;
        }
        .logo {
            display: block; margin: 0 auto 20px;
            width: 175px; /* 350x500 scaled down */
        }
        h1 {
            text-align: center; font-size: 2.5em;
            text-shadow: 2px 2px 8px #000;
            margin-bottom: 10px;
        }
        .card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 20px; margin: 20px 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        button, .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border: none;
            padding: 12px 24px; border-radius: 8px;
            font-size: 1em;
            cursor: pointer; margin: 5px;
            transition: transform 0.2s;
        }
        button:hover, .btn:hover { transform: scale(1.05); }
        input[type="file"] { margin: 10px 0; color: #eee; }
        #status {
            padding: 10px; border-radius: 8px;
            background: rgba(0,0,0,0.5);
            white-space: pre-wrap; font-size: 0.9em;
            max-height: 200px; overflow-y: auto;
        }
        .gallery { display: flex; flex-wrap: wrap; gap: 10px; }
        .gallery img { width: 100px; height: 100px; object-fit: cover; border-radius: 8px; }
        a { color: #bbb; }
    </style>
</head>
<body>
<div class="overlay"></div>
<div class="container">
    <img src="/logo" alt="DHA Vision Logo" class="logo">
    <h1>DHA Vision</h1>

    <div class="card">
        <h2>Upload Images</h2>
        <input type="file" id="fileInput" multiple accept="image/*">
        <button onclick="uploadImages()">Upload to input/</button>
        <div id="uploadStatus"></div>
    </div>

    <div class="card">
        <h2>NFT Generation</h2>
        <p>Run the full pipeline: styles + PCA + GAN training & generation.</p>
        <button onclick="runNFTGen()">🚀 Generate NFT Artworks</button>
        <p><small>This may take several minutes (GAN training).</small></p>
    </div>

    <div class="card">
        <h2>Generate from Saved Models</h2>
        <button onclick="runGANGen(5)">🧠 GAN: 5 new images</button>
        <button onclick="runGANGen(10)">🧠 GAN: 10 new images</button>
        <br><br>
        <span>PCA Variations of an image:</span><br>
        <input type="file" id="pcaInput" accept="image/*">
        <input type="number" id="pcaCount" value="3" min="1" max="10" style="width:60px">
        <button onclick="runPCAVar()">✨ Create PCA Variations</button>
    </div>

    <div class="card">
        <h2>Classify Image (Real/Fake)</h2>
        <input type="file" id="classifyInput" accept="image/*">
        <button onclick="classifyImage()">🔍 Classify</button>
        <div id="classifyResult"></div>
    </div>

    <div class="card">
        <h2>Output Gallery</h2>
        <button onclick="refreshGallery('output')">🖼️ NFT Styles</button>
        <button onclick="refreshGallery('ml_output')">🧬 ML Variations</button>
        <button onclick="refreshGallery('generated_images')">🎨 Generated Images</button>
        <div id="gallery" class="gallery"></div>
    </div>

    <div id="status"></div>
</div>

<script>
    function setStatus(msg) {
        document.getElementById('status').innerText = msg;
    }

    async function uploadImages() {
        const files = document.getElementById('fileInput').files;
        if (!files.length) return alert('Select files first.');
        const formData = new FormData();
        for (const f of files) formData.append('files', f);
        setStatus('Uploading...');
        const resp = await fetch('/upload', { method: 'POST', body: formData });
        const txt = await resp.text();
        document.getElementById('uploadStatus').innerText = txt;
        setStatus('Upload complete.');
    }

    async function runNFTGen() {
        setStatus('Running NFT generation (this may take several minutes)...');
        const resp = await fetch('/run_nft_gen');
        const txt = await resp.text();
        setStatus(txt);
    }

    async function runGANGen(n) {
        setStatus(`Generating ${n} GAN images...`);
        const resp = await fetch(`/generate_gan?n=${n}`);
        const txt = await resp.text();
        setStatus(txt);
    }

    async function runPCAVar() {
        const file = document.getElementById('pcaInput').files[0];
        if (!file) return alert('Select an image for PCA.');
        const count = document.getElementById('pcaCount').value;
        const formData = new FormData();
        formData.append('image', file);
        setStatus('Creating PCA variations...');
        const resp = await fetch(`/pca_variations?n=${count}`, { method: 'POST', body: formData });
        const txt = await resp.text();
        setStatus(txt);
    }

    async function classifyImage() {
        const file = document.getElementById('classifyInput').files[0];
        if (!file) return alert('Select an image to classify.');
        const formData = new FormData();
        formData.append('image', file);
        setStatus('Classifying...');
        const resp = await fetch('/classify', { method: 'POST', body: formData });
        const txt = await resp.text();
        document.getElementById('classifyResult').innerHTML = `<pre>${txt}</pre>`;
        setStatus('Classification done.');
    }

    async function refreshGallery(folder) {
        const resp = await fetch(`/list_images?folder=${folder}`);
        const data = await resp.json();
        const gallery = document.getElementById('gallery');
        gallery.innerHTML = '';
        data.images.forEach(img => {
            const imgEl = document.createElement('img');
            imgEl.src = `/image?folder=${folder}&name=${encodeURIComponent(img)}`;
            gallery.appendChild(imgEl);
        });
    }
</script>
</body>
</html>
"""

# =============================================================================
# Helper functions
# =============================================================================
function encode_image(path)
    if !isfile(path)
        return nothing
    end
    img = load(path)
    # Resize background if needed (original kept, but CSS covers)
    buf = IOBuffer()
    save(Stream(format"PNG", buf), img)
    return base64encode(take!(buf))
end

# =============================================================================
# HTTP Request Handlers
# =============================================================================
function serve_static(req)
    path = HTTP.URI(req.target).path
    if path == "/bg"
        bg_data = encode_image(joinpath(APP_DIR, BG_IMAGE))
        if bg_data !== nothing
            img_bytes = base64decode(bg_data)
            return HTTP.Response(200, ["Content-Type" => "image/png"], body = img_bytes)
        end
    elseif path == "/logo"
        logo_data = encode_image(joinpath(APP_DIR, LOGO))
        if logo_data !== nothing
            img_bytes = base64decode(logo_data)
            return HTTP.Response(200, ["Content-Type" => "image/png"], body = img_bytes)
        end
    end
    return HTTP.Response(404)
end

function handle_request(req)
    # Main page
    if req.method == "GET" && req.target == "/"
        return HTTP.Response(200, ["Content-Type" => "text/html"], body = HTML_PAGE)
    end

    # Serve background / logo
    if req.target in ["/bg", "/logo"]
        return serve_static(req)
    end

    # Upload images
    if req.method == "POST" && req.target == "/upload"
        try
            HTTP.handle_form_data(req) do part
                filename = part.filename
                if !isnothing(filename) && !isempty(filename)
                    data = part.data
                    open(joinpath(APP_DIR, "input", filename), "w") do io
                        write(io, data)
                    end
                end
            end
            return HTTP.Response(200, "Files uploaded successfully.")
        catch e
            return HTTP.Response(500, "Upload error: $e")
        end
    end

    # Run nft_ml_gen2.jl
    if req.method == "GET" && req.target == "/run_nft_gen"
        try
            cmd = `$(Base.julia_cmd()) nft_ml_gen2.jl`
            run(cmd, wait = true)
            return HTTP.Response(200, "NFT generation complete. Check output/, ml_output/, and saved_models/.")
        catch e
            return HTTP.Response(500, "Error: $e")
        end
    end

    # Generate GAN images via generate_from_models.jl
    if req.method == "GET" && startswith(req.target, "/generate_gan")
        query = HTTP.queryparams(HTTP.URI(req.target))
        n = parse(Int, get(query, "n", "5"))
        try
            cmd = `$(Base.julia_cmd()) generate_from_models.jl --gan $n`
            run(cmd, wait = true)
            return HTTP.Response(200, "GAN images generated in generated_images/")
        catch e
            return HTTP.Response(500, "Error: $e")
        end
    end

    # PCA variations (upload image, call script)
    if req.method == "POST" && startswith(req.target, "/pca_variations")
        query = HTTP.queryparams(HTTP.URI(req.target))
        n = parse(Int, get(query, "n", "3"))
        tempfile = joinpath(APP_DIR, "temp_pca_input.png")
        try
            HTTP.handle_form_data(req) do part
                if part.name == "image" && !isnothing(part.filename)
                    open(tempfile, "w") do io
                        write(io, part.data)
                    end
                end
            end
            if !isfile(tempfile)
                return HTTP.Response(400, "No image received.")
            end
            cmd = `$(Base.julia_cmd()) generate_from_models.jl --pca $tempfile $n`
            run(cmd, wait = true)
            rm(tempfile; force=true)
            return HTTP.Response(200, "PCA variations created in generated_images/")
        catch e
            rm(tempfile; force=true)
            return HTTP.Response(500, "Error: $e")
        end
    end

    # Classify image
    if req.method == "POST" && req.target == "/classify"
        tempfile = joinpath(APP_DIR, "temp_classify_input.png")
        try
            HTTP.handle_form_data(req) do part
                if part.name == "image" && !isnothing(part.filename)
                    open(tempfile, "w") do io
                        write(io, part.data)
                    end
                end
            end
            if !isfile(tempfile)
                return HTTP.Response(400, "No image received.")
            end
            cmd = `$(Base.julia_cmd()) generate_from_models.jl --classify $tempfile`
            result = read(cmd, String)
            rm(tempfile; force=true)
            return HTTP.Response(200, result)
        catch e
            rm(tempfile; force=true)
            return HTTP.Response(500, "Error: $e")
        end
    end

    # List images in a folder (for gallery)
    if req.method == "GET" && startswith(req.target, "/list_images")
        query = HTTP.queryparams(HTTP.URI(req.target))
        folder = get(query, "folder", "output")
        dir = joinpath(APP_DIR, folder)
        if !isdir(dir)
            return HTTP.Response(200, "[]")
        end
        images = filter(f -> occursin(r"\.(png|jpg|jpeg)$"i, f), readdir(dir))
        return HTTP.Response(200, ["Content-Type" => "application/json"], body = JSON.json(Dict("images" => images)))
    end

    # Serve individual images from folders
    if req.method == "GET" && startswith(req.target, "/image")
        query = HTTP.queryparams(HTTP.URI(req.target))
        folder = get(query, "folder", "output")
        name   = get(query, "name", "")
        path = joinpath(APP_DIR, folder, name)
        if !isfile(path)
            return HTTP.Response(404, "File not found")
        end
        img = load(path)
        buf = IOBuffer()
        save(Stream(format"PNG", buf), img)
        return HTTP.Response(200, ["Content-Type" => "image/png"], body = take!(buf))
    end

    # Fallback
    return HTTP.Response(404, "Not found")
end

# =============================================================================
# Start server and open browser
# =============================================================================
println("DHA Vision server starting at http://localhost:$PORT")
println("Open your browser to that address.")
@async try
    server = HTTP.serve(handle_request, "0.0.0.0", PORT)
catch e
    @error "Server error" e
end

# Try to open browser automatically (platform dependent)
if Sys.iswindows()
    run(`cmd /c start http://localhost:$PORT`)
elseif Sys.isapple()
    run(`open http://localhost:$PORT`)
elseif Sys.islinux() || Sys.isfreebsd()
    run(`xdg-open http://localhost:$PORT`)
end

# Keep the main thread alive
while true
    sleep(1)
end