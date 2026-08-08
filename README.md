# DHA‑Vision — Generative Art Pipeline (v3)
![logo](DHA_Vision_logo.png)

**Post‑modern, high‑resolution mathematical art from any photograph.**

DHA‑Vision is a Julia‑based pipeline that transforms a small set of input images into a vast gallery of **post‑modern generative art**.  
It combines classical NFT style filters, machine‑learning (PCA & GAN) and a suite of mathematically‑driven generators (fractals, chaotic maps, dynamical systems) with duotone colouring, FFT texture enhancement, and fluid motion warping.  
The result is a collection of **1024×1024**, harsh, minimalist artworks that retain a visual connection to the original photograph.

---

## ✨ Features

- **10 classical image styles** – cyberpunk, vaporwave, glitch, oil, pop art, etc.
- **PCA latent‑space variations** – new images created by exploring the principal components of the input + style set.
- **DCGAN generative model** – a convolutional GAN trained on the image collection, capable of producing novel images.
- **Smart retraining** – the pipeline detects whether the input directory has changed; if not, it loads pre‑trained PCA/GAN models and skips the lengthy training step.
- **9 image‑conditioned mathematical generators** that use the source image as a **parameter field** rather than a mere palette:
  - Julia fractal
  - Burning Ship fractal
  - Orbit‑trap fractal
  - Chaos‑game IFS (Barnsley fern)
  - Markov texture synthesis
  - Gradient‑guided random walk
  - Chirikov (standard) map
  - Hénon map
  - Logistic bifurcation diagram
- **Duotone palette** – a two‑colour palette is automatically extracted from each source image, giving the entire output a cohesive, editorial feel.
- **FFT high‑frequency boost** – edges and fine structures are emphasised in the frequency domain, adding grit and contrast.
- **Fluid motion warp** – smooth, semi‑random flow fields distort the generated shapes, making each piece feel alive and unique.
- **Morphisms & composites** – continuous blends between generators, domain‑warped variations, and layered composites produce dozens of extra artworks per input.

---

## 📁 Directory Structure
