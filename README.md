# DHA Vision

**NFT & ML Art Generation Suite**  
*by [Digital__Hana__Arts](https://github.com/DigitalHanaArts)*  
*Author: Ale Deragschan*

[![Julia](https://img.shields.io/badge/Julia-1.9.4-blueviolet?logo=julia)](https://julialang.org/)
[![License](https://img.shields.io/badge/license-Proprietary-red)](./LICENSE)

DHA Vision is a desktop application for experimenting with AI-assisted digital art creation. Starting from a single photo, it can transform images into a variety of NFT-inspired artistic styles, learn visual patterns from generated artwork, and create entirely new images using a built-in machine learning pipeline.

Everything runs locally on your computer, so once the application is installed, no internet connection is needed to generate artwork or train models.

---

## ✨ Features

- **10 artistic NFT-inspired styles** including Cyberpunk, Vaporwave, Glitch, Pixel Art, Comic, Oil Painting, Holographic, Gold Foil, Pop Art, and Abstract.
- **Built-in machine learning pipeline** combining PCA for feature extraction with a DCGAN for generating original artwork.
- **Image authenticity prediction** using the trained discriminator to classify generated images.
- **Simple browser-based interface** for uploading images, generating artwork, viewing results, and downloading your favorites.
- **Standalone application** built with PackageCompiler.jl, allowing the software to run without requiring a separate Julia installation.

---

## 🚀 How it works

The workflow is designed to be straightforward:

1. Upload a photo.
2. Choose one of the available artistic styles.
3. Generate a stylized NFT-inspired version of the image.
4. Train the machine learning model on the generated artwork.
5. Create entirely new images inspired by the learned visual style.

Whether you're experimenting with generative art, exploring GANs, or simply creating unique digital artwork, DHA Vision brings the entire workflow together in a single application.

---

## 📂 Project Structure