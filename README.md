# 🦷 Dental Surface Reconstruction App v4.0

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/MalikShehram/Reconstruction-of-Tooth)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

### 🚀 Live Demo
**Click the button below to try the application live on Hugging Face Spaces:**

[![Run on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Open%20in%20Spaces-Blue?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/MalikShehram/Reconstruction-of-Tooth)

*(Replace the link above with your actual Hugging Face Space URL)*

---


## 📖 Overview
The **Dental Surface Reconstruction App v4.0** is an advanced image processing tool designed to reconstruct missing tooth surfaces in dental models. Based on insights from previous versions, this tool moves away from unreliable auto-detection and utilizes a **user-guided masking system** to ensure high-precision results regardless of camera angles or image cropping.

The system intelligently "fills" dental gaps using a combination of symmetry patching, color-matched texture transfer, and seamless Poisson blending.

---

## ✨ Key Features
* **Precision Masking:** Includes an integrated Image Editor that allows users to paint directly over the missing tooth gap to define the reconstruction zone.
* **Symmetry-Based Reconstruction:** Extracts a "mirror" tooth from the opposite side of the dental arch to ensure anatomical accuracy.
* **Intelligent Alignment:** Uses ORB feature detection and Homography to align healthy reference images with the target "gap" image.
* **Seamless Blending:** Features LAB color statistics transfer and Poisson seamless cloning to match the new tooth's shade and lighting to the surrounding gums.
* **AI-Enhanced (Optional):** Supports Stable Diffusion inpainting for generating high-fidelity enamel textures when running on GPU-enabled systems.

---

## 🛠️ Technical Pipeline
1.  **Input:** User uploads two complete dental reference images and one target image with a gap.
2.  **Mask Extraction:** The user paints the gap; the app converts these strokes into a binary inpaint mask.
3.  **Warping:** Reference images are aligned to the target's perspective using RANSAC-based homography.
4.  **Synthesis:** The engine selects the best reference tooth, mirrors it (symmetry), and performs LAB color matching.
5.  **Blending:** Poisson cloning and Telea inpainting refine the seams for a natural look.

---

## 🚀 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/MalikShehram512/dental-reconstruction.git](https://github.com/MalikShehram512/dental-reconstruction.git)
cd dental-reconstruction
