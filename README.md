# Multi-View SAM: Multi-View Image Segmentation by Combining SAM and VGGT

MATLAB implementation of **Superposed IM-OFDM (S-IM-OFDM)** - an enhanced OFDM waveform for Integrated Sensing and Communications (ISAC).

## 📋 Project Overview

**Multi-View** SAM is a framework for multi-view image segmentation based on pre-trained **SAM (Segment Anything Model)** and **VGGT**. This project enables simultaneous semantic segmentation across multiple viewpoints using a single point prompt from the first image.

## 🎯 Core Task

Given a set of multi-view images, achieve:
-Accurate segmentation of the first image using a point prompt
-Consistent segmentation results across all viewpoints
-High-quality segmentation masks generated simultaneously for all views

## 🏗️ Architecture

This project explores two different method combinations:

### Method 1: SAM + VGGT

**Pipeline:**
```text
First image (point prompt) → SAM segmentation → Select highest confidence mask → VGGT track to other views
```

**Characteristics:**
    ✅ Advantages: Stable segmentation results, accurate point tracking
    ❌ Disadvantages:
        High GPU memory usage (~20GB for large masks)
        Slow inference speed
        Potential mask discontinuity issues

### Method 2: VGGT + SAM

**Pipeline:**
```text
First image (point prompt) → VGGT track to other views → Separate SAM segmentation for each view
```

**Characteristics:**
    ✅ Advantages:  Low GPU memory usage, fast inference
    ❌ Disadvantages:  Segmentation consistency may be compromised
