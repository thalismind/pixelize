---
title: Pixel Art Converter
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# Pixel Art Converter

Convert your images into pixel art with customizable palettes! This application supports both dynamic and fixed color palettes, with various color matching modes.

## Features

- Convert images to pixel art with customizable pixel sizes
- Two palette modes:
  - Dynamic: Automatically generates an optimal palette using K-means clustering
  - Fixed: Uses colors from a provided palette image
- Multiple color matching modes:
  - RGB: Matches colors based on RGB distance
  - Hue: Matches colors based on hue similarity
  - Brightness: Matches colors based on grayscale brightness

## How to Use

1. Upload an input image using the "Input Image" component
2. Choose between dynamic or fixed palette mode
3. For dynamic palette: Set the number of colors (2-32)
4. For fixed palette: Upload a palette image containing the colors you want to use
5. Select the color matching mode (RGB, Hue, or Brightness)
6. Adjust the pixel size if desired
7. Click "Convert to Pixel Art" to process the image

## Creating a Palette Image

For fixed palette mode, create an image containing the colors you want to use. Each color should be represented by at least one pixel. The order of colors doesn't matter, and you can use any arrangement of the colors in the image.