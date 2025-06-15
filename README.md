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

A Python application that converts images into pixel art using Gradio for the user interface. The application supports both dynamic and fixed color palettes, with various color matching modes.

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)

## Features

- Convert images to pixel art with customizable pixel sizes
- Two palette modes:
  - Dynamic: Automatically generates an optimal palette using K-means clustering
  - Fixed: Uses colors from a provided palette image
- Multiple color matching modes:
  - RGB: Matches colors based on RGB distance
  - Hue: Matches colors based on hue similarity
  - Brightness: Matches colors based on grayscale brightness
- Interactive Gradio interface with drag-and-drop support

## Local Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Local Usage

1. Run the application:
```bash
python pixelize.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://127.0.0.1:7860)

3. Using the interface:
   - Upload an input image using the "Input Image" component
   - Choose between dynamic or fixed palette mode
   - For dynamic palette: Set the number of colors (2-32)
   - For fixed palette: Upload a palette image containing the colors you want to use
   - Select the color matching mode (RGB, Hue, or Brightness)
   - Adjust the pixel size if desired
   - Click "Convert to Pixel Art" to process the image

## Creating a Palette Image

For fixed palette mode, create an image containing the colors you want to use. Each color should be represented by at least one pixel. The order of colors doesn't matter, and you can use any arrangement of the colors in the image.

## HuggingFace Spaces

This application is also available as a HuggingFace Space. You can try it out directly in your browser without any installation required.

## License

This project is open source and available under the MIT License.