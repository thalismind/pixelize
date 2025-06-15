---
title: Pixel Art Converter
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.34.0
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

## Prompts

This application was generated using the following prompt:

```none
Please plan out and implement a Python project using a Gradio interface to convert images into pixel art. Dependencies will be fairly minimal, mostly Gradio and Pillow, but you can use a mix of scipy, numpy, and other numeric libraries for the image manipulation. The image conversion process should operate in two main modes: fixed palette and dynamic palette. In dynamic palette, a number of colors will be given, and a palette should be generated from the image to provide the best color fit. In fixed palette, a second image will be uploaded, which will contain the palette colors as multi-pixel squares (simply list the colors present in the second image to get the palette, order is not important). Please offer options to match colors based on closest RGB, closest hue, and closest brightness/grayscale. The interface should be able to convert a single image at a time and let the user drag and drop into the Gradio image upload component.
```

```none
Please write scripts to set up the virtual environment and another script to run the program.
```

```none
Please prepare the repository for use with HuggingFace Spaces. Make sure it has the right configuration for a Gradio space.
```

Followed by manually updating the Gradio version in the README.md frontmatter and the requirements.txt file to use the latest version.

The pixelization process was not implemented correctly and was resizing the image. That was fixed with the following prompt:

```none
The pixel_size parameter currently makes the image larger, but it should make the pixels larger within the image, by combining groups of pixels on the grid given by the pixel_size. For example, if pixel_size is 2, then every 2x2 group should be the same color. If pixel_size is 4, then the groups are 4x4. Use the best color for each group based on the pixels within it.
```

```none
The average color may not be the best color for each pixel. Try both the mean and mode, and see which one is closer to the palette colors.
```

## License

This project is open source and available under the MIT License.