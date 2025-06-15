import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import gradio as gr
from collections import Counter

def extract_palette_from_image(palette_image):
    """Extract unique colors from a palette image."""
    img = Image.open(palette_image)
    img_array = np.array(img)
    # Reshape to get all pixels
    pixels = img_array.reshape(-1, 3)
    # Get unique colors
    unique_colors = np.unique(pixels, axis=0)
    return unique_colors

def generate_dynamic_palette(image, n_colors):
    """Generate a palette using K-means clustering."""
    img = Image.open(image)
    img_array = np.array(img)
    pixels = img_array.reshape(-1, 3)

    # Use K-means to find the most representative colors
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_.astype(int)

def rgb_distance(color1, color2):
    """Calculate Euclidean distance between two RGB colors."""
    return np.sqrt(np.sum((color1 - color2) ** 2))

def hue_distance(color1, color2):
    """Calculate distance based on hue."""
    # Convert RGB to HSV
    hsv1 = rgb_to_hsv(color1)
    hsv2 = rgb_to_hsv(color2)
    # Calculate hue difference (considering circular nature of hue)
    hue_diff = min(abs(hsv1[0] - hsv2[0]), 1 - abs(hsv1[0] - hsv2[0]))
    return hue_diff

def brightness_distance(color1, color2):
    """Calculate distance based on brightness (grayscale)."""
    # Convert to grayscale using standard weights
    gray1 = np.dot(color1, [0.299, 0.587, 0.114])
    gray2 = np.dot(color2, [0.299, 0.587, 0.114])
    return abs(gray1 - gray2)

def rgb_to_hsv(rgb):
    """Convert RGB to HSV."""
    rgb = rgb / 255.0
    r, g, b = rgb

    maxc = max(r, g, b)
    minc = min(r, g, b)
    v = maxc

    if maxc == minc:
        return 0, 0, v

    s = (maxc - minc) / maxc
    rc = (maxc - r) / (maxc - minc)
    gc = (maxc - g) / (maxc - minc)
    bc = (maxc - b) / (maxc - minc)

    if r == maxc:
        h = bc - gc
    elif g == maxc:
        h = 2.0 + rc - bc
    else:
        h = 4.0 + gc - rc

    h = (h / 6.0) % 1.0
    return h, s, v

def get_mode_color(pixel_group):
    """Calculate the mode color of a pixel group."""
    # Reshape to get all pixels
    pixels = pixel_group.reshape(-1, 3)

    # Convert to tuple for counting
    pixel_tuples = [tuple(pixel) for pixel in pixels]

    # Count occurrences of each color
    color_counts = Counter(pixel_tuples)

    # Get the most common color
    mode_color = np.array(color_counts.most_common(1)[0][0])
    return mode_color

def pixelize_image(image, palette, mode='rgb', pixel_size=1):
    """Convert image to pixel art using the given palette."""
    img = Image.open(image)
    img_array = np.array(img)

    # Get image dimensions
    height, width = img_array.shape[:2]

    # Calculate new dimensions based on pixel_size
    new_height = height // pixel_size
    new_width = width // pixel_size

    # Initialize output array
    output_array = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Choose distance function based on mode
    if mode == 'rgb':
        distance_func = rgb_distance
    elif mode == 'hue':
        distance_func = hue_distance
    else:  # brightness
        distance_func = brightness_distance

    # Process each pixel group
    for y in range(new_height):
        for x in range(new_width):
            # Get the pixel group
            y_start = y * pixel_size
            y_end = min((y + 1) * pixel_size, height)
            x_start = x * pixel_size
            x_end = min((x + 1) * pixel_size, width)

            pixel_group = img_array[y_start:y_end, x_start:x_end]

            # Calculate mean and mode colors
            mean_color = np.mean(pixel_group, axis=(0, 1)).astype(int)
            mode_color = get_mode_color(pixel_group)

            # Find closest palette color for mean
            mean_distances = np.array([distance_func(mean_color, palette_color) for palette_color in palette])
            mean_closest_color = palette[np.argmin(mean_distances)]
            mean_min_distance = np.min(mean_distances)

            # Find closest palette color for mode
            mode_distances = np.array([distance_func(mode_color, palette_color) for palette_color in palette])
            mode_closest_color = palette[np.argmin(mode_distances)]
            mode_min_distance = np.min(mode_distances)

            # Choose the color with the smaller distance to palette
            if mean_min_distance <= mode_min_distance:
                output_array[y, x] = mean_closest_color
            else:
                output_array[y, x] = mode_closest_color

    # Create output image
    output = Image.fromarray(output_array)

    # Resize back to original dimensions
    output = output.resize((width, height), Image.NEAREST)

    return output

def process_image(input_image, palette_image, n_colors, mode, pixel_size, use_dynamic_palette):
    """Process the image with the given parameters."""
    if use_dynamic_palette:
        palette = generate_dynamic_palette(input_image, n_colors)
    else:
        palette = extract_palette_from_image(palette_image)

    result = pixelize_image(input_image, palette, mode, pixel_size)
    return result

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Pixel Art Converter") as interface:
        gr.Markdown("# Pixel Art Converter")
        gr.Markdown("Convert your images into pixel art with customizable palettes!")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="filepath", label="Input Image")
                palette_image = gr.Image(type="filepath", label="Palette Image (for fixed palette mode)")
                use_dynamic_palette = gr.Checkbox(label="Use Dynamic Palette", value=True)
                n_colors = gr.Slider(minimum=2, maximum=32, value=8, step=1, label="Number of Colors (for dynamic palette)")
                mode = gr.Radio(["rgb", "hue", "brightness"], label="Color Matching Mode", value="hue")
                pixel_size = gr.Slider(minimum=1, maximum=32, value=4, step=1, label="Pixel Size")
                process_btn = gr.Button("Convert to Pixel Art")

            with gr.Column():
                output_image = gr.Image(label="Pixel Art Result")

        process_btn.click(
            fn=process_image,
            inputs=[input_image, palette_image, n_colors, mode, pixel_size, use_dynamic_palette],
            outputs=output_image
        )

    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()