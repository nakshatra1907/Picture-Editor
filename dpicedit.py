import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import convolve
from collections import deque


def change_brightness(image, value):
    # Ensure the value is between -255 and +255
    value = np.clip(value, -255, 255)

    # Create a copy of the image to modify
    img = image.copy().astype(np.int32)
    img += value

    # Return the new image, clipped between 0 and 255
    return np.clip(img, 0, 255).astype(np.uint8)


def change_contrast(image, value):
    # Ensure the value is between -255 and +255
    value = np.clip(value, -255, 255)

    # Calculate contrast correction factor F
    F = (259 * (value + 255)) / (255 * (259 - value))

    # Create a copy of the image to modify
    img = image.copy().astype(np.float32)
    img = F * (img - 128) + 128

    # Return the new image, clipped between 0 and 255
    return np.clip(img, 0, 255).astype(np.uint8)


def grayscale(image):
    # Create a copy of the image to modify
    img = image.copy().astype(np.float32)

    # Calculate grayscale value using the weighted average formula
    gray_value = np.dot(img[..., :3], [0.3, 0.59, 0.11])
    gray_value = np.clip(gray_value, 0, 255).astype(np.uint8)

    # Assign the grayscale value to each of the RGB components
    img[..., 0] = gray_value
    img[..., 1] = gray_value
    img[..., 2] = gray_value

    # Return the new grayscale image
    return img


def blur_effect(image):
    # Create a copy of the input image to avoid modifying the original
    blurred_image = image.copy()

    # Define the Gaussian blur kernel
    kernel = np.array([[0.0625, 0.125, 0.0625],
                       [0.125, 0.25, 0.125],
                       [0.0625, 0.125, 0.0625]])

    # Get the dimensions of the image
    rows, cols, _ = image.shape

    # Apply the kernel to each pixel in the image
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            # Initialize the new pixel value
            new_pixel = np.zeros(3)

            # Apply the kernel to the surrounding pixels
            for kr in range(-1, 2):  # kernel row
                for kc in range(-1, 2):  # kernel column
                    # Get the pixel value from the image
                    pixel_value = image[r + kr, c + kc]
                    # Update the new pixel value using the kernel
                    new_pixel += pixel_value * kernel[kr + 1, kc + 1]

            # Assign the new pixel value to the blurred image
            blurred_image[r, c] = np.clip(new_pixel, 0, 255)  # Ensure values are within [0, 255]

    return blurred_image



def edge_detection(image):
    # Create a copy of the input image to avoid modifying the original
    edge_image = image.copy()

    # Define the edge detection kernel
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

    # Get the dimensions of the image
    rows, cols, _ = image.shape

    # Apply the kernel to each pixel in the image
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            # Initialize the new pixel value
            new_pixel = np.zeros(3)

            # Apply the kernel to the surrounding pixels
            for kr in range(-1, 2):  # kernel row
                for kc in range(-1, 2):  # kernel column
                    # Get the pixel value from the image
                    pixel_value = image[r + kr, c + kc]
                    # Update the new pixel value using the kernel
                    new_pixel += pixel_value * kernel[kr + 1, kc + 1]

            # Add 128 to brighten the result
            new_pixel += 128

            # Ensure pixel values are within [0, 255]
            edge_image[r, c] = np.clip(new_pixel, 0, 255)

    # Handle the edges by setting them to zero (or any other value as per your requirement)

    return edge_image.astype(np.uint8)


def embossed(image):
    # Create a copy of the input image to avoid modifying the original
    embossed_image = image.copy()
    # Define the emboss kernel
    kernel = np.array([[-1, -1, 0],
                       [-1, 0, 1],
                       [0, 1, 1]])

    # Get the dimensions of the image
    rows, cols, _ = image.shape

    # Apply the kernel to each pixel in the image
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            new_pixel = np.zeros(3)
            for kr in range(-1, 2):
                for kc in range(-1, 2):
                    pixel_value = image[r + kr, c + kc]
                    new_pixel += pixel_value * kernel[kr + 1, kc + 1]

            new_pixel += 128
            embossed_image[r, c] = np.clip(new_pixel, 0, 255).astype(np.uint8)

    return embossed_image


def apply_effect_on_mask(image, mask, effect_function, *args):
    # Create a copy of the image to modify
    img_copy = image.copy()

    # Apply the effect function on the entire image (will modify only the mask area)
    effect_img = effect_function(img_copy, *args)

    # Replace only the masked areas in the original image
    img_copy[mask == 1] = effect_img[mask == 1]

    return img_copy


def rectangle_select(image, top_left, bottom_right):
    # Create a new mask for the selected rectangle area
    new_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    new_mask[top_left[0]:bottom_right[0]+1,top_left[1]:bottom_right[1]+1] = 1
    return new_mask

def magic_wand_select(image, coords, threshold):
    # Initialize the mask with the same size as the image, set to 0
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Define the stack for flood fill (starting with the given pixel coords)
    stack = deque([coords])

    # Get the target color from the starting pixel coords
    target_color = image[coords[1], coords[0]]

    # Define directions for neighbors (left, right, up, down)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Function to calculate color distance using the given formula
    def color_distance(pixel1, pixel2):
        p1 = pixel1.astype(np.float32)
        p2 = pixel2.astype(np.float32)
        r_mean = (p1[0] + p2[0]) / 2.0
        delta_r = p1[0] - p2[0]
        delta_g = p1[1] - p2[1]
        delta_b = p1[2] - p2[2]
        return np.sqrt((2.0 + r_mean / 256.0) * delta_r * delta_r +
                       4.0 * delta_g * delta_g +
                       (2.0 + (255.0 - r_mean) / 256.0) * delta_b * delta_b)

    # Flood fill algorithm
    while stack:
        current_pixel = stack.pop()
        r, c = current_pixel

        # If the current pixel is out of bounds or already visited, skip it
        if r < 0 or r >= image.shape[0] or c < 0 or c >= image.shape[1] or mask[r, c] == 1:
            continue

        # Calculate the color distance
        if color_distance(image[r, c], target_color) <= threshold:
            # Mark the pixel as part of the selection
            mask[r, c] = 1

            # Add neighboring pixels to the stack
            for dr, dc in directions:
                stack.append((r + dr, c + dc))

    # Return the updated mask
    return mask


def save_image(filename, image):
    img = image.astype(np.uint8)
    mpimg.imsave(filename, img)


def load_image(filename):
    img = mpimg.imread(filename)
    if len(img.shape) == 3 and img.shape[2] == 4:  # if png file with alpha channel
        img = img[:, :, :3]  # Remove the alpha channel
    if img.dtype == np.float32:  # if stored as float in [0,..,1] instead of integers in [0,..,255]
        img = (img * 255).astype(np.uint8)
    mask = np.ones((img.shape[0], img.shape[1]),
                   dtype=np.uint8)  # create a mask full of "1" of the same size of the loaded image
    img = img.astype(np.int32)
    return img, mask


def compute_edge(mask):
    rsize, csize = len(mask), len(mask[0])
    edge = np.zeros((rsize, csize))
    if np.all((mask == 1)): return edge
    for r in range(rsize):
        for c in range(csize):
            if mask[r][c] != 0:
                if r == 0 or c == 0 or r == len(mask) - 1 or c == len(mask[0]) - 1:
                    edge[r][c] = 1
                    continue

                is_edge = False
                for var in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
                    r_temp = r + var[0]
                    c_temp = c + var[1]
                    if 0 <= r_temp < rsize and 0 <= c_temp < csize:
                        if mask[r_temp][c_temp] == 0:
                            is_edge = True
                            break

                if is_edge:
                    edge[r][c] = 1

    return edge


def display_image(image, mask):
    # if using Spyder, please go to "Tools -> Preferences -> IPython console -> Graphics -> Graphics Backend" and select "inline"
    tmp_img = image.copy()
    edge = compute_edge(mask)
    for r in range(len(image)):
        for c in range(len(image[0])):
            if edge[r][c] == 1:
                tmp_img[r][c][0] = 255
                tmp_img[r][c][1] = 0
                tmp_img[r][c][2] = 0

    plt.imshow(tmp_img.astype(np.uint8))
    plt.axis('off')
    plt.show()
    print("Image size is", str(len(image)), "x", str(len(image[0])))


def menu():
    img, mask = np.array([]), np.array([])
    while True:
        if img.size == 0:
            print("\nWhat do you want to do?\ne - exit\nl - load a picture")
        else:
            print("\nWhat do you want to do?")
            print("e - exit\nl - load a picture\ns - save the current picture")
            print("1 - adjust brightness\n2 - adjust contrast\n3 - apply grayscale\n4 - apply blur")
            print("5 - edge detection\n6 - embossed\n7 - rectangle select\n8 - magic wand select")
        choice = input("Your choice: ")

        if choice == 'e':
            break
        elif choice == 'l':
            filename = input("Enter filename to load: ")
            img, mask = load_image(filename)
        elif img.size == 0:
            print("No image loaded. Please load an image first.")
        elif choice == 's':
            filename = input("Enter filename to save: ")
            save_image(filename, img)
        elif choice == '1':
            value = int(input("Enter brightness value (-255 to 255): "))
            img = apply_effect_on_mask(img, mask, change_brightness, value)
        elif choice == '2':
            value = int(input("Enter contrast value (-255 to 255): "))
            img = apply_effect_on_mask(img, mask, change_contrast, value)
        elif choice == '3':
            img = apply_effect_on_mask(img, mask, grayscale)
        elif choice == '4':
            img = apply_effect_on_mask(img, mask, blur_effect)
        elif choice == '5':
            img = apply_effect_on_mask(img, mask, edge_detection)
        elif choice == '6':
            img = apply_effect_on_mask(img, mask, embossed)
        elif choice == '7':
            x1, y1 = map(int, input("Enter top-left corner (x y): ").split())
            x2, y2 = map(int, input("Enter bottom-right corner (x y): ").split())
            mask = rectangle_select(mask, (x1, y1), (x2, y2))
        elif choice == '8':
            x, y = map(int, input("Enter pixel position (x y): ").split())
            threshold = int(input("Enter threshold value: "))
            mask = magic_wand_select(img, (x, y), threshold)
        else:
            print("Invalid choice. Please try again.")
        if img.size != 0:
            display_image(img, mask)


if __name__ == "__main__":
    menu()