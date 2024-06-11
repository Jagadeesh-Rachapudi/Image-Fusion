import pywt
import cv2
import numpy as np

def fuseCoeff(coeffs1, coeffs2, method='mean'):
    """Fuses wavelet coefficients using the specified method."""
    coeffs2 = coeffs2.T if coeffs1.shape != coeffs2.shape else coeffs2 

    if method == 'mean':
        return (coeffs1 + coeffs2) / 2
    elif method == 'min':
        return np.minimum(coeffs1, coeffs2)
    elif method == 'max':
        return np.maximum(coeffs1, coeffs2)
    else:
        raise ValueError("Invalid fusion method. Choose from 'mean', 'min', or 'max'.")

def fuse_images(image1_path, image2_path, method='mean', wavelet='db1', saturation_factor=2.0, value_factor=1.1, crop_factor=0.015):
    """Fuses two color images using wavelet decomposition, enhances colors, and crops edges based on a factor."""

    # Load images in color (BGR format)
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    # Get the original dimensions of both images
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape

    # Calculate the aspect ratios
    aspect_ratio1 = w1 / h1
    aspect_ratio2 = w2 / h2

    # Determine the new dimensions based on the aspect ratios
    if aspect_ratio1 > aspect_ratio2:
        new_h2 = h1
        new_w2 = int(new_h2 * aspect_ratio2)
    else:
        new_w2 = w1
        new_h2 = int(new_w2 / aspect_ratio2)

    # Resize while maintaining aspect ratio and minimizing the white gap
    img2 = cv2.resize(img2, (new_w2, new_h2))

    # Now, you need to determine which dimension needs padding to match the other image
    if new_h2 < h1:
        # Pad the height of img2
        padding_top = (h1 - new_h2) // 2
        padding_bottom = h1 - new_h2 - padding_top
        img2 = cv2.copyMakeBorder(img2, padding_top, padding_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    elif new_w2 < w1:
        # Pad the width of img2
        padding_left = (w1 - new_w2) // 2
        padding_right = w1 - new_w2 - padding_left
        img2 = cv2.copyMakeBorder(img2, 0, 0, padding_left, padding_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Calculate the number of pixels to crop based on the crop factor
    crop_pixels1 = round(w1 * crop_factor)
    crop_pixels2 = round(w2 * crop_factor)

    # Crop out a few pixels from each side of both images to reduce potential gaps
    img1 = img1[:, :-crop_pixels1]
    img2 = img2[:, crop_pixels2:]

    # Split images into color channels (B, G, R)
    img1_channels = cv2.split(img1)
    img2_channels = cv2.split(img2)

    # Fuse each color channel separately
    fused_channels = []
    for c1, c2 in zip(img1_channels, img2_channels):
        # Wavelet decomposition
        coeffs1 = pywt.wavedec2(c1, wavelet)
        coeffs2 = pywt.wavedec2(c2, wavelet)

        # Fusion of coefficients at each level
        fused_coeffs = []
        for i, (level1, level2) in enumerate(zip(coeffs1, coeffs2)):
            if i == 0:  # Approximation coefficients
                fused_coeffs.append(fuseCoeff(level1, level2, method))
            else:       # Detail coefficients
                fused_coeffs.append(tuple(fuseCoeff(d1, d2, method) for d1, d2 in zip(level1, level2)))

        # Wavelet reconstruction
        fused_channel = pywt.waverec2(fused_coeffs, wavelet)
        fused_channel = np.clip(fused_channel, 0, 255).astype(np.uint8)
        fused_channels.append(fused_channel)

    # Merge fused channels back into a color image
    fused_image = cv2.merge(fused_channels)
    
    # Enhance colors
    hsv_image = cv2.cvtColor(fused_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    s = np.clip(s * saturation_factor, 0, 255).astype(np.uint8)
    v = np.clip(v * value_factor, 0, 255).astype(np.uint8)
    enhanced_hsv = cv2.merge([h, s, v])
    fused_image = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    return fused_image

# Main execution
image1_path = 'Car_Front.png'
image2_path = 'Car_Back.png'
fused_image = fuse_images(image1_path, image2_path, method='mean', saturation_factor=2.0, crop_factor=0.015)  # Adjust crop_factor as needed

# Display or save
cv2.imshow("Fused Image", fused_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("fused_output_color.png", fused_image)
