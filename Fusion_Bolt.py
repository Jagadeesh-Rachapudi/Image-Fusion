import cv2
import numpy as np

def fuse_bolt_images(top_image_path, bottom_image_path, threshold=127):
    """Fuses top and bottom bolt images with precise alignment and masking."""
    
    # Load images
    top_img = cv2.imread(top_image_path)
    bottom_img = cv2.imread(bottom_image_path)
    
    # Convert to grayscale
    gray_top = cv2.cvtColor(top_img, cv2.COLOR_BGR2GRAY)
    gray_bottom = cv2.cvtColor(bottom_img, cv2.COLOR_BGR2GRAY)
    
    # Thresholding to isolate the bolt (assumes white bolt on darker background)
    _, mask_top = cv2.threshold(gray_top, threshold, 255, cv2.THRESH_BINARY_INV)
    _, mask_bottom = cv2.threshold(gray_bottom, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Find the bottom-most white pixel in the top image and the top-most white pixel in the bottom image
    top_bottom_y = np.max(np.where(mask_top > 0)[0])
    bottom_top_y = np.min(np.where(mask_bottom > 0)[0])

    # Crop images based on the found pixels
    cropped_top = top_img[:top_bottom_y, :]
    cropped_bottom = bottom_img[bottom_top_y:, :]

    # Combine the images 
    fused_image = np.vstack((cropped_top, cropped_bottom))

    return fused_image

# Main execution
top_image_path = 'Bottom.png'
bottom_image_path = 'Top.png'
fused_image = fuse_bolt_images(top_image_path, bottom_image_path)

# Display and save
cv2.imshow("Fused Image", fused_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("fused_bolt.png", fused_image)
