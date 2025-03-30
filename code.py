import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = r"/Users/yadyneshsonale/Desktop/2.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError("Could not read the image. Check the file path.")
image = cv2.resize(image, (600, 800))
h, w = image.shape[:2]



# Smooth gradient for blending
gradient = np.linspace(0, 1, h, dtype=np.float32).reshape(h, 1, 1)
gradient_inv = 1 - gradient

# Glow Effect
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hsv[..., 1] = np.clip(hsv[..., 1] * 1.5, 0, 255)
hsv[..., 2] = np.clip(hsv[..., 2] * 1.2, 0, 255)
cyberpunk = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Vintage Film Look
vintage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
vintage = cv2.cvtColor(vintage, cv2.COLOR_GRAY2BGR)
sepia = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
vintage = cv2.transform(vintage, sepia)
vintage = np.clip(vintage, 0, 255).astype(np.uint8)

# Detect bright areas
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
bright_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)[1]

def adaptive_pixelation(img, mask, min_size=8, max_size=20):
    result = img.copy()
    for y in range(0, h, min_size):
        for x in range(0, w, min_size):
            region = mask[y:y+min_size, x:x+min_size]
            brightness = np.mean(region)
            pixel_size = int(min_size + (brightness / 255) * (max_size - min_size))
            pixel_size = max(min_size, min(pixel_size, max_size))
            
            y_end, x_end = min(y + pixel_size, h), min(x + pixel_size, w)
            small = cv2.resize(img[y:y_end, x:x_end], (1, 1), interpolation=cv2.INTER_LINEAR)
            large = cv2.resize(small, (x_end - x, y_end - y), interpolation=cv2.INTER_NEAREST)
            result[y:y_end, x:x_end] = large
    return result

# Apply pixelation
pixelated = adaptive_pixelation(image, bright_mask)

# Enhance sparkle effect
sparkle = cv2.GaussianBlur(bright_mask, (5, 5), 0)
sparkle = cv2.cvtColor(sparkle, cv2.COLOR_GRAY2BGR)
sparkle_intensity = np.where(sparkle > 200, 255, sparkle * 2).astype(np.uint8)

# Blend effects
blended1 = (cyberpunk * gradient + cyberpunk * gradient_inv).astype(np.uint8)
blended2 = (blended1 * gradient + vintage * gradient_inv).astype(np.uint8)
final_image = (blended2 * gradient + pixelated * gradient_inv).astype(np.uint8)

# Final Cyberpunk Glow
hsv_final = cv2.cvtColor(final_image, cv2.COLOR_BGR2HSV)
hsv_final[..., 1] = np.clip(hsv_final[..., 1] * 1.5, 0, 255)
hsv_final[..., 2] = np.clip(hsv_final[..., 2] * 1.2, 0, 255)
final_cyberpunk = cv2.cvtColor(hsv_final, cv2.COLOR_HSV2BGR)

# sparkle effect
final_cyberpunk = cv2.addWeighted(final_cyberpunk, 1, sparkle_intensity, 0.5, 0)


plt.figure(figsize=(8, 10))
plt.imshow(cv2.cvtColor(final_cyberpunk, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()