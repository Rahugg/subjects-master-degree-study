import numpy as np
from PIL import Image

def load_and_crop(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    I = arr[:64, :64, :]
    return I

def center_pixel(I):
    return I[32, 32, :]

def red_channel(I):
    return I[:, :, 0]

def invert_image(I):
    return 255 - I

def to_grayscale(I):
    R, G, B = I[:,:,0].astype(float), I[:,:,1].astype(float), I[:,:,2].astype(float)
    gray = 0.299*R + 0.587*G + 0.114*B
    gray_norm = (gray - gray.min()) / (gray.max() - gray.min() + 1e-9)
    avg_brightness = float(gray_norm.mean())
    return gray_norm, avg_brightness

def flip_rotate(I):
    flip = I[:, ::-1, :]
    rot = np.rot90(I, k=1, axes=(0,1))
    return flip, rot

def downscale_32(I):
    H, W, C = I.shape
    I = I[:H - H%2, :W - W%2, :]
    I2 = I.reshape(I.shape[0]//2, 2, I.shape[1]//2, 2, C).mean(axis=(1,3)).astype(I.dtype)
    return I2

def demo(path="image.jpg"):
    I = load_and_crop(path)
    c = center_pixel(I)
    R = red_channel(I)
    I_inv = invert_image(I)
    gray, avg_b = to_grayscale(I)
    flip, rot = flip_rotate(I)
    I32 = downscale_32(I)
    print("C(32,32):", c.tolist())
    print("Red shape:", R.shape)
    print("Inverted shape:", I_inv.shape)
    print("Gray shape:", gray.shape, "Avg brightness:", round(avg_b, 4))
    print("Flip shape:", flip.shape, "Rot shape:", rot.shape)
    print("Downscaled shape:", I32.shape)

if __name__ == "__main__":
    demo()
