import argparse
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="sample.jpg")
parser.add_argument("--colors", type=int, default=16)
args = parser.parse_args()

image = Image.open(args.input).convert("RGB")
arr = np.asarray(image)
pixels = arr.reshape(-1, 3).astype(np.float32)

model = KMeans(n_clusters=args.colors, random_state=42, n_init=10)
labels = model.fit_predict(pixels)
centers = np.clip(model.cluster_centers_, 0, 255).astype(np.uint8)
compressed = centers[labels].reshape(arr.shape)

out_path = f"compressed_{args.colors}_colors.png"
Image.fromarray(compressed).save(out_path)

mse = np.mean((arr.astype(np.float32) - compressed.astype(np.float32)) ** 2)
print(f"Image size: {image.size}")
print(f"Colors: {args.colors}")
print(f"MSE: {mse:.2f}")
print(f"Saved: {out_path}")

plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.imshow(arr); plt.title("Original"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(compressed); plt.title(f"{args.colors} Colors"); plt.axis("off")
plt.tight_layout(); plt.savefig("comparison.png", dpi=150); plt.show()
