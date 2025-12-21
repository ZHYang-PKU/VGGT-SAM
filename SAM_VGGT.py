import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from points_transform import *




def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

###### Segment the first image by SAM

device = "cuda" if torch.cuda.is_available() else "cpu"
## load images from multiple perspectives
image_names = ["examples/kitchen/images/00.png", "examples/kitchen/images/01.png", "examples/kitchen/images/02.png"]
image_list = [Image.open(name) for name in image_names]
images_np_list = [np.array(img.convert('RGB')) for img in image_list]
images_np = np.stack(images_np_list, axis=0)
print(f"Shape of images_np: {images_np.shape}")



"""
images = load_and_preprocess_images(image_names).to(device)
print(f"Shape of images: {images.shape}")
print(f"Preprocessed images shape: {images.shape}")
print(f"Type of images: {images.dtype}")
"""


### load the SAM model
print("Initializing and loading SAM model...")
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

images_np = np.array(images_np)

predictor.set_image(images_np[0,:,:,:])  # image embedding


prompt_point = np.array([[440, 140]])      # each point : ( [0, W-1] , [0, H-1] )
#prompt_point = np.array([[778, 519]])     # bottom right point in the original image
prompt_label = np.array([1])


plt.figure(figsize=(10,10))
plt.imshow(images_np[0,:,:,:])
show_points(prompt_point, prompt_label, plt.gca())
plt.axis('on')
plt.show()


masks, scores, logits = predictor.predict(
    point_coords=prompt_point,
    point_labels=prompt_label,
    multimask_output=True,
)
#print(f"Shape of masks: {masks.shape}")


for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(images_np[0,:,:,:])
    show_mask(mask, plt.gca())
    show_points(prompt_point, prompt_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()   




###### VGGT tracker module
## Using the pre-trained VGGT to track the points from the first image

print("Initializing and loading VGGT model...")
# model = VGGT.from_pretrained("facebook/VGGT-1B")  # another way to load the model
model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
model.eval()
model = model.to(device)


images = load_and_preprocess_images(image_names).to(device)
#print(f"Preprocessed images shape: {images.shape}")
#print(f"Type of images: {images.dtype}")


## obtain the query points from the predicted mask
max_score_idx = np.argmax(scores)
query_mask = masks[max_score_idx]
rows, cols = np.where(query_mask == True)
#print(rows)
#print(cols)
query_points = np.stack((cols, rows), axis=1)  # [N, 2]
#print(f"Shape of query_points: {query_points.shape}")

preprocessed_tensor, transform_info = preprocess_image_with_tracking(image_names[0], 'crop')
transformed_query_points = original_to_preprocessed(query_points, transform_info)
#print(f"Shape of transformed_query_points: {transformed_query_points.shape}")


transformed_query_points = torch.from_numpy(transformed_query_points).to(torch.float32).to(device)
#print(f"Shape of query_points: {transformed_query_points.shape}")
#print(f"Type of query_points: {transformed_query_points.dtype}")


"""
#### Visualize the query points
images_np = np.array(images.cpu())
print(f"Images shape after converting to numpy: {images_np.shape}")

image_to_display = images_np[0]  # shape: (3, H, W)
image_to_display = np.transpose(image_to_display, (1, 2, 0))  # 调整为 (H, W, 3)
print(f"Image shape after transpose: {image_to_display.shape}")

plt.figure(figsize=(10,10))
plt.imshow(image_to_display)
show_points(query_points[0], np.array([1]), plt.gca())
plt.axis('on')
plt.show()
"""


# Run inference
print("Running inference...")
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images, transformed_query_points)



# Obtain the tracked points in all images
print("Computing tracked points in all perspectives...")
tracked_points = predictions["track"]   # [1, S(num of images), N(num of query points), 2]
tracked_points_confidence = predictions["conf"]
#print(f"Shape of tracked points: {tracked_points.shape}")

## Visualize the tracked masks in each perspective
tracked_points = tracked_points.squeeze(0)  # Shape: [S, N, 2]
S, N, _ = tracked_points.shape
_, _, H, W = images.shape
predicted_masks = torch.zeros((S, H, W), dtype=torch.bool)
for s in range(S):
    # obtain the points to be tracked
    points = tracked_points[s]  # [N, 2]

    # int
    x_coords = torch.round(points[:, 0]).long()  # column index（width）
    y_coords = torch.round(points[:, 1]).long()  # 行row index（height）

    # make sure the points are in the right range
    valid_mask = (x_coords >= 0) & (x_coords < W) & (y_coords >= 0) & (y_coords < H)
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]

    # Mask these points
    # mask index:  [y, x]
    predicted_masks[s, y_coords, x_coords] = True




images_vggt_np = np.array(images.cpu())
#print(f"Shape of predicted masks: {predicted_masks.shape}")
#print(f"Shape of images_vggt_np: {images_vggt_np.shape}")

for i in range(images.shape[0]):
    # visualize the i-th image and its mask
    image_to_display = images_vggt_np[i]  # shape: (3, H, W)
    image_to_display = np.transpose(image_to_display, (1, 2, 0))  #  (H, W, 3)
    plt.figure(figsize=(10,10))
    plt.imshow(image_to_display)
    show_mask(predicted_masks[i], plt.gca())
    plt.title(f"Mask {i+1}", fontsize=18)
    plt.axis('off')
    plt.show()