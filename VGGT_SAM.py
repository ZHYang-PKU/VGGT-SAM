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



device = "cuda" if torch.cuda.is_available() else "cpu"
## load images from multiple perspectives
image_names = ["examples/kitchen/images/00.png", "examples/kitchen/images/01.png", "examples/kitchen/images/02.png"]


###### VGGT tracker module
## Using the pre-trained VGGT to track the prompt points from the first image
print("Initializing and loading VGGT model...")
# model = VGGT.from_pretrained("facebook/VGGT-1B")  # another way to load the model
model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
model.eval()
model = model.to(device)


images = load_and_preprocess_images(image_names).to(device)  # [S, 3, 350, 518]
#print(f"Preprocessed images shape: {images.shape}")
#print(f"Type of images: {images.dtype}")


preprocessed_tensor, transform_info = preprocess_image_with_tracking(image_names[0], 'crop')
prompt_point = np.array([[440, 140]])      # each point : ( [0, W-1] , [0, H-1] )
prompt_point_transformed = original_to_preprocessed(prompt_point, transform_info)
#print(prompt_point_transformed)
#print(f"Shape of prompt_point_transformed: {prompt_point_transformed.shape}")


prompt_label = np.array([1])
prompt_point_transformed = torch.from_numpy(prompt_point_transformed).to(torch.float32).to(device)
prompt_point_transformed = prompt_point_transformed.unsqueeze(0)
#print(f"Shape of prompt_point_transformed: {prompt_point_transformed.shape}")


print("Running VGGT inference...")
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images, prompt_point_transformed)


# Obtain the tracked points in all images
print("Computing tracked points in all perspectives...")
tracked_points_transformed = predictions["track"]   # [1, S(num of images), N(num of query points), 2]
tracked_points_confidence = predictions["conf"]
#print(f"Shape of tracked points: {tracked_points_transformed.shape}")

tracked_points_transformed = np.array(tracked_points_transformed.cpu())

S = tracked_points_transformed.shape[1]
recovered_points = np.zeros_like(tracked_points_transformed)
for s in range(S):
    recovered_points[:, s, :, :] = preprocessed_to_original(tracked_points_transformed.squeeze(0)[s], transform_info)
#print(f"Shape of recovered_points: {recovered_points.shape}")
#print(recovered_points)


image_list = [Image.open(name) for name in image_names]
images_np_list = [np.array(img.convert('RGB')) for img in image_list]
images_np = np.stack(images_np_list, axis=0)
#print(f"Shape of images_np: {images_np.shape}")


### load the SAM model
print("Initializing and loading SAM model...")
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

images_np = np.array(images_np)  # 转换为numpy数组
S, H, W, _ = images_np.shape
predicted_masks = torch.zeros((S, H, W), dtype=torch.bool)
for s in range(S):
    predictor.set_image(images_np[s,:,:,:])  # image embedding
    masks, scores, logits = predictor.predict(
        point_coords=recovered_points[:, s, :, :].squeeze(0),
        point_labels=prompt_label,
        multimask_output=True,
    )
    max_score_idx = np.argmax(scores)
    predicted_masks[s] = torch.tensor(masks[max_score_idx])

#print(f"Shape of predicted masks: {predicted_masks.shape}")
#print(f"Shape of images_np: {images_np.shape}")

for i in range(images.shape[0]):
    # visualize the i-th image and its mask
    image_to_display = images_np[i]  # shape: (3, H, W)
    plt.figure(figsize=(10,10))
    plt.imshow(image_to_display)
    show_mask(predicted_masks[i], plt.gca())
    plt.title(f"Mask {i+1}", fontsize=18)
    plt.axis('off')
    plt.show()













