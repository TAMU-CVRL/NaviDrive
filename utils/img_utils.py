import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from nuscenes.utils.geometry_utils import view_points

def resize_with_aspect_ratio(img, size=256):
    """Resize image while maintaining aspect ratio and then center crop to (size, size).
    Args:
        img: PIL Image, numpy array, or torch tensor
        size: Desired output size (size x size)
    Returns:
        Resized and center-cropped PIL Image
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        img = Image.fromarray(img)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    elif not isinstance(img, Image.Image):
        raise TypeError(f"Unsupported image type: {type(img)}")
    
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    # Center crop
    new_img = Image.new("RGB", (size, size))
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    new_img.paste(img_resized, (left, top))
    return new_img

def resize_long_edge(image, long_edge=256):
    """
    Resize image so that the longer edge matches long_edge, maintaining aspect ratio.
    Args:
        image: PIL Image or numpy array
        long_edge: Desired size of the longer edge
    Returns:
        Resized PIL Image
    """
    # Convert numpy → PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    w, h = image.size
    if w >= h:
        new_w = long_edge
        new_h = int(h * long_edge / w)
    else:
        new_h = long_edge
        new_w = int(w * long_edge / h)

    # Use high-quality downsampling
    image = image.resize((new_w, new_h), Image.BICUBIC)
    return image

image_transform = transforms.Compose([
    transforms.Lambda(lambda img: resize_with_aspect_ratio(img, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),  # CLIP mean
        std=(0.26862954, 0.26130258, 0.27577711)   # CLIP std
    )
])

def inverse_image_transform(img_tensor):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)

    return transforms.ToPILImage()(img)

# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py, render_annotation
def crop_annotation(nusc, ann_token, sample_record, margin=5, min_ratio=0.8):
    assert 'LIDAR_TOP' in sample_record['data'].keys(), 'Error: No LIDAR_TOP in data, unable to render.'

    # Figure out which camera the object is fully visible in (this may return nothing).
    boxes, cam = [], []
    cams = [key for key in sample_record['data'].keys() if 'CAM' in key]
    for cam in cams:
        _, boxes, _ = nusc.get_sample_data(sample_record['data'][cam], box_vis_level=1, selected_anntokens=[ann_token])
        if len(boxes) > 0:
            break  # We found an image that matches. Let's abort.
    if len(boxes) == 0:
        return None  # skip if not visible

    cam_token = sample_record['data'][cam]

    # Plot CAMERA view.
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(cam_token, selected_anntokens=[ann_token])
    im = Image.open(data_path)

    # Crop the box from the image
    box = boxes[0]
    corners = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :]
    x_min, y_min = corners.min(axis=1)
    x_max, y_max = corners.max(axis=1)
    
    # Calculate the area inside the image, if too small, skip this box
    x_min_clip = max(int(x_min), 0)
    y_min_clip = max(int(y_min), 0)
    x_max_clip = min(int(x_max), im.width)
    y_max_clip = min(int(y_max), im.height)

    area_box = max(int(x_max - x_min), 1) * max(int(y_max - y_min), 1)
    area_clipped = max(x_max_clip - x_min_clip, 0) * max(y_max_clip - y_min_clip, 0)
    ratio_inside = area_clipped / area_box

    if ratio_inside < min_ratio:
        return None

    # Add margin and ensure within image bounds
    x_min_final = max(int(x_min) - margin, 0)
    y_min_final = max(int(y_min) - margin, 0)
    x_max_final = min(int(x_max) + margin, im.width)
    y_max_final = min(int(y_max) + margin, im.height)

    cropped_im = im.crop((x_min_final, y_min_final, x_max_final, y_max_final)) 
    
    return cropped_im
