from __future__ import annotations
import cv2
import numpy as np
import torch
from typing import Sequence

SYNAPSE_COLORMAP = {
    1: [0, 0, 255],
    2: [0, 255, 0],
    3: [255, 0, 0],
    4: [0, 255, 255],
    5: [255, 0, 255],
    6: [255, 255, 0],
    7: [63, 208, 244],
    8: [241, 240, 234],
}

ACDC_COLORMAP = {
    1: [0, 0, 255],
    2: [0, 255, 0],
    3: [255, 0, 0],
}

class2colormap = {
    9: SYNAPSE_COLORMAP,
    4: ACDC_COLORMAP
}



def make_rgb_darker(color: Sequence[int, int, int], percentage: float = 0.5) -> tuple[int, int, int]:
    def _dark(c: int) -> int:
        return int(max(0., c - c * percentage))
    r, g, b = color
    return _dark(r), _dark(g), _dark(b)

def is_grayscale(image: np.ndarray | torch.Tensor) -> bool:
    return not (len(image.shape) > 2 and image.shape[2] > 1)

def save_x_y_tensor(x: torch.Tensor, y: torch.Tensor, colormap: dict, out: str) -> None:
    """
    input ndarray shape:
        x: [h, w, [c]]; y: [h, w];
    """
    x = x if is_grayscale(x) else x.permute(1, 2, 0)
    x = x.detach().cpu().numpy().astype(np.uint8)
    y = y.detach().cpu().numpy().astype(np.uint8)
    save_x_y(x, y, colormap, out)


def save_x_y(x: np.ndarray, y: np.ndarray, colormap: dict, out: str) -> None:
    """
    input ndarray shape:
        x: [h, w, [c]]; y: [h, w];
    colormap: {class_id: [R, G, B]}
    """
    assert all([x.dtype == np.uint8, y.dtype == np.uint8])

    if is_grayscale(x):
        x_bgr_canvas = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    elif x.shape[2] == 1 and len(x.shape) == 3: # (H, W, 1)
        x_bgr_canvas = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    else:
        x_bgr_canvas = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


    for class_id, color_rgb in colormap.items():
        mask_for_class_i = (y == class_id)

        color_bgr_for_opencv = (color_rgb[2], color_rgb[1], color_rgb[0])

        x_bgr_canvas[mask_for_class_i] = color_bgr_for_opencv

    cv2.imwrite(out, x_bgr_canvas)

def save_x_y_hat(x: np.ndarray, y: np.ndarray, y_hat: np.ndarray, colormap: dict, out: str) -> None:
    assert all([x.dtype == np.uint8, y.dtype == np.uint8, y_hat.dtype == np.uint8])

    if is_grayscale(x):
        x_bgr_canvas = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    elif x.shape[2] == 1 and len(x.shape) == 3:
        x_bgr_canvas = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    else:
        x_bgr_canvas = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

    for class_id, color_rgb in colormap.items():
        mask_for_pred_class_i = (y_hat == class_id)
        color_bgr_for_pred = (color_rgb[2], color_rgb[1], color_rgb[0]) # RGB to BGR
        x_bgr_canvas[mask_for_pred_class_i] = color_bgr_for_pred

        # contours, _ = cv2.findContours(np.array(y == class_id).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # darker_color_rgb = make_rgb_darker(color_rgb, percentage=0.5)
        # darker_color_bgr_for_contour = (darker_color_rgb[2], darker_color_rgb[1], darker_color_rgb[0]) # RGB to BGR
        # cv2.drawContours(x_bgr_canvas, contours, -1, darker_color_bgr_for_contour, thickness=2) # 这里的thickness可以调整

    cv2.imwrite(out, x_bgr_canvas)
