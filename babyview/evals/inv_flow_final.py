"""
INV Script for generating step-wise factual predictions from a video using a CCWM model,
and tracking how an initial perturbation “flows” from one frame to the next,
with multiple rollouts at each step, *averaging the diff maps before computing argmax*.
Now also generates an image for each frame in the sequence with a red circle at the flow location.


conda activate dinov3
cd /ccn2/u/khaiaw/Code/baselines/dinov3/

PERTURB_STD=2
ZOOM_ITERS=4
python /ccn2/u/khaiaw/Code/baselines/dinov3/babyview/evals/inv_flow_final.py \
    --out_dir "/ccn2/u/khaiaw/Code/ccwm/viz/flow_counterfactuals/full_tapvid_davis_first/std_${PERTURB_STD}_zoom_${ZOOM_ITERS}" \
    --num_rollouts 1 \
    --perturb_std $PERTURB_STD \
    --zoom_iters $ZOOM_ITERS \
    --mask_ratio 0.9 \
    --no_blur \
    --model_type dinov3 \
    --flat_points_start_idx 0 \
    --num_flat_points_to_process 100 \
    --data_path=/ccn2/u/ksimon12/flow/miniflow/full_tapvid_davis_first/dataset.json \
    --model_name awwkl/dinov3-vitl-babyview-gradaccum1 \
    --model_name facebook/dinov3-vitl16-pretrain-lvd1689m \
    --log_interval=10 \
    --viz_all \
    --squish
"""

import copy
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

import socket

import yaml

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import tqdm
import pickle 
from PIL import Image, ImageOps
import json
# from timm.data import constants
from glob import glob
import sys
# from einops import rearrange
import time 
from datetime import datetime, timedelta
# import moviepy.editor as mpy
import logging 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def fig_to_array(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.get_renderer().buffer_rgba(), dtype=np.uint8)
    img = buf.reshape((h, w, 4))[:, :, :3]  # Drop alpha channel if not needed
    return img


def get_args(extend=False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name', 
        type=str,
        default='CCWM7B_RGB_causal/model_00415000.pt',
        help='CCWM Model Name (from gcloud)'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['dinov3', 'vjepa2', 'ccwm', 'cwm', 'psi', 'mask_psi'],
        default='ccwm',
    )
    parser.add_argument(
        '--model_size',
        type=str,
        choices=['vit_large', 'vit_huge', 'vit_giant_xformers'],
        default='vit_large',
        help='Size of the model (from timm)'
    )
    parser.add_argument(
        '--quantizer_name', 
        type=str, 
        default='LPQ_ps-4_vs-65536_nc-1_eb-1_db-11-medium-all_data/model_best.pt', 
        help='Quantizer Model Name (from gcloud)',
    )
    parser.add_argument(
        '--video_path', 
        type=str, 
        # default="/ccn2a/dataset/davis-videos/car-roundabout.mp4",
        default="/ccn2/u/klemenk/Code/ccwm/examples/camel-walk.mp4",
        # default="/ccn2a/dataset/davis-videos/camel.mp4",
        help='Path to the video file'
    )
    parser.add_argument(
        '--video_frame_skip',
        type=int,
        default=5,
        help='Number of frames to skip when loading the video (passed to video_to_frames).'
    )
    parser.add_argument(
        '--out_dir', 
        type=str, 
        default="/ccn2/u/klemenk/Code/ccwm/viz/flow_counterfactuals", 
        help='Path to output directory'
    )
    parser.add_argument(
        '--start_frame',
        type=int,
        default=240,
        help='Index of frame 0 in video'
    )
    parser.add_argument(
        '--num_future_frames',
        type=int,
        default=8,
        help='Number of consecutive frames to predict and track flow.'
    )
    parser.add_argument(
        '--mask_ratio', 
        type=float, 
        default=0.9, 
        help='Ratio of masked patches in the predicted frame'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to run on'
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1110,
        help="Base random seed"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for sampling"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1000,
        help="Top-k for sampling"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling"
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=5,
        help="Number of rollouts (different seeds) **per step** to generate"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="seq2par",
        help="Mode of prediction (seq2par, parallel, patchwise_parallel, etc.)"
    )    
    parser.add_argument(
        '--num_seq_patches',
        type=int,
        default=0,
        help='Number of sequence patches (for patchwise modes)'
    )
    parser.add_argument(
        '--perturb_x',
        type=float,
        default=200,
        help='X-coordinate for the center of the perturbation on the first step.'
    )
    parser.add_argument(
        '--perturb_y',
        type=float,
        default=120,
        help='Y-coordinate for the center of the perturbation on the first step.'
    )
    parser.add_argument(
        '--perturb_std',
        type=float,
        default=2.0,
        help='Standard deviation for the Gaussian perturbation.'
    )
    parser.add_argument(
        '--perturb_magnifier',
        type=float,
        default=1.0,
        help='Multiply perturb_std by this factor every zoom iteration'
    )

    # ADD DAVIS ARGS
    parser.add_argument(
        '--davis_pkl', 
        default=None,
        help='To run on DAVIS examples.'
    )

    parser.add_argument(
        '--davis_vidx',
        default=0,
        type=int,
        help='Which DAVIS video to run.'
    )
    
    parser.add_argument(
        '--flat_points_start_idx',
        default=0,
        type=int,
        help='Which point to start from within the list of flat points.'
    )
    parser.add_argument(
        '--num_flat_points_to_process',
        default=200,
        type=int,
        help='How many points to process from the list of flat points.'
    )

    parser.add_argument(
        '--prev_cache_root_dir',
        type=str
    )

    parser.add_argument(
        '--viz_all',
        action='store_true',
        help='Visualize Everything'
    )

    parser.add_argument(
        '--chained',
        action='store_true',
        help='Run Chained'
    )

    parser.add_argument(
        '--squish',
        action='store_true',
        help='Squish Resize'
    )

    parser.add_argument(
        '--occ_thresh',
        type=float,
        default=0.4
    )

    parser.add_argument(
        '--saved_perts',
        type=str,
        default=None
    )

    parser.add_argument(
        '--select_points_demo',
        action='store_true',
        help="Check for valid points"
    )

    parser.add_argument(
        '--larger_select_points_demo',
        action='store_true',
        help="Check for valid points"
    )

    parser.add_argument(
        '--put_wb_last',
        action='store_true',
        help="Re-order input sequence to put WB patches last"
    )

    parser.add_argument(
        '--clump_mask_size',
        type=int, 
        default=0,
        help="Clump unmask indices together to form larger mask"
    )

    parser.add_argument(
        '--num_t',
        type=int, 
        default=-1,
        help="Num Timesteps"
    )

    parser.add_argument(
        '--max_t',
        type=int, 
        default=100000,
        help="Max timesteps"
    )

    parser.add_argument(
        '--fixed_timesteps',
        action='store_true',
        help="Run on 0->(5,10,15) fixed"
    )

    parser.add_argument(
        '--kl_all',
        action='store_true',
        help="KL using all four tokens in patch"
    )

    parser.add_argument(
        '--zoom_iters', 
        type=int,
        default=4,
        help="Zoom iterations"
    )

    parser.add_argument(
        '--offset_center', 
        action='store_true',
        help="When scaling KL predictions, point to center instead of top left"
    )

    parser.add_argument(
        '--no_blur',
        action="store_true",
        help="Remove diff map blurring"
    )

    parser.add_argument(
        '--zoom_stride',
        default=1,
        type=int,
        help="Go straight to N x zoom."
    )

    parser.add_argument(
        '--data_path',
        type=str,
        help="Path to miniflow dataset."
    )

    parser.add_argument(
        '--no_viz', 
        action='store_true'
    )

    parser.add_argument(
        '--pert_cache_path',
        type=str
    )

    parser.add_argument(
        '--viz_interval',
        type=int,
        default=10
    )

    parser.add_argument(
        '--log_interval',
        type=int,
        default=5
    )

    parser.add_argument(
        '--first_logits_only',
        action='store_true'
    )

    parser.add_argument(
        '--compile',
        action='store_true'
    )

    parser.add_argument(
        '--frame_gap',
        type=int
    )
    
    parser.add_argument('--vjepa2_config_path', type=str, default='evals/optical_flow/config.yaml')

    if extend: 
        return parser

    return parser.parse_args()

def zoom_into_frame(frame, center_x, center_y, reduce_pct=0.25, zoom_stride=1, rect=False, img_size=256):
    """
    Zoom into frame for given center. If zoom_stride is set to, e.g. N, 
    it will be equivalent to zooming in Nx times at a stride of 1. 
    """ 
    h, w, _ = frame.shape
    
    keep = (1 - reduce_pct) ** zoom_stride

    rh, rw = int(h * keep), int(w * keep)

    if not rect:
        # assert h == w == img_size, f"Expected square crop of {(img_size, img_size)}, got {frame.shape}"
        assert h == w
    else:
        rh = rw = min(h, w)

    left = int(max(center_x - rw//2, 0))
    right = min(left + rw, w)
    left = right - rw

    top = int(max(center_y - rh//2, 0))
    bottom = min(top + rh, h)
    top = bottom - rh
    
    frame = frame[top:bottom, left:right]

    assert frame.shape[:-1] == (rh, rw), f"Expected {(rh, rw)} after crop, got {frame.shape[:-1]}"

    frame = np.array(Image.fromarray(frame).resize((img_size, img_size)))
    
    wscale, hscale = (img_size / rw), (img_size / rh)
    return frame, left, top, wscale, hscale



def recover_og_coordinates(gt_query_x,
                           gt_query_y,
                           gt_target_x,
                           gt_target_y, 
                           pred_x,
                           pred_y,
                           frame0_x_scales,
                           frame0_x_offsets,
                           frame0_y_scales,
                           frame0_y_offsets,
                           frame1_x_scales,
                           frame1_x_offsets,
                           frame1_y_scales,
                           frame1_y_offsets):
    """
    Given zoom sequence, rescale points back to origina scale.
    """

    n = len(frame0_x_scales)

    for i in range(n - 1, -1, -1): 

        gt_query_x = gt_query_x * (1/frame0_x_scales[i])
        gt_query_x += frame0_x_offsets[i]

        gt_query_y = gt_query_y * (1/frame0_y_scales[i])
        gt_query_y += frame0_y_offsets[i]

        gt_target_x = gt_target_x * (1/frame1_x_scales[i])
        gt_target_x += frame1_x_offsets[i]
        
        gt_target_y = gt_target_y * (1/frame1_y_scales[i])
        gt_target_y += frame1_y_offsets[i]

        pred_x = pred_x * (1/frame1_x_scales[i])
        pred_x += frame1_x_offsets[i]

        pred_y = pred_y * (1/frame1_y_scales[i])
        pred_y += frame1_y_offsets[i]
    
    return gt_query_x, gt_query_y, gt_target_x, gt_target_y, pred_x, pred_y
            

def resize(np_img, patch_size=8, fixed_size=None, smart=False):
    """
    Resize frame. 
    """
    h, w, _ = np_img.shape
    # realign to patch size
    rh, rw = (h//patch_size)*patch_size, (w//patch_size)*patch_size

    size = min(rh, rw) 

    if fixed_size is not None:
        size = fixed_size
        if h == w == size: 
            return np_img
    
    if smart: 
        img = ImageOps.fit(Image.fromarray(np_img), (size, size))
    else: 
        img = Image.fromarray(np_img).resize((size, size))

    return np.array(img)


def crop_and_rescale_points(points_xy_raster, og_size):
    """
    Used for select_points demo. Given the original DAVIS raster, 
    crop and rescale each point to fit the select_points demo input. 
    Then, re-scale back to [0, 1].
    """
    og_h, og_w = og_size 

    points_xy = points_xy_raster * [[[og_w, og_h]]]

    # assume center crop, wide image
    points_xy[..., 0] = points_xy[..., 0] - (og_w - og_h) // 2
    points_xy = points_xy / og_h

    return points_xy


def get_pred_and_epe(heatmap, gt_x, gt_y, img_size=256, offset_center=False):
    """
    Given a heatmap, get its argmax location and compute EPE. 
    
    The predictions will be scaled to the original img_size, e.g. 
    for KL maps (32x32), predictions will be scaled by 8x.
    """ 
    y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)

    if isinstance(img_size, tuple): 
        img_size_y, img_size_x = img_size 
    else:
        img_size_y = img_size_x = img_size

    # Compute pred_x and y in original image space
    y = (img_size_y / heatmap.shape[0]) * y 
    x = (img_size_x / heatmap.shape[1]) * x 
    
    if offset_center: 
        assert img_size_y == img_size_x
        half_patch_size = (img_size // heatmap.shape[0]) // 2
        y = y + half_patch_size
        x = x + half_patch_size

    epe = np.sqrt((x - gt_x) ** 2 + (y - gt_y) ** 2)

    return x, y, epe


def compute_2d_argmax(prob_heatmap):
    b,h,w = prob_heatmap.shape

    flat_idx = prob_heatmap.reshape(b, -1).argmax(axis=1)   # (B,) linear indices
    rows = flat_idx // w                                    # (B,) row indices
    cols = flat_idx %  w                                    # (B,) col indices

    return np.stack([rows, cols], axis=1)                   # (B, 2)


def viz_rollout(out_dir: str,
             out_name: str,
             ground_truths: tuple[int], 
             kl_predictions: tuple[int],
             frame_curr,
             frame_next,
             cos_sim_map, 
             rgb_predictions=None,
             perturbation=None):

    gt_query_x, gt_query_y, gt_target_x, gt_target_y = ground_truths 
    kl_pred_x, kl_pred_y = kl_predictions
    rgb_pred_x, rgb_pred_y = rgb_predictions if rgb_predictions is not None else (float('inf'), float('inf'))

    kl_epe = np.sqrt((kl_pred_x - gt_target_x) ** 2 + (kl_pred_y - gt_target_y) ** 2)
    rgb_epe = np.sqrt((rgb_pred_x - gt_target_x) ** 2 + (rgb_pred_y - gt_target_y) ** 2)

    # Plot frame0, frame1, cos_sim
    fig, axes = plt.subplots(1, 3, figsize=(9, 9))
    
    axes = axes.flatten()
    axes[0].imshow(frame_curr)
    axes[0].set_title("Frame 0 (r=RGB,b=KL,g=GT)") 

    axes[0].arrow(gt_query_x, gt_query_y, gt_target_x - gt_query_x, gt_target_y - gt_query_y, head_width=3, head_length=3, fc='green', ec='green') 
    axes[0].arrow(gt_query_x, gt_query_y, kl_pred_x - gt_query_x, kl_pred_y - gt_query_y, head_width=3, head_length=3, fc='blue', ec='blue')
    axes[0].arrow(gt_query_x, gt_query_y, rgb_pred_x - gt_query_x, rgb_pred_y - gt_query_y, head_width=3, head_length=3, fc='red', ec='red')

    axes[1].imshow(frame_next)
    axes[1].set_title("Frame 1")

    axes[1].scatter(gt_target_x, gt_target_y, c='g', s=5)
    axes[1].scatter(kl_pred_x, kl_pred_y, c='b', s=5)
    axes[1].scatter(rgb_pred_x, rgb_pred_y, c='r', s=5)

    axes[2].imshow(cos_sim_map, cmap='viridis')
    axes[2].set_title("Cosine Similarity Map")

    suptitle = f"EPE: KL - {kl_epe:.3f}"
    suptitle += f" / RGB - {rgb_epe:.3f}"

    fig.suptitle(suptitle)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, out_name))
    plt.close(fig)


def viz_basic(out_dir,
              ground_truths: np.ndarray, 
              kl_predictions: np.ndarray, 
              frame_curr,
              frame_next,
              rgb_predictions=None):

    gt_query_x, gt_query_y, gt_target_x, gt_target_y = ground_truths 

    kl_pred_x, kl_pred_y = kl_predictions
    rgb_pred_x, rgb_pred_y = rgb_predictions if rgb_predictions is not None else (float('inf'), float('inf'))

    kl_epe = np.sqrt((kl_pred_x - gt_target_x) ** 2 + (kl_pred_y - gt_target_y) ** 2)
    rgb_epe = np.sqrt((rgb_pred_x - gt_target_x) ** 2 + (rgb_pred_y - gt_target_y) ** 2)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))

    axs[0].imshow(frame_curr)
    axs[0].set_title("Frame 0 (r=RGB,b=KL,g=GT)")
    axs[1].imshow(frame_next)
    axs[1].set_title("Frame 1") 

    axs[0].arrow(gt_query_x, gt_query_y, gt_target_x - gt_query_x, gt_target_y - gt_query_y, head_width=3, head_length=3, fc='green', ec='green') 
    axs[0].arrow(gt_query_x, gt_query_y, kl_pred_x - gt_query_x, kl_pred_y - gt_query_y, head_width=3, head_length=3, fc='blue', ec='blue')
    if rgb_predictions is not None:
        axs[0].arrow(gt_query_x, gt_query_y, rgb_pred_x - gt_query_x, rgb_pred_y - gt_query_y, head_width=3, head_length=3, fc='red', ec='red')

    axs[1].scatter(gt_target_x, gt_target_y, c='g', s=5)
    axs[1].scatter(kl_pred_x, kl_pred_y, c='b', s=5)
    if rgb_predictions is not None:
        axs[1].scatter(rgb_pred_x, rgb_pred_y, c='r', s=5)

    suptitle = f"EPE: KL - {kl_epe:.3f}"
    if rgb_predictions is not None: 
        suptitle += f" / RGB - {rgb_epe:.3f}"

    fig.suptitle(suptitle)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'final.png'))
    plt.close(fig)



def viz_multimask(out_dir: str,
                  ground_truths: tuple[int],
                  kl_predictions: tuple[int],
                  frame_curr: np.ndarray,
                  frame_next: np.ndarray,
                  avg_rgb_diff_map: np.ndarray,
                  avg_kl_map: np.ndarray,
                  rgb_predictions=None,
                  img_size=256
                  ): 
    
    fig, axs = plt.subplots(1, 4, figsize=(12,3))

    gt_query_x, gt_query_y, gt_target_x, gt_target_y = ground_truths
    kl_pred_x, kl_pred_y = kl_predictions
    rgb_pred_x, rgb_pred_y = rgb_predictions if rgb_predictions is not None else (float('inf'), float('inf'))

    kl_epe = np.sqrt((kl_pred_x - gt_target_x) ** 2 + (kl_pred_y - gt_target_y) ** 2)
    rgb_epe = np.sqrt((rgb_pred_x - gt_target_x) ** 2 + (rgb_pred_y - gt_target_y) ** 2)

    axs[0].imshow(frame_curr)
    axs[0].set_title("Frame 0 (r=RGB,b=KL,g=GT)") 

    axs[0].arrow(gt_query_x, gt_query_y, gt_target_x - gt_query_x, gt_target_y - gt_query_y, head_width=3, head_length=3, fc='green', ec='green') 
    axs[0].arrow(gt_query_x, gt_query_y, kl_pred_x - gt_query_x, kl_pred_y - gt_query_y, head_width=3, head_length=3, fc='blue', ec='blue')
    if rgb_predictions is not None:
        axs[0].arrow(gt_query_x, gt_query_y, rgb_pred_x - gt_query_x, rgb_pred_y - gt_query_y, head_width=3, head_length=3, fc='red', ec='red')

    axs[1].imshow(frame_next)
    axs[1].set_title("Frame 1")
    axs[1].scatter(gt_target_x, gt_target_y, c='g', s=5)
    axs[1].scatter(kl_pred_x, kl_pred_y, c='b', s=5)
    if rgb_predictions is not None:
        axs[1].scatter(rgb_pred_x, rgb_pred_y, c='r', s=5)
    
    axs[2].imshow(avg_rgb_diff_map, cmap="hot")
    axs[2].set_title(f"Avg Diff Map", fontsize=12)

    axs[3].imshow(avg_kl_map)
    axs[3].set_title("Avg KL Map", fontsize=12)

    for j, ax in enumerate(axs[2:]):
        if j == 0 and rgb_predictions is None:
            continue 
            
        if j == 0: 
            pred_x, pred_y = rgb_predictions
            gt_x, gt_y = gt_target_x, gt_target_y
            color = 'red'
            radius = 15
        else:
            pred_x, pred_y = kl_predictions
            scale_x, scale_y = avg_kl_map.shape[1]/img_size, avg_kl_map.shape[0]/img_size
            pred_x, pred_y = pred_x * scale_x, pred_y * scale_y
            gt_x, gt_y = gt_target_x * scale_x, gt_y * scale_y
            color = 'blue'
            radius = 2

        circ1 = plt.Circle((gt_x, gt_y), radius=radius, edgecolor="green", facecolor='none', linewidth=2, linestyle='--')
        circ2 = plt.Circle((pred_x, pred_y), radius=radius, edgecolor=color, facecolor='none', linewidth=2, linestyle='--')

        ax.add_patch(circ1)
        ax.add_patch(circ2)
    
    suptitle = f"EPE: KL - {kl_epe:.3f}"
    if rgb_predictions is not None: 
        suptitle += f" / RGB - {rgb_epe:.3f}"
    fig.suptitle(suptitle)
    
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, f"averaged_preds.png"))
    plt.close(fig)

def build_pt_video_transform():
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    eval_transform = video_transforms.Compose(
        [
            video_transforms.Resize(256, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(256, 256)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )
    return eval_transform

def main(args):

    if args.zoom_stride > 1: 
        assert args.zoom_iters == 1, \
            f"Currently, when zoom_stride > 1, assumes that we have a single zoom that zooms directly to Nx"

    with open(args.data_path, 'r') as f:
        dataset = json.load(f)


    pretrained_model_name = args.model_name
    if args.model_type == 'dinov3':
        args.num_rollouts = 1
        
        from transformers import AutoImageProcessor, AutoModel
        from transformers.image_utils import load_image
        
        processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        model = AutoModel.from_pretrained(
            pretrained_model_name, 
            device_map="auto", 
        )
        
    img_size = 256
    
    logger.info(f"Running in resolution {img_size}x{img_size}")

    def should_viz(c): 
        if args.no_viz:
            return False
        
        return args.viz_all or c % args.viz_interval == 0

    def should_log(c): 
        return c % args.log_interval == 0

    # ------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # ------------------------------------------------
    out_dir = os.path.join(
        args.out_dir, 
        args.model_type,
        args.model_name.replace("/", "_").replace(".pt", "")
    )
    out_dir = f"{out_dir}_mask_ratio_{args.mask_ratio}"
    out_dir += f"_t{args.temperature:.2f}_k{args.top_k}_p{args.top_p:.2f}"
    os.makedirs(out_dir, exist_ok=True)

    viz_dir = os.path.join(out_dir, 'viz') 
    os.makedirs(viz_dir, exist_ok=True)

    flags_dir = os.path.join(out_dir, 'flags') 
    os.makedirs(flags_dir, exist_ok=True)

    args_dir = os.path.join(out_dir, 'args') 
    os.makedirs(args_dir, exist_ok=True)

    results_dir = os.path.join(out_dir, 'results') 
    os.makedirs(results_dir, exist_ok=True)

    data_range_str = f'[{args.flat_points_start_idx},{args.flat_points_start_idx + args.num_flat_points_to_process})' 

    tapvid_formatted_results = []    
    seen = set()

    if args.prev_cache_root_dir is not None:
        assert out_dir != args.prev_cache_root_dir, f'For sanity, please write to new directory and combine later.'

        print(f"Removing points already computed!")

        all_json_files = glob(os.path.join(args.prev_cache_root_dir, 'results' 'tapvid_*.json'))

        print(f'Found {len(all_json_files)} JSON files.')

        for json_file in all_json_files:
            with open(json_file, 'r') as f:
                seen = seen.update([x['uid'] for x in json.load(f)])
        
        print(f'Found {len(seen)} evals in cache.')

    with open(os.path.join(args_dir, f'args_{data_range_str}.json'), 'w') as f:
        args_dict = vars(args)
        hostname = socket.gethostname()
        args_dict["hostname"] = hostname
        json.dump(args_dict, f, indent=4)

    kl_batch_average = 0
    rgb_batch_average = 0
    counts = 0

    dataset = dataset[args.flat_points_start_idx : args.flat_points_start_idx + args.num_flat_points_to_process]
    og_len = len(dataset)
    dataset = [x for x in dataset if x['uid'] not in seen]

    print(f'Evaluating on {og_len} -> {len(dataset)} points after cache.')

    del seen

    kl_epe_logs = {} 
    rgb_epe_logs = {}
    rgb_pred_logs = {}

    start = time.time()

    for data in tqdm.tqdm(dataset, total=len(dataset)):
        
        # start = time.time()
        query_frame_file = data["query_frame_file"]
        target_frame_file = data["target_frame_file"] 

        query_frame = np.array(Image.open(query_frame_file))
        target_frame = np.array(Image.open(target_frame_file))

        query_y_raster = data["query_y_raster"] 
        query_x_raster = data["query_x_raster"]

        target_y_raster = data["target_y_raster"] 
        target_x_raster = data["target_x_raster"]

        data_uid = data["uid"]

        start_uid, end_uid, *_ = data_uid.split(",")
        st = int(start_uid.split("_")[-1])
        et = int(end_uid.split("_")[-1])

        kl_epe_logs[data_uid] = {} 
        rgb_epe_logs[data_uid] = {} 
        rgb_pred_logs[data_uid] = {}

        points = np.array([[
            [query_x_raster, query_y_raster], 
            [target_x_raster, target_y_raster]
        ]])

        # Save for later - all data before resize
        query_frame_og = query_frame 
        target_frame_og = target_frame 

        og_h, og_w = query_frame_og.shape[:-1]
        points_og = (query_x_raster * og_w, 
                     query_y_raster * og_h, 
                     target_x_raster * og_w, 
                     target_y_raster * og_h)

        if not args.squish:
            if counts == 0: logger.warning("Running in Square Crop Mode")
            points = crop_and_rescale_points(points, query_frame.shape[:2]) * img_size
        else: 
            if counts == 0: logger.info("Running in Squish Mode")
            points = points * img_size

        query_x, query_y = points[0, 0]
        target_x, target_y = points[0, 1]

        query_frame = resize(query_frame, fixed_size=img_size, smart=not args.squish)
        target_frame = resize(target_frame, fixed_size=img_size, smart=not args.squish)

        if not args.squish:
            query_frame_og = query_frame 
            target_frame_og = target_frame
            og_h, og_w = query_frame_og.shape[:-1]
            points_og = tuple(map(lambda x: x.item(), (query_x, query_y, target_x, target_y)))

        # end = time.time()

        # print(f'image processing; {end-start:.3f}')

        # start = time.time()

        frame0_x_offsets, frame0_y_offsets = [], [] 
        frame0_x_scales, frame0_y_scales = [], []

        frame1_x_offsets, frame1_y_offsets = [], [] 
        frame1_x_scales, frame1_y_scales = [], []

        cached_pert = None
        pred_ofs = None
        
        if args.pert_cache_path is not None: 
            qff, tff, pidx = data_uid.split(',')

            vidname = '_'.join(qff.split('_')[:-1])

            st = int(qff.split('_')[-1])
            et = int(tff.split('_')[-1])
            pidx = int(pidx)

            if 'davis' in args.pert_cache_path:
                with open('/ccn2/u/ksimon12/tapvid_davis/tapvid_davis.pkl', 'rb') as f:
                    davis_dataset = pickle.load(f)
                
                davis_vidnames = sorted(list(davis_dataset.keys()))
                vidx = davis_vidnames.index(vidname)
                assert vidx >= 0
            else:
                # FOR KINETICS AND KUBRIC, STORE VIDX, NOT VIDNAME
                vidx = int(vidname)

            # M, 3, 20, 20
            cached_pert = np.load(os.path.join(args.pert_cache_path, f'{vidx:03d}', f'{vidx:03d}_{pidx:03d}_{st:03d}_{et:03d}.npy'))

            # precompute offsets
            pert_norms = np.linalg.norm(cached_pert, axis=1) 
            pert_ofs = compute_2d_argmax(pert_norms) #m, 2
            midpoint = np.array([*cached_pert.shape[-2:]]) // 2

            # given 256x256 input (M offsets)
            pred_ofs = midpoint - pert_ofs
            pred_ofs = pred_ofs[:args.num_rollouts].mean(0)


        occ_pred = False # default occ prediction

        # + 1 including initial run
        for zoom_itr in range(args.zoom_iters + 1): 
            # zoom_start = time.time()
            kl_epe_logs[data_uid][zoom_itr] = {"iters": []}
            rgb_epe_logs[data_uid][zoom_itr] = {"iters": []}
            rgb_pred_logs[data_uid][zoom_itr] = {"iters": []}

            cos_sim_maps = []  # collect a diff map per seed for the current step

            for rollout_idx in range(args.num_rollouts): 
                current_seed = args.seed + rollout_idx

                # --- DINOv3 patch features (returns [num_patches, hidden_dim]) ---
                device = next(model.parameters()).device
                num_register = int(getattr(model.config, "num_register_tokens", 0) or getattr(model.config, "num_registers", 0))

                def _to_pil_rgb(np_img: np.ndarray) -> Image.Image:
                    if np_img.ndim == 3 and np_img.shape[-1] >= 3:
                        np_img = np_img[..., :3]
                    return Image.fromarray(np_img.astype(np.uint8))

                @torch.inference_mode()
                def _extract_patch_feats(np_img: np.ndarray):
                    pil = _to_pil_rgb(np_img)

                    inputs = processor(images=pil, return_tensors="pt").to(device)
                    pv = inputs["pixel_values"]                  # [1,3,H,W]
                    H, W = int(pv.shape[-2]), int(pv.shape[-1])  # e.g. 224,224

                    out = model(**inputs, return_dict=True)
                    toks = out.last_hidden_state

                    start = 1 + num_register if toks.shape[1] >= (1 + num_register) else 0
                    patch = toks[:, start:, :].squeeze(0).float()  # [num_patches, D]

                    return patch

                frame1_feat = _extract_patch_feats(query_frame)   # [196, 1024]
                frame2_feat = _extract_patch_feats(target_frame)  # [196, 1024]

                query_x_224 = int((query_x / query_frame.shape[1]) * 224)
                query_y_224 = int((query_y / query_frame.shape[0]) * 224)

                # the image uses a 16x16 patch size on a 224x224 image, so we have 14x14 patches
                n_patches_per_side = int(frame1_feat.shape[0] ** 0.5) # 14
                query_point_idx = int(query_x_224 // 16 + (query_y_224 // 16) * n_patches_per_side)
                frame1_feat_query = frame1_feat[query_point_idx] # [1024]
                
                # find the idx of the frame2_feat with the highest cosine similarity
                cos_sim = F.cosine_similarity(frame1_feat_query.unsqueeze(0), frame2_feat, dim=-1) # [256]
                frame2_target_idx = cos_sim.argmax().item() # get the index of the most similar frame2_feat
                
                # convert this to x, y coordinates, by using the centre of the patch
                itr_rgb_x = (frame2_target_idx % n_patches_per_side) * 16 + 8
                itr_rgb_y = (frame2_target_idx // n_patches_per_side) * 16 + 8
                itr_rgb_x = (itr_rgb_x / 224) * query_frame.shape[1]
                itr_rgb_y = (itr_rgb_y / 224) * query_frame.shape[0]
                itr_rgb_epe = np.sqrt((itr_rgb_x - target_x) ** 2 + (itr_rgb_y - target_y) ** 2)

                kl = torch.zeros((img_size // 8, img_size // 8))
                kl_np = kl.cpu().numpy()
                itr_kl_x, itr_kl_y = itr_rgb_x, itr_rgb_y
                itr_kl_epe = itr_rgb_epe

                kl_epe_logs[data_uid][zoom_itr]['iters'].append(itr_kl_epe)
                rgb_epe_logs[data_uid][zoom_itr]['iters'].append(itr_rgb_epe)
                rgb_pred_logs[data_uid][zoom_itr]['iters'].append((itr_rgb_x, itr_rgb_y))
                
                # convert cos_sim into a 16x16 map
                cos_sim_map = cos_sim.reshape(n_patches_per_side, n_patches_per_side).detach().cpu().numpy() # [16, 16]
                cos_sim_maps.append(cos_sim_map)

                if should_viz(counts): 
                    data_viz_dir = os.path.join(viz_dir, data_uid)
                    os.makedirs(data_viz_dir, exist_ok=True)

                    # zoom_dir = os.path.join(data_viz_dir, f'zoom={zoom_itr}')
                    # os.makedirs(zoom_dir, exist_ok=True)

                    viz_rollout(data_viz_dir,
                                f"zoom_{zoom_itr:03d}.png",
                                (query_x, query_y, target_x, target_y), 
                                (itr_kl_x, itr_kl_y), 
                                query_frame,
                                target_frame, 
                                cos_sim_map, 
                                (itr_rgb_x, itr_rgb_y))

            # VJEPA2 occ computation
            metric = float(cos_sim.max())
            occ_pred = bool(metric < 0.05)

            # No need to average over rollouts for VJEPA2 because it is deterministic
            rgb_pred_x, rgb_pred_y, rgb_epe = itr_rgb_x, itr_rgb_y, itr_rgb_epe
            kl_pred_x, kl_pred_y, kl_epe = itr_kl_x, itr_kl_y, itr_kl_epe

            kl_epe_logs[data_uid][zoom_itr]['multi_mask'] = kl_epe
            rgb_epe_logs[data_uid][zoom_itr]['multi_mask'] = rgb_epe
            rgb_pred_logs[data_uid][zoom_itr]['multi_mask'] = (rgb_pred_x, rgb_pred_y)
            
            if zoom_itr == args.zoom_iters:
                break
            
            # zoom_end = time.time()
            # print(f'1 iteration zoom: {zoom_end-zoom_start:.3f}s')
            if zoom_itr == 0: 
                h_scale = query_frame_og.shape[0] / query_frame.shape[0]
                w_scale = query_frame_og.shape[1] / query_frame.shape[1]

                query_frame = query_frame_og
                target_frame = target_frame_og

                query_x, query_y = query_x * w_scale, query_y * h_scale 
                kl_pred_x, kl_pred_y = kl_pred_x * w_scale, kl_pred_y * h_scale 
                rgb_pred_x, rgb_pred_y = rgb_pred_x * w_scale, rgb_pred_y * h_scale 

                target_x, target_y = target_x * w_scale, target_y * h_scale

            is_rect = query_frame.shape[0] != query_frame.shape[1]

            if is_rect: assert zoom_itr == 0, f'Zooming in should only produce squares'

            query_frame, curr_left, curr_top, curr_wscale, curr_hscale = zoom_into_frame(query_frame, query_x, query_y, zoom_stride=args.zoom_stride, rect=is_rect, img_size=img_size)
            target_frame, next_left, next_top, next_wscale, next_hscale = zoom_into_frame(target_frame, kl_pred_x, kl_pred_y, zoom_stride=args.zoom_stride, rect=is_rect, img_size=img_size)

            query_x = (query_x - curr_left) * curr_wscale
            query_y = (query_y - curr_top) * curr_hscale

            target_x = (target_x - next_left) * next_wscale
            target_y = (target_y - next_top) * next_hscale

            frame0_x_offsets.append(curr_left)
            frame0_x_scales.append(curr_wscale)
            frame0_y_offsets.append(curr_top)
            frame0_y_scales.append(curr_hscale)

            frame1_x_offsets.append(next_left)
            frame1_x_scales.append(next_wscale)
            frame1_y_offsets.append(next_top)
            frame1_y_scales.append(next_hscale)


        # recover 
        kl_pred_x, kl_pred_y = recover_og_coordinates(
            query_x,
            query_y, 
            target_x, 
            target_y, 
            kl_pred_x, 
            kl_pred_y, 
            frame0_x_scales, 
            frame0_x_offsets,
            frame0_y_scales,
            frame0_y_offsets,
            frame1_x_scales,
            frame1_x_offsets,
            frame1_y_scales,
            frame1_y_offsets
        )[-2:]

        rgb_pred_x, rgb_pred_y = recover_og_coordinates(
            query_x,
            query_y, 
            target_x, 
            target_y, 
            rgb_pred_x, 
            rgb_pred_y, 
            frame0_x_scales, 
            frame0_x_offsets,
            frame0_y_scales,
            frame0_y_offsets,
            frame1_x_scales,
            frame1_x_offsets,
            frame1_y_scales,
            frame1_y_offsets
        )[-2:]
        
        query_x, query_y, target_x, target_y = points_og
        
        kl_epe = np.sqrt((target_x - kl_pred_x) ** 2 + (target_y - kl_pred_y) ** 2)
        rgb_epe = np.sqrt((target_x - rgb_pred_x) ** 2 + (target_y - rgb_pred_y) ** 2)

        kl_batch_average += kl_epe
        rgb_batch_average += rgb_epe

        kl_epe_logs[data_uid]['final'] = kl_epe
        rgb_epe_logs[data_uid]['final'] = rgb_epe
        rgb_pred_logs[data_uid]['final'] = (rgb_pred_x, rgb_pred_y)

        res = {
            "uid": data["uid"],
            "gt_query_x": data["query_x_raster"] * 256, 
            "gt_query_y": data["query_y_raster"] * 256, 
            "gt_target_x": data["target_x_raster"] * 256, 
            "gt_target_y": data["target_y_raster"] * 256,
            "gt_occ": data.get("occluded", False),                                 # TODO: add occ to dataset 
            "pred_target_x": kl_pred_x * (256 / og_w), 
            "pred_target_y": kl_pred_y * (256 / og_h), 
            "pred_occ": occ_pred,
        }
        res['occ_metric'] = metric

        tapvid_formatted_results.append(res)
    
        if should_log(counts): 
            with open(os.path.join(results_dir, f'kl_epe_results_{data_range_str}.json'), 'w') as f:
                json.dump(kl_epe_logs, f, indent=4)

            with open(os.path.join(results_dir, f'rgb_epe_results_{data_range_str}.json'), 'w') as f:
                json.dump(rgb_epe_logs, f, indent=4)
                
            with open(os.path.join(results_dir, f'rgb_pred_results_{data_range_str}.json'), 'w') as f:
                json.dump(rgb_pred_logs, f, indent=4)
            
            with open(os.path.join(results_dir, f'tapvid_formatted_results_256res_{data_range_str}.json'), 'w') as f:
                json.dump(tapvid_formatted_results, f, indent=4)
                      
        
        if should_viz(counts):
            viz_basic(data_viz_dir,
                        (query_x, query_y, target_x, target_y), 
                        (kl_pred_x, kl_pred_y), 
                        query_frame_og,
                        target_frame_og, 
                        (rgb_pred_x, rgb_pred_y))
    
        counts += 1

        end = time.time()

        total_elapsed = end - start
        avg_time_per_data = total_elapsed / counts

        expected_remaining = (len(dataset) - counts) * avg_time_per_data

        # Add 30 seconds (for example)
        now = datetime.now()
        expected_time = now + timedelta(seconds=expected_remaining)

        # Format to string
        now_formatted = now.strftime('%Y-%m-%d %H:%M:%S')
        formatted = expected_time.strftime('%Y-%m-%d %H:%M:%S')

        with open(os.path.join(flags_dir, f'PROGRESS_{data_range_str}.txt'), 'w') as f:
            f.write((f"Logging @ {now_formatted} ({hostname})\n"
                     f"- Progress     : {counts}/{len(dataset)} done\n"
                     f"- Total elapsed: {total_elapsed:.3f}s\n"
                     f"- Avg sec / itr: {avg_time_per_data:.3f}s\n"
                     f"- Expected done: {formatted}\n"))
        

        # query_y_raster = data["query_y_raster"] 
        # query_x_raster = data["query_x_raster"]

        # target_y_raster = data["target_y_raster"] 
        # target_x_raster = data["target_x_raster"]


    with open(os.path.join(results_dir, f'kl_epe_results_{data_range_str}.json'), 'w') as f:
        json.dump(kl_epe_logs, f, indent=4)
    
    with open(os.path.join(results_dir, f'rgb_epe_results_{data_range_str}.json'), 'w') as f:
        json.dump(rgb_epe_logs, f, indent=4)
        
    with open(os.path.join(results_dir, f'rgb_pred_results_{data_range_str}.json'), 'w') as f:
        json.dump(rgb_pred_logs, f, indent=4)

    with open(os.path.join(results_dir, f'tapvid_formatted_results_256res_{data_range_str}.json'), 'w') as f:
        json.dump(tapvid_formatted_results, f, indent=4)
                      
    with open(os.path.join(flags_dir, f'FLAG_{data_range_str}.txt'), 'w') as f: 
        f.write("Done!")

if __name__ == "__main__":
    args = get_args()
    main(args)