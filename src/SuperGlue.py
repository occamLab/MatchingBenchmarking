from pathlib import Path
import argparse
import matplotlib.cm as cm
import torch
import matplotlib.pyplot as plt

from models.matching import Matching
from models.utils import (
    AverageTimer,
    VideoStreamer,
    make_matching_plot_fast,
    frame2tensor,
)

torch.set_grad_enabled(False)


global out
global confidence
# Sets initial config for superglue
parser = argparse.ArgumentParser(
    description="SuperGlue demo", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--input",
    type=str,
    default="0",
    help="ID of a USB webcam, URL of an IP camera, "
    "or path to an image directory or movie file",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory where to write output frames (If None, no output)",
)

parser.add_argument(
    "--image_glob",
    type=str,
    nargs="+",
    default=["*.png", "*.jpg", "*.jpeg"],
    help="Glob if a directory of images is specified",
)
parser.add_argument(
    "--skip",
    type=int,
    default=1,
    help="Images to skip if input is a movie or directory",
)
parser.add_argument(
    "--max_length",
    type=int,
    default=1000000,
    help="Maximum length if input is a movie or directory",
)
parser.add_argument(
    "--resize",
    type=int,
    nargs="+",
    default=[640, 480],
    help="Resize the input image before running inference. If two numbers, "
    "resize to the exact dimensions, if one number, resize the max "
    "dimension, if -1, do not resize",
)
parser.add_argument(
    "--superglue",
    choices={"indoor", "outdoor"},
    default="indoor",
    help="SuperGlue weights",
)
parser.add_argument(
    "--max_keypoints",
    type=int,
    default=-1,
    help="Maximum number of keypoints detected by Superpoint"
    " ('-1' keeps all keypoints)",
)
parser.add_argument(
    "--keypoint_threshold",
    type=float,
    default=0.005,
    help="SuperPoint keypoint detector confidence threshold",
)
parser.add_argument(
    "--nms_radius",
    type=int,
    default=4,
    help="SuperPoint Non Maximum Suppression (NMS) radius" " (Must be positive)",
)
parser.add_argument(
    "--sinkhorn_iterations",
    type=int,
    default=20,
    help="Number of Sinkhorn iterations performed by SuperGlue",
)
parser.add_argument(
    "--match_threshold", type=float, default=0.2, help="SuperGlue match threshold"
)

parser.add_argument(
    "--show_keypoints", action="store_true", help="Show the detected keypoints"
)
parser.add_argument(
    "--no_display",
    action="store_true",
    help="Do not display images to screen. Useful if running remotely",
)
parser.add_argument(
    "--force_cpu", action="store_true", help="Force pytorch to run in CPU mode."
)

opt = parser.parse_args()
print(opt)

if len(opt.resize) == 2 and opt.resize[1] == -1:
    opt.resize = opt.resize[0:1]
if len(opt.resize) == 2:
    print("Will resize to {}x{} (WxH)".format(opt.resize[0], opt.resize[1]))
elif len(opt.resize) == 1 and opt.resize[0] > 0:
    print("Will resize max dimension to {}".format(opt.resize[0]))
elif len(opt.resize) == 1:
    print("Will not resize images")
else:
    raise ValueError("Cannot specify more than two integers for --resize")

# device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
device = "cpu"
print('Running inference on device "{}"'.format(device))
config = {
    "superpoint": {
        "nms_radius": opt.nms_radius,
        "keypoint_threshold": opt.keypoint_threshold,
        "max_keypoints": opt.max_keypoints,
    },
    "superglue": {
        "weights": opt.superglue,
        "sinkhorn_iterations": opt.sinkhorn_iterations,
        "match_threshold": opt.match_threshold,
    },
}
matching = Matching(config).eval().to(device)
keys = ["keypoints", "scores", "descriptors"]


def get_superglue_matches(query_image, train_image):

    frame_1 = query_image
    frame_2 = train_image

    frame_1_tensor = frame2tensor(frame_1, device)

    frame_1_data = matching.superpoint({"image": frame_1_tensor})
    frame_1_data = {k + "0": frame_1_data[k] for k in keys}
    frame_1_data["image0"] = frame_1_tensor

    frame_2_tensor = frame2tensor(frame_2, device)

    pred = matching({**frame_1_data, "image1": frame_2_tensor})
    kpts0 = frame_1_data["keypoints0"][0].cpu().numpy()
    kpts1 = pred["keypoints1"][0].cpu().numpy()
    matches = pred["matches0"][0].cpu().numpy()
    global confidence
    confidence = pred["matching_scores0"][0].cpu().detach().numpy()

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    color = cm.jet(confidence[valid])
    text = []

    global out
    out = make_matching_plot_fast(
        frame_1,
        frame_2,
        kpts0,
        kpts1,
        mkpts0,
        mkpts1,
        color,
        text,
        path=None,
        show_keypoints=opt.show_keypoints,
    )

    final_mkpts = list(zip(mkpts0, mkpts1))

    final_list = []

    for pair_of_coords in final_mkpts:
        for coords in pair_of_coords:
            for coord in coords:
                final_list.append(str(int(coord)))

    return final_list, mkpts0, mkpts1


def draw_superglue_matches():
    plt.imshow(out)
    plt.show()

def get_confidence():
    return confidence
