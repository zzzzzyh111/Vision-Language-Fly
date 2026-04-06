import argparse
import os
import time

import clip
import numpy as np
import rospy
import torch
import yaml
from PIL import Image as PILImage
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Bool, Float32MultiArray
from transformers import pipeline

from topic_names import IMAGE_TOPIC, REACHED_GOAL_TOPIC, WAYPOINT_TOPIC
from utils import load_model, msg_to_pil, to_numpy, transform_images

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEPLOYMENT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

TOPOMAP_IMAGES_DIR = os.path.join(DEPLOYMENT_DIR, "topomaps", "images")
GOAL_DIR = ["topomap1", "topomap2", "topomap3"]
ROBOT_CONFIG_PATH = os.path.join(DEPLOYMENT_DIR, "config", "robot.yaml")
MODEL_CONFIG_PATH = os.path.join(DEPLOYMENT_DIR, "config", "models.yaml")

with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
RATE = robot_config["frame_rate"]

context_queue = []
context_size = None
obs_img = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def normalize_goal_prompt(goal_text, options_list):
    cleaned_text = " ".join(goal_text.strip().split())
    lowered_text = cleaned_text.lower()

    if lowered_text.startswith("goal image:"):
        cleaned_text = cleaned_text.split(":", 1)[1].strip()
        lowered_text = cleaned_text.lower()

    if lowered_text.startswith("a photo of"):
        cleaned_text = cleaned_text[len("a photo of") :].strip()

    cleaned_text = cleaned_text.strip("\"' .,")
    options_by_lower = {option.lower(): option for option in options_list}

    if cleaned_text.lower() in options_by_lower:
        selected_option = options_by_lower[cleaned_text.lower()]
    else:
        selected_option = next(
            (option for option in options_list if option.lower() in lowered_text),
            None,
        )
        if selected_option is None:
            raise ValueError(
                "LLaMA returned an unsupported goal description. "
                f"Expected one of: {', '.join(options_list)}. Got: {goal_text!r}"
            )

    return selected_option, f"a photo of {selected_option}"


def validate_topomap_folders(basefolder, subfolders):
    if not os.path.isdir(basefolder):
        raise FileNotFoundError(
            "Topomap image directory not found at "
            f"{basefolder}. Add the required topomap images or update "
            "TOPOMAP_IMAGES_DIR before running navigation."
        )

    missing_folders = [
        subfolder for subfolder in subfolders if not os.path.isdir(os.path.join(basefolder, subfolder))
    ]
    if missing_folders:
        raise FileNotFoundError(
            "Missing required topomap folders under "
            f"{basefolder}: {', '.join(missing_folders)}"
        )


def llama_select_goal(user_instruction, options_list):
    prompt = f"""
You are an assistant for a UAV navigation task.
Based on the user's instruction, choose exactly one most relevant item from the given options.
Reply strictly in this format: "Goal Image: a photo of [option]".
Only use items from the options list.
Do not explain or use any other words or symbols.

Options: {', '.join(options_list)}

Instruction: "{user_instruction}"
"""
    pipeline_kwargs = {"device_map": "auto"}
    if torch.cuda.is_available():
        pipeline_kwargs["torch_dtype"] = torch.float16
    pipe = pipeline(
        "text-generation",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        **pipeline_kwargs,
    )
    messages = [{"role": "user", "content": prompt.strip()}]
    output = pipe(messages, max_new_tokens=30, temperature=0.2)
    dialog = output[0]["generated_text"]
    return next((turn["content"] for turn in dialog if turn["role"] == "assistant"), "").strip()


def load_images_from_folder(basefolder, subfolders, preprocess, clip_device):
    validate_topomap_folders(basefolder, subfolders)
    image_paths = []
    image_tensors = []
    for subfolder in subfolders:
        folder = os.path.join(basefolder, subfolder)
        files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not files:
            continue
        files.sort(key=lambda f: int(os.path.splitext(f)[0]))
        last_img_path = os.path.join(folder, files[-1])
        last_img = preprocess(PILImage.open(last_img_path).convert("RGB")).unsqueeze(0).to(clip_device)
        image_paths.append(last_img_path)
        image_tensors.append(last_img)
    if not image_tensors:
        raise RuntimeError(
            "No valid images found in the configured topomap folders. "
            "Expected numbered .png/.jpg/.jpeg files in each topomap directory."
        )
    return image_paths, torch.cat(image_tensors)


def callback_obs(msg):
    global obs_img, context_queue
    obs_img = msg_to_pil(msg)
    if context_size is None:
        return
    if len(context_queue) < context_size + 1:
        context_queue.append(obs_img)
    else:
        context_queue.pop(0)
        context_queue.append(obs_img)


def main(args: argparse.Namespace):
    global context_size
    options = ["AprilTag", "blue backpack", "pink pig"]

    user_input = input("Navigation Task Description: ").strip()
    print("Running LLaMA to interpret instruction...")
    goal_text = llama_select_goal(user_input, options)
    print("\n LLaMA selected text:", goal_text)
    selected_option, text_for_clip = normalize_goal_prompt(goal_text, options)
    print(" Selected option:", selected_option)
    print(" Final CLIP prompt:", text_for_clip)

    start_time = time.time()
    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", clip_device)
    clip_model, preprocess = clip.load("ViT-B/32", device=clip_device)
    image_paths, image_inputs = load_images_from_folder(
        TOPOMAP_IMAGES_DIR, GOAL_DIR, preprocess, clip_device
    )
    text_inputs = clip.tokenize([text_for_clip]).to(clip_device)

    with torch.no_grad():
        logits_per_image, _ = clip_model(image_inputs, text_inputs)
        probs = logits_per_image.softmax(dim=0).squeeze(1).cpu().numpy()
    print("Inference time =", time.time() - start_time)

    print("\n--- Image Match Results ---")
    print("image path = ", image_paths)
    for filename, logit, prob in zip(image_paths, logits_per_image, probs):
        print(f"{filename:<25s} | Logit: {logit.item():.4f} | Probability: {prob:.4f}")

    best_idx = probs.argmax()
    best_filename = image_paths[best_idx]
    print(f"\n Best match image: {best_filename}")
    print(f"   -> Logit: {logits_per_image[best_idx].item():.4f}")
    print(f"   -> Probability: {probs[best_idx]:.4f}")

    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)
    if args.model not in model_paths:
        raise KeyError(
            f"Unknown model '{args.model}'. Available models: {', '.join(model_paths.keys())}"
        )
    model_config_path = os.path.join(BASE_DIR, model_paths[args.model]["config_path"])
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)
    context_size = model_params["context_size"]

    ckpt_path = os.path.join(BASE_DIR, model_paths[args.model]["ckpt_path"])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "Model weights not found at "
            f"{ckpt_path}. Download the VLFly checkpoint to this path, or update "
            "deployment/config/models.yaml to point to the correct file."
        )
    print(f"Loading model from {ckpt_path}")
    model = load_model(ckpt_path, model_params, device).to(device)
    model.eval()

    selected_dir = GOAL_DIR[best_idx]
    print("topomap path is:", selected_dir)
    topomap_dir = os.path.join(TOPOMAP_IMAGES_DIR, selected_dir)
    if not os.path.isdir(topomap_dir):
        raise FileNotFoundError(
            f"Selected topomap directory does not exist: {topomap_dir}"
        )
    topomap_filenames = sorted(
        [
            filename
            for filename in os.listdir(topomap_dir)
            if filename.lower().endswith((".png", ".jpg", ".jpeg"))
        ],
        key=lambda x: int(os.path.splitext(x)[0]),
    )
    if not topomap_filenames:
        raise RuntimeError(
            "The selected topomap directory is empty. "
            f"Expected numbered image files under {topomap_dir}."
        )
    topomap = [
        PILImage.open(os.path.join(topomap_dir, filename))
        for filename in topomap_filenames
    ]

    closest_node = 0
    assert -1 <= args.goal_node < len(topomap), "Invalid goal index"
    goal_node = len(topomap) - 1 if args.goal_node == -1 else args.goal_node
    reached_goal = False

    rospy.init_node("vlfly_navigation", anonymous=False)
    rate = rospy.Rate(RATE)
    rospy.Subscriber(IMAGE_TOPIC, RosImage, callback_obs, queue_size=1)
    waypoint_pub = rospy.Publisher(WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)
    goal_pub = rospy.Publisher(REACHED_GOAL_TOPIC, Bool, queue_size=1)

    print("Registered with master node. Waiting for image observations...")

    while not rospy.is_shutdown():
        chosen_waypoint = np.zeros(4)
        if len(context_queue) > model_params["context_size"]:
            start = max(closest_node - args.radius, 0)
            end = min(closest_node + args.radius + 1, goal_node)
            batch_obs_imgs = []
            batch_goal_data = []

            for subgoal_image in topomap[start : end + 1]:
                batch_obs_imgs.append(transform_images(context_queue, model_params["image_size"]))
                batch_goal_data.append(transform_images(subgoal_image, model_params["image_size"]))

            batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(device)
            batch_goal_data = torch.cat(batch_goal_data, dim=0).to(device)

            distances, waypoints = model(batch_obs_imgs, batch_goal_data)
            distances = to_numpy(distances)
            waypoints = to_numpy(waypoints)
            min_dist_idx = np.argmin(distances)
            print("distances[min_dist_idx] =", distances[min_dist_idx])

            if distances[min_dist_idx] > args.close_threshold:
                chosen_waypoint = waypoints[min_dist_idx][args.waypoint]
                closest_node = start + min_dist_idx
            else:
                chosen_waypoint = waypoints[min(min_dist_idx + 1, len(waypoints) - 1)][args.waypoint]
                closest_node = min(start + min_dist_idx + 1, goal_node)
            print("closest_node =", closest_node)

        if model_params["normalize"]:
            chosen_waypoint[:2] *= MAX_V / RATE

        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint
        waypoint_pub.publish(waypoint_msg)

        reached_goal = closest_node == goal_node
        goal_pub.publish(Bool(data=reached_goal))
        if reached_goal:
            print("Reached goal! Stopping...")
            if obs_img is not None:
                final_image_path = os.path.join(DEPLOYMENT_DIR, "topomaps", "Final.png")
                obs_img.save(final_image_path)
                print(f"Final image saved at {final_image_path}")
            return
        rate.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        default="vlfly",
        type=str,
        help="model name (default: vlfly)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=4,
        type=int,
        help="index of the waypoint used for navigation (default: 4)",
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="goal node index in the selected topomap (default: last node)",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=5,
        type=int,
        help="distance threshold before advancing to the next node",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=2,
        type=int,
        help="number of nearby nodes to inspect for localization",
    )
    args = parser.parse_args()
    print(f"Using {device}")
    main(args)
