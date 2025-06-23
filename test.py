import os
import json
from pathlib import Path

def sample_rewards_by_folder_name(dataset_root, step=10):
    for video_folder in sorted(os.listdir(dataset_root)):
        folder_path = Path(dataset_root) / video_folder
        if not folder_path.is_dir() or not video_folder.isdigit():
            continue

        video_id = video_folder
        reward_filename = f"{video_id}_states.json"
        reward_path = folder_path / reward_filename

        # Output file: e.g., "10_sampled_rewards.json"
        output_filename = f"{video_id}_sampled_states.json"
        output_path = folder_path / output_filename

        if not reward_path.exists():
            print(f"⚠️ Missing file: {reward_filename} in folder {video_folder}")
            continue

        # Load original rewards
        with open(reward_path, "r") as f:
            rewards = json.load(f)

        # Sample rewards every `step`
        sampled_rewards = rewards[::step]

        sampled_rewards.append(rewards[-1])

        # Save sampled rewards
        with open(output_path, "w") as f:
            json.dump(sampled_rewards, f)

        print(f"✅ {video_folder}: {len(sampled_rewards)} rewards saved to {output_filename}")

# Example usage
dataset_root = "/home/lianniello/egocentric_dataset/frames/valid/gripper"  # Change this path
sample_rewards_by_folder_name(dataset_root)
