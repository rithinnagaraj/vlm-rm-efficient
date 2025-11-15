import gymnasium as gym
import numpy as np
import h5py
import os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Define the goal string for CLIP reward
goal_string = 'a car on top of a mountain near a flag'
baseline_prompt = 'a car on a mountain'

# Load CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)

# Precompute text embedding
inputs_text = processor(text=goal_string, return_tensors="pt", padding=True)
text_embeddings = model.get_text_features(**inputs_text)
embedding_text = text_embeddings.cpu().detach().numpy()

# Precompute baseline embedding
inputs_baseline = processor(text=baseline_prompt, return_tensors="pt", padding=True)
baseline_embeddings = model.get_text_features(**inputs_baseline)
embedding_baseline = baseline_embeddings.cpu().detach().numpy()

# Generate CLIP reward
def clip_reward(obs):
    image = Image.fromarray(obs)
    inputs_img = processor(images=image, return_tensors="pt", padding=True)
    outputs_img = model.get_image_features(**inputs_img)
    embedding_img = outputs_img.cpu().detach().numpy()

    cosine_similarity_goal = np.dot(embedding_text, embedding_img.T) / (np.linalg.norm(embedding_text) * np.linalg.norm(embedding_img))
    cosine_similarity_baseline = np.dot(embedding_baseline, embedding_img.T) / (np.linalg.norm(embedding_baseline) * np.linalg.norm(embedding_img))
    cosine_similarity = cosine_similarity_goal - cosine_similarity_baseline
    
    return cosine_similarity.item()

def generate_dataset(env_name, num_episodes, max_steps, output_file):
    env = gym.make(env_name, render_mode='rgb_array')
    observations = []
    rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        for step in range(max_steps):
            action = env.action_space.sample()
            next_obs, reward, done, _, _ = env.step(action)
            observations.append(obs)
            rewards.append(clip_reward(obs))

            obs = next_obs
            if done:
                break

    env.close()

    # Convert lists to numpy arrays
    observations = np.array(observations)
    rewards = np.array(rewards)

    print("Observations shape:", observations.shape)
    print("Rewards shape:", rewards.shape)

    # Save to HDF5 file
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('observations', data=observations)
        f.create_dataset('rewards', data=rewards)
    
    print("Generated dataset saved to", output_file)

if __name__ == "__main__":
    env_name = 'MountainCarContinuous-v0'
    num_episodes = 30
    max_steps = 1000
    output_file = 'Datasets\\Student Dataset\\mountain_car_continuous_dataset.hdf5'

    generate_dataset(env_name, num_episodes, max_steps, output_file)