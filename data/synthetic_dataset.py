"""
Generate synthetic dataset for multimodal content moderation.

Creates 10K text+image pairs with labels:
- Safe: 70% (neutral content)
- Unsafe: 30% (toxic, harmful, explicit)

Features realistic content diversity.
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import random

# Text templates for synthetic data
SAFE_TEXT_TEMPLATES = [
    "Check out this beautiful sunset! 🌅",
    "Just finished a great workout at the gym",
    "Making pasta for dinner tonight, looks delicious",
    "Love spending time with my family",
    "Just got back from a hike, gorgeous views",
    "Coffee and morning reading ☕📚",
    "New album just dropped, can't stop listening",
    "Planning a road trip next month, any suggestions?",
    "My puppy is so cute and fluffy 🐕",
    "Finishing up a project, feeling productive",
    "Morning jog before work",
    "Homemade pizza night with friends",
    "Just started reading this amazing book",
    "Beach day tomorrow, can't wait!",
    "Cooking class was so fun today",
    "Concert tickets for next month secured!",
    "New haircut, feeling refreshed",
    "Gardening in the backyard",
    "Movie night with the family",
    "Learning to play the guitar",
]

UNSAFE_TEXT_TEMPLATES = [
    "I hate people who [negative stereotype]",
    "You're an idiot if you believe that",
    "Kill yourself, nobody likes you anyway",
    "[Slur] should not exist in society",
    "I'm going to hurt you when I find you",
    "Death to all [group]",
    "You deserve to suffer",
    "Sexual violence is hilarious",
    "Let's lynch [target]",
    "You're disgusting and worthless",
    "I want to see you dead",
    "Rape is not that bad",
    "Hitler did nothing wrong",
    "All [group] are criminals",
    "You should commit suicide",
    "I hope your family dies",
    "That's so gay (as insult)",
    "Women are inferior to men",
    "Go back to your country",
    "You deserve to be assaulted",
]

NEUTRAL_WORDS = ["the", "a", "an", "is", "are", "was", "were", "this", "that", "of", "to", "in", "at", "by"]


def generate_text(is_unsafe: bool) -> str:
    """Generate synthetic text content."""
    templates = UNSAFE_TEXT_TEMPLATES if is_unsafe else SAFE_TEXT_TEMPLATES
    text = random.choice(templates)
    
    # Add some variation
    if random.random() < 0.3:
        extra = " " + random.choice(NEUTRAL_WORDS) + " " + random.choice(["today", "yesterday", "tomorrow"])
        text = text + extra
    
    return text


def generate_image(is_unsafe: bool, size: tuple = (224, 224)) -> Image.Image:
    """Generate synthetic image content."""
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    if is_unsafe:
        # Dark, aggressive colors for unsafe content
        bg_color = (100 + random.randint(0, 50), 20 + random.randint(0, 30), 20 + random.randint(0, 30))
        text_color = (255, 100, 100)
        text = "⚠️ UNSAFE"
    else:
        # Bright, positive colors for safe content
        bg_color = (150 + random.randint(0, 100), 180 + random.randint(0, 70), 100 + random.randint(0, 100))
        text_color = (100, 200, 100)
        text = "✓ SAFE"
    
    # Fill background with gradient-like effect
    for y in range(size[1]):
        intensity = int(200 + 50 * np.sin(y / size[1] * np.pi))
        draw.rectangle(
            [(0, y), (size[0], y + 1)],
            fill=tuple(min(255, max(0, c + intensity - 200)) for c in bg_color)
        )
    
    # Add some visual noise/shapes
    for _ in range(random.randint(3, 8)):
        x = random.randint(0, size[0])
        y = random.randint(0, size[1])
        radius = random.randint(10, 50)
        draw.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)],
            fill=tuple(random.randint(50, 200) for _ in range(3)),
            outline=text_color,
            width=2
        )
    
    # Add label text (simplified without font)
    try:
        draw.text((size[0]//2 - 40, size[1]//2 - 10), text, fill=text_color)
    except:
        pass  # Font not available, skip text
    
    return img


def create_dataset(num_samples: int = 10000, output_dir: str = "data", seed: int = 42) -> dict:
    """
    Create synthetic dataset.
    
    Args:
        num_samples: Total number of samples
        output_dir: Where to save dataset
        seed: Random seed
    
    Returns:
        Dataset statistics
    """
    random.seed(seed)
    np.random.seed(seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_path / "images").mkdir(exist_ok=True)
    (output_path / "text").mkdir(exist_ok=True)
    
    # Split: 70% safe, 30% unsafe
    num_safe = int(num_samples * 0.7)
    num_unsafe = num_samples - num_safe
    
    dataset = []
    
    print(f"Generating {num_samples} samples...")
    print(f"  Safe: {num_safe} (70%)")
    print(f"  Unsafe: {num_unsafe} (30%)")
    
    # Generate safe samples
    for idx in tqdm(range(num_safe), desc="Safe samples"):
        sample_id = f"safe_{idx:06d}"
        
        text = generate_text(is_unsafe=False)
        image = generate_image(is_unsafe=False)
        
        # Save image
        img_path = output_path / "images" / f"{sample_id}.png"
        image.save(img_path)
        
        # Save text
        txt_path = output_path / "text" / f"{sample_id}.txt"
        txt_path.write_text(text)
        
        dataset.append({
            "id": sample_id,
            "text": text,
            "image_path": str(img_path),
            "text_path": str(txt_path),
            "label": 0,  # Safe
            "label_name": "safe",
        })
    
    # Generate unsafe samples
    for idx in tqdm(range(num_unsafe), desc="Unsafe samples"):
        sample_id = f"unsafe_{idx:06d}"
        
        text = generate_text(is_unsafe=True)
        image = generate_image(is_unsafe=True)
        
        # Save image
        img_path = output_path / "images" / f"{sample_id}.png"
        image.save(img_path)
        
        # Save text
        txt_path = output_path / "text" / f"{sample_id}.txt"
        txt_path.write_text(text)
        
        dataset.append({
            "id": sample_id,
            "text": text,
            "image_path": str(img_path),
            "text_path": str(txt_path),
            "label": 1,  # Unsafe
            "label_name": "unsafe",
        })
    
    # Shuffle
    random.shuffle(dataset)
    
    # Save metadata
    metadata_path = output_path / "dataset.json"
    with open(metadata_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    # Save split information
    split_path = output_path / "splits.json"
    splits = {
        "train": {
            "indices": list(range(0, int(len(dataset) * 0.7))),
            "size": int(len(dataset) * 0.7)
        },
        "val": {
            "indices": list(range(int(len(dataset) * 0.7), int(len(dataset) * 0.85))),
            "size": int(len(dataset) * 0.15)
        },
        "test": {
            "indices": list(range(int(len(dataset) * 0.85), len(dataset))),
            "size": int(len(dataset) * 0.15)
        },
        "unlabeled": {
            "indices": list(range(0, len(dataset))),
            "size": len(dataset)
        }
    }
    with open(split_path, 'w') as f:
        json.dump(splits, f, indent=2)
    
    stats = {
        "total_samples": len(dataset),
        "safe_samples": num_safe,
        "unsafe_samples": num_unsafe,
        "safe_ratio": num_safe / len(dataset),
        "unsafe_ratio": num_unsafe / len(dataset),
        "image_dir": str(output_path / "images"),
        "text_dir": str(output_path / "text"),
        "metadata_file": str(metadata_path),
        "splits_file": str(split_path),
    }
    
    print(f"\n✓ Dataset created successfully!")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Safe: {stats['safe_samples']} ({stats['safe_ratio']:.1%})")
    print(f"  Unsafe: {stats['unsafe_samples']} ({stats['unsafe_ratio']:.1%})")
    print(f"  Location: {output_path}")
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic dataset")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    stats = create_dataset(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
