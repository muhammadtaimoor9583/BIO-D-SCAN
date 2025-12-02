"""
YOLO Dataset Splitter
Splits images and labels into train, val, and test sets.
"""

import os
import shutil
from pathlib import Path
import random
from typing import List, Tuple
import yaml


def get_image_label_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    """Get pairs of image and label files."""
    pairs = []
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(f'*{ext}')))
        image_files.extend(list(images_dir.glob(f'*{ext.upper()}')))
    
    # Match with corresponding label files
    for image_path in image_files:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))
        else:
            print(f"Warning: No label found for {image_path.name}")
    
    return pairs


def split_dataset(
    source_images_dir: str,
    source_labels_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    Split dataset into train, val, and test sets.
    
    Args:
        source_images_dir: Directory containing all images
        source_labels_dir: Directory containing all labels
        output_dir: Base output directory for the split dataset
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.2)
        test_ratio: Proportion of data for testing (default: 0.1)
        seed: Random seed for reproducibility
    """
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Set random seed
    random.seed(seed)
    
    # Convert to Path objects
    source_images = Path(source_images_dir)
    source_labels = Path(source_labels_dir)
    output_base = Path(output_dir)
    
    # Get all image-label pairs
    print("Finding image-label pairs...")
    pairs = get_image_label_pairs(source_images, source_labels)
    print(f"Found {len(pairs)} image-label pairs")
    
    if len(pairs) == 0:
        print("Error: No image-label pairs found!")
        return
    
    # Shuffle pairs
    random.shuffle(pairs)
    
    # Calculate split indices
    total = len(pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    splits = {
        'train': pairs[:train_end],
        'val': pairs[train_end:val_end],
        'test': pairs[val_end:]
    }
    
    # Create directory structure and copy files
    for split_name, split_pairs in splits.items():
        print(f"\nProcessing {split_name} split ({len(split_pairs)} samples)...")
        
        # Create directories
        images_dir = output_base / 'images' / split_name
        labels_dir = output_base / 'labels' / split_name
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        for img_path, label_path in split_pairs:
            # Copy image
            shutil.copy2(img_path, images_dir / img_path.name)
            # Copy label
            shutil.copy2(label_path, labels_dir / label_path.name)
        
        print(f"  Copied {len(split_pairs)} images and labels to {split_name}")
    
    # Create/update dataset.yaml
    create_dataset_yaml(output_base, source_labels)
    
    # Print summary
    print("\n" + "="*60)
    print("Dataset Split Summary:")
    print("="*60)
    print(f"Total samples: {total}")
    print(f"Train: {len(splits['train'])} ({len(splits['train'])/total*100:.1f}%)")
    print(f"Val:   {len(splits['val'])} ({len(splits['val'])/total*100:.1f}%)")
    print(f"Test:  {len(splits['test'])} ({len(splits['test'])/total*100:.1f}%)")
    print(f"\nOutput directory: {output_base}")
    print("="*60)


def create_dataset_yaml(output_dir: Path, source_labels_dir: Path):
    """Create dataset.yaml file for YOLO."""
    
    # Try to infer class names from existing dataset.yaml or create default
    yaml_path = output_dir.parent / 'dataset.yaml'
    
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            existing_yaml = yaml.safe_load(f)
            class_names = existing_yaml.get('names', {})
    else:
        # Analyze label files to determine number of classes
        max_class = -1
        for label_file in Path(source_labels_dir).glob('*.txt'):
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.split()[0])
                            max_class = max(max_class, class_id)
            except:
                pass
        
        # Create default class names
        class_names = {i: f'class_{i}' for i in range(max_class + 1)}
    
    # Create YAML content
    yaml_content = {
        'path': str(output_dir.absolute()),
        'train': './images/train',
        'val': './images/val',
        'test': './images/test',
        'names': class_names,
        'nc': len(class_names)
    }
    
    # Write YAML file
    output_yaml = output_dir / 'dataset.yaml'
    with open(output_yaml, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nCreated {output_yaml}")


def main():
    """Main function with configuration."""
    
    # Configuration
    SOURCE_IMAGES_DIR = '/home/muneeb/fyp/dataset_2000/images/val'
    SOURCE_LABELS_DIR = '/home/muneeb/fyp/dataset_2000/labels/val'
    OUTPUT_DIR = '/home/muneeb/fyp/dataset_split'
    
    # Split ratios (must sum to 1.0)
    TRAIN_RATIO = 0.7  # 70% for training
    VAL_RATIO = 0.2    # 20% for validation
    TEST_RATIO = 0.1   # 10% for testing
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    print("YOLO Dataset Splitter")
    print("=" * 60)
    print(f"Source images: {SOURCE_IMAGES_DIR}")
    print(f"Source labels: {SOURCE_LABELS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Split ratios - Train: {TRAIN_RATIO}, Val: {VAL_RATIO}, Test: {TEST_RATIO}")
    print("=" * 60 + "\n")
    
    # Split the dataset
    split_dataset(
        source_images_dir=SOURCE_IMAGES_DIR,
        source_labels_dir=SOURCE_LABELS_DIR,
        output_dir=OUTPUT_DIR,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=RANDOM_SEED
    )
    
    print("\n✓ Dataset split completed successfully!")


if __name__ == "__main__":
    main()
