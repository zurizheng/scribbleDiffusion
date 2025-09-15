#!/usr/bin/env python3
"""
Analyze the diversity of the COCO training subset
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

def analyze_coco_subset():
    """Analyze what's in our 1000-image COCO subset"""
    
    # COCO annotation file
    coco_root = Path("/workspace/scribbleDiffusion/data/coco")
    ann_file = coco_root / "annotations" / "instances_train2017.json"
    
    if not ann_file.exists():
        print("❌ COCO annotations not found")
        return
    
    print("📊 Analyzing COCO subset diversity...")
    
    # Load annotations
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # Get first 1000 images (same as our dataset)
    images = coco_data['images'][:1000]
    image_ids = {img['id'] for img in images}
    
    # Category mapping
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Analyze annotations for our subset
    category_counts = Counter()
    image_category_counts = defaultdict(set)
    
    for ann in coco_data['annotations']:
        if ann['image_id'] in image_ids:
            cat_name = categories[ann['category_id']]
            category_counts[cat_name] += 1
            image_category_counts[ann['image_id']].add(cat_name)
    
    print(f"\n📈 DIVERSITY ANALYSIS:")
    print(f"Total images: {len(images)}")
    print(f"Unique categories: {len(category_counts)}")
    print(f"Total object instances: {sum(category_counts.values())}")
    print(f"Avg objects per image: {sum(category_counts.values()) / len(images):.1f}")
    
    print(f"\n🏆 TOP 20 CATEGORIES:")
    for cat, count in category_counts.most_common(20):
        print(f"  {cat}: {count}")
    
    # Images per category count
    images_per_cat = Counter(len(cats) for cats in image_category_counts.values())
    print(f"\n🎯 IMAGES BY CATEGORY COUNT:")
    for cat_count, img_count in sorted(images_per_cat.items()):
        print(f"  {img_count} images have {cat_count} categories")
    
    # Check coverage of common object types
    important_categories = [
        'person', 'car', 'dog', 'cat', 'bird', 'horse', 'bicycle', 'motorcycle',
        'airplane', 'bus', 'train', 'truck', 'boat', 'chair', 'couch', 'table',
        'bed', 'tv', 'laptop', 'mouse', 'keyboard', 'cell phone', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    print(f"\n✅ COVERAGE OF IMPORTANT CATEGORIES:")
    covered = 0
    for cat in important_categories:
        if cat in category_counts:
            print(f"  ✅ {cat}: {category_counts[cat]} instances")
            covered += 1
        else:
            print(f"  ❌ {cat}: missing")
    
    print(f"\nCoverage: {covered}/{len(important_categories)} ({covered/len(important_categories)*100:.1f}%)")
    
    return {
        'total_images': len(images),
        'unique_categories': len(category_counts),
        'category_counts': dict(category_counts),
        'coverage_ratio': covered/len(important_categories)
    }

def assess_diversity_for_scribblediffusion():
    """Assess if 1000 images is enough for ScribbleDiffusion"""
    
    stats = analyze_coco_subset()
    
    print(f"\n🎯 SCRIBBLEDIFFUSION ASSESSMENT:")
    
    # ScribbleDiffusion needs good edge/sketch diversity
    if stats['unique_categories'] >= 40:
        print(f"✅ Category diversity: EXCELLENT ({stats['unique_categories']} categories)")
    elif stats['unique_categories'] >= 25:
        print(f"🟡 Category diversity: GOOD ({stats['unique_categories']} categories)")
    else:
        print(f"❌ Category diversity: LIMITED ({stats['unique_categories']} categories)")
    
    if stats['coverage_ratio'] >= 0.7:
        print(f"✅ Important objects coverage: EXCELLENT ({stats['coverage_ratio']*100:.1f}%)")
    elif stats['coverage_ratio'] >= 0.5:
        print(f"🟡 Important objects coverage: GOOD ({stats['coverage_ratio']*100:.1f}%)")
    else:
        print(f"❌ Important objects coverage: LIMITED ({stats['coverage_ratio']*100:.1f}%)")
    
    if stats['total_images'] >= 1000:
        print(f"✅ Dataset size: GOOD for proof-of-concept ({stats['total_images']} images)")
    else:
        print(f"🟡 Dataset size: SMALL ({stats['total_images']} images)")
    
    print(f"\n📋 RECOMMENDATIONS:")
    if stats['coverage_ratio'] >= 0.6 and stats['unique_categories'] >= 30:
        print("🎯 1000 images should be SUFFICIENT for ScribbleDiffusion proof-of-concept")
        print("🚀 Good diversity for learning sketch-to-image mapping")
        print("💡 Can scale up to 5K-10K later for production quality")
    else:
        print("⚠️  1000 images might be LIMITED for robust ScribbleDiffusion")
        print("📈 Consider increasing to 2000-5000 images")
        print("🎯 Focus on categories with clear, recognizable shapes")

if __name__ == "__main__":
    assess_diversity_for_scribblediffusion()