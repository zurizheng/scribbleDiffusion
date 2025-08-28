#!/bin/bash
# Quick script to monitor COCO download/extraction progress
# Run this in a separate terminal while download_coco.sh is running

COCO_DIR="data/coco"

echo "🔍 COCO Dataset Progress Monitor"
echo "=" * 40

while true; do
    clear
    echo "🔍 COCO Dataset Progress Monitor - $(date)"
    echo "=" * 50
    
    if [ -d "$COCO_DIR" ]; then
        cd "$COCO_DIR"
        
        # Check download progress
        echo "📥 Download Status:"
        if [ -f "train2017.zip" ]; then
            train_size=$(du -h train2017.zip 2>/dev/null | cut -f1)
            echo "   ✅ train2017.zip: $train_size (target: ~18GB)"
        else
            echo "   ⏳ train2017.zip: downloading..."
        fi
        
        if [ -f "val2017.zip" ]; then
            val_size=$(du -h val2017.zip 2>/dev/null | cut -f1)
            echo "   ✅ val2017.zip: $val_size (target: ~1GB)"
        else
            echo "   ⏳ val2017.zip: downloading..."
        fi
        
        if [ -f "annotations_trainval2017.zip" ]; then
            ann_size=$(du -h annotations_trainval2017.zip 2>/dev/null | cut -f1)
            echo "   ✅ annotations_trainval2017.zip: $ann_size (target: ~241MB)"
        else
            echo "   ⏳ annotations_trainval2017.zip: downloading..."
        fi
        
        echo ""
        
        # Check extraction progress
        echo "📦 Extraction Status:"
        if [ -d "train2017" ]; then
            train_count=$(ls train2017/ 2>/dev/null | wc -l)
            echo "   ✅ train2017/: $train_count files (target: ~118,287)"
            
            if [ "$train_count" -gt 0 ] && [ "$train_count" -lt 118287 ]; then
                percent=$(echo "scale=1; $train_count * 100 / 118287" | bc -l 2>/dev/null || echo "calculating...")
                echo "      Progress: $percent% extracted"
            fi
        else
            echo "   ⏳ train2017/: not started"
        fi
        
        if [ -d "val2017" ]; then
            val_count=$(ls val2017/ 2>/dev/null | wc -l)
            echo "   ✅ val2017/: $val_count files (target: ~5,000)"
        else
            echo "   ⏳ val2017/: not started"
        fi
        
        if [ -d "annotations" ]; then
            ann_count=$(ls annotations/ 2>/dev/null | wc -l)
            echo "   ✅ annotations/: $ann_count files (target: ~6)"
        else
            echo "   ⏳ annotations/: not started"
        fi
        
        echo ""
        echo "💾 Disk Usage:"
        du -sh . 2>/dev/null | head -1
        
    else
        echo "❌ COCO directory not found: $COCO_DIR"
        echo "💡 Make sure you're running this from the project root"
    fi
    
    echo ""
    echo "🔄 Updating every 10 seconds... (Ctrl+C to stop)"
    sleep 10
done
