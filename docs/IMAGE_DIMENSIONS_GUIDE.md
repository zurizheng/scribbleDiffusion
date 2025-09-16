# Image Dimensions Optimization Guide for ScribbleDiffusion

## 🎯 **Recommendation: 256x256 Fixed Size**

### **Why 256x256 is Optimal:**

1. **Memory Efficiency**
   - 4x less memory than 512x512
   - Allows larger batch sizes
   - Faster training iterations

2. **Quality vs Speed Balance**
   - Still high enough for detailed sketches
   - Fast enough for rapid experimentation
   - Good for focused datasets (like apples)

3. **Stable Diffusion Compatibility**
   - VAE works well at 256x256
   - Latent space is 32x32 (manageable)
   - Less prone to artifacts

### **Comparison Analysis:**

| Dimension | Memory | Speed | Quality | Batch Size | Recommendation |
|-----------|--------|-------|---------|------------|----------------|
| 100x100   | Low    | Fast  | Poor    | Large      | ❌ Too small for details |
| 256x256   | Medium | Good  | Good    | Medium     | ✅ **OPTIMAL** |
| 512x512   | High   | Slow  | Best    | Small      | ⚠️ Only if you have 8GB+ VRAM |
| Variable  | ???    | Slow  | ???     | Complex    | ❌ Complex, inconsistent |

### **Why NOT 100x100:**
- Sketches lose important details
- VAE doesn't work well at low res
- Generated images look pixelated
- Hard to see if model is working

### **Why NOT Variable Sizes:**
- Inconsistent batch processing
- Complex data loading
- Memory usage unpredictable
- Harder to debug issues

## 🛠️ **Implementation:**

```yaml
# Optimal config for focused training
data:
  image_size: 256        # Fixed size
  sketch_size: 256       # Match image size
  batch_size: 8          # Can increase with 256x256
  
training:
  max_train_steps: 2000  # Faster convergence with focused dataset
  gradient_accumulation_steps: 2  # Effective batch size = 16
```

## 🔧 **For Your Setup:**

Given you're switching to better hardware, I recommend:

1. **Start with 256x256** for initial experiments
2. **Use focused apple dataset** (much better than COCO)
3. **Once working, scale to 512x512** if needed

## 📊 **Expected Results with 256x256 + Apple Dataset:**

- ✅ Clear apple generation (no kaleidoscope)
- ✅ Fast training (2-3 hours vs 20+ hours)
- ✅ Good sketch conditioning
- ✅ Manageable memory usage
- ✅ Easy to debug and iterate

## 🚀 **Next Steps:**

1. Fix model saving ✅ (Done above)
2. Set dimensions to 256x256
3. Create focused apple dataset
4. Train on better hardware
5. Scale up if needed

The key is starting with something that works reliably, then scaling up!