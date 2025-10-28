# Custom ML Model Integration Guide

This guide shows you how to train and use **your own ML model** with the RAG system instead of using Ollama's default embeddings.

## Quick Overview

The system has 3 main files for custom model integration:

1. **`train_custom_model.py`** - Train your own model on your books/documents
2. **`use_custom_model.py`** - Simple configuration to use your trained model
3. **`custom_model_integration.py`** - Advanced integration examples

## Step-by-Step Instructions

### Step 1: Prepare Your Training Data

1. Create a folder called `training_documents`:
```bash
mkdir training_documents
```

2. Add your books, PDFs, thesis, and documents to this folder
   - Supports: PDF, TXT, images (with OCR)

### Step 2: Train Your Model

**Option A: Simple TF-IDF Model (Recommended for beginners)**
```python
# Open train_custom_model.py
# Set TRAINING_METHOD = "tfidf"
# Run:
python train_custom_model.py
```

**Option B: Advanced Sentence-BERT Model (Better quality)**
```python
# First install: pip install sentence-transformers
# Open train_custom_model.py
# Set TRAINING_METHOD = "sentence-bert"
# Run:
python train_custom_model.py
```

**Option C: HuggingFace Transformers**
```python
# First install: pip install transformers torch
# Open train_custom_model.py
# Set TRAINING_METHOD = "huggingface"
# Run:
python train_custom_model.py
```

Your model will be saved to `./my_custom_model`

### Step 3: Configure the System to Use Your Model

Open `use_custom_model.py` and update the configuration:

```python
# Line 18-21: Update these settings
self.MODEL_TYPE = "sklearn"  # or "huggingface", "pytorch"
self.MODEL_PATH = "./my_custom_model"  # Path to your saved model
```

### Step 4: Activate Your Custom Model

Open `vector_db.py` and change this line:

**Before:**
```python
from embedding_function import OllamaEmbeddingFunction
```

**After:**
```python
from use_custom_model import CustomModelEmbedding as OllamaEmbeddingFunction
```

### Step 5: Restart the App

Restart the Streamlit application and your custom model will be used!

## Model Training Methods Explained

### 1. TF-IDF (sklearn)
- **Best for**: Domain-specific documents, offline use
- **Pros**: Fast, no GPU needed, works offline
- **Cons**: Not as sophisticated as neural models
- **When to use**: You have specific domain documents (medical, legal, technical)

### 2. Sentence-BERT
- **Best for**: General-purpose, high-quality embeddings
- **Pros**: State-of-the-art quality, good for diverse documents
- **Cons**: Requires more compute, needs GPU for large datasets
- **When to use**: You want the best quality and have compute resources

### 3. HuggingFace Transformers
- **Best for**: Customization, specific transformer models
- **Pros**: Flexible, many pre-trained options
- **Cons**: Complex, requires understanding of transformers
- **When to use**: You want to use a specific transformer model

## How It Works

```
Your Books/Documents
        ↓
Train Custom Model (train_custom_model.py)
        ↓
Saved Model (./my_custom_model)
        ↓
Configure (use_custom_model.py)
        ↓
RAG System uses YOUR model for embeddings
        ↓
Answers are based on YOUR trained model + YOUR data
```

## Example: Training on Medical Documents

Let's say you have medical books and papers:

1. Put all PDFs in `training_documents/`
2. Train TF-IDF model:
```python
python train_custom_model.py
```
3. Model learns medical terminology and concepts
4. When users ask medical questions, embeddings are optimized for medical text
5. Better retrieval of relevant medical information

## Advanced: Using Your Own PyTorch/TensorFlow Model

If you've already trained a custom PyTorch or TensorFlow model:

1. Save your model to a file
2. Update `use_custom_model.py`:
   - Set `MODEL_TYPE = "pytorch"` or create new type
   - Implement the embedding generation method
3. See `custom_model_integration.py` for examples

## Troubleshooting

**Model file not found**
- Make sure you ran `train_custom_model.py` first
- Check that `MODEL_PATH` matches where model was saved

**Import errors**
- Install required libraries: `pip install scikit-learn` or `pip install sentence-transformers`

**Poor retrieval quality**
- Train on more documents
- Try different training method (sentence-bert for quality)
- Adjust chunk size in document_processor.py

**Model too slow**
- Use TF-IDF instead of neural models
- Reduce embedding dimensions
- Enable batch processing

## Benefits of Custom Model

1. **Domain Expertise**: Model learns your specific domain language
2. **Privacy**: All training happens locally, no cloud services
3. **Optimization**: Embeddings optimized for your specific documents
4. **Control**: Full control over model architecture and training

## Comparison: Default vs Custom Model

| Feature | Default (Ollama) | Custom (Your Model) |
|---------|------------------|---------------------|
| Setup Time | Instant | Requires training |
| Quality | Good for general | Optimized for your domain |
| Offline | Requires Ollama running | Fully offline |
| Customization | Limited | Full control |
| Training Data | Pre-trained | Your documents |

## Next Steps

1. Start with TF-IDF for quick results
2. If quality is important, move to Sentence-BERT
3. For production, fine-tune on your specific use case
4. Monitor retrieval quality and retrain as needed

## Questions?

Check these files for more details:
- `custom_model_integration.py` - Advanced integration examples
- `train_custom_model.py` - Training script with detailed comments
- `use_custom_model.py` - Configuration and usage

Happy training! 🚀
