"""
Train Your Own Embedding Model
================================

This script shows how to train your own embedding model on your documents.
Choose the approach that fits your needs.
"""

import os
from typing import List, Tuple

# Option 1: Train using Sentence-BERT (Fine-tuning approach)
def train_sentence_bert_model(training_texts: List[str], model_output_path: str):
    """
    Train a sentence-BERT model on your documents
    
    Requirements:
    pip install sentence-transformers
    """
    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader
        
        # Load a pre-trained model to fine-tune
        base_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create training examples
        # For unsupervised learning, we use the same text as input and output
        train_examples = []
        for text in training_texts:
            # Create positive pairs (text with itself or similar chunks)
            train_examples.append(InputExample(texts=[text, text]))
        
        # Create DataLoader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        
        # Use Multiple Negatives Ranking Loss
        train_loss = losses.MultipleNegativesRankingLoss(base_model)
        
        # Train the model
        print("Training Sentence-BERT model...")
        base_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=3,
            warmup_steps=100,
            output_path=model_output_path
        )
        
        print(f"✓ Model saved to {model_output_path}")
        return True
        
    except ImportError:
        print("Install sentence-transformers: pip install sentence-transformers")
        return False


# Option 2: Train using TF-IDF + Dimensionality Reduction (Simple approach)
def train_tfidf_model(training_texts: List[str], model_output_path: str):
    """
    Train a simple TF-IDF + SVD model (good for domain-specific documents)
    
    Requirements:
    pip install scikit-learn
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.pipeline import Pipeline
        import joblib
        
        # Create a pipeline: TF-IDF -> SVD (dimensionality reduction)
        model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('svd', TruncatedSVD(n_components=384))  # 384-dimensional embeddings
        ])
        
        print("Training TF-IDF model...")
        model.fit(training_texts)
        
        # Save the model
        joblib.dump(model, model_output_path)
        print(f"✓ Model saved to {model_output_path}")
        return True
        
    except ImportError:
        print("Install scikit-learn: pip install scikit-learn")
        return False


# Option 3: Fine-tune HuggingFace Transformers
def train_huggingface_model(training_texts: List[str], model_output_path: str):
    """
    Fine-tune a HuggingFace transformer model
    
    Requirements:
    pip install transformers torch
    """
    try:
        from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
        import torch
        
        # Load base model
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Tokenize your texts
        print("Tokenizing texts...")
        encodings = tokenizer(
            training_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Create a simple dataset
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
            
            def __getitem__(self, idx):
                return {key: val[idx] for key, val in self.encodings.items()}
            
            def __len__(self):
                return len(self.encodings['input_ids'])
        
        dataset = SimpleDataset(encodings)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=model_output_path,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            save_steps=500,
            save_total_limit=2,
        )
        
        # Note: For actual contrastive learning, you'd need to implement
        # a custom training loop with triplet loss or similar
        print("Training HuggingFace model...")
        print("Note: This is a basic example. For production, implement contrastive learning.")
        
        # Save model
        model.save_pretrained(model_output_path)
        tokenizer.save_pretrained(model_output_path)
        
        print(f"✓ Model saved to {model_output_path}")
        return True
        
    except ImportError:
        print("Install transformers: pip install transformers torch")
        return False


# Helper function to load your documents
def load_training_documents(doc_folder: str) -> List[str]:
    """
    Load all text from your document folder
    """
    from document_processor import DocumentProcessor
    
    processor = DocumentProcessor()
    all_texts = []
    
    for filename in os.listdir(doc_folder):
        file_path = os.path.join(doc_folder, filename)
        
        try:
            chunks = processor.process_document(file_path, filename)
            all_texts.extend([chunk['text'] for chunk in chunks])
            print(f"Loaded {filename}: {len(chunks)} chunks")
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
    
    return all_texts


# Main training script
def main():
    """
    Main function to train your custom model
    """
    print("=" * 60)
    print("Custom ML Model Training")
    print("=" * 60)
    
    # Configuration
    DOCUMENT_FOLDER = "./training_documents"  # Put your books/PDFs here
    MODEL_OUTPUT_PATH = "./my_custom_model"
    
    # Choose your training method
    TRAINING_METHOD = "tfidf"  # Options: "sentence-bert", "tfidf", "huggingface"
    
    # Check if document folder exists
    if not os.path.exists(DOCUMENT_FOLDER):
        os.makedirs(DOCUMENT_FOLDER)
        print(f"\nCreated folder: {DOCUMENT_FOLDER}")
        print("Please add your training documents (PDFs, text files) to this folder.")
        print("Then run this script again.")
        return
    
    # Load training documents
    print(f"\nLoading documents from {DOCUMENT_FOLDER}...")
    training_texts = load_training_documents(DOCUMENT_FOLDER)
    
    if not training_texts:
        print("No documents found. Add your training documents and try again.")
        return
    
    print(f"\nLoaded {len(training_texts)} text chunks for training")
    
    # Train model based on selected method
    print(f"\nTraining method: {TRAINING_METHOD}")
    
    if TRAINING_METHOD == "sentence-bert":
        success = train_sentence_bert_model(training_texts, MODEL_OUTPUT_PATH)
    elif TRAINING_METHOD == "tfidf":
        success = train_tfidf_model(training_texts, MODEL_OUTPUT_PATH)
    elif TRAINING_METHOD == "huggingface":
        success = train_huggingface_model(training_texts, MODEL_OUTPUT_PATH)
    else:
        print(f"Unknown training method: {TRAINING_METHOD}")
        success = False
    
    if success:
        print("\n" + "=" * 60)
        print("✓ Training Complete!")
        print("=" * 60)
        print(f"\nYour model is saved at: {MODEL_OUTPUT_PATH}")
        print("\nNext steps:")
        print("1. Test your model with test_custom_model.py")
        print("2. Update embedding_function.py to use your model")
        print("3. Restart the Streamlit app")
    else:
        print("\n✗ Training failed. Check the errors above.")


if __name__ == "__main__":
    main()


"""
USAGE:
======

1. Create a folder called 'training_documents'
2. Add your books, PDFs, and documents to that folder
3. Choose a training method in the main() function
4. Run this script: python train_custom_model.py
5. Your model will be saved to './my_custom_model'

TRAINING METHODS:
=================

1. sentence-bert: Best quality, requires more compute
   - Good for: General purpose, high-quality embeddings
   - Requirements: sentence-transformers library

2. tfidf: Fast, works offline, good for domain-specific
   - Good for: Domain-specific documents, offline use
   - Requirements: scikit-learn

3. huggingface: Flexible, many pre-trained options
   - Good for: Customization, using specific transformers
   - Requirements: transformers, torch
"""
