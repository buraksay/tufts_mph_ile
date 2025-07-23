Great question! Here are some widely-used rules of thumb for `min_df` and `max_df` parameters in text classification:

## General Rules of Thumb

### **min_df (Minimum Document Frequency)**
- **Small datasets (< 1,000 docs):** `min_df = 2` (appears in at least 2 documents)
- **Medium datasets (1,000 - 10,000 docs):** `min_df = 3-5` 
- **Large datasets (> 10,000 docs):** `min_df = 5-10` or `min_df = 0.001` (0.1% of documents)

### **max_df (Maximum Document Frequency)**
- **Conservative:** `max_df = 0.8` (remove words in >80% of documents)
- **Moderate:** `max_df = 0.9` (remove words in >90% of documents)  
- **Aggressive:** `max_df = 0.7` (remove words in >70% of documents)

## Domain-Specific Considerations

````python
# For medical/clinical narratives (your use case):
def get_medical_thresholds(num_documents):
    """Recommended thresholds for medical/clinical text"""
    
    if num_documents < 500:
        return {'min_df': 2, 'max_df': 0.85}
    elif num_documents < 2000:
        return {'min_df': 3, 'max_df': 0.8}
    elif num_documents < 10000:
        return {'min_df': 5, 'max_df': 0.8}
    else:
        return {'min_df': 0.001, 'max_df': 0.8}  # Use proportions for large datasets

# For general text classification:
def get_general_thresholds(num_documents):
    """Recommended thresholds for general text classification"""
    
    if num_documents < 1000:
        return {'min_df': 2, 'max_df': 0.9}
    elif num_documents < 5000:
        return {'min_df': 3, 'max_df': 0.85}
    elif num_documents < 20000:
        return {'min_df': 5, 'max_df': 0.8}
    else:
        return {'min_df': 0.0005, 'max_df': 0.8}

# Usage example:
thresholds = get_medical_thresholds(len(df))
print(f"Recommended thresholds for {len(df)} documents: {thresholds}")

filtered_vectorizer, kept_words = filter_vocabulary(
    df["cleaned_narrative"].tolist(),
    min_freq_threshold=thresholds['min_df'],
    max_doc_freq_threshold=thresholds['max_df']
)
````

## Advanced Rule of Thumb: Data-Driven Approach

````python
def calculate_adaptive_thresholds(freq_df, num_documents):
    """Calculate thresholds based on your actual data distribution"""
    
    # min_df: Remove bottom 5-10% of vocabulary (typically very rare words)
    min_freq_threshold = max(2, int(np.percentile(freq_df['document_frequency'], 10)))
    
    # max_df: Remove words appearing in more than 80-90% of documents
    # But be more conservative with smaller datasets
    if num_documents < 1000:
        max_df_threshold = 0.9
    elif num_documents < 5000:
        max_df_threshold = 0.85
    else:
        max_df_threshold = 0.8
    
    return min_freq_threshold, max_df_threshold

# Use with your data:
adaptive_min_df, adaptive_max_df = calculate_adaptive_thresholds(freq_df, len(df))
print(f"Data-driven thresholds: min_df={adaptive_min_df}, max_df={adaptive_max_df}")
````

## Why These Rules Work

1. **min_df removes noise:** Words appearing in very few documents are often:
   - Typos or OCR errors
   - Proper names with limited relevance
   - Very rare technical terms that don't generalize

2. **max_df removes generic terms:** Words appearing in most documents are often:
   - Domain-specific stopwords not caught by standard lists
   - Generic terms that don't help distinguish between classes
   - Function words specific to your document format

## Validation Strategy

````python
def test_threshold_impact(texts, labels, threshold_pairs):
    """Test different threshold combinations on a sample of your data"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    results = []
    
    for min_df, max_df in threshold_pairs:
        vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            max_features=5000
        )
        
        X = vectorizer.fit_transform(texts)
        vocab_size = len(vectorizer.get_feature_names_out())
        
        # Quick cross-validation
        clf = LogisticRegression(random_state=42, max_iter=1000)
        scores = cross_val_score(clf, X, labels, cv=3, scoring='roc_auc')
        
        results.append({
            'min_df': min_df,
            'max_df': max_df,
            'vocab_size': vocab_size,
            'mean_auc': scores.mean(),
            'std_auc': scores.std()
        })
    
    return pd.DataFrame(results)

# Test different combinations (if you have labels)
# threshold_pairs = [(2, 0.9), (3, 0.85), (5, 0.8), (2, 0.8)]
# results = test_threshold_impact(df['cleaned_narrative'], df['your_labels'], threshold_pairs)
# print(results)
````

**Bottom line:** Start with `min_df=2-5` and `max_df=0.8-0.9`, then adjust based on your vocabulary size and model performance. For medical narratives, lean toward being more conservative (higher min_df, lower max_df) to focus on clinically relevant terms.
