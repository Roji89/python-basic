
# Vector similarity metrics — guide

This document explains L2 (Euclidean) distance, cosine distance, and dot product, with real‑world uses and practical guidance.

---

## L2 (Euclidean) distance
What it is  
- Straight‑line distance between two vectors in d‑dimensional space. It measures absolute difference in all coordinates.

Intuition  
- “How far apart are these two points?” — both direction and magnitude matter.

When to use (real world)  
- Image retrieval using raw CNN features (find visually closest images).  
- Clustering numeric sensor or feature data where absolute differences matter.  
- Anomaly detection where a large deviation from a centroid indicates an outlier.

Pros / Cons  
- Pros: intuitive, suitable when magnitude/scale matters.  
- Cons: sensitive to vector length; long vectors dominate distances.

---

## Cosine similarity / cosine distance
What it is  
- Cosine similarity measures the angle between vectors (range −1..1). Cosine distance is commonly defined as 1 − cosine_similarity.

Intuition  
- “Do these vectors point in the same direction?” — ignores magnitude, focuses on orientation/shape.

When to use (real world)  
- Semantic search and document retrieval (match meaning regardless of document length).  
- Comparing sentence or paragraph embeddings in NLP (RAG systems).  
- Use when you want similarity independent of vector length.

Pros / Cons  
- Pros: robust to differing lengths and token counts; good for semantic comparisons.  
- Cons: ignores magnitude, so absolute scale differences are lost.

Practical note  
- Typically implemented by normalizing vectors to unit length and using dot product (fast and equivalent).

---

## Dot product (inner product)
What it is  
- Sum of elementwise products of two vectors; a raw similarity score combining direction and magnitude.

Intuition  
- “How aligned are they, weighted by magnitude?” — large when vectors point similarly and are large.

When to use (real world)  
- Transformer attention scores and neural network scoring (raw inner products used before softmax).  
- Recommendation systems where vector magnitude encodes popularity/confidence (so popular relevant items score higher).  
- Fast retrieval when vectors are already normalized (dot on unit vectors = cosine similarity).

Pros / Cons  
- Pros: simple, fast; preserves magnitude information if that is meaningful.  
- Cons: unbounded and not directly comparable across different scales unless normalized or otherwise adjusted.

---

## Comparison and selection guidance
- Semantic text/document search / RAG → cosine similarity (normalize vectors first) or dot product on normalized vectors.  
- Image retrieval / clustering raw features → L2 (Euclidean) distance.  
- Attention mechanisms or systems where magnitude encodes signal → dot product.  
- If you need a bounded similarity in [−1,1] → cosine similarity.

---

## Practical tips & pitfalls
- Consistency: apply the same preprocessing (normalization or not) for stored vectors and queries.  
- Normalization: convert vectors to unit length to make dot product equal cosine similarity; guard against zero vectors.  
- Metric choice: choose index and DB settings according to metric (L2 vs inner product).  
- Scale: for large datasets use ANN indexes (FAISS, HNSW, Annoy, etc.) configured for the chosen metric.  
- Conversion: to convert similarity → distance use a monotonic transform (e.g., distance = 1 − cosine_similarity).  
- Edge cases: detect and handle zero vectors or NaNs before normalization.

---

## Short summary
- L2 (Euclidean): use when absolute closeness matters.  
- Cosine: use when direction/meaning matters, independent of length.  
- Dot product: use when magnitude should influence score or when working with normalized vectors.

