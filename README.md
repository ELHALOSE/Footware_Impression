# 👣 Footwear Impression Retrieval using Siamese Network and Triplet Loss

## 📌 Problem Statement
Crime scene footwear impressions are often partial, noisy, and vary in quality. Matching them against a clean reference gallery is a vital step in forensic investigations.

This project proposes a **deep learning-based approach** for retrieving the most similar reference shoeprint using a **Siamese Network trained with Triplet Loss**.

---

## 📂 Dataset: CSFID-300
We use the CSFID-300 dataset, containing:
- **Reference images**: clean full shoe impressions.
- **Crime scene images**: partial and noisy shoe impressions.

---

## 🧠 Methodology

### 🏗 Model Architecture
- Siamese network with shared CNN branches.
- Output: 128-D embedding vector for each image.
- Loss: **Triplet Loss** to ensure:
  
  ```bash
  distance(anchor, positive)} + {margin} < {distance(anchor, negative)}
  

### 🔁 Triplet Generation
- **Anchor**: crime scene image
- **Positive**: same class reference image
- **Negative**: different class reference image
- Triplets generated dynamically per epoch for better generalization.

### 🧪 Retrieval Phase
- Extract embeddings for all reference and probe images.
- Calculate Euclidean distance:
  ```python
  distance = np.linalg.norm(probe_feat - ref_feat)


###  📊 Evaluation Metrics:
- **Top-1 Accuracy**: Is the closest result correct?
- **Top-5 Accuracy**: Is the correct image among the first 5 results?
- **Mean Average Precision** (optional): Evaluating the full ranking.

###  🎨 Visualization:
-We display the probe image alongside the expected results.
![Image Alt Text](path_to_image)
