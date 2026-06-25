# Facial Detection Models (Deep Learning-Based)

This folder contains deep-learning-centric face grouping pipelines implemented in Jupyter notebooks.  
The implementations combine pretrained face models (MTCNN + FaceNet) and learned latent representations (convolutional autoencoder) with unsupervised clustering (DBSCAN/KMeans).

---

## 1) Folder Overview

The folder explores two complementary deep-learning clustering pipelines:

1. **Pretrained embedding pipeline**:
   - MTCNN for detection/alignment
   - InceptionResnetV1 (FaceNet-style) for embeddings
   - DBSCAN for identity grouping

2. **Learned latent embedding pipeline**:
   - Custom convolutional autoencoder (PyTorch)
   - KMeans clustering in latent space
   - Reconstruction and cluster visualization

---

## 2) Problem Being Solved

The notebooks address **unsupervised face grouping**:

- Detect faces in raw images
- Encode faces into compact, discriminative representations
- Cluster similar faces automatically without name labels

This is useful for photo organization, identity grouping, and exploratory biometric analysis.

---

## 3) Implemented Algorithms / Models

### `face_grouping_using_mtcnn_and_dbscan.ipynb`
- **MTCNN** (face detection + alignment)
- **InceptionResnetV1** pretrained on **VGGFace2** (embedding extraction)
- **DBSCAN** (`eps=0.9`, `min_samples=1`, Euclidean metric)
- Grid visualization of images with assigned cluster IDs

### `faceclustering_autoencoder.ipynb`
- `fetch_lfw_people(min_faces_per_person=60, color=True)` dataset loading
- Custom **convolutional autoencoder**:
  - Encoder: Conv2d stack (3→32→64→128→256)
  - Latent linear projection (`latent_dim` set to 32 during training run)
  - Decoder: ConvTranspose2d stack back to RGB image
- **MSE reconstruction loss** with **Adam**
- **KMeans** (`n_clusters=16`) on latent vectors
- Reconstruction comparison and clustered sample plotting

---

## 4) Detailed Methodology and Workflow

### A. MTCNN + FaceNet + DBSCAN pipeline

1. Upload user images via Colab widget  
2. Load images and force RGB conversion  
3. Initialize device (`cuda` if available else CPU)  
4. Detect/align face per image using MTCNN  
5. Remove images where no face is detected  
6. Stack aligned faces and compute deep embeddings (`resnet(aligned)`)  
7. Cluster embeddings via DBSCAN  
8. Print image-to-cluster mapping  
9. Visualize all valid images with cluster labels in a grid

### B. Autoencoder + KMeans pipeline

1. Load LFW subset with minimum samples per identity  
2. Build balanced subset (up to 60 images per class)  
3. Define PyTorch `Dataset` and transformation pipeline:
   - scale → PIL conversion → resize 64×64 → tensor → normalization
4. Train convolutional autoencoder for 50 epochs  
5. Plot training reconstruction loss curve  
6. Visualize original vs reconstructed images  
7. Extract latent vectors with `autoencoder.encode(...)`  
8. Cluster latent vectors using KMeans (`n_clusters=16`)  
9. Visualize cluster assignments

---

## 5) Important Files and Their Roles

| File | Role |
|---|---|
| `face_grouping_using_mtcnn_and_dbscan.ipynb` | End-to-end face detection/alignment + deep embedding + DBSCAN clustering on uploaded images |
| `faceclustering_autoencoder.ipynb` | Learns compressed face representations and clusters latent embeddings with KMeans |

---

## 6) Technologies and Libraries Used

- **Python**
- **PyTorch**, `torchvision`
- **facenet-pytorch** (`MTCNN`, `InceptionResnetV1`)
- **scikit-learn** (`DBSCAN`, `KMeans`, `fetch_lfw_people`, `TSNE` imported)
- **Pillow**
- **NumPy**, **Pandas**
- **OpenCV**
- **Matplotlib**
- Runtime contexts:
  - Google Colab (upload-based notebook)
  - Kaggle-style notebook environment (LFW/autoencoder notebook)

---

## 7) Input and Output Formats

### Inputs

#### `face_grouping_using_mtcnn_and_dbscan.ipynb`
- User-uploaded image files (`.jpg`, `.jpeg`, `.png`)
- Faces should be visible enough for MTCNN detection

#### `faceclustering_autoencoder.ipynb`
- LFW dataset from `fetch_lfw_people(...)`
- Color images resized to `64×64` in preprocessing transform

### Outputs

- Cluster labels per image (console mapping)
- Visual cluster grids (`Cluster <id>` on each image)
- Autoencoder:
  - Epoch-wise training loss
  - Original vs reconstructed sample grids
  - Latent vectors (`[N, latent_dim]`) for clustering

---

## 8) How the Implementation Works Internally

- **Detection stage** filters invalid samples early (`x is None` branch)
- **Embedding stage** transforms aligned faces into dense vectors via pretrained InceptionResnetV1
- **DBSCAN stage** groups embeddings by density (no fixed cluster count)
- **Autoencoder stage** learns task-specific representation by reconstruction
- **KMeans stage** imposes fixed-number partitioning over learned latent space

This gives two different unsupervised clustering paradigms:
- Density-based on pretrained semantic embeddings
- Centroid-based on self-learned latent embeddings

---

## 9) Steps to Run

### Environment setup

```bash
pip install torch torchvision facenet-pytorch scikit-learn matplotlib pillow numpy pandas opencv-python
```

### Run notebook 1 (MTCNN + DBSCAN)

1. Open `face_grouping_using_mtcnn_and_dbscan.ipynb` (Colab recommended)
2. Run all cells sequentially
3. Upload face images when prompted
4. Review printed cluster mapping and visualization grid

### Run notebook 2 (Autoencoder + KMeans)

1. Open `faceclustering_autoencoder.ipynb` (Kaggle/local with internet dataset access)
2. Run all cells in order
3. Wait for training (50 epochs)
4. Inspect:
   - loss curve
   - reconstructions
   - clustering outputs

---

## 10) Results / Visualizations Generated

- Cluster assignment text output for each valid image
- Large grid visualization of clustered images
- Autoencoder training loss trajectory
- Side-by-side original/reconstructed face samples
- Cluster distribution from KMeans labels

---

## 11) Strengths and Limitations

### Strengths
- Uses strong pretrained face embeddings (FaceNet variant)
- Fully unsupervised grouping workflow
- Includes both pretrained and learned-representation clustering approaches
- Visual outputs make model behavior easy to inspect

### Limitations
- No explicit quantitative clustering metrics (e.g., NMI/ARI/silhouette) in shown cells
- DBSCAN sensitivity to `eps` and data scale
- KMeans requires manual `n_clusters` selection
- Some noisy cluster assignments visible in printed outputs
- Notebook workflow is not yet packaged into reusable scripts/modules

---

## 12) Folder Structure

```text
facial_detection_models_deeplearning_based/
├── face_grouping_using_mtcnn_and_dbscan.ipynb
└── faceclustering_autoencoder.ipynb
```

---

## 13) Suggested Documentation Improvements (for this folder)

- Add **hyperparameter table** (DBSCAN eps/min_samples, latent_dim, n_clusters, epochs)
- Add **quantitative evaluation** for clustering quality
- Document **hardware/runtime expectations** (CPU vs GPU time)
- Add **failure-case gallery** (missed detections, mixed-identity clusters)
- Add **reproducibility section** (random seeds, exact package versions)
- Provide scriptized pipeline (`src/` modules + CLI) for production-style reuse
