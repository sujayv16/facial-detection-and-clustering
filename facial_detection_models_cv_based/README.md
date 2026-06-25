# Facial Detection Models (CV-Based)

This folder contains classical computer-vision and machine-learning approaches for face analysis and grouping, implemented in Jupyter notebooks.  
The work focuses on handcrafted-feature pipelines and statistical learning methods (HOG, PCA/LDA, KNN/SVM) for recognition and clustering-style organization tasks.

---

## 1) Folder Overview

The `facial_detection_models_cv_based` folder includes experiments that rely on:

- **Feature engineering** (e.g., HOG descriptors)
- **Dimensionality reduction** (Kernel PCA, LDA)
- **Traditional ML classifiers** (KNN, SVM)
- Interactive notebook-based workflows in **Google Colab** and **Kaggle**

---

## 2) Problem Being Solved

The notebooks aim to solve two related problems:

1. **Face representation and recognition** using classical CV/ML pipelines  
2. **Grouping/organizing facial images** by visual similarity without requiring an end-to-end deep recognition stack

This is useful for understanding baseline methods and interpretable feature pipelines before using heavier deep models.

---

## 3) Implemented Algorithms / Models

### `Face_Detection_and_Clustering.ipynb`
- **HOG (Histogram of Oriented Gradients)** feature extraction and visualization
- Imports indicate optional/trial usage of:
  - `SVC`
  - `GridSearchCV`
  - `train_test_split`
  - `accuracy_score`, `confusion_matrix`

### `face_recognition_pca_lda.ipynb`
- **Kernel PCA** (`sklearn.decomposition.KernelPCA`)
- **LDA** (`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`)
- **KNN classifier** (`sklearn.neighbors.KNeighborsClassifier`)
- Dataset preparation from **AT&T Face Dataset** style directory layout (`s1` to `s40`)

---

## 4) Methodology and Workflow

### A. HOG-based visual feature workflow (`Face_Detection_and_Clustering.ipynb`)

1. Install and import required packages  
2. Upload multiple images through Colab widget  
3. Convert each image to grayscale (`PIL -> numpy`)  
4. Extract HOG descriptors (`orientations=9`, `pixels_per_cell=(8,8)`, `cells_per_block=(2,2)`)  
5. Visualize:
   - Original grayscale image
   - HOG visualization map  
6. Iterate for all uploaded images

This notebook is primarily exploratory/diagnostic and demonstrates classical texture-gradient representations for faces.

### B. PCA/LDA recognition workflow (`face_recognition_pca_lda.ipynb`)

1. Load facial images from per-person folders
2. Flatten each image into a 1D vector (10304 features/image)
3. Build:
   - `Data` matrix of shape `(400, 10304)`
   - `labels` vector for 40 classes
4. Apply dimensionality reduction (PCA/LDA pipeline)
5. Train/evaluate classifier (KNN used in imports/code flow)

---

## 5) Important Files and Their Roles

| File | Role |
|---|---|
| `Face_Detection_and_Clustering.ipynb` | HOG feature extraction and side-by-side visualization for uploaded face images |
| `face_recognition_pca_lda.ipynb` | Recognition-oriented pipeline using AT&T faces with PCA/LDA and KNN-based classification |

---

## 6) Technologies and Libraries Used

- **Python**
- **NumPy**, **Pandas**
- **Matplotlib**, **Seaborn**
- **OpenCV (`cv2`)**
- **Pillow (`PIL`)**
- **scikit-image** (`hog`)
- **scikit-learn** (`KNN`, `SVC`, `KernelPCA`, `LDA`, metrics/model selection)
- Notebook environments:
  - **Google Colab** (file upload flow)
  - **Kaggle** (dataset path usage in PCA/LDA notebook)

---

## 7) Input and Output Formats

### Inputs
- Image files: `.jpg`, `.jpeg`, `.png` (Colab upload in HOG notebook)
- AT&T style dataset directory structure:
  - `/kaggle/input/att-database-of-faces/s1 ... s40`
  - 10 images/class expected by code

### Outputs
- HOG notebook:
  - Per-image plots: grayscale vs HOG visualization
- PCA/LDA notebook:
  - Feature matrix + labels
  - Recognition pipeline outputs (classification performance/analysis in notebook flow)

---

## 8) Internal Implementation Notes

- Grayscale conversion is applied before HOG (`convert('L')`)
- HOG parameters are fixed in code and not hyperparameter-searched in provided cells
- AT&T pipeline assumes fixed image dimensions (flattened length `10304`)
- Labels are generated deterministically by folder index (`1..40`)

---

## 9) How to Run

### Option A — Google Colab (HOG notebook)

```bash
# Open notebook in Colab
# Run cells in order
# Upload face images when prompted by files.upload()
```

### Option B — Kaggle / local Jupyter (PCA/LDA notebook)

```bash
pip install numpy pandas matplotlib pillow scikit-learn opencv-python
```

Then:

1. Ensure AT&T dataset is available in expected path format (`s1`...`s40`)
2. Open `face_recognition_pca_lda.ipynb`
3. Run all cells sequentially

---

## 10) Results / Visualizations Generated

- HOG gradient maps for each uploaded sample
- Data-shape and dataset sanity checks
- PCA/LDA-based recognition analysis workflow (classification stage via KNN)

---

## 11) Strengths and Limitations

### Strengths
- Interpretable, educational CV pipeline
- Lightweight compared with deep models
- Clear step-by-step feature engineering workflow
- Good baseline for comparison against deep-learning methods

### Limitations
- Sensitive to pose/illumination/expression changes
- Flat-vector representation can lose spatial structure
- Dataset-specific assumptions (fixed size/class count)
- Notebook-centric execution; limited modular packaging

---

## 12) Folder Structure

```text
facial_detection_models_cv_based/
├── Face_Detection_and_Clustering.ipynb
└── face_recognition_pca_lda.ipynb
```

---

## 13) Suggested Documentation Improvements (for this folder)

- Add explicit **dataset acquisition + licensing** section (AT&T dataset details)
- Add a **reproducibility table** (exact versions + runtime type)
- Add **evaluation section** with numeric metrics (accuracy/confusion matrix screenshots)
- Add **parameter sensitivity notes** (e.g., HOG params, KNN k-value, PCA components)
- Export reusable functions into `.py` modules for maintainability
