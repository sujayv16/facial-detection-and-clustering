# import the necessary packages
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from imutils import build_montages
import numpy as np
import argparse
import pickle
import cv2
import os
import matplotlib.pyplot as plt
import face_recognition
from constants import FACE_DATA_PATH, ENCODINGS_PATH, CLUSTERING_RESULT_PATH

def move_image(image, id, labelID, method):
    path = os.path.join(CLUSTERING_RESULT_PATH, 'output', method, 'labels', f'label{labelID}')
    os.makedirs(path, exist_ok=True)
    filename = f'{id}.jpg'
    cv2.imwrite(os.path.join(path, filename), image)

def save_cluster_visualization(embeddings, labels, method, output_folder):
    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        if label == -1:
            continue
        idx = np.where(labels == label)
        plt.scatter(embeddings[idx, 0], embeddings[idx, 1], label=f'Cluster {label}')
    plt.title(f'Cluster Visualization ({method})')
    plt.legend()
    plt.savefig(os.path.join(output_folder, f'cluster_visualization_{method}.png'))
    plt.close()

def process_clusters(data, encodings, clt, method, known_encodings, known_names):
    output_method_folder = os.path.join(CLUSTERING_RESULT_PATH, 'output', method)
    montages_folder = os.path.join(output_method_folder, 'montages')
    labels_folder = os.path.join(output_method_folder, 'labels')
    os.makedirs(montages_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    labelIDs = np.unique(clt.labels_)
    for labelID in labelIDs:
        print(f"[INFO] Processing label ID: {labelID} using {method}")
        label_folder_path = os.path.join(labels_folder, f'label{labelID}')
        os.makedirs(label_folder_path, exist_ok=True)
        idxs = np.where(clt.labels_ == labelID)[0]
        faces = []
        for i in idxs:
            print(f"[INFO] Processing image: {i} using {method}")
            image = cv2.imread(data[i]["imagePath"])
            (top, right, bottom, left) = data[i]["loc"]
            face = image[top:bottom, left:right]
            move_image(image, i, labelID, method)
            face = cv2.resize(face, (96, 96))
            faces.append(face)

            encoding = encodings[i]
            matches = face_recognition.compare_faces(known_encodings, encoding)
            if any(matches):
                matched_indices = [j for j, match in enumerate(matches) if match]
                names = [known_names[j] for j in matched_indices]
                print(f"[INFO] Image {i} matched known individuals: {', '.join(names)}")
            else:
                print(f"[INFO] Image {i} did not match any known individuals")

        if faces:
            montage = build_montages(faces, (96, 96), (5, 5))[0]
            montage_path = os.path.join(montages_folder, f'montage_label{labelID}.jpg')
            cv2.imwrite(montage_path, montage)
            print(f"[INFO] Montage saved for label ID: {labelID} using {method}")

    silhouette_avg = silhouette_score(encodings, clt.labels_)
    print(f"[INFO] Silhouette score for {method}: {silhouette_avg:.3f}")

    # Convert encodings to NumPy array and compute valid perplexity
    encodings_array = np.array(encodings)
    n_samples = encodings_array.shape[0]
    # TSNE's perplexity must be less than the number of samples; if too few, set to n_samples - 1.
    valid_perplexity = min(30, n_samples - 1) if n_samples > 1 else 1

    tsne = TSNE(n_components=2, random_state=42, perplexity=valid_perplexity)
    embeddings = tsne.fit_transform(encodings_array)
    save_cluster_visualization(embeddings, clt.labels_, method, output_method_folder)

    return silhouette_avg, len(np.where(labelIDs > -1)[0])

def main(encodings_path=ENCODINGS_PATH):
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--encodings", default=encodings_path, help="Path to serialized DB of facial encodings")
    ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of parallel jobs to run (-1 will use all CPUs)")
    args = vars(ap.parse_args())

    print("[INFO] Loading encodings...")
    data = pickle.loads(open(args["encodings"], "rb").read())
    data = np.array(data)
    encodings = [d["encoding"] for d in data]

    known_encodings = []
    known_names = []  # Load actual known face data here if available

    os.makedirs(os.path.join(CLUSTERING_RESULT_PATH, 'output'), exist_ok=True)

    silhouette_scores = {}

    # --- DBSCAN with Hyperparameter Tuning ---
    print("[INFO] Tuning DBSCAN...")
    eps_values = np.arange(0.1, 1.0, 0.1)
    min_samples_values = range(2, 11)
    best_score = -1
    best_params = None

    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=args["jobs"])
            labels = dbscan.fit_predict(encodings)

            if len(np.unique(labels)) > 1:
                score = silhouette_score(encodings, labels)
                print(f"[INFO] Silhouette for DBSCAN (eps={eps}, min_samples={min_samples}): {score:.3f}")
                if score > best_score:
                    best_score = score
                    best_params = {'eps': eps, 'min_samples': min_samples}

    print(f"[INFO] Best DBSCAN params: {best_params}, Score: {best_score:.3f}")
    best_dbscan = DBSCAN(**best_params, metric="euclidean", n_jobs=args["jobs"])
    dbscan_result = best_dbscan.fit(encodings)

    dbscan_score, dbscan_clusters = process_clusters(data, encodings, dbscan_result, "dbscan", known_encodings, known_names)
    silhouette_scores["dbscan"] = dbscan_score

    # --- KMeans and Agglomerative ---
    clustering_methods = {
        "kmeans": KMeans(n_clusters=dbscan_clusters, random_state=42),
        "agglomerative": AgglomerativeClustering(n_clusters=dbscan_clusters, linkage="ward")
    }

    for method, clt in clustering_methods.items():
        print(f"[INFO] Clustering using {method}...")
        clt.fit(encodings)
        score, _ = process_clusters(data, encodings, clt, method, known_encodings, known_names)
        silhouette_scores[method] = score

    return silhouette_scores

if __name__ == "__main__":
    main()
