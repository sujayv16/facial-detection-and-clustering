import streamlit as st
from cluster_faces1 import main as cluster_main
from PIL import Image
import os
import pickle
import cv2
import face_recognition
import shutil

# -----------------------------
# Page configuration and custom CSS
# -----------------------------
st.set_page_config(page_title="Facial Clustering App", layout="wide")

custom_css = """
<style>
    /* General body style with dark background */
    .reportview-container {
        background-color: #2c3e50;
    }
    .main {
        background-color: #2c3e50;
    }
    body {
        color: #ecf0f1;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Segoe UI', sans-serif;
        color: #ecf0f1;
    }
    /* Header styling */
    .header {
        text-align: center;
        padding: 2rem 0;
        background: #34495e;
        color: #ecf0f1;
        margin-bottom: 2rem;
        border-bottom: 3px solid #2980b9;
    }
    /* Sidebar styling */
    .css-1d391kg { 
        background-color: #34495e;
        color: #ecf0f1;
    }
    .sidebar .sidebar-content {
        background: #34495e;
        border-radius: 8px;
        padding: 1rem;
    }
    /* Uploaded images gallery styling */
    .gallery img {
        border: 3px solid #2980b9;
        border-radius: 5px;
        margin: 0.5rem;
    }
    /* Button styling */
    .stButton>button {
        background-color: #2980b9;
        color: #ecf0f1;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1.5rem;
        font-size: 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1abc9c;
    }
    /* Expander header styling */
    .streamlit-expanderHeader {
        font-size: 1.1rem;
        color: #ecf0f1;
    }
    /* Slider styling */
    .stSlider>div>div {
        color: #ecf0f1;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# -----------------------------
# Utility functions
# -----------------------------
def create_encodings_from_uploaded_images(upload_folder, output_pickle='encodings.pickle'):
    data = []
    for image_name in os.listdir(upload_folder):
        image_path = os.path.join(upload_folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model='hog')
        encodings = face_recognition.face_encodings(rgb, boxes)
        for (box, encoding) in zip(boxes, encodings):
            d = {
                "loc": box,
                "encoding": encoding,
                "imagePath": image_path
            }
            data.append(d)
    with open(output_pickle, "wb") as f:
        pickle.dump(data, f)

def run_clustering_and_montages(encodings_path):
    # Clear old clustering outputs
    output_folder = os.path.join("output")
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    return cluster_main(encodings_path)

def display_uploaded_images(image_paths):
    st.markdown("### Uploaded Images")
    cols = st.columns(5)
    for idx, img_path in enumerate(image_paths):
        with cols[idx % 5]:
            st.image(Image.open(img_path), use_column_width=True)

def display_silhouette_scores(silhouette_scores):
    st.markdown("### Silhouette Scores")
    for method, score in silhouette_scores.items():
        st.write(f"**{method.capitalize()}:** {score:.3f}")

# -----------------------------
# Main Application Layout
# -----------------------------
def main():
    st.markdown("<div class='header'><h1>Facial Clustering App</h1><p>Sujay, Pareen, Aiswarya, Suresh, Manoj</p></div>", unsafe_allow_html=True)
    
    # Sidebar instructions
    st.sidebar.markdown("## Instructions")
    st.sidebar.info(
        """
        1. **Upload** your images using the uploader below.
        2. Click **Cluster Images** to encode and cluster faces.
        3. **Explore** results in the Clustering Results section.
        """
    )
    
    uploaded_files = st.file_uploader("Upload images (JPG, JPEG, PNG)", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])
    
    if uploaded_files:
        os.makedirs('uploaded_images', exist_ok=True)
        image_paths = []
        
        # Clear old images
        for file in os.listdir('uploaded_images'):
            os.remove(os.path.join('uploaded_images', file))
        
        for uploaded_file in uploaded_files:
            path = os.path.join("uploaded_images", uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.getvalue())
            image_paths.append(path)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("## Preview of Uploaded Images")
        display_uploaded_images(image_paths)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("Cluster Images"):
            st.info("Encoding faces from uploaded images...")
            create_encodings_from_uploaded_images(upload_folder="uploaded_images", output_pickle="encodings.pickle")
            
            st.info("Clustering images...")
            silhouette_scores = run_clustering_and_montages(encodings_path="encodings.pickle")
            st.success("Clustering complete!")
            display_silhouette_scores(silhouette_scores)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("## Explore Clustering Results")
        for method in ["DBSCAN", "KMeans", "Agglomerative"]:
            with st.expander(f"{method} Results", expanded=False):
                st.markdown(f"### {method} Clusters")
                montages_folder = os.path.join("output", method.lower(), "montages")
                if os.path.exists(montages_folder):
                    image_files = sorted([f for f in os.listdir(montages_folder) if f.endswith(".jpg")])
                    if image_files:
                        selected_image = st.select_slider(f"Select an image for {method}", options=image_files)
                        st.image(Image.open(os.path.join(montages_folder, selected_image)), caption=selected_image, use_column_width=True)
                
                viz_path = os.path.join("output", method.lower(), f"cluster_visualization_{method.lower()}.png")
                if os.path.exists(viz_path):
                    st.markdown(f"### {method} Cluster Visualization")
                    st.image(Image.open(viz_path), caption=f"{method} Cluster Visualization", use_column_width=True)

if __name__ == "__main__":
    main()
