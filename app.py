import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Face Detection",
    page_icon="👤",
    layout="wide"
)

st.title("👤 Face Detection App")
st.markdown("Detect human faces in any image "
            "using OpenCV Haar Cascade classifier.")
st.markdown("---")

# Load cascade classifier
@st.cache_resource
def load_cascade():
    cascade_path = cv2.data.haarcascades + \
        'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)

face_cascade = load_cascade()

def detect_faces(image, scale_factor,
                 min_neighbors, min_size):
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(
            img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size)
    )
    return faces, img_array

def draw_faces(img_array, faces,
               box_color, show_count):
    result = img_array.copy()
    color_map = {
        "Blue":   (66,  133, 244),
        "Green":  (52,  168, 83),
        "Red":    (234, 67,  53),
        "Yellow": (251, 188, 4),
        "Purple": (156, 39,  176)
    }
    color = color_map.get(box_color,
                          (66, 133, 244))

    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(result,
                      (x, y),
                      (x+w, y+h),
                      color, 3)
        if show_count:
            cv2.putText(
                result,
                f"Face {i+1}",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2
            )
    return result

# Sidebar settings
st.sidebar.header("⚙️ Detection Settings")

scale_factor  = st.sidebar.slider(
    "Scale Factor:",
    1.05, 1.5, 1.1, 0.05,
    help="Lower = more detections, "
         "slower processing")
min_neighbors = st.sidebar.slider(
    "Min Neighbors:",
    1, 10, 5, 1,
    help="Higher = fewer but more "
         "accurate detections")
min_size      = st.sidebar.slider(
    "Min Face Size (px):",
    10, 100, 30, 5)
box_color     = st.sidebar.selectbox(
    "Box Color:",
    ["Blue", "Green", "Red",
     "Yellow", "Purple"])
show_labels   = st.sidebar.checkbox(
    "Show face labels", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Parameter Guide")
st.sidebar.info("""
**Scale Factor:** How much image
size is reduced at each scale.
1.1 = thorough, 1.5 = fast

**Min Neighbors:** How many
neighbors each rectangle should
have. Higher = more strict.

**Min Face Size:** Minimum face
size in pixels to detect.
""")

# Tabs
tab1, tab2, tab3 = st.tabs([
    "📷 Upload Image",
    "🎨 Sample Images",
    "📊 About"
])

# Tab 1 — Upload
with tab1:
    st.markdown("### Upload Your Image")

    uploaded = st.file_uploader(
        "Upload an image:",
        type=['jpg', 'jpeg', 'png',
              'webp', 'bmp']
    )

    if uploaded:
        image = Image.open(uploaded)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Original Image")
            st.image(image,
                     use_column_width=True)
            st.caption(
                f"Size: {image.size[0]}x"
                f"{image.size[1]} pixels")

        with col2:
            st.markdown("#### Detected Faces")
            with st.spinner(
                "Detecting faces..."
            ):
                faces, img_array = \
                    detect_faces(
                        image,
                        scale_factor,
                        min_neighbors,
                        min_size
                    )

                if len(faces) > 0:
                    result = draw_faces(
                        img_array, faces,
                        box_color, show_labels)
                    st.image(
                        result,
                        use_column_width=True)
                    st.success(
                        f"✅ {len(faces)} "
                        f"face(s) detected!")
                else:
                    st.image(
                        image,
                        use_column_width=True)
                    st.warning(
                        "No faces detected. "
                        "Try adjusting the "
                        "parameters.")

        if len(faces) > 0:
            st.markdown("---")
            st.markdown("### 📊 Detection Results")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Faces Detected",
                      len(faces))
            c2.metric("Image Width",
                      f"{image.size[0]}px")
            c3.metric("Image Height",
                      f"{image.size[1]}px")
            c4.metric("Scale Factor",
                      scale_factor)

            import pandas as pd
            face_data = []
            for i, (x, y, w, h) in \
                    enumerate(faces, 1):
                face_data.append({
                    'Face':   f"Face {i}",
                    'X':      x,
                    'Y':      y,
                    'Width':  w,
                    'Height': h,
                    'Area':   w * h
                })
            face_df = pd.DataFrame(face_data)
            st.dataframe(
                face_df,
                use_container_width=True,
                hide_index=True)

# Tab 2 — Sample Images
with tab2:
    st.markdown("### Test with Sample Scenarios")
    st.info(
        "Upload your own photos to test. "
        "Try group photos for best results!")

    st.markdown("### 💡 Best Practices")
    tips = [
        ("Good lighting",
         "Front-facing, well-lit faces "
         "detect most accurately"),
        ("Face angle",
         "Frontal faces work best with "
         "Haar cascade classifiers"),
        ("Image resolution",
         "Higher resolution = better "
         "detection accuracy"),
        ("Multiple faces",
         "Works great with group photos "
         "— detects all faces"),
        ("Adjust parameters",
         "If missing faces, lower "
         "Min Neighbors to 3"),
    ]
    for title, desc in tips:
        with st.expander(f"💡 {title}"):
            st.write(desc)

# Tab 3 — About
with tab3:
    st.markdown("### 📊 How It Works")
    st.markdown("""
    #### Haar Cascade Classifier
    This app uses **OpenCV's Haar Cascade**
    algorithm — one of the most widely used
    face detection methods in computer vision.

    **How it works:**
    1. Scans image at multiple scales
    2. Uses pre-trained features to identify
       face-like patterns
    3. Applies sliding window technique
    4. Returns bounding box coordinates

    **Parameters explained:**
    - **Scale Factor** — reduces image by this
      factor at each step. 1.1 = 10% reduction
    - **Min Neighbors** — each candidate
      rectangle must have this many neighbors
    - **Min Size** — ignores faces smaller
      than this pixel size

    #### Why Haar Cascade?
    - Extremely fast — runs in milliseconds
    - No GPU required
    - Works on CPU
    - 20+ years proven in production
    - Used in cameras worldwide
    """)

    import pandas as pd
    tech_df = pd.DataFrame({
        'Technology': [
            'OpenCV',
            'Haar Cascade',
            'Pillow (PIL)',
            'NumPy',
            'Streamlit'
        ],
        'Purpose': [
            'Computer vision processing',
            'Face detection algorithm',
            'Image loading and conversion',
            'Array operations on images',
            'Web app deployment'
        ]
    })
    st.dataframe(tech_df,
                 use_container_width=True,
                 hide_index=True)

st.markdown("---")
st.markdown(
    "Built by **Jyotiraditya** | "
    "Face Detection using OpenCV | "
    "Haar Cascade Classifier"
)