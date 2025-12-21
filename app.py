import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import pandas as pd
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Deep Video Analyst Pro", page_icon="👁️", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    h3 { color: #00e676; font-family: 'Helvetica Neue', sans-serif; }
    .metric-card { background-color: #1e1e1e; padding: 10px; border-radius: 8px; border-left: 4px solid #00e676; }
    .frame-caption { text-align: center; font-size: 11px; color: #aaa; margin-top: 5px; }
    div[data-testid="stExpander"] { background-color: #161b22; }
</style>
""", unsafe_allow_html=True)

st.title("👁️ Deep Video Analyst: Pro Dashboard")

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_models():
    # A. ENCODER
    try:
        encoder = tf.keras.models.load_model('model2_encoder.keras', compile=False)
        st.sidebar.success("✅ Encoder Loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Encoder Failed: {e}")
        return None, None

    # B. CLASSIFIER (Model 1)
    try:
        model1 = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(512,)), 
            tf.keras.layers.Dense(1, activation='linear') 
        ])
        model1.load_weights('model1.weights.h5')
        st.sidebar.success("✅ Model 1 Loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Model 1 Failed: {e}")
        model1 = None

    return encoder, model1

encoder, model1 = load_models()
IMG_SIZE = (224, 224) 

# --- 3. ADVANCED ANALYSIS ENGINE ---
def analyze_video(video_path, encoder, model1):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_data = []
    
    # UI Elements for Progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    metric_placeholder = st.empty()
    
    frame_idx = 0
    prev_frame_gray = None # For motion detection

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        try:
            # --- A. DEEP LEARNING ANALYSIS ---
            frame_resized = cv2.resize(frame, IMG_SIZE)
            frame_norm = frame_resized.astype('float32') / 255.0
            input_tensor = np.expand_dims(frame_norm, axis=0)
            
            # Encode & Bridge (8 -> 512 Fix)
            features = encoder.predict(input_tensor, verbose=0)
            features_flat = features.reshape(1, -1)
            
            if features_flat.shape[1] == 8:
                features_512 = np.tile(features_flat, (1, 64))
            elif features_flat.shape[1] == 512:
                features_512 = features_flat
            else:
                features_512 = np.zeros((1, 512))
            
            # Model Scores
            m1_score = 0.0
            if model1:
                raw = model1.predict(features_512, verbose=0)[0][0]
                m1_score = float(raw) if not np.isnan(raw) else 0.0
            
            m2_score = float(np.var(features_flat))
            if np.isnan(m2_score): m2_score = 0.0

            # --- B. COMPUTER VISION ANALYTICS ---
            
            # 1. Color Analysis (Mean RGB)
            # Frame is in BGR format from OpenCV
            mean_b = np.mean(frame[:, :, 0])
            mean_g = np.mean(frame[:, :, 1])
            mean_r = np.mean(frame[:, :, 2])
            
            # 2. Motion Flux (Pixel Difference)
            curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_score = 0.0
            if prev_frame_gray is not None:
                # Calculate absolute difference between frames
                diff = cv2.absdiff(curr_frame_gray, prev_frame_gray)
                motion_score = np.mean(diff)
            
            prev_frame_gray = curr_frame_gray

            # Store Data
            frame_data.append({
                "frame_id": frame_idx,
                "timestamp": frame_idx / fps if fps else 0,
                "m1_attention": m1_score,
                "m2_energy": m2_score,
                "motion_flux": motion_score,
                "color_r": mean_r,
                "color_g": mean_g,
                "color_b": mean_b
            })

            # --- C. LIVE UPDATES ---
            if frame_idx % 5 == 0:
                percent = int((frame_idx / total_frames) * 100)
                progress_bar.progress(min(percent / 100, 1.0))
                status_text.markdown(f"**Analyzing Frame:** `{frame_idx}/{total_frames}` ({percent}%)")
                
                # Show live metrics every 10 frames
                if frame_idx % 10 == 0:
                    metric_placeholder.caption(f"Last Score: {m1_score:.3f} | Motion: {motion_score:.1f}")

        except Exception as e:
            print(f"Skipped frame {frame_idx}: {e}")

        frame_idx += 1

    cap.release()
    progress_bar.empty()
    status_text.success("Analysis Complete!")
    metric_placeholder.empty()
    
    return pd.DataFrame(frame_data)

def get_images(video_path, indices):
    cap = cv2.VideoCapture(video_path)
    images = {}
    target = set(indices)
    idx = 0
    while cap.isOpened() and len(images) < len(target):
        ret, frame = cap.read()
        if not ret: break
        if idx in target:
            images[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        idx += 1
    cap.release()
    return images

# --- 4. VISUALIZATION DASHBOARD ---
uploaded_file = st.sidebar.file_uploader("📂 Upload Video", type=['mp4', 'avi', 'mov'])

if uploaded_file and encoder:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    st.markdown("### 🚀 Processing Pipeline")
    df = analyze_video(tfile.name, encoder, model1)
    
    if not df.empty:
        # --- TABBED INTERFACE FOR ANALYTICS ---
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Main Dashboard", "🎨 Color & Motion", "🎞️ Key Frames", "📋 Data Stats"])
        
        with tab1:
            st.markdown("#### Model Performance Timeline")
            # Dual axis plot
            fig, ax1 = plt.subplots(figsize=(10, 4))
            ax1.set_facecolor('#1e1e1e')
            fig.patch.set_facecolor('#0e1117')
            
            sns.lineplot(data=df, x='frame_id', y='m1_attention', ax=ax1, color='#00e676', label='M1 Attention')
            ax1.set_ylabel('Attention Score', color='#00e676')
            ax1.tick_params(axis='y', colors='#00e676')
            ax1.tick_params(axis='x', colors='white')
            
            ax2 = ax1.twinx()
            sns.lineplot(data=df, x='frame_id', y='m2_energy', ax=ax2, color='#29b5e8', label='M2 Latent Energy')
            ax2.set_ylabel('Latent Energy', color='#29b5e8')
            ax2.tick_params(axis='y', colors='#29b5e8')
            
            st.pyplot(fig)
            
            st.markdown("#### Correlation Analysis")
            # Correlation Heatmap
            corr = df[['m1_attention', 'm2_energy', 'motion_flux']].corr()
            fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax_corr, cbar=False)
            fig_corr.patch.set_facecolor('#0e1117')
            st.pyplot(fig_corr)

        with tab2:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### 🏃 Motion Flux (Scene Changes)")
                st.line_chart(df.set_index('frame_id')['motion_flux'], color='#ff4b4b')
                st.caption("Peaks indicate fast movement or camera cuts.")
                
            with col_b:
                st.markdown("#### 🎨 RGB Color Distribution")
                st.line_chart(df.set_index('frame_id')[['color_r', 'color_g', 'color_b']])
                st.caption("Tracks the average intensity of Red, Green, and Blue channels.")

        with tab3:
            st.header("Model 1 (Bi-LSTM / Attention)")
            top_m1 = df.nlargest(5, 'm1_attention')['frame_id'].tolist()
            top_m1 = sorted(top_m1)
            imgs_m1 = get_images(tfile.name, top_m1)
            
            cols1 = st.columns(5)
            for i, idx in enumerate(top_m1):
                if idx in imgs_m1:
                    with cols1[i]:
                        st.image(imgs_m1[idx], use_container_width=True)
                        st.markdown(f"<div class='frame-caption'>Frame {idx}<br>Score: {df.loc[df['frame_id']==idx, 'm1_attention'].values[0]:.2f}</div>", unsafe_allow_html=True)

            st.header("Model 2 (Autoencoder)")
            top_m2 = df.nlargest(5, 'm2_energy')['frame_id'].tolist()
            top_m2 = sorted(top_m2)
            imgs_m2 = get_images(tfile.name, top_m2)
            
            cols2 = st.columns(5)
            for i, idx in enumerate(top_m2):
                if idx in imgs_m2:
                    with cols2[i]:
                        st.image(imgs_m2[idx], use_container_width=True)
                        st.markdown(f"<div class='frame-caption'>Frame {idx}<br>Energy: {df.loc[df['frame_id']==idx, 'm2_energy'].values[0]:.2f}</div>", unsafe_allow_html=True)

        with tab4:
            st.markdown("#### Statistical Summary")
            st.dataframe(df.describe())
            
            st.markdown("#### Raw Data Export")
            st.dataframe(df)

    else:
        st.error("Analysis failed to produce data.")

elif not uploaded_file:
    st.info("👈 Please upload a video to start the analysis.")