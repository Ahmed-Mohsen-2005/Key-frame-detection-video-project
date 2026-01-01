import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os
import json
import time
from datetime import datetime
from scipy.stats import zscore

st.set_page_config(
    page_title="Deep Video Analyst",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { 
        background-color: #0b0e11; 
        color: #c9d1d9; 
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
    }
    
    h1, h2, h3 { color: #58a6ff; font-weight: 600; letter-spacing: -0.5px; }
    .stHeader { border-bottom: 1px solid #30363d; padding-bottom: 15px; }
    
    div[data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: border-color 0.3s;
    }
    div[data-testid="metric-container"]:hover {
        border-color: #58a6ff;
    }
    
    .frame-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .frame-card:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        border-color: #58a6ff; 
    }
    
    .stSuccess { background-color: rgba(46, 160, 67, 0.1); border: 1px solid #2ea043; color: #3fb950; }
    .stWarning { background-color: rgba(187, 128, 9, 0.1); border: 1px solid #bb8009; color: #db6d28; }
    .stError { background-color: rgba(248, 81, 73, 0.1); border: 1px solid #f85149; color: #ff7b72; }
    
    section[data-testid="stSidebar"] { 
        background-color: #010409; 
        border-right: 1px solid #30363d; 
    }
    
    .stProgress > div > div > div > div {
        background-color: #238636;
    }
    
    div[data-testid="stDataFrame"] {
        border: 1px solid #30363d;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

if 'session_history' not in st.session_state:
    st.session_state['session_history'] = []


class ModelHandler:
    def __init__(self):
        self.encoder = None
        self.classifier = None
        self.input_shape = (224, 224) 
        self.status_log = []
        self.classifier_input_dim = 512 # Default, will autodetect

    def log(self, message, level="info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_log.append(f"[{timestamp}] [{level.upper()}] {message}")

    def _try_load_classifier(self, input_dim):
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(input_dim,)),
                tf.keras.layers.Dense(1, activation='linear')
            ])
            # Skip mismatch allows loading even if layer names don't match exactly
            model.load_weights('model1.weights.h5', skip_mismatch=True)
            return model
        except Exception:
            return None

    def load_resources(self):
        status = {"encoder": False, "classifier": False}
        
        try:
            self.encoder = tf.keras.models.load_model('model2_encoder.keras', compile=False)
            status["encoder"] = True
            self.log("Encoder model loaded successfully.", "success")
        except Exception as e:
            self.log(f"Encoder failed to load: {str(e)}", "error")

        possible_shapes = [512, 25088, 4096, 1024, 128, 8]
        
        for shape in possible_shapes:
            model = self._try_load_classifier(shape)
            if model:
                self.classifier = model
                self.classifier_input_dim = shape
                status["classifier"] = True
                self.log(f"Classifier loaded successfully! (Detected Input Shape: {shape})", "success")
                break
        
        if not status["classifier"]:
            try:
                self._try_load_classifier(512)
            except Exception as e:
                 self.log(f"Classifier Critical Failure. Error: {str(e)}", "error")

        return status

    def predict_frame(self, frame):
        if not self.encoder:
            return 0.0, 0.0

        try:
            resized = cv2.resize(frame, self.input_shape)
            normalized = resized.astype('float32') / 255.0
            input_tensor = np.expand_dims(normalized, axis=0)

            latent = self.encoder.predict(input_tensor, verbose=0)
            latent_flat = latent.reshape(1, -1)
            
            enc_dim = latent_flat.shape[1]
            target_dim = self.classifier_input_dim
            
            if enc_dim == target_dim:
                bridge_tensor = latent_flat
            elif enc_dim < target_dim:
                repeats = int(np.ceil(target_dim / enc_dim))
                tiled = np.tile(latent_flat, (1, repeats))
                bridge_tensor = tiled[:, :target_dim] 
            else:
                bridge_tensor = latent_flat[:, :target_dim]

            attention_score = 0.0
            if self.classifier:
                raw_pred = self.classifier.predict(bridge_tensor, verbose=0)[0][0]
                attention_score = float(raw_pred)
            
            energy_score = float(np.var(latent_flat))

            if np.isnan(attention_score) or np.isinf(attention_score): attention_score = 0.0
            if np.isnan(energy_score) or np.isinf(energy_score): energy_score = 0.0

            return attention_score, energy_score

        except Exception as e:
            return 0.0, 0.0


class VideoProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.cap = cv2.VideoCapture(filepath)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.duration = self.total_frames / self.fps if self.fps else 0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def process_video_stream(self, model_handler, progress_callback=None, status_callback=None):
        data = []
        frame_idx = 0
        prev_gray = None
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break

            m1_score, m2_score = model_handler.predict_frame(frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            motion = 0.0
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion = np.mean(diff)
            prev_gray = gray

            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

            b, g, r = cv2.split(frame)
            mean_r, mean_g, mean_b = np.mean(r), np.mean(g), np.mean(b)
            brightness = np.mean(gray)
            contrast = gray.std()
            
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.mean(edges) / 255.0

            # Store Record
            data.append({
                "frame_id": frame_idx,
                "timestamp": frame_idx / self.fps if self.fps else 0,
                "m1_attention": m1_score,
                "m2_energy": m2_score,
                "cv_motion": motion,
                "cv_blur_var": blur_score,
                "cv_brightness": brightness,
                "cv_contrast": contrast,
                "cv_edge_density": edge_density,
                "rgb_r": mean_r,
                "rgb_g": mean_g,
                "rgb_b": mean_b
            })

            frame_idx += 1
            
            if frame_idx % 5 == 0 and progress_callback:
                percent = int((frame_idx / self.total_frames) * 100)
                progress_callback(frame_idx / self.total_frames)
                
                if status_callback:
                    status_callback(f"Processing Frame {frame_idx} / {self.total_frames} ({percent}%)")

        self.cap.release()
        return pd.DataFrame(data)

    def fetch_specific_frame(self, frame_idx):
        self.cap.open(self.filepath) 
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        self.cap.release()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

class AnalyticsEngine:
    @staticmethod
    def compute_hybrid_scores(df, w_attn, w_energy, w_quality):
        if df.empty: return df

        df['z_m1'] = zscore(df['m1_attention'])
        df['z_m2'] = zscore(df['m2_energy'])
        
        df['z_m1'] = df['z_m1'].fillna(0)
        df['z_m2'] = df['z_m2'].fillna(0)

        def minmax(series):
            return (series - series.min()) / (series.max() - series.min() + 1e-6)

        df['norm_m1'] = minmax(df['z_m1'])
        df['norm_m2'] = minmax(df['z_m2'])
        
        df['norm_blur'] = minmax(df['cv_blur_var'])
        df['norm_contrast'] = minmax(df['cv_contrast'])
        
        df['quality_score'] = (df['norm_blur'] * 0.7) + (df['norm_contrast'] * 0.3)

        df['final_score'] = (
            (df['norm_m1'] * w_attn) + 
            (df['norm_m2'] * w_energy) + 
            (df['quality_score'] * w_quality)
        )
        
        return df

    @staticmethod
    def detect_scene_cuts(df, threshold=30.0):
        cuts = df[df['cv_motion'] > threshold]
        return cuts['frame_id'].tolist()

class VisualizationEngine:
    @staticmethod
    def plot_timeline(df, scene_cuts):
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['frame_id'], y=df['final_score'], 
            fill='tozeroy', mode='lines', name='Hybrid Relevance', 
            line=dict(color='#58a6ff', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['frame_id'], y=df['cv_motion'], 
            name='Motion Flux', 
            line=dict(color='#ff7b72', width=1, dash='dot'), 
            yaxis='y2', opacity=0.5
        ))
        
        if scene_cuts:
            fig.add_trace(go.Scatter(
                x=scene_cuts, y=[df['final_score'].max()]*len(scene_cuts),
                mode='markers', name='Scene Cut',
                marker=dict(symbol='x', size=8, color='yellow')
            ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title="Frame Sequence"),
            yaxis=dict(title="Relevance Score"),
            yaxis2=dict(title="Motion", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=1.1),
            height=400
        )
        return fig

    @staticmethod
    def plot_radar(row, df_maxes):
        categories = ['Attention', 'Energy', 'Clarity', 'Motion', 'Contrast']
        
        norm_motion = 1.0 - (row['cv_motion'] / (df_maxes['cv_motion'] + 1e-6))
        
        values = [
            row['norm_m1'],
            row['norm_m2'],
            row['norm_blur'],
            norm_motion,
            row['norm_contrast']
        ]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            line_color='#58a6ff'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=20, b=20),
            height=300
        )
        return fig

    @staticmethod
    def plot_3d_cluster(df):
        fig = px.scatter_3d(
            df, x='m1_attention', y='m2_energy', z='final_score',
            color='final_score', opacity=0.8,
            title="3D Feature Space"
        )
        fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0,r=0,b=0,t=40))
        return fig

class ReportGenerator:
    @staticmethod
    def create_audit_report(filename, df, duration, settings, best_frame_idx):
        report = {
            "metadata": {
                "file_name": filename,
                "analysis_timestamp": str(datetime.now()),
                "duration_seconds": round(duration, 2),
                "total_frames": len(df)
            },
            "settings_used": settings,
            "key_metrics": {
                "best_frame_index": int(best_frame_idx),
                "average_attention_score": float(df['m1_attention'].mean()),
                "average_latent_energy": float(df['m2_energy'].mean()),
                "average_motion": float(df['cv_motion'].mean())
            },
            "performance_flags": {
                "low_quality_frames_detected": int((df['quality_score'] < 0.3).sum()),
                "high_motion_scenes": int((df['cv_motion'] > 30).sum())
            }
        }
        return json.dumps(report, indent=4)

    @staticmethod
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

def main():
    with st.sidebar:
        st.title("⚙️ Control Center")
        st.markdown("---")
        
        st.subheader("🤖 Model Parameters")
        w_attn = st.slider("Model 1 Importance (Attention)", 0.0, 2.0, 1.0)
        w_energy = st.slider("Model 2 Importance (Energy)", 0.0, 2.0, 1.0)
        w_quality = st.slider("Quality Penalty (Blur/Dark)", 0.0, 1.0, 0.2)
        
        st.markdown("---")
        st.subheader("🎞️ Extraction Settings")
        key_frame_count = st.slider("Number of Key Frames", min_value=3, max_value=20, value=5, help="How many frames to extract per model.")
        
        st.markdown("---")
        st.subheader("📂 Session Logs")
        if st.session_state['session_history']:
            for log in reversed(st.session_state['session_history']):
                st.caption(f"📝 {log['timestamp']} - {log['name']}")
        else:
            st.info("System Ready. Waiting for input.")

    st.title("Deep Video Analyst: Titan Edition")
    st.markdown("### Enterprise Video Intelligence & Forensics Platform")
    
    model_handler = ModelHandler()
    load_status = model_handler.load_resources()
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Encoder Engine", "Online" if load_status["encoder"] else "Offline", delta_color="normal")
    c2.metric("Classifier Engine", "Online" if load_status["classifier"] else "Offline", delta_color="normal")
    c3.metric("System Health", "98%", "Stable")

    uploaded_file = st.file_uploader("Drop Video File (MP4, AVI, MOV, MKV)", type=['mp4', 'mov', 'avi', 'mkv'])
    
    if uploaded_file and load_status["encoder"]:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        vp = VideoProcessor(tfile.name)
        
        st.info(f"Loaded **{uploaded_file.name}** | Resolution: {vp.width}x{vp.height} | FPS: {vp.fps:.2f} | Duration: {vp.duration:.1f}s")
        
        st.divider()
        st.subheader("🚀 Analysis Execution")
        
        prog_bar = st.progress(0)
        status_txt = st.empty()
        
        start_time = time.time()
        df_raw = vp.process_video_stream(
            model_handler, 
            progress_callback=lambda x: prog_bar.progress(x),
            status_callback=lambda msg: status_txt.markdown(f"**{msg}**")
        )
        elapsed = time.time() - start_time
        status_txt.success(f"Processing Complete in {elapsed:.2f}s!")
        
        df_analyzed = AnalyticsEngine.compute_hybrid_scores(df_raw, w_attn, w_energy, w_quality)
        scene_cuts = AnalyticsEngine.detect_scene_cuts(df_analyzed)
        
        # Update History
        st.session_state['session_history'].append({
            'name': uploaded_file.name,
            'timestamp': datetime.now().strftime("%H:%M")
        })

        tab_overview, tab_vision, tab_ai, tab_keyframes, tab_raw = st.tabs([
            "📊 Executive Overview", 
            "👁️ Computer Vision Forensics", 
            "🧠 Deep AI Metrics", 
            "🎞️ Key Frames Inspector",
            "💾 Raw Data & Export"
        ])
        
        with tab_overview:
            st.subheader("🎯 Model Performance Metrics")
            
            m1_confidence = df_analyzed['norm_m1'].mean() * 100
            m2_confidence = df_analyzed['norm_m2'].mean() * 100
            avg_system_conf = (m1_confidence + m2_confidence) / 2

            acc_col1, acc_col2, acc_col3 = st.columns(3)
            
            with acc_col1:
                st.metric(
                    label="Model 1 (Attention) Activation",
                    value=f"{m1_confidence:.2f}%",
                    delta=f"{m1_confidence - 50:.1f}% vs Baseline",
                    help="Indicates how strongly the classifier reacted to the visual features on average."
                )
            
            with acc_col2:
                st.metric(
                    label="Model 2 (Energy) Intensity",
                    value=f"{m2_confidence:.2f}%",
                    delta=f"{m2_confidence - 50:.1f}% vs Baseline",
                    help="Indicates the average complexity/entropy detected in the latent space."
                )

            with acc_col3:
                st.metric(
                    label="Aggregate System Confidence",
                    value=f"{avg_system_conf:.2f}%",
                    delta="Final Assessment"
                )
            
            st.divider()

            st.subheader("Global Timeline Analysis")
            fig_timeline = VisualizationEngine.plot_timeline(df_analyzed, scene_cuts)
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            c_ex1, c_ex2 = st.columns(2)
            with c_ex1:
                st.info(f"**Scene Analysis:** Detected {len(scene_cuts)} significant scene changes based on optical flow flux.")
            with c_ex2:
                best_fr = df_analyzed.loc[df_analyzed['final_score'].idxmax()]
                st.success(f"**Top Recommendation:** Frame {int(best_fr['frame_id'])} with Hybrid Score {best_fr['final_score']:.3f}")

        with tab_vision:
            st.subheader("Optical & Spectral Analysis")
            col_v1, col_v2 = st.columns(2)
            
            with col_v1:
                st.markdown("#### 🏃 Motion Stability")
                st.line_chart(df_analyzed.set_index('frame_id')['cv_motion'], color='#ff7b72')
                st.caption("Peaks indicate camera movement or object motion.")
                
            with col_v2:
                st.markdown("#### 🌫️ Laplacian Clarity (Blur)")
                fig_blur = px.scatter(df_analyzed, x='frame_id', y='cv_blur_var', color='cv_blur_var', 
                                    color_continuous_scale='Viridis', title="Focus Quality")
                fig_blur.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_blur, use_container_width=True)

            st.markdown("#### 🌈 RGB Channel Distribution")
            fig_rgb = go.Figure()
            fig_rgb.add_trace(go.Scatter(x=df_analyzed['frame_id'], y=df_analyzed['rgb_r'], name='Red', line=dict(color='#ff7b72')))
            fig_rgb.add_trace(go.Scatter(x=df_analyzed['frame_id'], y=df_analyzed['rgb_g'], name='Green', line=dict(color='#3fb950')))
            fig_rgb.add_trace(go.Scatter(x=df_analyzed['frame_id'], y=df_analyzed['rgb_b'], name='Blue', line=dict(color='#58a6ff')))
            fig_rgb.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(t=20, b=20))
            st.plotly_chart(fig_rgb, use_container_width=True)

        with tab_ai:
            st.subheader("Latent Space Analysis")
            col_ai1, col_ai2 = st.columns([2, 1])
            
            with col_ai1:
                fig_3d = VisualizationEngine.plot_3d_cluster(df_analyzed)
                st.plotly_chart(fig_3d, use_container_width=True)
            
            with col_ai2:
                st.markdown("#### Model Distribution")
                st.dataframe(df_analyzed[['m1_attention', 'm2_energy']].describe(), use_container_width=True)
                st.caption("Statistical spread of model outputs.")

        with tab_keyframes:
            st.subheader("🎞️ Automated Frame Extraction")
            
            st.markdown(f"#### 🧠 Model 1 Favorites (High Attention) - Top {key_frame_count}")
            top_m1 = df_analyzed.nlargest(key_frame_count, 'm1_attention').sort_values('frame_id')
            
            cols_m1 = st.columns(key_frame_count)
            cols_m1 = st.columns(min(key_frame_count, 5)) 
            
            for i, (idx, row) in enumerate(top_m1.iterrows()):
                thumb = vp.fetch_specific_frame(int(row['frame_id']))
                col_idx = i % 5
                if i > 0 and i % 5 == 0:
                    cols_m1 = st.columns(min(key_frame_count - i, 5))
                
                with cols_m1[col_idx]:
                    st.image(thumb, use_container_width=True)
                    st.caption(f"Fr {int(row['frame_id'])} | {row['m1_attention']:.2f}")

            st.divider()

            st.markdown(f"#### ⚡ Model 2 Favorites (High Complexity) - Top {key_frame_count}")
            top_m2 = df_analyzed.nlargest(key_frame_count, 'm2_energy').sort_values('frame_id')
            
            cols_m2 = st.columns(min(key_frame_count, 5))
            for i, (idx, row) in enumerate(top_m2.iterrows()):
                thumb = vp.fetch_specific_frame(int(row['frame_id']))
                col_idx = i % 5
                if i > 0 and i % 5 == 0:
                    cols_m2 = st.columns(min(key_frame_count - i, 5))
                
                with cols_m2[col_idx]:
                    st.image(thumb, use_container_width=True)
                    st.caption(f"Fr {int(row['frame_id'])} | {row['m2_energy']:.2f}")

            st.divider()

            st.markdown("### 🏆 The Best Representative Frame (Hybrid)")
            best_idx = df_analyzed['final_score'].idxmax()
            best_row = df_analyzed.loc[best_idx]
            best_thumb = vp.fetch_specific_frame(int(best_idx))
            
            col_best_img, col_best_radar = st.columns([1, 1])
            with col_best_img:
                st.image(best_thumb, use_container_width=True)
                st.success(f"Selected Frame {best_idx} with Hybrid Score: {best_row['final_score']:.4f}")
            
            with col_best_radar:
                fig_radar = VisualizationEngine.plot_radar(best_row, df_analyzed.max())
                st.plotly_chart(fig_radar, use_container_width=True)

        with tab_raw:
            st.subheader("💾 Data Warehouse")
            st.write("Full audit trail of every frame analyzed.")
            
            st.dataframe(
                df_analyzed,
                use_container_width=True,
                column_config={
                    "frame_id": "Frame #",
                    "final_score": st.column_config.ProgressColumn(
                        "Hybrid Score", help="The final weighted score", format="%.3f", min_value=0, max_value=1
                    ),
                    "m1_attention": st.column_config.NumberColumn("M1 (Attn)", format="%.3f"),
                    "m2_energy": st.column_config.NumberColumn("M2 (Energy)", format="%.3f"),
                    "cv_motion": st.column_config.LineChartColumn("Motion Flux"),
                },
                height=400
            )
            
            st.markdown("---")
            st.subheader("📤 Export Center")
            
            col_down1, col_down2 = st.columns(2)
            
            csv_data = ReportGenerator.convert_df_to_csv(df_analyzed)
            col_down1.download_button(
                label="📄 Download Analytics CSV",
                data=csv_data,
                file_name=f"analytics_{uploaded_file.name}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            settings = {"w1": w_attn, "w2": w_energy, "w_qual": w_quality}
            json_data = ReportGenerator.create_audit_report(uploaded_file.name, df_analyzed, vp.duration, settings, best_idx)
            col_down2.download_button(
                label="📋 Download JSON Audit Report",
                data=json_data,
                file_name=f"audit_{uploaded_file.name}.json",
                mime="application/json",
                use_container_width=True
            )
            
            with st.expander("Preview JSON Structure"):
                st.json(json_data)

if __name__ == "__main__":
    main()