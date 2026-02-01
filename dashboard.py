import streamlit as st
import pandas as pd
import numpy as np
import json
import subprocess
import os
import plotly.graph_objects as go

from datetime import datetime
from naive_bayes import predict_sentiment as nb_predict
from bert import predict_sentiment as bert_predict

# Set page config
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.markdown('<h1 style="color: #1f77b4;">📊 Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown("Analisis Sentimen Timnas Indonesia menggunakan Naive Bayes + TF-IDF vs IndoBERT")

# Sidebar
st.sidebar.title("Menu")
selected_page = st.sidebar.radio(
    "Pilih Halaman:",
    ["📈 Overview", "🤖 Naive Bayes", "🧠 IndoBERT", "📊 Perbandingan"]
)

# Helper function
def load_results(filename):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def get_dataset_info():
    """Baca vader_result.csv dan kembalikan informasi dataset"""
    try:
        df = pd.read_csv(
            'vader_result.csv',
            encoding='utf-8',
            on_bad_lines='skip',
            engine='python'
        )
        df.columns = df.columns.str.strip()
        
        # Hitung jumlah baris bersih
        total_rows = len(df)
        
        # Hitung jumlah per label
        label_counts = df['label'].str.strip().value_counts().to_dict()
        
        return {
            'total_samples': total_rows,
            'label_counts': label_counts,
            'classes': list(label_counts.keys())
        }
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return None

def run_preprocessing():
    """Run preprocessing.py to generate vader_result.csv"""
    script = "preprocessing.py"
    with st.spinner("Menjalankan preprocessing data..."):
        result = subprocess.run(
            ["/Users/surya/Documents/Projects/thesis/.venv/bin/python", script],
            cwd="/Users/surya/Documents/Projects/thesis",
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            st.success("✅ Preprocessing selesai!")
            return True
        else:
            st.error(f"❌ Error: {result.stderr}")
            return False

def run_training(model_type):
    if model_type == "naive_bayes":
        script = "naive_bayes.py"
        with st.spinner("Training Naive Bayes + TF-IDF..."):
            result = subprocess.run(
                ["/Users/surya/Documents/Projects/thesis/.venv/bin/python", script],
                cwd="/Users/surya/Documents/Projects/thesis",
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                st.success("✅ Training Naive Bayes selesai!")
            else:
                st.error(f"❌ Error: {result.stderr}")
    
    elif model_type == "indobert":
        script = "bert.py"
        with st.spinner("Training IndoBERT..."):
            result = subprocess.run(
                ["/Users/surya/Documents/Projects/thesis/.venv/bin/python", script],
                cwd="/Users/surya/Documents/Projects/thesis",
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                st.success("✅ Training IndoBERT selesai!")
            else:
                st.error(f"❌ Error: {result.stderr}")

# PAGE 1: OVERVIEW
if selected_page == "📈 Overview":
    st.header("📋 Ringkasan Proyek")
    
    # Get dataset info
    dataset_info = get_dataset_info()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if dataset_info:
            st.info(f"""
        ### 📝 Dataset
        - **Sumber**: YouTube & TikTok Comments
        - **Tema**: Timnas Indonesia
        - **Data Bersih**: {dataset_info['total_samples']} sampel
        - **Label**: {', '.join(dataset_info['classes'])}
        - **Distribusi**:
          - Negative: {dataset_info['label_counts'].get('negative', 0)}
          - Positive: {dataset_info['label_counts'].get('positive', 0)}
          - Neutral: {dataset_info['label_counts'].get('neutral', 0)}
        """)
        else:
            st.info("""
        ### 📝 Dataset
        - **Sumber**: YouTube & TikTok Comments
        - **Tema**: Timnas Indonesia
        - **Data Bersih**: ~ sampel
        - **Label**: Negative, Positive, Neutral
        """)
    
    with col2:
        st.info("""
        ### 🎯 Model
        1. **Naive Bayes + TF-IDF**
        2. **IndoBERT** (Pre-trained)
        
        Tujuan: Perbandingan akurasi dan performa
        """)

    # Preprocessing Button
    if os.path.exists('vader_result.csv'):
        st.success("✅ Preprocessing sudah dijalankan. vader_result.csv tersedia.")
    else:
        st.warning("⚠️ Preprocessing belum dijalankan. Silakan jalankan preprocessing terlebih dahulu.")

    if st.button("▶️ Jalankan Preprocessing", use_container_width=True):
        if run_preprocessing():
            st.rerun()
    
    st.divider()
    
    # Status
    st.subheader("📊 Status Model")
    
    nb_results = load_results("naive_bayes_tfidf_results.json")
    bert_results = load_results("indobert_results.json")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if nb_results:
            st.success("✅ Naive Bayes")
            st.metric("Accuracy", f"{nb_results['overall_metrics']['accuracy_percentage']:.2f}%")
        else:
            st.warning("⚠️ Belum dijalankan")

        if st.button("▶️ Train Naive Bayes", use_container_width=True):
            run_training("naive_bayes")
            st.rerun()
    
    with col2:
        if bert_results:
            st.success("✅ IndoBERT")
            st.metric("Accuracy", f"{bert_results['overall_metrics']['accuracy_percentage']:.2f}%")
        else:
            st.warning("⚠️ Belum dijalankan")
        if st.button("▶️ Train IndoBERT", use_container_width=True):
            run_training("indobert")
            st.rerun()
    
    st.divider()

    # Prediction Demo
    st.subheader("🧪 Demo Prediksi Sentimen")
    
    # Text input with session state
    default_text = st.session_state.get('demo_text', '')
    user_input = st.text_area(
        "Masukkan teks komentar:", 
        value=default_text,
        height=100,
        placeholder="Contoh: Timnas Indonesia main bagus banget hari ini!"
    )
    
    # Predict button
    if st.button("🔍 Prediksi Sentimen", use_container_width=True):
        if not user_input.strip():
            st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")
        else:
            # Get predictions
            nb_prediction = nb_predict(user_input)
            bert_prediction = bert_predict(user_input)
            
            st.markdown("#### Hasil Prediksi:")
            
            # Display results side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🤖 Naive Bayes + TF-IDF**")
                if isinstance(nb_prediction, dict) and 'error' in nb_prediction:
                    st.error(nb_prediction['error'])
                elif isinstance(nb_prediction, dict):
                    # Display sentiment with colored badge
                    sentiment = nb_prediction['prediction']
                    if sentiment == 'positive':
                        st.success(f"**✅ {sentiment.upper()}**")
                    elif sentiment == 'negative':
                        st.error(f"**❌ {sentiment.upper()}**")
                    else:
                        st.info(f"**➖ {sentiment.upper()}**")
                    
                    # Display confidence
                    st.metric("Confidence", f"{nb_prediction['confidence']*100:.2f}%")
                    
                    # Show probabilities
                    with st.expander("📊 Lihat Probabilitas"):
                        for label, prob in sorted(nb_prediction['probabilities'].items(), key=lambda x: x[1], reverse=True):
                            st.write(f"{label.capitalize()}: {prob*100:.2f}%")
                    
                    # Show processed text
                    if 'processed_text' in nb_prediction:
                        with st.expander("🔍 Teks Setelah Preprocessing"):
                            st.code(nb_prediction['processed_text'])
                else:
                    st.success(f"Prediksi: **{nb_prediction.capitalize()}**")
            
            with col2:
                st.markdown("**🧠 IndoBERT**")
                if isinstance(bert_prediction, dict) and 'error' in bert_prediction:
                    st.error(bert_prediction['error'])
                elif isinstance(bert_prediction, dict):
                    # Display sentiment with colored badge
                    sentiment = bert_prediction['prediction']
                    if sentiment == 'positive':
                        st.success(f"**✅ {sentiment.upper()}**")
                    elif sentiment == 'negative':
                        st.error(f"**❌ {sentiment.upper()}**")
                    else:
                        st.info(f"**➖ {sentiment.upper()}**")
                    
                    # Display confidence
                    st.metric("Confidence", f"{bert_prediction['confidence']*100:.2f}%")
                    
                    # Show probabilities
                    with st.expander("📊 Lihat Probabilitas"):
                        for label, prob in sorted(bert_prediction['probabilities'].items(), key=lambda x: x[1], reverse=True):
                            st.write(f"{label.capitalize()}: {prob*100:.2f}%")
                else:
                    st.success(f"Prediksi: **{bert_prediction.capitalize()}**")
            
            # Comparison chart
            if isinstance(nb_prediction, dict) and isinstance(bert_prediction, dict) and 'probabilities' in nb_prediction and 'probabilities' in bert_prediction:
                st.divider()
                st.markdown("#### 📊 Perbandingan Probabilitas")
                
                # Prepare data for comparison
                labels = list(nb_prediction['probabilities'].keys())
                nb_probs = [nb_prediction['probabilities'][label] for label in labels]
                bert_probs = [bert_prediction['probabilities'][label] for label in labels]
                
                # Create grouped bar chart
                fig = go.Figure(data=[
                    go.Bar(name='Naive Bayes', x=[l.capitalize() for l in labels], y=nb_probs, marker_color='#636EFA'),
                    go.Bar(name='IndoBERT', x=[l.capitalize() for l in labels], y=bert_probs, marker_color='#00CC96')
                ])
                
                fig.update_layout(
                    barmode='group',
                    yaxis_title='Probabilitas',
                    yaxis=dict(range=[0, 1]),
                    height=350,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)

# PAGE 2: NAIVE BAYES
elif selected_page == "🤖 Naive Bayes":
    st.header("🤖 Naive Bayes + TF-IDF")
    
    nb_results = load_results("naive_bayes_tfidf_results.json")
    
    if nb_results:
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{nb_results['overall_metrics']['accuracy_percentage']:.2f}%")
        with col2:
            st.metric("Train Samples", nb_results['dataset_info']['train_samples'])
        with col3:
            st.metric("Test Samples", nb_results['dataset_info']['test_samples'])
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            labels = list(nb_results['per_class_metrics'].keys())
            precision = [nb_results['per_class_metrics'][label]['precision'] for label in labels]
            recall = [nb_results['per_class_metrics'][label]['recall'] for label in labels]
            
            fig = go.Figure(data=[
                go.Bar(name='Precision', x=labels, y=precision),
                go.Bar(name='Recall', x=labels, y=recall)
            ])
            fig.update_layout(title="Precision vs Recall")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            conf_matrix = np.array(nb_results['confusion_matrix']['matrix'])
            labels_list = ['Negative', 'Positive', 'Neutral']
            fig = go.Figure(data=go.Heatmap(
                z=conf_matrix,
                x=labels_list,
                y=labels_list,
                text=conf_matrix,
                texttemplate='%{text}',
                colorscale='Blues'
            ))
            fig.update_layout(title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)
        
        # Metrics Table
        st.subheader("📋 Per-Class Metrics")
        metrics_data = []
        for label, metrics in nb_results['per_class_metrics'].items():
            metrics_data.append({
                'Label': label.capitalize(),
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}"
            })
        
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
    
    else:
        st.warning("⚠️ Hasil belum tersedia")
        if st.button("▶️ Train Sekarang"):
            run_training("naive_bayes")
            st.rerun()

# PAGE 3: INDOBERT
elif selected_page == "🧠 IndoBERT":
    st.header("🧠 IndoBERT")
    
    bert_results = load_results("indobert_results.json")
    
    if bert_results:
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{bert_results['overall_metrics']['accuracy_percentage']:.2f}%")
        with col2:
            st.metric("Train Samples", bert_results['dataset_info']['train_samples'])
        with col3:
            st.metric("Test Samples", bert_results['dataset_info']['test_samples'])
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            labels = list(bert_results['per_class_metrics'].keys())
            precision = [bert_results['per_class_metrics'][label]['precision'] for label in labels]
            recall = [bert_results['per_class_metrics'][label]['recall'] for label in labels]
            
            fig = go.Figure(data=[
                go.Bar(name='Precision', x=labels, y=precision),
                go.Bar(name='Recall', x=labels, y=recall)
            ])
            fig.update_layout(title="Precision vs Recall")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            conf_matrix = np.array(bert_results['confusion_matrix']['matrix'])
            labels_list = ['Negative', 'Positive', 'Neutral']
            fig = go.Figure(data=go.Heatmap(
                z=conf_matrix,
                x=labels_list,
                y=labels_list,
                text=conf_matrix,
                texttemplate='%{text}',
                colorscale='Greens'
            ))
            fig.update_layout(title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)
        
        # Metrics Table
        st.subheader("📋 Per-Class Metrics")
        metrics_data = []
        for label, metrics in bert_results['per_class_metrics'].items():
            metrics_data.append({
                'Label': label.capitalize(),
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}"
            })
        
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
    
    else:
        st.warning("⚠️ Hasil belum tersedia")
        if st.button("▶️ Train Sekarang"):
            run_training("indobert")
            st.rerun()

# PAGE 4: COMPARISON
elif selected_page == "📊 Perbandingan":
    st.header("📊 Perbandingan Model")
    
    nb_results = load_results("naive_bayes_tfidf_results.json")
    bert_results = load_results("indobert_results.json")
    
    if nb_results and bert_results:
        # Overall Comparison
        models = ['Naive Bayes', 'IndoBERT']
        accuracies = [
            nb_results['overall_metrics']['accuracy_percentage'],
            bert_results['overall_metrics']['accuracy_percentage']
        ]
        
        fig = go.Figure(data=[
            go.Bar(x=models, y=accuracies, marker_color=['#636EFA', '#00CC96'])
        ])
        fig.update_layout(
            title="Akurasi Model",
            yaxis_title="Accuracy (%)",
            yaxis=dict(range=[0, 100]),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Metrics Comparison Table
        st.subheader("📋 Perbandingan Metrik")
        
        comparison_data = {
            'Metrik': ['Accuracy', 
                      'Precision (Neg)', 'Recall (Neg)', 'F1 (Neg)',
                      'Precision (Pos)', 'Recall (Pos)', 'F1 (Pos)',
                      'Precision (Neutral)', 'Recall (Neutral)', 'F1 (Neutral)'],
            'Naive Bayes': [
                f"{nb_results['overall_metrics']['accuracy_percentage']:.2f}%",
                f"{nb_results['per_class_metrics']['negative']['precision']:.4f}",
                f"{nb_results['per_class_metrics']['negative']['recall']:.4f}",
                f"{nb_results['per_class_metrics']['negative']['f1']:.4f}",
                f"{nb_results['per_class_metrics']['positive']['precision']:.4f}",
                f"{nb_results['per_class_metrics']['positive']['recall']:.4f}",
                f"{nb_results['per_class_metrics']['positive']['f1']:.4f}",
                f"{nb_results['per_class_metrics']['neutral']['precision']:.4f}",
                f"{nb_results['per_class_metrics']['neutral']['recall']:.4f}",
                f"{nb_results['per_class_metrics']['neutral']['f1']:.4f}"
            ],
            'IndoBERT': [
                f"{bert_results['overall_metrics']['accuracy_percentage']:.2f}%",
                f"{bert_results['per_class_metrics']['negative']['precision']:.4f}",
                f"{bert_results['per_class_metrics']['negative']['recall']:.4f}",
                f"{bert_results['per_class_metrics']['negative']['f1']:.4f}",
                f"{bert_results['per_class_metrics']['positive']['precision']:.4f}",
                f"{bert_results['per_class_metrics']['positive']['recall']:.4f}",
                f"{bert_results['per_class_metrics']['positive']['f1']:.4f}",
                f"{bert_results['per_class_metrics']['neutral']['precision']:.4f}",
                f"{bert_results['per_class_metrics']['neutral']['recall']:.4f}",
                f"{bert_results['per_class_metrics']['neutral']['f1']:.4f}"
            ]
        }
        
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
        
        st.divider()
        
        # Confusion Matrix Comparison
        col1, col2 = st.columns(2)
        
        with col1:
            nb_conf = np.array(nb_results['confusion_matrix']['matrix'])
            labels_list = ['Negative', 'Positive', 'Neutral']
            fig = go.Figure(data=go.Heatmap(
                z=nb_conf,
                x=labels_list,
                y=labels_list,
                text=nb_conf,
                texttemplate='%{text}',
                colorscale='Blues'
            ))
            fig.update_layout(title="Naive Bayes", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            bert_conf = np.array(bert_results['confusion_matrix']['matrix'])
            labels_list = ['Negative', 'Positive', 'Neutral']
            fig = go.Figure(data=go.Heatmap(
                z=bert_conf,
                x=labels_list,
                y=labels_list,
                text=bert_conf,
                texttemplate='%{text}',
                colorscale='Greens'
            ))
            fig.update_layout(title="IndoBERT", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Kesimpulan
        nb_acc = nb_results['overall_metrics']['accuracy_percentage']
        bert_acc = bert_results['overall_metrics']['accuracy_percentage']
        diff = bert_acc - nb_acc
        
        if bert_acc > nb_acc:
            st.success(f"""
            **🏆 IndoBERT lebih baik: {bert_acc:.2f}% vs {nb_acc:.2f}% (+{diff:.2f}%)**
            
            IndoBERT adalah pre-trained model dengan transfer learning yang lebih powerful.
            """)
        else:
            st.info(f"Kedua model memberikan hasil kompetitif: {nb_acc:.2f}% vs {bert_acc:.2f}%")
    
    else:
        st.warning("⚠️ Jalankan kedua model terlebih dahulu")

st.divider()
st.caption(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
