import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Iris Classification", page_icon="🌸", layout="wide")

st.title("🌸 Iris Flower Classification")
st.markdown("A Machine Learning project to classify Iris flower species")

@st.cache_data
def load_data():
    df = pd.read_csv(r'C:\Users\sachi\Downloads\Iris.csv')
    return df

df = load_data()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "🤖 Model Training", "🔮 Prediction", "📈 Visualizations"])

with tab1:
    st.header("Dataset Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("First 10 Rows")
        st.dataframe(df.head(10))
    with col2:
        st.subheader("Dataset Statistics")
        st.dataframe(df.describe())
    
    st.subheader("Species Distribution")
    fig, ax = plt.subplots()
    df['Species'].value_counts().plot(kind='bar', ax=ax, color=['#3498db', '#2ecc71', '#e74c3c'])
    st.pyplot(fig)

with tab2:
    st.header("Train ML Models")
    
    X = df.drop(['Id', 'Species'], axis=1)
    le = LabelEncoder()
    y = le.fit_transform(df['Species'])
    
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='rbf'),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }
    
    selected_models = st.multiselect("Select Models to Train", list(models.keys()), default=list(models.keys()))
    
    if st.button("🚀 Train Models", type="primary"):
        results = {}
        progress = st.progress(0)
        
        for idx, name in enumerate(selected_models):
            model = models[name]
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            results[name] = {"accuracy": acc, "model": model, "y_pred": y_pred}
            progress.progress((idx + 1) / len(selected_models))
        
        st.success("Training Complete!")
        
        st.subheader("Model Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            for name, res in results.items():
                st.metric(name, f"{res['accuracy']*100:.2f}%")
        
        with col2:
            fig, ax = plt.subplots()
            bars = ax.bar(results.keys(), [r['accuracy'] for r in results.values()], 
                         color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
            ax.set_ylabel('Accuracy')
            ax.set_title('Model Accuracy Comparison')
            ax.set_ylim([0.7, 1.05])
            for bar, acc in zip(bars, [r['accuracy'] for r in results.values()]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{acc*100:.1f}%', ha='center', fontweight='bold')
            st.pyplot(fig)
        
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        best_result = results[best_model_name]
        
        st.subheader(f"Best Model: {best_model_name} ({best_result['accuracy']*100:.2f}%)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.text("Classification Report:")
            st.code(classification_report(y_test, best_result['y_pred'], target_names=le.classes_))
        
        with col2:
            st.text("Confusion Matrix:")
            cm = confusion_matrix(y_test, best_result['y_pred'])
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            st.pyplot(fig)
        
        st.session_state['trained_model'] = best_result['model']
        st.session_state['scaler'] = scaler
        st.session_state['label_encoder'] = le

with tab3:
    st.header("Make Prediction")
    
    if 'trained_model' in st.session_state:
        st.subheader("Enter Flower Measurements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
            sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
        
        with col2:
            petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
            petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
        
        if st.button("🌸 Predict Species", type="primary"):
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            input_scaled = st.session_state['scaler'].transform(input_data)
            prediction = st.session_state['trained_model'].predict(input_scaled)
            species = st.session_state['label_encoder'].inverse_transform(prediction)[0]
            
            st.success(f"Predicted Species: **{species}**")
            
            species_emoji = {
                'Iris-setosa': '🌼',
                'Iris-versicolor': '🌺',
                'Iris-virginica': '🌻'
            }
            st.info(f"{species_emoji.get(species, '🌸')} This flower is classified as **{species}**")
    else:
        st.warning("⚠️ Please train a model first in the Model Training tab!")

with tab4:
    st.header("Data Visualizations")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    sns.scatterplot(data=df, x='SepalLengthCm', y='SepalWidthCm', hue='Species', ax=axes[0, 0])
    axes[0, 0].set_title('Sepal Length vs Sepal Width')
    
    sns.scatterplot(data=df, x='PetalLengthCm', y='PetalWidthCm', hue='Species', ax=axes[0, 1])
    axes[0, 1].set_title('Petal Length vs Petal Width')
    
    sns.boxplot(data=df, x='Species', y='SepalLengthCm', ax=axes[1, 0])
    axes[1, 0].set_title('Sepal Length Distribution')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    sns.boxplot(data=df, x='Species', y='PetalWidthCm', ax=axes[1, 1])
    axes[1, 1].set_title('Petal Width Distribution')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots()
    numeric_df = df.drop(['Id', 'Species'], axis=1)
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

st.sidebar.header("About")
st.sidebar.info("This app classifies Iris flowers using Machine Learning")
st.sidebar.markdown("---")
st.sidebar.markdown("**Models Used:**")
st.sidebar.markdown("- Logistic Regression")
st.sidebar.markdown("- Random Forest")
st.sidebar.markdown("- SVM")
st.sidebar.markdown("- KNN")
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit & Scikit-learn")
