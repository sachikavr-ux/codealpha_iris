# codealpha_iris

# 🌸 Iris Classification Web App

An interactive **Machine Learning web application** that predicts the species of an Iris flower based on its sepal and petal measurements.

The application uses a trained machine learning classification model to classify an Iris flower into one of three species:

* 🌱 Iris Setosa
* 🌼 Iris Versicolor
* 🌸 Iris Virginica

## 🚀 Live Demo

**Try the application:**
https://23dddcee-5d84-4e11-98a7-379129735958-00-2wq7h330j41cf.pike.replit.dev/

## 📌 Project Overview

The Iris Classification project demonstrates how Machine Learning can be integrated into a web application to make real-time predictions.

Users enter the measurements of an Iris flower, and the application processes the input using a trained classification model and displays the predicted flower species.

## ✨ Features

* 🌸 Iris flower species prediction
* 📊 Machine Learning-based classification
* 🖥️ Simple and user-friendly web interface
* ⚡ Real-time prediction
* 📱 Responsive interface
* 🤖 Automated classification based on flower measurements
* 🌐 Deployed as a web application

## 🧠 Machine Learning

The application is based on the popular **Iris Dataset**, which contains measurements of Iris flowers.

### Input Features

The model uses four features:

| Feature      | Description         |
| ------------ | ------------------- |
| Sepal Length | Length of the sepal |
| Sepal Width  | Width of the sepal  |
| Petal Length | Length of the petal |
| Petal Width  | Width of the petal  |

### Output

The model predicts one of the following classes:

```text
Iris Setosa
Iris Versicolor
Iris Virginica
```

## 🛠️ Technologies Used

* **Python**
* **Machine Learning**
* **Scikit-learn**
* **Pandas**
* **NumPy**
* **HTML**
* **CSS**
* **JavaScript**
* **Web Framework**
* **Replit** for deployment

## 📂 Project Structure

```text
Iris-Classification/
│
├── app.py                 # Main application
├── model.py               # Machine learning model
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html         # Web interface
├── static/
│   ├── style.css          # Styling
│   └── script.js          # Frontend JavaScript
│
└── README.md              # Project documentation
```

> The exact file structure may vary depending on your implementation.

## ⚙️ How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

### 2. Navigate to the project

```bash
cd Iris-Classification
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python app.py
```

Then open the local URL displayed in your terminal.

## 🔍 How It Works

```text
User Input
    ↓
Flower Measurements
    ↓
Data Preprocessing
    ↓
Machine Learning Model
    ↓
Classification
    ↓
Predicted Iris Species
```

## 🎯 Example

Suppose the user enters:

```text
Sepal Length : 5.1
Sepal Width  : 3.5
Petal Length : 1.4
Petal Width  : 0.2
```

The model can classify the flower as:

```text
Iris Setosa
```

## 📚 Dataset

This project uses the well-known **Iris dataset**, commonly used for demonstrating classification algorithms and introductory machine learning concepts.

The dataset contains three Iris species with four numerical features for each sample.

## 💡 Future Improvements

* Add prediction probability
* Add graphical visualization of the input
* Compare multiple ML algorithms
* Add model performance metrics
* Add confusion matrix visualization
* Improve UI/UX
* Add mobile-friendly design
* Deploy using additional cloud platforms
* Add an API endpoint for predictions

## 👩‍💻 Author

**Sachika VR Rajalakshmi V**

Artificial Intelligence & Data Science Student

## ⭐ Support

If you found this project useful, consider giving the repository a ⭐ on GitHub!

---

### 🔗 Live Application

https://23dddcee-5d84-4e11-98a7-379129735958-00-2wq7h330j41cf.pike.replit.dev/
