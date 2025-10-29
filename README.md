# PyCaret Machine Learning Tutorials

#Note: Please open the colab links for the 7 colabs in this readme file, pdf for some reason aren't loading, also check the videos

Welcome to the complete collection of PyCaret machine learning tutorials! This repository contains 7 comprehensive Jupyter notebooks covering various machine learning tasks using PyCaret - a low-code Python library that automates machine learning workflows.

---

## 📺 Video Tutorials

Watch the complete video series on YouTube:

1. **Anomaly Detection** - https://youtu.be/9vZRlXzHL_o
2. **Association Rules Mining** - https://youtu.be/2RS-6d39m3g
3. **Binary Classification** - https://youtu.be/gyRUSbDrgHQ
4. **Clustering Analysis** - https://youtu.be/H3-dnJNpJaU
5. **Multi-Class Classification** - https://youtu.be/on5-MiCkmrA
6. **Regression Analysis** - https://youtu.be/nwO2j4ZDb6A
7. **Time Series Forecasting** - https://youtu.be/y6F3BVDBQRk

---

## 📚 Notebooks Overview

### 1. Anomaly Detection (`Anomaly_Detection_Pycaret.ipynb`)
https://colab.research.google.com/drive/1izxHqUqjosMeBfoCGNAuAY8PKaVgj4KC?usp=sharing

**What it does:** Detects fraudulent credit card transactions using unsupervised learning.

**Key Features:**
- Uses credit card transaction dataset
- Implements Isolation Forest algorithm
- GPU-accelerated processing
- UMAP visualization for anomaly patterns
- Automatically assigns anomaly scores to each transaction

**Use Cases:** Fraud detection, network intrusion detection, quality control, system health monitoring

**Dataset:** Credit Card Transactions (`creditcard.csv`)

---

### 2. Association Rules Mining (`Association_Pycaret.ipynb`)
https://colab.research.google.com/drive/1ONQC9YYdI_DE93OB1CswI84v9za8Qu0X?usp=sharing

**What it does:** Discovers hidden patterns and relationships in grocery shopping data.

**Key Features:**
- Market basket analysis using Apriori algorithm
- Identifies frequently purchased product combinations
- Generates association rules with support, confidence, and lift metrics
- Visualizes top product associations
- Helps optimize product placement and cross-selling strategies

**Use Cases:** Retail analytics, recommendation systems, inventory management, promotional bundling

**Dataset:** Grocery Store Transactions (`Groceries_dataset.csv`)

---

### 3. Binary Classification (`Binary_Classification_Pycaret.ipynb`)
https://colab.research.google.com/drive/1N-5fKthnErW4JLPns7xOetwAJv-QSlTD?usp=sharing

**What it does:** Predicts presence or absence of heart disease (two-class problem).

**Key Features:**
- Compares 15+ classification algorithms automatically
- Handles train-test split
- Interactive model evaluation dashboard
- Provides predictions with probability scores
- GPU-accelerated model training

**Use Cases:** Medical diagnosis, spam detection, customer churn prediction, loan approval

**Dataset:** Heart Disease Dataset (`HeartDiseaseTrain-Test.csv`)

**Target Variable:** `target` (0 = No disease, 1 = Disease present)

---

### 4. Clustering Analysis (`Clustering_Pycaret.ipynb`)
https://colab.research.google.com/drive/1N-5fKthnErW4JLPns7xOetwAJv-QSlTD?usp=sharing

**What it does:** Segments wholesale customers into distinct groups based on purchasing behavior.

**Key Features:**
- Unsupervised learning with K-Means clustering
- Automatic feature normalization
- Customer segmentation based on product categories
- 2D cluster visualization
- Identifies natural customer groups

**Use Cases:** Customer segmentation, image compression, document clustering, anomaly detection

**Dataset:** Wholesale Customers Dataset (`Wholesale customers data.csv`)

**Features:** Annual spending on Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen

---

### 5. Multi-Class Classification (`MultiClass_Pycaret.ipynb`)
https://colab.research.google.com/drive/1-UmxZn8agfW9wd1lrK2E3zaJaCh9Xkip?usp=sharing

**What it does:** Predicts wine quality ratings across multiple categories (3-8 scale).

**Key Features:**
- Handles imbalanced multi-class classification
- Compares multiple algorithms with multi-class metrics
- Automatic preprocessing and encoding
- Predicts quality with confidence scores
- GPU-accelerated training

**Use Cases:** Product quality grading, sentiment analysis (positive/neutral/negative), disease stage classification

**Dataset:** Wine Quality Dataset (`WineQT.csv`)

**Target Variable:** `quality` (ratings from 3 to 8)

**Features:** Chemical properties including acidity, sugar, alcohol content, pH, sulfates

---

### 6. Regression Analysis (`Regression_Pycaret.ipynb`)
https://colab.research.google.com/drive/1tKKCW3PQXsSsShXXgA7VF47QhVrJ_Bhu?usp=sharing

**What it does:** Predicts continuous house prices based on property features.

**Key Features:**
- Compares 20+ regression algorithms
- Automatic feature engineering and preprocessing
- Handles categorical and numerical features
- Provides metrics like R², RMSE, MAE
- Production-ready model finalization

**Use Cases:** Price prediction, demand forecasting, risk assessment, sales forecasting

**Dataset:** Housing Dataset (`Housing.csv`)

**Target Variable:** `price` (house price in currency units)

**Features:** Area, bedrooms, bathrooms, stories, parking, furnishing status, location factors

---

### 7. Time Series Forecasting (`Time_Series_Forecasting.ipynb`)
https://colab.research.google.com/drive/127-XSHyKVJp3ib3ShIrUjQS6w0Qe-h_n?usp=sharing

**What it does:** Forecasts future daily temperatures using historical time series data.

**Key Features:**
- Specialized time series algorithms (ARIMA, ETS, Prophet)
- Handles seasonal patterns (365-day cycle)
- 5-fold time series cross-validation
- Future predictions with confidence intervals
- Visual forecast plots with historical context

**Use Cases:** Weather forecasting, stock price prediction, demand forecasting, energy consumption planning

**Dataset:** Daily Temperature Dataset (`DailyTemp.csv`)

**Target Variable:** `Temp` (daily temperature)

**Forecast Horizon:** 30 days ahead

---

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# GPU support (optional but recommended)
# CUDA-compatible GPU with TensorFlow
```

### Installation

```bash
# Install PyCaret with all dependencies
pip install pycaret[full]

# For association rules mining
pip install mlxtend pandas

# For GPU support
pip install tensorflow-gpu
```

### Running the Notebooks

1. Clone this repository
2. Install required dependencies
3. Launch Jupyter Notebook or Google Colab
4. Open any notebook and run cells sequentially
5. Replace dataset paths with your own data locations

```bash
jupyter notebook
```

---

## 📊 Common Workflow Across All Notebooks

Each PyCaret notebook follows a consistent, intuitive workflow:

1. **GPU Verification** - Check hardware acceleration availability
2. **Installation** - Install PyCaret and dependencies
3. **Data Loading** - Import and explore dataset
4. **Setup** - Initialize PyCaret environment with target variable
5. **Model Training** - Create or compare models automatically
6. **Evaluation** - Assess model performance with visualizations
7. **Prediction** - Generate predictions on new data

---

## 🎯 Key PyCaret Advantages

- **Low-Code:** Accomplish complex ML tasks with few lines of code
- **AutoML:** Automatic model comparison and selection
- **GPU Support:** Accelerated training on compatible hardware
- **Comprehensive:** Covers classification, regression, clustering, anomaly detection, and time series
- **Production-Ready:** Easy model finalization and deployment
- **Visualizations:** Built-in plots for model evaluation and interpretation
