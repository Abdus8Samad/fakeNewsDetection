# Fake News Detection using NLP, Machine Learning, and Deep Learning

## Motivation
In recent years, the rapid spread of fake news on social media and other digital platforms has become a significant problem, often leading to misinformation and chaos. Our project aims to address this issue by applying Machine Learning techniques to identify and classify fake news, ensuring the dissemination of accurate information.

## The Team 😀
All of us are CS undergrads at `NSUT Delhi`:
- Abdus Samad
- Samad Ahmed

## Dataset and Publications Used
- The data is obtained from the [`LIAR dataset`](https://paperswithcode.com/dataset/liar), which contains 12.8K manually labeled short statements collected over a decade from POLITIFACT.COM.
- Key publications include:
  - [`“Liar, Liar Pants on Fire”: A New Benchmark Dataset for Fake News Detection`](https://arxiv.org/pdf/1705.00648.pdf)
  - [`Fake News Detection Using Machine Learning approaches: A systematic Review`](https://www.researchgate.net/publication/336436870_Fake_News_Detection_Using_Machine_Learning_approaches_A_systematic_Review)

## Goals of the Project
This project aims to classify news articles or statements as fake or real. The steps include:
1. Data Preprocessing: Splitting the data into training, testing, and validation sets (70:15:15).
2. Running Decision Trees with varying depths (4 to 20) using GINI Gain and Entropy criteria.
3. Implementing a Random Forest by ensembling weak decision tree classifiers.
4. Applying Adaboost to improve the performance of Decision Trees.
5. Using Logistic Regression and SVM for classification.
6. Implementing Artificial Neural Networks for improved accuracy.

### File Structure  
- `FakeNewsDetectionModel.ipynb`: Jupyter Notebook with ML Model  
- `Validation_data.tsv`, `test_data.tsv`, `train_data.tsv`: Dataset files
- `Project Report`: Detailed project report
- `Project Presentation.pptx`: Project presentation

## 1. Introduction to the Problem Statement
Technology has revolutionized information access, but it has also enabled the spread of fake news. This misinformation, often biased and politically motivated, can lead to significant societal issues. Our project focuses on analyzing news from social media, classifying it as fake or real using NLP and various Machine Learning models. By preprocessing the data and training multiple models, we aim to improve accuracy in fake news detection.

## 2. Literature Review
We reviewed several research papers to understand existing methodologies for fake news detection. Common techniques include converting text to numerical values using vectorization techniques like TF-IDF and BOW. Various models, such as SVM, Random Forest, Naïve Bayes, and deep learning algorithms like CNN, have been employed in these studies. The LIAR dataset is a common choice among researchers, and we also observed the use of different feature extraction methods and the impact of dataset biases.

## 3. Dataset with Pre-Processing Techniques
We used the LIAR dataset, containing various features such as statement, speaker, and party affiliation.

### 3.1. Data Description
The dataset has 16 columns and 12788 rows, with features including statement ID, label, statement, subject, speaker, party affiliation, and credit history account.

### 3.2. Data Cleaning
We cleaned the dataset by merging and dropping unnecessary columns, converting labels into binary classifications, and preprocessing the text data.

### 3.3. Data Preprocessing [Use of NLP]
We refined the text data by removing punctuation, links, and stop words, and then applied tokenization and lemmatization. We used NLP algorithms like BOW and TF-IDF for vectorization and label encoding to convert categorical data into numeric form.

## 4. Methodology and Model Details
### 4.1. Methodology
We divided the data into four parts, applied TF-IDF and BOW, and used Grid Search for hyper-parameter optimization. PCA and t-SNE were employed for dimensionality reduction. The models used include Logistic Regression, Naïve Bayes, Decision Tree, Random Forest, AdaBoost, SVM, and deep learning with Neural Networks.

### 4.2. Model Details
The models were applied to all four cases, and we observed their performance in terms of accuracy. Logistic Regression, Naïve Bayes, Decision Tree, and Random Forest were used, with Random Forest showing the highest accuracy.

## 5. Results and Analysis
### 5.1. Results
![Results](https://user-images.githubusercontent.com/76804249/210210409-0378da78-4f25-452b-a691-38255f32b1e3.png)

### 5.2. Analysis
Random Forest with BOW and considering both the speaker and party affiliation provided the highest accuracy (62.93%). Logistic Regression showed the best precision, recall, and F1-score. The inclusion of speaker and party data improved model performance, highlighting the complexity of fake news detection and the need for further improvements.

By applying multiple algorithms and preprocessing techniques, we aim to contribute to the accurate classification of news, mitigating the impact of fake news in society.