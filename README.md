# Hate Speech Detection

## Overview
This project is a Hate Speech Detection model that analyzes text data from Twitter to classify it into three categories:
- **Hate Speech**
- **Offensive Language**
- **No Hate or Offensive Language**

The project uses a Decision Tree Classifier for building the model and includes data cleaning, feature extraction, and evaluation.

---

## Table of Contents
- [Project Workflow](#project-workflow)
- [Technologies Used](#technologies-used)
- [Dataset Information](#dataset-information)
- [Setup Instructions](#setup-instructions)
- [Key Features](#key-features)
- [Model Evaluation](#model-evaluation)
- [Usage](#usage)
- [Output](#output)
- [Contributors](#contributors)

---

## Project Workflow
1. **Data Loading**: Load the dataset (`TwitterHate.csv`) containing tweets and their labels.
2. **Data Preprocessing**:
   - Remove null values.
   - Map numerical labels to their respective categories.
   - Clean the text data (remove URLs, punctuation, digits, and stopwords; perform stemming).
3. **Feature Extraction**:
   - Convert text data into a numerical format using `CountVectorizer`.
4. **Model Building**:
   - Use a Decision Tree Classifier to train the model.
5. **Model Evaluation**:
   - Evaluate the model using a confusion matrix and accuracy score.
6. **Prediction**:
   - Test the model on a sample text input.

---

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - `pandas`
  - `numpy`
  - `nltk`
  - `sklearn`
  - `matplotlib`
  - `seaborn`

---

## Dataset Information
- **File Name**: `TwitterHate.csv`
- **Columns**:
  - `tweet`: The text of the tweet.
  - `label`: The classification of the tweet into the categories listed above.

---

## Setup Instructions
1. Clone this repository.
   ```bash
   git clone https://github.com/yourusername/hate-speech-detection.git
   ```
2. Navigate to the project directory.
   ```bash
   cd hate-speech-detection
   ```
3. Install required libraries.
   ```bash
   pip install -r requirements.txt
   ```
4. Place the `TwitterHate.csv` file in the project directory.
5. Run the `HateSpeechDetection.ipynb` file to execute the code.

---

## Key Features
- **Data Cleaning**: Preprocess tweets by removing unnecessary characters, URLs, and stopwords, and apply stemming to standardize words.
- **Feature Engineering**: Convert text data into a bag-of-words representation using `CountVectorizer`.
- **Machine Learning Model**: Build and train a Decision Tree Classifier.
- **Visualization**: Generate a confusion matrix heatmap to visualize the model’s performance.

---

## Model Evaluation
- **Confusion Matrix**:
  ```
  [[9572,  234],
   [ 367,  375]]
  ```
- **Accuracy Score**: 94.3%

---

## Usage
### Sample Prediction
Use the following sample code to test a custom input:
```python
sample = "Let's unite and kill all the people who are protesting against the government"
sample = clean_data(sample)
data1 = cv.transform([sample]).toarray()
prediction = dt.predict(data1)
print(prediction)
```
---

## Output
The model predicts the category of the input text correctly. For the given sample, it detects hate speech.

---

## Contributors
This project was developed by **Akanksha Sawant**. Feel free to contribute or reach out with any suggestions or feedback.

---

## Future Work
- Integrate deep learning models (e.g., LSTM, BERT) for better performance.
- Deploy the model using Flask or Django for real-time hate speech detection.
- Expand the dataset to include multilingual hate speech.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements
- Dataset providers and open-source libraries.
- Inspiration from various hate speech detection research and projects.
