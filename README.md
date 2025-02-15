# 👻 Ghost Type Prediction

## 📚 Introduction

In this project, the goal is to predict the **Ghost** type from a dataset containing various features such as `bone_length`, `rotting_flesh`, `hair_length`, `has_soul`, and `color`. The dataset includes different types of creatures, including **Ghoul**, **Goblin**, and **Ghost**. By leveraging machine learning algorithms, we aim to classify the creatures into one of these three types based on the given features, with a specific focus on predicting the **Ghost** type. 

The dataset contains key characteristics of these creatures that can help us distinguish between the types, including physical attributes (e.g., bone length, rotting flesh, hair length) and more intangible qualities (e.g., soul presence, color). This predictive model will be built and evaluated using machine learning techniques to determine its accuracy in classifying the **Ghost** type.

## 🎯 Aim

The aim of this project is to develop a classification model that accurately predicts the **Ghost** type from the dataset. We will preprocess the data, engineer relevant features, train machine learning models, and evaluate their performance. Specifically, we will focus on:

- Exploring the dataset and performing feature engineering.
- Training classification models to predict the type of the creatures.
- Evaluating the model performance with metrics such as accuracy, confusion matrix, and classification report.
- Fine-tuning the model to improve its predictive power.

## 📊 Dataset Columns Description

- **id**: A unique identifier for each observation.
- **bone_length**: The length of the creature's bones, a numerical feature that may be indicative of its type.
- **rotting_flesh**: A numerical feature representing the amount of rotting flesh, potentially distinguishing certain types.
- **hair_length**: The length of the creature's hair, used to differentiate between various types.
- **has_soul**: A numerical value representing whether the creature has a soul, with values ranging from 0 to 1 (0 indicating no soul and 1 indicating the presence of a soul).
- **color**: The color of the creature, which might be correlated with its type (e.g., clear, green, black).
- **type**: The class label, indicating the type of creature. This is the target variable we aim to predict, with possible values being **Ghoul**, **Goblin**, and **Ghost**.

## 📝 Conclusion

In this project, we aimed to predict the type of creature, specifically focusing on the **Ghost** type, using various machine learning classifiers. After training several models on the dataset, we evaluated their performance based on accuracy scores.

### 🔑 Key Takeaways:

- The **Random Forest Classifier** emerged as the top performer, with an accuracy of 70.67%, followed closely by **Logistic Regression** at 69.33%.
- Models like **Gradient Boosting** and **LightGBM** showed competitive results, achieving 68% accuracy.
- **Gaussian Naive Bayes**, **Decision Tree**, and **KNeighbors** classifiers had relatively lower accuracy, with **KNeighbors Classifier** performing the worst at 58.67%.
- **Bernoulli Naive Bayes** showed the lowest accuracy at 29.33%, making it unsuitable for this classification task.

### 💡 Insights:
- The **Random Forest Classifier** is the most reliable model for predicting the Ghost type.
- **Gradient Boosting** and **LightGBM** can still provide useful insights with further fine-tuning.
- Feature engineering and model tuning will help improve accuracy across all models.

## 🔗 Links

- **Kaggle Notebook**: [Ghost Type Prediction on Kaggle](https://www.kaggle.com/code/senasudemir/ghost-type-prediction?scriptVersionId=222671483)
- **Hugging Face Space**: [Ghost Type Prediction on Hugging Face](https://huggingface.co/spaces/Senasu/Ghost_Type_Prediction)
