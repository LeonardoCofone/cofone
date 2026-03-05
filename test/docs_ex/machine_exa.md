# Machine Learning — Overview

Machine learning (ML) is a branch of artificial intelligence that enables systems to learn from data and improve their performance over time without being explicitly programmed.
Instead of following hard-coded rules, a machine learning model identifies patterns in training data and uses those patterns to make predictions or decisions on new, unseen data.

## Core Concepts

### Supervised Learning
In supervised learning, the model is trained on labeled data — input-output pairs where the correct answer is known.
Common tasks include classification (predicting a category) and regression (predicting a continuous value).
Examples: spam detection, house price prediction, image recognition.

### Unsupervised Learning
Unsupervised learning involves finding hidden patterns in data without predefined labels.
Common techniques include clustering (grouping similar data points) and dimensionality reduction (compressing data while preserving structure).
Examples: customer segmentation, anomaly detection, topic modeling.

### Reinforcement Learning
In reinforcement learning, an agent learns by interacting with an environment and receiving rewards or penalties for its actions.
The goal is to maximize cumulative reward over time.
Examples: game-playing AI (AlphaGo, OpenAI Five), robotic control, recommendation systems.

## Key Algorithms

- **Linear Regression**: predicts a continuous output as a linear combination of input features.
- **Logistic Regression**: used for binary classification, outputs a probability between 0 and 1.
- **Decision Trees**: hierarchical models that split data based on feature thresholds.
- **Random Forest**: ensemble of decision trees that reduces overfitting through averaging.
- **Support Vector Machines (SVM)**: finds the optimal hyperplane separating classes in high-dimensional space.
- **Neural Networks**: layers of interconnected nodes that learn complex representations through backpropagation.
- **Gradient Boosting (XGBoost, LightGBM)**: builds models sequentially, each correcting errors of the previous one.

## Deep Learning

Deep learning is a subfield of machine learning based on neural networks with many layers (hence "deep").
These models excel at tasks involving unstructured data such as images, audio, and text.
Key architectures include Convolutional Neural Networks (CNNs) for vision, Recurrent Neural Networks (RNNs) for sequences, and Transformers for language.

Large Language Models (LLMs) like GPT-4, Claude, and Gemini are deep learning models trained on vast amounts of text.
They are capable of generating coherent text, answering questions, writing code, and reasoning across diverse domains.

## Evaluation Metrics

Choosing the right evaluation metric depends on the task:
- **Accuracy**: percentage of correct predictions (classification).
- **Precision / Recall / F1**: useful when classes are imbalanced.
- **ROC-AUC**: measures the ability to distinguish between classes.
- **MAE / RMSE**: measure prediction error for regression tasks.

## Overfitting and Regularization

Overfitting occurs when a model performs well on training data but poorly on new data.
Common remedies include regularization techniques (L1, L2), dropout in neural networks, early stopping, and cross-validation.

## Tools and Frameworks

The most widely used ML frameworks are TensorFlow, PyTorch, and scikit-learn.
For data manipulation, pandas and NumPy are standard.
Experiment tracking tools like MLflow and Weights & Biases help manage model versions and metrics.