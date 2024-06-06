## Streamlit Link :- [Link](https://predictionsurvivaltitanic123.streamlit.app/)

### Project Description
**Titanic Survival Prediction**
- **Objective**: The goal of this project is to predict the survival of passengers on the Titanic based on their attributes such as age, fare, class, sex, and other relevant features.
- **Tools**: Python, Pandas, NumPy, Scikit-Learn, Jupyter Notebook, Streamlit
- ---
- ### Flowchart
Below is a simplified flowchart for the Titanic Survival Prediction project:

```plaintext
Start
  |
  V
Data Collection
  |
  V
Data Preprocessing
  |  - Handle Missing Values
  |  - Encode Categorical Features
  |  - Create New Features (Age Groups)
  V
Feature Engineering
  |  - Select Relevant Features
  |  - Prepare Features for Model Training
  V
Model Training
  |  - Train Model with Selected Features
  |  - Use Algorithms like Random Forest, Logistic Regression
  V
Model Evaluation
  |  - Evaluate Model Performance (Accuracy, Precision, Recall)
  |  - Tune Model Parameters
  V
Deployment
  |  - Build Web App using Streamlit
  |  - Integrate Model for Real-time Prediction
  V
End
```
---
### Detailed Steps
1. **Data Collection**:
   - Obtain the Titanic dataset from Kaggle or another trusted source.

2. **Data Preprocessing**:
   - **Handle Missing Values**: Fill missing values in columns like `Age`, `Embarked`, etc.
   - **Encode Categorical Features**: Convert features like `Sex`, `Embarked`, and `Title` into numerical values.
   - **Create New Features**: Generate age group features (Child, Teen, Adult, Middle_Age, Old) based on the `Age` column.

3. **Feature Engineering**:
   - **Select Relevant Features**: Choose features like `Pclass`, `Fare`, `Age Group`, `Sex`, `Embarked`, `Family_member`, and `Title`.
   - **Prepare Features**: Ensure features are in the correct format for model training.

4. **Model Training**:
   - **Train Model**: Use machine learning algorithms such as Random Forest, Logistic Regression, etc., to train the model.
   - **Use Algorithms**: Choose the best algorithm based on performance metrics.

5. **Model Evaluation**:
   - **Evaluate Performance**: Use metrics like accuracy, precision, recall, and F1-score to evaluate the model.
   - **Tune Parameters**: Optimize model parameters to improve performance.

6. **Deployment**:
   - **Build Web App**: Use Streamlit to create an interactive web application.
   - **Integrate Model**: Load the trained model into the web app for real-time predictions.
