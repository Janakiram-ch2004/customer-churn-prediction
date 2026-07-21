
Project Overview

This project predicts whether a telecom customer is likely to leave the company based on customer information and subscribed services. The model takes customer details as input and predicts either *Churn* or *No Churn*. A Streamlit web application is used to make predictions through a simple user interface.



Features

- Customer churn prediction using Deep Learning
- Data preprocessing with Scikit-learn
- Interactive Streamlit web application
- Saved preprocessing pipeline and trained model
- Real-time prediction



Dataset

The project uses the Telco Customer Churn dataset.

Target Variable:
- Churn

Input Features include:
- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges



 ata Preprocessing

The dataset was cleaned before training the model.

Steps performed:

- Handled missing values
- Separated numerical and categorical features
- Applied StandardScaler for numerical columns
- Applied OneHotEncoder for categorical columns
- Combined preprocessing using ColumnTransformer
- Saved the preprocessing pipeline as preprocessor.pkl



Deep Learning Model

The model was built using TensorFlow and Keras.

Architecture:

- Dense(128, ReLU)
- Batch Normalization
- Dropout(0.2)
- Dense(64, ReLU)
- Batch Normalization
- Dropout(0.2)
- Dense(32, ReLU)
- Dense(1, Sigmoid)

Training:

- Optimizer: Adam
- Learning Rate: 0.0005
- Loss: Binary Crossentropy
- EarlyStopping used to prevent overfitting



Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Keras
- Streamlit
- Joblib
- Matplotlib
- Seaborn



Project Structure


customer_prediction/
 app.py
 train.py
 churn_dl_model.keras
 preprocessor.pkl
 requirements.txt
 WA_Fn-UseC_-Telco-Customer-Churn.csv
 .gitignore




 How to Run

Clone the repository.

bash
git clone https://github.com/your-username/customer-churn-prediction.git


Move to the project folder.

bash
cd customer-churn-prediction


Install the required packages.

bash
pip install -r requirements.txt


Run the application.

bash
streamlit run app.py




 Output

The application predicts:

- Churn
- No Churn

It also displays the prediction probability.



 Author

Janakiram
