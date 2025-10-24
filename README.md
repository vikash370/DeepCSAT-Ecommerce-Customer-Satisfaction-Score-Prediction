DeepCSAT: E-Commerce Customer Satisfaction Predictor 🤖
A deep learning project that uses Natural Language Processing (NLP) to analyze customer feedback and predict a Customer Satisfaction (CSAT) score in real-time.

🚀 Live Demo & Screenshot
You can test the live application here: [https://[YOUR_STREAMLIT_APP_URL_HERE]](https://deepcsat-ecommerce-customer-satisfaction-score-prediction-yxsm.streamlit.app/)

<img width="1904" height="946" alt="Screenshot 2025-10-24 120929" src="https://github.com/user-attachments/assets/4b3f1d7e-ba2c-4a6b-a59e-83669ce4e4a8" />


PROBLEM STATEMENT : 
In the e-commerce sector, understanding customer satisfaction is critical but traditionally slow. 
Businesses rely on post-interaction surveys, which are often delayed, have low response rates, and are reactive. 
This means by the time a bad experience is reported, the customer is already dissatisfied.

This project solves this by creating a proactive early warning system. 
By analyzing a customer's raw text feedback (like a chat message or a support remark) as it happens, this model instantly predicts their satisfaction score, allowing the business to identify and help unhappy customers before they churn.

🛠️ Technology Stack:- 
  1. Backend & Modeling: Python 3.11, TensorFlow (Keras)

  2. Web Framework: Streamlit

  3. Data Processing: Pandas, NLTK (for text preprocessing), Scikit-learn

  4. Deployment: Streamlit Community Cloud (streamlit.io)

🧠 Project Methodology
   1. Data Preparation: Loaded the eCommerce_Customer_support_data.csv dataset.

   2. Text Preprocessing (NLP): Cleaned the Customer Remarks text by:
      Converting to lowercase.
      Removing punctuation and numbers.
      Removing common stopwords (e.g., 'the', 'is', 'a').
      Lemmatizing words to their root form (e.g., 'running' -> 'run').

   3. Feature Engineering: Used the Keras Tokenizer to convert the cleaned text into numerical sequences, which were then padded to a uniform length.

   4. Model Development: Built and trained a Bidirectional LSTM (Long Short-Term Memory) network. This type of network is excellent at understanding the context of a sentence by reading it both forwards and backward.

   5. Deployment: The final trained model (.h5) and tokenizer (.json) were saved and are used to power the Streamlit web app.

Local Installation & Setup
To run this project on your local machine, follow these steps:

1. Clone the Repository:

2. Create a Virtual Environment: (Recommended to avoid conflicts. Make sure you have Python 3.11 installed.)

3. Activate the Environment:

On Windows:

On macOS/Linux:

4. Install Required Libraries:

5. Run the Streamlit App:

Your browser will automatically open to the app at http://localhost:8501.

🚀 How to Deploy on Streamlit Cloud
Create a public GitHub repository.

Upload these 4 files:

app.py (The Streamlit app code)

requirements.txt (The list of libraries)

deep_csat_text_model.h5 (Your saved model)

tokenizer.json (Your saved tokenizer)

Go to and sign in with GitHub.

Click "New app" and select your new repository.

Ensure the "Main file path" is set to app.py.

Click "Deploy!"

🏁 Conclusion
This project successfully demonstrates the ability of deep learning to extract valuable, real-time insights from unstructured text. The DeepCSAT tool shifts customer service from a reactive to a proactive model, providing a direct way to monitor customer sentiment and improve loyalty.
