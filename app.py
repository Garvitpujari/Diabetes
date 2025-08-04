import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset from GitHub
url = "https://raw.githubusercontent.com/Garvitpujari/Diabetes/main/diabetes_data.csv"
df = pd.read_csv(url)

# Prepare data
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction function
def predict(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    input_data = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
    prediction = model.predict(input_data)[0]
    return "Diabetes: Yes" if prediction == 1 else "Diabetes: No"

# Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="Blood Pressure"),
        gr.Number(label="Skin Thickness"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="Diabetes Pedigree Function"),
        gr.Number(label="Age")
    ],
    outputs="text",
    title="Diabetes Predictor",
    description="Predict diabetes based on medical input values using Logistic Regression."
)

# Launch app
if __name__ == "__main__":
    interface.launch()
