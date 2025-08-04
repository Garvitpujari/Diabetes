import pandas as pd
import gradio as gr
import xgboost as xgb

# Load dataset from GitHub
url = "https://raw.githubusercontent.com/Garvitpujari/Diabetes/main/diabetes_012_health_indicators_BRFSS2015.csv"
df = pd.read_csv(url)

# Selected features
features = [
    "Age", "Sex", "HighBP", "HighChol", "BMI", "Smoker", 
    "PhysActivity", "Veggies", "HvyAlcoholConsump", "GenHlth"
]
target = "Diabetes_012"

# Manual train/test split (no sklearn)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
split = int(0.8 * len(df))
X_train, y_train = df.loc[:split, features], df.loc[:split, target]

# Train XGBoost model
model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, eval_metric="mlogloss")
model.fit(X_train, y_train)

# Output messages
messages = {
    0: "🟢 No Diabetes – You seem healthy. Keep up the good lifestyle!",
    1: "🟡 Little Diabetic – Warning signs. Cut down sugar and stay active!",
    2: "🔴 Diabetic – High risk. Please consult a doctor immediately!"
}

# Prediction function
def predict(Age, Sex, HighBP, HighChol, BMI, Smoker, PhysActivity, Veggies, HvyAlcoholConsump, GenHlth):
    input_df = pd.DataFrame([[
        Age, Sex, HighBP, HighChol, BMI, Smoker,
        PhysActivity, Veggies, HvyAlcoholConsump, GenHlth
    ]], columns=features)
    prediction = int(model.predict(input_df)[0])
    return messages[prediction]

# Gradio UI inputs
inputs = [
    gr.Number(label="Age (in years)"),
    gr.Radio([0, 1], label="Sex (0 = Female, 1 = Male)"),
    gr.Radio([0, 1], label="High Blood Pressure? (0 = No, 1 = Yes)"),
    gr.Radio([0, 1], label="High Cholesterol? (0 = No, 1 = Yes)"),
    gr.Number(label="BMI (Body Mass Index)"),
    gr.Radio([0, 1], label="Smoker? (0 = No, 1 = Yes)"),
    gr.Radio([0, 1], label="Physically Active? (0 = No, 1 = Yes)"),
    gr.Radio([0, 1], label="Eats Vegetables Regularly? (0 = No, 1 = Yes)"),
    gr.Radio([0, 1], label="Heavy Alcohol Consumption? (0 = No, 1 = Yes)"),
    gr.Radio([1, 2, 3, 4, 5], label="General Health (1 = Excellent, 5 = Poor)")
]

# Launch Gradio app
gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs="text",
    title="🧠 Diabetes Risk Predictor",
    description=(
        "📊 Estimate your diabetes risk using lifestyle and health indicators. "
        "**No blood tests required.**\n\n"
        "**Legend:**\n"
        "- Binary: 0 = No, 1 = Yes\n"
        "- General Health: 1 = Excellent, 5 = Poor"
    )
).launch()
