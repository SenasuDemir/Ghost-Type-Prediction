import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import streamlit as st

# Load your dataset
df = pd.read_csv("train.csv")

# Feature engineering
def feature_engineering(df):
    df['hair_soul'] = df['hair_length'] * df['has_soul']
    df['hair_bone'] = df['hair_length'] * df['bone_length']

# Function to predict ghost type
def ghost_classification(pipeline, input_df):
    # Use the trained pipeline to make predictions on the input data
    prediction = pipeline.predict(input_df)
    
    # Return the predicted ghost type (the class label)
    return prediction[0]

# Feature engineering on the dataset
feature_engineering(df)

# Define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ['bone_length', 'rotting_flesh', 'hair_length', 'has_soul', 'hair_soul', 'hair_bone']),
        ("cat", OneHotEncoder(), ['color'])
    ]
)

# Split the data
X = df[['bone_length', 'rotting_flesh', 'hair_length', 'has_soul', 'color', 'hair_soul', 'hair_bone']]
y = df["type"]

# Define the model and pipeline
model = RandomForestClassifier()
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

# Train the model
pipeline.fit(X, y)

# Save the trained model
joblib.dump(pipeline, 'best_model.pkl')

print("Model retrained and saved as 'best_model.pkl'")

# Streamlit app logic
def main():
    st.set_page_config(page_title="Ghost Prediction", page_icon="👻", layout="centered")

    # Title with emoji
    st.title("👻 Ghost Prediction App")
    st.markdown("### Predict the type of ghost based on its characteristics.")
    st.markdown("---")

    # Sidebar design with a neat header and spacing
    st.sidebar.header("Input Parameters")
    st.sidebar.markdown("Adjust the sliders and choose the options below to predict the ghost type.")

    # Group the input sliders together with a more structured look
    with st.sidebar:
        bone_length = st.slider("Bone Length", 0.0, 1.0, step=0.01)
        rotting_flesh = st.slider("Rotting Flesh", 0.0, 1.0, step=0.01)
        hair_length = st.slider("Hair Length", 0.0, 1.0, step=0.01)
        has_soul = st.slider("Has Soul", 0.0, 1.0, step=0.01)
        color = st.selectbox('Color', df['color'].unique())  # Make sure 'color' is in df

    # Create the input data frame for prediction
    input_data = {
        "bone_length": [bone_length],
        "rotting_flesh": [rotting_flesh],
        "hair_length": [hair_length],
        "has_soul": [has_soul],
        "color": [color]
    }

    # Feature engineering on the input data
    input_df = pd.DataFrame(input_data)
    feature_engineering(input_df)
    
    # Prediction button with a clear label
    if st.sidebar.button("Predict Ghost Type"):
        ghost_type = ghost_classification(pipeline, input_df)
        
        st.markdown("### Prediction Result")
        st.write(f"**The predicted ghost type is:** {ghost_type}")

    # Add some spacing and footer for visual appeal
    st.markdown("---")
    st.markdown("Made with 💖 by Senasu Demir")

if __name__ == "__main__":
    main()
