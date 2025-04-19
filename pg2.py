import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Extended_Mental_Health_Training_Data.csv")

# Label encoding
features = ["Thoughts", "Emotions", "Symptoms", "Behaviours"]
target = "Suggestions"
encoders = {}

for col in features + [target]:
    enc = LabelEncoder()
    df[col] = enc.fit_transform(df[col])
    encoders[col] = enc

# Train/Test split and model training
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

# Suggestion explanation map
explanations = {
    "Consider talking to a mental health professional": "Professional help can provide tailored therapeutic strategies for your specific concerns.",
    "Engage in physical activities": "Regular exercise boosts mood-regulating chemicals and reduces stress.",
    "Practice mindfulness and relaxation exercises": "Mindfulness helps manage overwhelming emotions and negative thought cycles.",
    "Establish a support system with friends and family": "Talking with loved ones builds emotional strength and reduces isolation.",
    "Seek professional help for anxiety or depression": "Therapists can help manage anxiety and depression with proven methods.",
    "Consider journaling or expressive writing": "Writing down thoughts can clarify emotions and aid in emotional release."
}

# Streamlit UI
st.set_page_config(page_title="Mental Health Support Assistant", layout="centered")
st.title("🧠 Mindfulness")
st.caption("Powered by Machine Learning and Psychological Insights")

# Session state for chat memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input widgets
with st.form("input_form"):
    col1, col2 = st.columns(2)
    thought = col1.selectbox("💭 Thoughts", encoders["Thoughts"].classes_)
    emotion = col2.selectbox("❤️ Emotions", encoders["Emotions"].classes_)
    symptom = col1.selectbox("🩺 Physical Symptoms", encoders["Symptoms"].classes_)
    behaviour = col2.selectbox("🧍 Behaviours", encoders["Behaviours"].classes_)
    submitted = st.form_submit_button("Get Suggestion")

if submitted:
    encoded_input = [
        encoders["Thoughts"].transform([thought])[0],
        encoders["Emotions"].transform([emotion])[0],
        encoders["Symptoms"].transform([symptom])[0],
        encoders["Behaviours"].transform([behaviour])[0]
    ]
    pred = model.predict([encoded_input])
    suggestion = encoders["Suggestions"].inverse_transform(pred)[0]
    explanation = explanations.get(suggestion, "No explanation available.")

    # Append to chat
    user_msg = f"**User:** {thought}, {emotion}, {symptom}, {behaviour}"
    bot_msg = f"**Assistant:** Suggestion - {suggestion}\n\n_Explanation_: {explanation}"
    st.session_state.chat_history.append((user_msg, bot_msg))

# Display chat history
st.subheader("🗨️ Chat History")
for user_msg, bot_msg in st.session_state.chat_history[::-1]:
    st.markdown(user_msg)
    st.markdown(bot_msg)
    st.markdown("---")

# Download history
if st.session_state.chat_history:
    if st.download_button("📤 Export Chat History", data=pd.DataFrame(st.session_state.chat_history).to_csv(index=False),
                          file_name="chat_history.csv", mime="text/csv"):
        st.success("Chat history exported!")

# Show model accuracy
st.info(f"🔍 Model Accuracy: {accuracy*100:.2f}% (based on validation dataset)")
