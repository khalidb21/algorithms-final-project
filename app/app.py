import streamlit as st
import pandas as pd
from pathlib import Path
from google import genai
from dotenv import load_dotenv

from app_helper import *

# Model path - get from config file
model_path = Path("../app/lr_model.pk1")

# Load environment variables
load_dotenv()
# The client gets the API key from the environment variable `GEMINI_API_KEY`
client = genai.Client() 

# Page Configuration
st.set_page_config(
    page_title="Battery SOH Chatbot",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Battery SOH Predictor & Chatbot")

# CSS styling for SOH Classifications
st.markdown("""
    <style>
    .healthy {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        font-weight: bold;
    }
    .unhealthy {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        font-weight: bold;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = None
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.6

# Function for chatbot battery queries
def get_battery_insights(soh, status, threshold):
    """Get Gemini insights about battery health"""
    prompt = f"""
    A battery has the following State of Health (SOH) metrics:
    - Predicted SOH: {soh:.4f}
    - Status: {status}
    - Threshold: {threshold}
    
    Provide a brief (2-3 sentences) professional assessment of this battery's condition 
    and one recommendation for the user.
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Could not generate insights: {e}"


# Sidebar used for threshold
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.session_state.threshold = st.slider(
        "SOH Threshold for Classification",
        min_value=0.5,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="Battery is 'Healthy' if SOH ≥ threshold"
    )
    
    st.info(f"""
    📊 **Current Threshold: {st.session_state.threshold:.2f}**
    - ✅ Healthy: SOH ≥ {st.session_state.threshold:.2f}
    - ❌ Unhealthy: SOH < {st.session_state.threshold:.2f}
    """)
    
    # Model Loading
    st.session_state.model = load_model(model_path)
    if st.session_state.model:
        st.success("Model loaded successfully!")
    else:
        st.error("Failed to load model")


# Split functionality into two tabs
tab1, tab2 = st.tabs(["🔮 Predict", "💬 Chat"])

# Tab 1: Prediction
with tab1:
    st.header("Battery SOH Prediction")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please load a model first using the sidebar settings")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Enter Cell Voltage Values")
            st.write("**Format Option 1:** `U1:0.85 U2:0.87 U3:0.84 ...` (space-separated with cell names)")
            st.write("**Format Option 2:** `0.85 0.87 0.84 ...` (21 comma or space-separated values)")
            
            cell_input = st.text_area(
                "Cell Values",
                value="3.4867,3.5053,3.5311,3.5128,3.4889,3.4688,3.4402,3.4607,3.4858,3.5255,3.5704,3.5301,3.4905,3.4564,3.3903,3.4281,3.4849,3.5503,3.6402,3.568,3.4939",
                height=100,
                label_visibility="collapsed"
            )
        
        with col2:
            st.metric("Threshold", f"{st.session_state.threshold:.2f}")
            st.metric("Status", "Ready")
        
        # Prediction button
        if st.button("🚀 Predict Battery Health", width='stretch', type="primary"):
            cell_values = parse_cell_input(cell_input)
            
            if cell_values:
                # Make prediction
                soh = predict_soh(st.session_state.model, cell_values)
                
                if soh is not None:
                    # Classify battery
                    status, color, emoji = classify_battery(soh, st.session_state.threshold)
                    
                    # Display results
                    st.divider()
                    st.subheader("Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Predicted SOH", f"{soh:.4f}")
                    
                    with col2:
                        st.metric("Status", f"{emoji} {status}")
                    
                    with col3:
                        health_percent = soh * 100
                        st.metric("Health %", f"{health_percent:.1f}%")
                    
                    # Status box
                    if status == "Healthy":
                        st.markdown(
                            f"""<div class='healthy'>
                            ✅ HEALTHY BATTERY
                            <br>This battery is in good condition (SOH ≥ {st.session_state.threshold:.2f})
                            <br>It is suitable for continued use or remanufacturing.
                            </div>""",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""<div class='unhealthy'>
                            ❌ UNHEALTHY BATTERY
                            <br>This battery needs attention (SOH < {st.session_state.threshold:.2f})
                            <br>Consider recycling or proper disposal.
                            </div>""",
                            unsafe_allow_html=True
                        )
                    
                    # Get AI insights
                    st.subheader("AI Insights")
                    with st.spinner("Generating insights..."):
                        insights = get_battery_insights(soh, status, st.session_state.threshold)
                        st.info(insights)
                    
                    # Display cell values table
                    with st.expander("View Cell Values"):
                        cell_df = pd.DataFrame({
                            'Cell': list(cell_values.keys()),
                            'Voltage': list(cell_values.values())
                        })
                        st.dataframe(cell_df, width='stretch', hide_index=True)

# Tab 2: Chatbot
with tab2:
    st.header("Battery Assistant Chatbot")
    
    st.write("""
    Ask questions about battery health, maintenance, recycling, or general battery-related topics.
    The chatbot is powered by Google Gemini AI.
    """)
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Chat input
    user_input = st.chat_input("Ask a question about batteries...")
    
    if user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        # Generate Gemini response
        try:
            # Create a system prompt for battery-related context
            system_prompt = """You are an expert battery technician and sustainability consultant. 
            Answer questions about battery health, maintenance, State of Health (SOH), recycling, 
            and environmental impact. Provide clear, concise, and practical advice.
            Keep responses under 300 words."""
            
            full_prompt = f"{system_prompt}\n\nUser question: {user_input}"
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt
            )
            bot_reply = response.text
        except Exception as e:
            bot_reply = f"❌ Error communicating with Gemini: {e}"
        
        # Add bot message to history
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        
        # Display bot response
        with st.chat_message("assistant"):
            st.write(bot_reply)
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", width='stretch'):
        st.session_state.messages = []
        st.rerun()

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.9rem;'>
    <p>
    Battery SOH Prediction with Chatbot Integration
    <br>
    Built with Streamlit, Scikit-learn, and Google Gemini API
    <br>
    Predicts battery health from PulseBat voltage measurements (U1-U21)
    </p>
    </div>
""", unsafe_allow_html=True)