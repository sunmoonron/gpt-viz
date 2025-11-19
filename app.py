import streamlit as st
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="AI Stress Test: Context Dilution", layout="wide")

# Custom CSS to make it look like a lab notebook
st.markdown("""
<style>
    .main-title {font-size: 3rem; font-weight: 800; color: #9E1E1E; margin-bottom: 0px;}
    .subtitle {font-size: 1.2rem; color: #555; margin-bottom: 20px;}
    .highlight-box {
        background-color: #f0f2f6; 
        border-left: 5px solid #FF4B4B; 
        padding: 15px; 
        border-radius: 5px;
        margin: 20px 0;
    }
    .success-box {
        background-color: #4306b3; 
        border-left: 5px solid #00cc66; 
        padding: 15px; 
        border-radius: 5px;
    }
    .failure-box {
        background-color: #8306b3; 
        border-left: 5px solid #ff3333; 
        padding: 15px; 
        border-radius: 5px;
    }
    .vocab-term {
        font-weight: bold;
        color: #2c3e50;
        text-decoration: underline;
        cursor: help;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. LOAD MODEL (CACHED) ---
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2', output_attentions=True)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# --- 3. SIDEBAR: EDUCATIONAL CHEAT SHEET ---
with st.sidebar:
    st.header("📝 Glossary")
    st.info("""
    **Token:**
    A chunk of text. A token can be a word, part of a word, or even a space. (i.e `[[` might be one token, `100` another).

    **Feature:**
    Specific concept AI learned to recognize. (e.g., "opening brackets", "numbers", "commas", "Are we inside a list?"). In the code it looks like a neuron activating(e.g., 'Neuron 512 in Layer 5').

    **Representation:**
    A list of numbers (vectors) that AI uses to describe a Feature. (e.g., `[0.2, -1.3, 0.7, ...]` might represent "inside a double list"). It is the 'brain activity' associated with that concept.
    
    **Attention:**
    How the AI connects words. When predicting the last word, it "attends" (looks back) at previous words to gather context. (e.g., when seeing `]]`, it might look back at `[[` to remember to close two brackets).
    
    **Logits / Probability:**
    The score the AI assigns to every possible next word in the dictionary. Higher score = more likely prediction. (e.g., logits for `]` vs `]]` tell us which closing bracket it thinks is more appropriate; higher logit = more likely to be chosen).
    
    **Layer:** 
    A stage in the AI's processing pipeline. Each layer refines the representation further. (e.g., Layer 0 sees raw tokens, Layer 11 produces final predictions).
            
    **Head:**
    A sub-component within a layer that focuses on different patterns. (e.g., Head 0 might focus on syntax, Head 8 on matching brackets).
    """)
    
    st.header("⚙️ Debugger Controls")
    st.write("Use these if you want to explore different parts of the brain.")
    layer_idx = st.slider("Layer (Depth)", 0, 11, 5, help="Deeper layers understand more complex abstract concepts.")
    head_idx = st.slider("Head (Perspective)", 0, 11, 8, help="Different heads look for different patterns.")

# --- 4. HERO SECTION ---
st.markdown('<div class="main-title">Breaking GPT-2: The "Context Dilution" Attack</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">An interactive demonstration of how "distractor" tokens break an AI\'s ability to count.</div>', unsafe_allow_html=True)

st.markdown("### 1. The Mission")
st.write("""
We are going to give the AI a simple Python coding task: **Close the brackets.**
If we start a list with `[[`, the AI must eventually close it with `]]`.
To do this, it must "remember" how deep the nesting is.
""")

# --- 5. THE ATTACK CONTROLLER ---
st.markdown("### 2. The Experiment")
st.write("Use the slider below to inject **'Noise'** (random numbers) into the list. Watch how the AI handles it.")

# The Slider
distraction_level = st.slider("👇 DRAG THIS: Number of Distractor Items", 1, 150, 2, key="distraction_slider")

# Construct the prompt dynamically
# Base: x = [[1, 
# Distractors: 100, 101, 102...
# We leave it open-ended for the model to complete
garbage_list = [str(i) for i in range(100, 100 + distraction_level)]
garbage_str = ", ".join(garbage_list)

if distraction_level == 0:
    prompt = "x = [[1,"
else:
    prompt = f"x = [[1, {garbage_str}"

# Display the Prompt
st.code(prompt, language="python")

# --- 6. RUNNING THE MODEL ---
inputs = tokenizer(prompt, return_tensors="pt")
tokens = [t.replace('Ġ', ' ') for t in tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])]

with torch.no_grad():
    outputs = model(**inputs)
    
    # 1. Get Attention Data
    # Layer 5, Head 8 is often a "Induction Head" in GPT-2 Small 
    # It likes to look back at previous similar tokens.
    attention_tensor = outputs.attentions[layer_idx][0, head_idx] # [seq_len, seq_len]
    tensor_data = attention_tensor.detach().cpu()
    list_data = tensor_data.tolist()
    attention_matrix = np.array(list_data) # Convert to NumPy for easier slicing
    # attention_matrix = attention_tensor.detach().cpu().numpy()
    
    # 2. Get Predictions
    logits = outputs.logits[0, -1, :] # Prediction for the NEXT token
    probs = torch.softmax(logits, dim=0)

# --- 7. VISUALIZING ATTENTION (THE DIAGNOSTIC) ---
st.markdown("### 3. Inside the Brain: Attention Map")

col_viz1, col_viz2 = st.columns([2, 1])

with col_viz1:
    # Extract attention from the LAST token (the one currently thinking)
    # looking back at all previous tokens
    last_token_attn = attention_matrix[-1, :]
    
    # Plotting with Matplotlib
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # We plot the attention intensity across the sentence
    ax.plot(last_token_attn, color='#1E88E5', linewidth=2.5)
    ax.fill_between(range(len(last_token_attn)), last_token_attn, color='#1E88E5', alpha=0.3)
    
    # Fix the Y-axis so we can actually SEE the dilution happening
    # If we don't fix this, the chart rescales and hides the effect!
    ax.set_ylim(0, 1.0) 
    
    ax.set_title(f"Attention Intensity (Layer {layer_idx}, Head {head_idx})", fontsize=14)
    ax.set_xlabel("Token Position in Sequence")
    ax.set_ylabel("Focus Strength (0-1)")
    
    # Highlight the critical "[[ " region (usually at the start)
    ax.axvspan(2, 4, color='red', alpha=0.1, label="Critical Signal ([[)")
    ax.legend(loc='upper right')
    
    st.pyplot(fig)

with col_viz2:
    st.markdown("""
    #### 🧐 What am I looking at?
    This line shows how much the AI is focusing on previous words to guess the next one.
    
    - **The Peak:** Ideally, there should be a spike near the start (the `[[`). That is the signal telling the AI "You are inside a double list."
    - **The Drop:** As you drag the slider right, watch the peak **shrink**. This is **Context Dilution**. The signal is getting drowned out by the noise.
    """)

# --- 8. THE RESULT (DID IT BREAK?) ---
st.markdown("### 4. The Prediction: Did it Fail?")

# Find probabilities for "]" and "]]"
# We need the specific token IDs for these
token_id_single = tokenizer.encode("]") [0]
token_id_double = tokenizer.encode("]]")[0]

prob_single = probs[token_id_single].item()
prob_double = probs[token_id_double].item()

# Normalize them relative to each other for the bar chart (just for visual comparison)
total_mass = prob_single + prob_double
pct_single = prob_single / total_mass
pct_double = prob_double / total_mass

col_res1, col_res2 = st.columns(2)

with col_res1:
    # Custom HTML Bar Chart
    st.markdown(f"""
    <div style="margin-bottom: 10px;">
        <strong>Probability of "]]" (Correct)</strong><br>
        <div style="background-color: #ddd; width: 100%; border-radius: 5px;">
            <div style="background-color: #00cc66; width: {pct_double*100}%; height: 24px; border-radius: 5px; text-align: right; padding-right: 5px; color: white; font-weight: bold;">
                {prob_double:.4f}
            </div>
        </div>
    </div>
    <div style="margin-bottom: 10px;">
        <strong>Probability of "]" (Wrong)</strong><br>
        <div style="background-color: #ddd; width: 100%; border-radius: 5px;">
            <div style="background-color: #ff3333; width: {pct_single*100}%; height: 24px; border-radius: 5px; text-align: right; padding-right: 5px; color: white; font-weight: bold;">
                {prob_single:.4f}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_res2:
    # Logic to determine status
    if prob_double > prob_single:
        st.markdown("""
        <div class="success-box">
            <h3>✅ SYSTEM STABLE</h3>
            The AI correctly remembers it needs to close two brackets.
            The signal from the start of the sentence is strong enough.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="failure-box">
            <h3>❌ SYSTEM COMPROMISED</h3>
            <strong>The attack worked.</strong>
            The AI has "forgotten" the double bracket `[[`. 
            The attention mechanism averaged out the signal over too many numbers, and now it thinks a single `]` is sufficient.
        </div>
        """, unsafe_allow_html=True)

# --- 9. CONCLUSION ---
st.divider()
st.markdown("### Why this matters for AI Safety")
st.write("""
This demonstrates **Context Dilution**, a real phenomenon in Transformer models. 
Even though the model *technically* can see the `[[` at the start, the mathematical operation it uses (Softmax Attention) forces it to "spend" its attention budget. 
When the context gets too long, the "budget" spent on the critical instruction (`[[`) becomes too small to trigger the correct circuit.
""")