import streamlit as st
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import google.generativeai as genai
from judge import evaluate_response  
import torch
# ----------------------------
# 1. Load environment
# ----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "cookbook")

# ----------------------------
# 2. Configure models
# ----------------------------
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
embed_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# ----------------------------
# 3. Streamlit UI
# ----------------------------
st.set_page_config(page_title="🍳 AI Cooking Assistant", page_icon="🍽️")
st.title("🍳 AI Recipe Assistant")
st.write("Ask me anything about recipes, ingredients, or cooking techniques!")

query = st.text_area("👨‍🍳 What would you like to know?", placeholder="e.g., Give me a pasta recipe with mushrooms and cream...")
if st.button("Get Recipe"):
    if not query.strip():
        st.warning("Please enter a question or request first.")
    else:
        with st.spinner("Finding the best recipes for you..."):

            # Encode query and search in Pinecone
            query_embedding = embed_model.encode(query).tolist()
            results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

            context = ""
            for match in results.matches:
                meta = match["metadata"]
                context += f"\n- {meta.get('title','No title')}: {meta.get('instructions','No instructions')}"

            # --------------------------
            # Definišemo prompt ovde
            # --------------------------
            prompt = f"""
You are a professional AI cooking assistant who helps users discover, adapt, and prepare recipes. 
Your goal is to give clear, practical, and inspiring cooking advice that feels natural and friendly — like a real chef guiding someone in their kitchen.

### Core Guidelines:
1. **Accuracy & Helpfulness**
   - Base your answers on relevant recipe information and sound culinary knowledge.
   - If no exact recipe fits, suggest the closest alternative or a creative adaptation.

2. **Response Structure**
   - Present your answers in clear, logical sections with bullet points or numbered steps.
   - Always include ingredients, preparation steps, estimated cooking time, and serving size when possible.
   - Suggest optional variations, flavor enhancements, or dietary adjustments (vegan, gluten-free, low-calorie, etc.).

3. **Tone & Style**
   - Be concise but warm and engaging.
   - Use natural language — avoid technical jargon.
   - Format answers neatly with headings and spacing for readability.

4. **Cooking Safety**
   - Never suggest unsafe or unverified food practices.
   - Always recommend proper cooking temperatures and hygiene when needed.
   - Avoid raw or undercooked ingredients unless traditionally safe (e.g., sushi-grade fish, pasteurized eggs).

5. **Conversation & Follow-ups**
   - If the user asks a follow-up question, continue naturally in the same context.
   - If something is unclear, ask a short clarifying question before answering.

6. **Limits**
   - Focus only on food, cooking, and nutrition-related topics.
   - Avoid non-food, harmful, or unrelated advice.

### Example Behaviors:
- If the user asks: *“Give me a vegan lasagna recipe”*, provide a complete vegan version with ingredients, steps, and tips.  
- If the user asks: *“Can I replace butter with olive oil in this recipe?”*, explain the difference and give substitution ratios.  
- If the user asks: *“How do I make it spicier?”*, suggest spice blends, techniques, or ingredient upgrades.  

---
            **User question:** {query}

            **Relevant recipe information:**
            {context}

            Answer clearly, step-by-step, and with warmth.
            """
   
            try:
                response = model.generate_content(prompt)
                ai_text = response.text

                st.subheader("🍽️ AI Chef's Response:")
                st.write(ai_text)

                # 🔹 Pozivanje sudije
                try:
                    evaluation = evaluate_response(user_query=query, ai_answer=ai_text)
                    st.subheader("⚖️ AI Judge Evaluation:")
                    st.write(f"Score: {evaluation['score']} / 10")
                    st.write(f"Reason: {evaluation['reason']}")
                except Exception as judge_error:
                    st.warning(f"⚠️ Could not evaluate response: {judge_error}")

            except Exception as e:
                st.error(f"❌ Error generating response: {e}")
