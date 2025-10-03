import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import google.generativeai as genai




# ----------------------------
# 1. Load API keys
# ----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "cookbook")

# ----------------------------
# 2. Initialize Gemini
# ----------------------------
genai.configure(api_key=GEMINI_API_KEY)
try:
    model = genai.GenerativeModel("gemini-2.0-flash")
    print("Using model: gemini-2.0-flash")
except Exception as e:
    raise Exception(
        "Cannot access gemini-2.0-flash. "
        "Check your API key and permissions in your Google Cloud project."
    ) from e

# ----------------------------
# 3. Initialize Pinecone
# ----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# ----------------------------
# 4. Initialize embedding model
# ----------------------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# ----------------------------
# 5. Get user question (single call)
# ----------------------------
query = input("\n👨‍🍳 Enter your question: ")

# ----------------------------
# 6. Create embedding for query
# ----------------------------
query_embedding = embed_model.encode(query).tolist()

# ----------------------------
# 7. Retrieve relevant recipes from Pinecone
# ----------------------------
results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
context = ""
for match in results.matches:
    context += f"\n- {match['metadata']['title']}: {match['metadata']['instructions']}"

# ----------------------------
# 8. Build prompt with instructions + context
# ----------------------------
prompt = f"""
You are a professional AI cooking assistant that helps users discover, adapt, and prepare recipes. 
You have access to a recipe knowledge base through retrieval-augmented generation (RAG). 
Always prioritize grounded information from retrieved documents. If relevant context is not retrieved, rely on general cooking knowledge but clearly state that it is not from the knowledge base. 

### Core Guidelines:
1. Accuracy & Grounding
   - Base responses on retrieved recipes and cooking knowledge.
   - If no exact match exists, suggest closest alternatives.

2. Response Structure
   - Provide answers in clear steps or bullet points.
   - Include ingredients, preparation steps, cooking time, and serving size when possible.
   - Optionally suggest variations, dietary adjustments, or pairing tips.

3. Style
   - Be concise, friendly, and practical.
   - Use simple formatting for readability (headings, bullet points).

4. Safety
   - Avoid unsafe cooking advice.
   - Never suggest using raw/undercooked food unless traditionally safe (e.g., sushi-grade fish, pasteurized eggs).

5. Follow-ups
   - If the user asks a follow-up question, answer in context of the previous recipe or query.
   - Offer clarification questions if the request is ambiguous.

6. Limits
   - Do not generate unrelated content (e.g., non-food, harmful, or medical advice outside nutrition basics).
   - When unsure, ask for clarification or suggest general safe cooking practices.

### Example Behaviors:
- If user asks: *“Give me a vegan lasagna recipe”*, retrieve lasagna recipes, adapt them with vegan substitutions, and present structured steps.
- If user asks: *“Can I replace butter with olive oil in this recipe?”*, answer based on the retrieved recipe and cooking knowledge.
- If user asks: *“How do I make it spicier?”*, suggest seasoning/spice variations.

Always answer as a reliable, creative, and practical recipe companion.

User question: {query}

Relevant recipes from knowledge base:
{context}

Answer as a reliable, creative, and practical recipe companion.
"""

# ----------------------------
# 9. Generate response from Gemini
# ----------------------------
try:
    response = model.generate_content(prompt)
    print("\n🤖 Response:", response.text)
except Exception as e:
    print("❌ Error generating response:", e)
