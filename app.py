import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import google.generativeai as genai
from judge import evaluate_response

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "cookbook")

genai.configure(api_key=GEMINI_API_KEY)
try:
    model = genai.GenerativeModel("gemini-2.0-flash")
except Exception as e:
    raise Exception("Cannot access gemini-2.0-flash. Check your API key and permissions.") from e

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)


embed_model = SentenceTransformer('all-MiniLM-L6-v2')

questions = [
    "How can I make pasta with chicken and mushrooms?",
    "How can I reheat pasta without it becoming dry?",
    "What is a good substitute for cream in a sauce?",
    "How can I make pizza dough crispy?",
    "How do I know if chicken is properly cooked?",
    "How can I make a healthy dinner with salmon?",
    "How can I thicken soup without using flour?",
    "What are the best spices for chicken curry?",
    "How can I make a vegan chocolate cake?",
    "How should I store cooked rice to keep it from spoiling?",
    "How can I prevent onions from burning while frying?",
    "What’s the best way to cook steak medium-rare?",
    "How can I make scrambled eggs fluffy?",
    "How do I properly season a cast iron pan?",
    "What’s the difference between baking powder and baking soda?",
    "How can I make bread rise faster?",
    "How can I stop pasta from sticking together after cooking?",
    "How do I clarify butter at home?",
    "How can I tell if an egg is still fresh?",
    "How can I freeze soup without ruining its texture?",
    "What are some low-carb dinner ideas?",
    "How can I reduce the amount of salt in a recipe without losing flavor?",
    "What are healthy snacks that are high in protein?",
    "Is olive oil or butter better for frying?",
    "What is the best way to cook vegetables to keep nutrients?",
    "Why does my cake sink in the middle?",
    "How can I make cookies soft and chewy instead of crunchy?",
    "How can I make whipped cream without a mixer?",
    "How do I melt chocolate without burning it?",
    "How can I make gluten-free pancakes?"
]

def run_batch_evaluation():
    print("\n🍳 Running batch evaluation...\n")
    results = []

    for i, question in enumerate(questions, 1):
        print(f"🔹 [{i}] Question: {question}")

      
        query_embedding = embed_model.encode(question).tolist()

       
        context = ""
        try:
            matches = index.query(vector=query_embedding, top_k=3, include_metadata=True).matches
            for match in matches:
                context += f"\n- {match['metadata']['title']}: {match['metadata']['instructions']}"
        except Exception:
            context = "No relevant recipes found."

     
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
User question: {question}

Related recipe information:
{context}
"""

       
        try:
            response = model.generate_content(prompt)
            ai_answer = response.text.strip()
            print(f"🧠 Answer: {ai_answer[:300]}{'...' if len(ai_answer) > 300 else ''}")

            
            evaluation = evaluate_response(user_query=question, ai_answer=ai_answer)
            print(f"🏅 Score: {evaluation['score']} / 10 → {evaluation['reason']}\n")

          
            results.append({
                "question": question,
                "answer": ai_answer,
                "score": evaluation["score"],
                "reason": evaluation["reason"]
            })

        except Exception as e:
            print(f"❌ Error on question {i}: {e}\n")

    
    with open("judge_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("💾 Batch evaluation finished! Results saved to judge_results.json\n")



if __name__ == "__main__":
    run_batch_evaluation()
