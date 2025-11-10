import os
import json
import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def evaluate_response(user_query, ai_answer, reference_answer=None):
    """
    Evaluates the quality of an AI-generated answer using Gemini 2.5 Flash as an impartial LLM judge.

    Returns:
        dict: {"score": int, "reason": str}
    """
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
You are an impartial and consistent **AI evaluator** designed to assess how well an AI-generated answer responds to a user's question.
Act like a professional human evaluator, not like a chatbot.

### Context
User question:
\"\"\"{user_query}\"\"\"

AI-generated answer:
\"\"\"{ai_answer}\"\"\"

Reference (ideal) answer:
\"\"\"{reference_answer or "N/A"}\"\"\"

---

### Evaluation Criteria
Score the AI answer **from 1 to 10** based on these weighted criteria:
1. **Accuracy (40%)** — factual correctness and realism.
2. **Relevance (25%)** — does it directly address the question?
3. **Completeness (20%)** — does it cover all necessary details?
4. **Clarity (15%)** — is the answer clear, structured, and readable?

---

### Instructions
- Be strict and objective — an average-quality response should get around 6–7.
- If a reference answer is provided, compare the AI’s answer against it.
- If the AI’s answer is off-topic, factually wrong, or incomplete, reduce points accordingly.
- Output **only valid JSON**, no additional text.

---

### JSON output format
{{
  "score": <integer between 1 and 10>,
  "reason": "<one-sentence justification explaining the score>"
}}
"""

    try:
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2, 
                response_mime_type="application/json"
            )
        )

       
        try:
            result = json.loads(response.text)
        except json.JSONDecodeError:
            result = {"score": 0, "reason": "Invalid JSON output from model."}

        
        if not isinstance(result.get("score"), int) or not (1 <= result["score"] <= 10):
            result["score"] = 0
            result["reason"] = "Invalid or missing score."

        return result

    except Exception as e:
        return {"score": 0, "reason": f"Evaluation failed: {str(e)}"}
