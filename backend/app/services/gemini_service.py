import google.generativeai as genai
import os

api_key = os.getenv("GEMINI_API_KEY", "")
if api_key:
    genai.configure(api_key=api_key)

def ask_the_principal(user_query: str, retrieved_context: str) -> dict:
    """Detects input language automatically, enforces Class 5 simplicity, and applies strict financial safety rules."""
    detector_model = genai.GenerativeModel('gemini-2.5-pro')
    lang_prompt = f"Analyze the following text and reply with ONLY one word specifying its language (Example: English, Hindi, Bengali). Text: {user_query}"
    
    try:
        lang_response = detector_model.generate_content(lang_prompt)
        detected_lang = lang_response.text.strip()
    except Exception:
        detected_lang = "English"

    system_instruction = (
        "You are 'Appna Bank AI', a warm financial companion for farmers, students, and rural users.\n"
        f"CRITICAL RULE 1: You must reply completely in {detected_lang}.\n"
        "CRITICAL RULE 2: Explain things like a teacher talking to a Class 5 student. Use simple words and fun examples. No complex jargon.\n"
        "CRITICAL RULE 3: Financial Safety Rules:\n"
        "  - Always recommend saving an Emergency Fund first before anything else!\n"
        "  - Tell them to get basic Insurance before buying stocks or investments.\n"
        "  - Suggest helpful Indian Government schemes (like PM-Kisan, Sukanya Samriddhi) if applicable.\n"
        "  - NEVER give risky stock tips to grandparents, senior citizens, or families with low income."
    )

    full_prompt = f"""
    Context provided to help you answer:
    {retrieved_context}

    User Question: {user_query}

    Remember to speak like a Class 5 helper and use the language: {detected_lang}.
    """

    model = genai.GenerativeModel(
        model_name='gemini-2.5-pro',
        system_instruction=system_instruction
    )

    response = model.generate_content(full_prompt)
    
    return {
        "detected_language": detected_lang,
        "answer": response.text
    }
