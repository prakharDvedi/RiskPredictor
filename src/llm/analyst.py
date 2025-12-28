import os
from groq import Groq

class IntelligenceAnalyst:
    def __init__(self):
        # 1. Setup API Client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("⚠️ GROQ_API_KEY not found. Briefings will be disabled.")
            self.client = None
        else:
            self.client = Groq(api_key=api_key)

    def generate_briefing(self, context_text, risk_score):
        """
        Generates a military-style briefing based on the raw text and risk level.
        """
        if not self.client:
            return "⚠️ API Key Missing. Cannot generate briefing."

        # 2. Define the Persona & Prompt
        system_prompt = (
            "You are a military intelligence officer. "
            "Your goal is to write a concise, high-priority threat briefing."
        )
        
        user_prompt = f"""
        **INTEL DATA:**
        - Threat Level: {risk_score:.1%}
        - Intercepted Chatter/News: "{context_text}"
        
        **MISSION:**
        Write a warning briefing (max 150 words).
        1. Start with "SITUATION".
        2. Explain WHY the model triggered (connect the text to the risk).
        3. Recommend a defensive action.
        """

        # 3. Call the LLM
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.1-8b-instant", # Fast & Free on Groq
                temperature=0.5,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"⚠️ Intel Failure: {str(e)}"