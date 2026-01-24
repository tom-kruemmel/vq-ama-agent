GENERATE_QUERIES_PROMPT = """
You are an AI assistant. Generate five different rewrites of the user's question to help retrieve relevant documents from a vector database. Separate each rewritten question with a newline. Only use newlines after the first four questions. Just create the questions without any additional text.

User's question: {question}
"""

GENERATE_ANSWER_PROMPT = """
Context:
{context}

Question: {question}

Answer:"""

JUDGE_QUESTION_DOMAIN_PROMPT = """
You are a strict domain gatekeeper. Decide whether the USER QUESTION is IN-DOMAIN for the DOMAIN.

Return ONLY a compact JSON object without any markdown or such wrapping it, starting with curly braces, with keys:
- "in_domain": true or false
- "score": a number from 0.0 to 1.0 reflecting confidence that it is in-domain
- "rationale": a short one-sentence explanation

DOMAIN:
{domain_desc}

USER QUESTION:
{question}

JSON:
"""
