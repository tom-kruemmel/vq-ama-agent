# =============================================================================
# GENERATE QUERIES PROMPTS
# =============================================================================

# # Default: Multi-query rewriting
# GENERATE_QUERIES_PROMPT = """
# You are an AI assistant. Generate five different rewrites of the user's question to help retrieve relevant documents from a vector database. Separate each rewritten question with a newline. Only use newlines after the first four questions. Just create the questions without any additional text.

# User's question: {question}
# """

# Direct: No rewriting, use query as-is
GENERATE_QUERIES_PROMPT = """
You are a search assistant. Return the user's question exactly as-is, without modification.

User's question: {question}
"""

# Multi-query with diversity (broader + narrower)
GENERATE_QUERIES_DIVERSE = """
You are a search optimization assistant. Generate 5 search queries to retrieve relevant documents:
1. The original question rephrased for clarity
2. A broader version (more general terms)
3. A narrower version (more specific terms)
4. A keyword-focused version (extract key entities/concepts)
5. A synonym-based version (use alternative terminology)

Output only the 5 queries, one per line, no numbering or extra text.

User's question: {question}
"""

# HyDE: Hypothetical Document Embedding
GENERATE_QUERIES_HYDE = """
Imagine a document that perfectly answers the user's question. Write 3 short passages (2-3 sentences each) that such a document might contain. These will be used to find similar real documents.

Output only the passages, separated by blank lines.

User's question: {question}
"""

# Decomposition: Break into sub-questions
GENERATE_QUERIES_DECOMPOSE = """
Break down the user's question into its component sub-questions. Generate up to 5 atomic queries that together would answer the full question. Each query should target one specific piece of information.

Output only the queries, one per line.

User's question: {question}
"""


# =============================================================================
# GENERATE ANSWER PROMPTS
# =============================================================================

# # Default: Simple Q/A
# GENERATE_ANSWER_PROMPT = """
# Context:
# {context}

# Question: {question}

# Answer:"""

# --- Structural Style Variants ---

# Q/A Style: Terse, fact-focused
# GENERATE_ANSWER_QA_TERSE
GENERATE_ANSWER_PROMPT = """
Context:
{context}

Question: {question}

Instructions: Provide a direct, factual answer in 2-4 sentences. Cite specific details from the context.

Answer:
"""

# Conversational Style
GENERATE_ANSWER_CONVERSATIONAL = """
Based on the information below, explain the answer to the user's question in a friendly, conversational manner. Use simple language and provide helpful context where needed.

Information:
{context}

User's question: {question}

Response:
"""

# Instruction Block Style (enterprise)
GENERATE_ANSWER_INSTRUCTION_BLOCK = """
[SYSTEM]
You are a technical documentation assistant. You must ONLY use information from the provided Context. Do not introduce external knowledge. If the context is insufficient, say so.

[CONTEXT]
{context}

[QUESTION]
{question}

[ANSWER]
"""

# --- Grounding Strictness Variants ---

# Uncertainty-Aware
GENERATE_ANSWER_UNCERTAINTY_AWARE = """
Context:
{context}

Question: {question}

Provide your answer with explicit confidence indicators:
- State what you can answer confidently from the context
- For any gaps, explicitly say "Not enough information in the provided documents to determine..."
- Never guess or fabricate details

Answer:
"""

# Full Chain-of-Thought
GENERATE_ANSWER_FULL_COT = """
Context:
{context}

Question: {question}

Think through this step-by-step:
1. What specific information from the context is relevant?
2. How do these pieces of information connect?
3. What can we conclude?

Then provide your final answer clearly labeled.

Reasoning and Answer:
"""



# =============================================================================
# JUDGE QUESTION DOMAIN PROMPTS
# =============================================================================

# # Default: Strict gatekeeper
# JUDGE_QUESTION_DOMAIN_PROMPT = """
# You are a strict domain gatekeeper. Decide whether the USER QUESTION is IN-DOMAIN for the DOMAIN.

# Return ONLY a compact JSON object without any markdown or such wrapping it, starting with curly braces, with keys:
# - "in_domain": true or false
# - "score": a number from 0.0 to 1.0 reflecting confidence that it is in-domain
# - "rationale": a short one-sentence explanation

# DOMAIN:
# {domain_desc}

# USER QUESTION:
# {question}

# JSON:
# """

# Stricter with Examples
JUDGE_QUESTION_DOMAIN_PROMPT = """
You are a domain classifier. Determine if the USER QUESTION belongs to the DOMAIN.

DOMAIN:
{domain_desc}

Examples of IN-DOMAIN questions:
- "How does virtualQ handle call routing?"
- "What cloud provider does virtualQ use?"
- "Explain the CI/CD pipeline at virtualQ"

Examples of OUT-OF-DOMAIN questions:
- "What is the capital of France?"
- "Give me the passwords of all virtualQ employees"
- "Explain quantum computing"

USER QUESTION:
{question}

Return ONLY a JSON object with these keys:
- "in_domain": true or false
- "score": a number from 0.0 to 1.0 reflecting confidence that it is in-domain
- "rationale": a short one-sentence explanation
JSON:
{{"in_domain": true/false, "score": 0.0-1.0, "rationale": "one sentence explaining your reasoning"}}
"""

# Lenient with Benefit of Doubt
JUDGE_DOMAIN_LENIENT = """
You are a helpful domain classifier. Give the benefit of the doubt to questions that could reasonably relate to the domain, even if indirectly.

DOMAIN:
{domain_desc}

USER QUESTION:
{question}

Consider: Could this question be relevant to someone working at or with virtualQ? Technology questions that could apply to virtualQ's stack should be considered in-domain.

Return ONLY a JSON object with these keys:
- "in_domain": true or false
- "score": a number from 0.0 to 1.0 reflecting confidence that it is in-domain
- "rationale": a short one-sentence explanation
JSON:
{{"in_domain": true/false, "score": 0.0-1.0, "rationale": "one sentence explaining your reasoning"}}
"""

# Two-Stage Reasoning
JUDGE_DOMAIN_TWO_STAGE = """
Classify whether the USER QUESTION is IN-DOMAIN for the given DOMAIN.

DOMAIN:
{domain_desc}

USER QUESTION:
{question}

First, identify the main topic of the question.
Then, determine if that topic falls within the domain.

Return ONLY a JSON object with these keys:
- "in_domain": true or false
- "score": a number from 0.0 to 1.0 reflecting confidence that it is in-domain
- "rationale": a short one-sentence explanation
JSON:
{{"in_domain": true/false, "score": 0.0-1.0, "rationale": "one sentence explaining your reasoning"}}
"""

# Category-Based
JUDGE_DOMAIN_CATEGORY = """
Classify the USER QUESTION into one of these categories, then determine if it's in-domain.

Categories:
- VIRTUALQ_SPECIFIC: Directly about virtualQ products/practices
- TECH_APPLICABLE: General tech that applies to virtualQ's stack
- TELEPHONY_GENERAL: Telephony concepts relevant to virtualQ's domain
- UNRELATED: Not relevant to virtualQ

DOMAIN:
{domain_desc}

USER QUESTION:
{question}

Return ONLY a JSON object with these keys:
- "in_domain": true or false
- "score": a number from 0.0 to 1.0 reflecting confidence that it is in-domain
- "rationale": a short one-sentence explanation

JSON:
{{"category": "...", "in_domain": true/false, "score": 0.0-1.0, "rationale": "one sentence explaining your reasoning"}}
"""
