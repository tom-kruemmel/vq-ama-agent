from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

# Import your existing agent and components
from .agent import RAGAgent
from .embeddings import Embeddings
from .rank_fusion import RankFusion
from .confidence_checker import ConfidenceChecker
from .prompt_templates import GENERATE_QUERIES_PROMPT, GENERATE_ANSWER_PROMPT, JUDGE_QUESTION_DOMAIN_PROMPT

CHAT_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAG Agent Chat</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; padding: 0; display: flex; flex-direction: column; height: 100vh; }
    #chat { flex: 1; overflow-y: auto; padding: 1em; border-bottom: 1px solid #ccc; }
    .message { 
      margin: 0.5em 0; 
      white-space: pre-wrap;
    }
    .user { color: #2a6f97; }
    .agent { color: #6f972a; }
    #input { display: flex; padding: 1em; }
    #input textarea { flex: 1; resize: none; padding: 0.5em; font-size: 1em; }
    #input button { margin-left: 1em; padding: 0.5em 1em; font-size: 1em; }
  </style>
</head>
<body>
  <div id="chat"></div>
  <div id="input">
    <textarea id="message" rows="2" placeholder="Type your message..."></textarea>
    <button id="send">Send</button>
  </div>
<script>
  const chat = document.getElementById('chat');
  const messageInput = document.getElementById('message');
  const sendBtn = document.getElementById('send');

  function appendMessage(sender, text, id = null) {
    const div = document.createElement('div');
    div.className = 'message ' + (sender === 'You' ? 'user' : 'agent');
    if (id) div.id = id;

    const escaped = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
    const html = `<strong>${sender}:</strong> ${escaped.replace(/\\n/g, '<br>')}`;

    div.innerHTML = html;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
  }

  function showThinkingAnimation(id) {
    let dotCount = 0;
    const maxDots = 3;
    const element = document.getElementById(id);
    return setInterval(() => {
      dotCount = (dotCount + 1) % (maxDots + 1);
      const dots = '.'.repeat(dotCount);
      if (element) {
        element.innerHTML = `<strong>Agent:</strong> Thinking${dots}`;
      }
    }, 500);
  }

  async function sendMessage() {
    const text = messageInput.value.trim();
    if (!text) return;
    appendMessage('You', text);
    messageInput.value = '';

    const thinkingId = `thinking-${Date.now()}`; // unique ID per message
    appendMessage('Agent', 'Thinking', thinkingId);
    const thinkingInterval = showThinkingAnimation(thinkingId);

    try {
      const resp = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: text })
      });
      const data = await resp.json();

      clearInterval(thinkingInterval);
      const placeholder = document.getElementById(thinkingId);
      if (placeholder) {
        const escaped = data.answer
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;");
        placeholder.innerHTML = `<strong>Agent:</strong> ${escaped.replace(/\\n/g, '<br>')}`;
      }
    } catch (error) {
      clearInterval(thinkingInterval);
      const placeholder = document.getElementById(thinkingId);
      if (placeholder) {
        placeholder.innerHTML = `<strong>Agent:</strong> Error receiving response.`;
      }
    }
  }

  sendBtn.addEventListener('click', sendMessage);
  messageInput.addEventListener('keypress', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
</script>
</body>
</html>
"""


def create_app(agent: RAGAgent, user_roles: list[str], headings: list[str],
         generate_queries_prompt: str = GENERATE_QUERIES_PROMPT,
         generate_answer_prompt: str = GENERATE_ANSWER_PROMPT,
         judge_question_domain_prompt: str = JUDGE_QUESTION_DOMAIN_PROMPT) -> Flask:
    app = Flask(__name__)
    CORS(app)

    @app.route('/')
    def index():
        return render_template_string(CHAT_HTML)

    @app.route('/chat', methods=['POST'])
    def chat():
        data = request.get_json()
        question = data.get('question', '')
        # RAG workflow
        in_domain, dq_score, dq_rationale = agent.judge_question_domain(
            question,
            min_score=0.60,  # tune as you like
        )
        print(f"Domain judge: {in_domain} (score: {dq_score:.3f}) -- {dq_rationale}")
        if not in_domain:
            return jsonify({
                'answer': "I can help with questions about virtualQ and its technology stack. Please ask a question related to that.",
          })
        queries = agent.generate_queries(question)
        retriever = Embeddings()
        retrieved_docs = retriever.retrieve_documents(queries, user_roles, headings)
        fusion = RankFusion()
        fused_docs_with_scores = fusion.reciprocal_rank_fusion(retrieved_docs)
        fused_docs = [doc for doc, _ in fused_docs_with_scores]
        checker = ConfidenceChecker(
          top_k=5,
          min_chunks=3,
          min_unique_chunks=2,
          min_total_chars=450,
          min_top1_score=0.015,
          min_avg_top3_score=0.011,
        )

        confident, confidence_details = checker.evaluate(fused_docs_with_scores)
        if not confident:
          print(f"Abstain gate triggered: {confidence_details}")
          return jsonify({
            'answer': (
              "I don’t have enough reliable context to answer that confidently. "
              "Please rephrase your question or provide a bit more detail."
            )
          })

        # Only if confident, generate an answer
        answer = agent.generate_answer(question, fused_docs)
        return jsonify({'answer': answer})

    return app


def run_chat_server(agent: RAGAgent, user_roles: list[str], headings: list[str],
           host: str = '127.0.0.1', port: int = 5000,
           generate_queries_prompt: str = GENERATE_QUERIES_PROMPT,
           generate_answer_prompt: str = GENERATE_ANSWER_PROMPT,
           judge_question_domain_prompt: str = JUDGE_QUESTION_DOMAIN_PROMPT):
  """
  Launches the chat web server on localhost.
  agent: An instance of your RAGAgent
  user_roles: List of roles to filter retrieval
  headings: List of document headings to query
  host: Host interface (default: 127.0.0.1)
  port: Port number (default: 5000)
  """
  app = create_app(
    agent, user_roles, headings,
    generate_queries_prompt=generate_queries_prompt,
    generate_answer_prompt=generate_answer_prompt,
    judge_question_domain_prompt=judge_question_domain_prompt
  )
  app.run(host=host, port=port)
