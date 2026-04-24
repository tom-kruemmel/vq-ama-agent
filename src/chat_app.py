import logging
import threading
from collections import deque

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

# Import your existing agent and components
from .agent import RAGAgent
from .domain_judge import DomainJudge
from .embeddings import Embeddings
from .rank_fusion import RankFusion
from .reranker import CrossEncoderReranker
from .confidence_checker import ConfidenceChecker
from .pipeline import run_pipeline

logger = logging.getLogger(__name__)

# --------------- chat history helpers ---------------
MAX_HISTORY_TURNS = 5  # keep last N Q/A pairs per session

# session_id -> deque([(question, answer), ...])
_session_histories: dict[str, deque] = {}
_session_lock = threading.Lock()


def _get_history(session_id: str) -> deque:
    with _session_lock:
        if session_id not in _session_histories:
            _session_histories[session_id] = deque(maxlen=MAX_HISTORY_TURNS)
        return _session_histories[session_id]


def _format_history(history: deque) -> list[dict[str, str]]:
    """Format recent turns into structured messages for the Converse API."""
    messages = []
    for q, a in history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    return messages

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


def create_app(
    agent: RAGAgent,
    domain_judge: DomainJudge,
    headings: list[str],
    retriever: Embeddings | None = None,
    fusion: RankFusion | None = None,
    reranker: CrossEncoderReranker | None = None,
    confidence_checker: ConfidenceChecker | None = None,
    rerank_top_n: int = 10,
) -> Flask:
    app = Flask(__name__)
    CORS(app)

    # Use injected instances or create defaults
    _retriever = retriever or Embeddings()
    _fusion = fusion or RankFusion()
    _reranker = reranker or CrossEncoderReranker()
    _checker = confidence_checker or ConfidenceChecker()

    @app.route('/health')
    def health():
        return jsonify({'status': 'ok'})

    @app.route('/')
    def index():
        return render_template_string(CHAT_HTML)

    @app.route('/chat', methods=['POST'])
    def chat():
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'Missing "question" field'}), 400
        question = data['question']
        if not question or not question.strip():
            return jsonify({'error': 'Question cannot be empty'}), 400

        result = run_pipeline(
            question,
            agent=agent,
            headings=headings,
            retriever=_retriever,
            fusion=_fusion,
            reranker=_reranker,
            confidence_checker=_checker,
            domain_judge=domain_judge,
            rerank_top_n=rerank_top_n,
            enforce_gates=True,
            sanitize=True,
        )

        if result.answer is None:
            return jsonify({'error': 'Question cannot be empty'}), 400
        return jsonify({'answer': result.answer})

    return app


def run_chat_server(
    agent: RAGAgent,
    domain_judge: DomainJudge,
    headings: list[str],
    host: str = '127.0.0.1',
    port: int = 8000,
    retriever: Embeddings | None = None,
    fusion: RankFusion | None = None,
    reranker: CrossEncoderReranker | None = None,
    confidence_checker: ConfidenceChecker | None = None,
):
  """
  Launches the chat web server on localhost.
  """
  app = create_app(
    agent, domain_judge, headings,
    retriever=retriever,
    fusion=fusion,
    reranker=reranker,
    confidence_checker=confidence_checker,
  )
  app.run(host=host, port=port)
