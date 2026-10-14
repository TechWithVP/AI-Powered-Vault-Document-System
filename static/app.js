/**
 * Minimal chat client for the Vault Document System.
 *
 * Plain JavaScript with no build step: the page is served directly by FastAPI,
 * so there is nothing to compile or install before the UI runs.
 */

const BASE = "/api/v1";
const POLL_INTERVAL_MS = 2500;
const KEY_STORAGE = "vault.apiKey";

/** Thin wrapper over fetch that attaches auth and unwraps the error envelope. */
class ApiClient {
  constructor() {
    this.apiKey = localStorage.getItem(KEY_STORAGE) || "dev-key-change-me";
  }

  setKey(key) {
    this.apiKey = key.trim();
    localStorage.setItem(KEY_STORAGE, this.apiKey);
  }

  async request(path, options) {
    const { method = "GET", body, isForm = false } = options || {};
    const headers = { "X-API-Key": this.apiKey };
    if (body && !isForm) headers["Content-Type"] = "application/json";

    const response = await fetch(BASE + path, {
      method,
      headers,
      body: isForm ? body : body ? JSON.stringify(body) : undefined,
    });

    if (response.status === 204) return null;

    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = payload && payload.error ? payload.error.message : response.statusText;
      throw new Error(detail || "Request failed");
    }
    return payload;
  }

  listDocuments() {
    return this.request("/documents?limit=50");
  }

  upload(file) {
    const form = new FormData();
    form.append("file", file);
    return this.request("/documents/upload", { method: "POST", body: form, isForm: true });
  }

  createSession(documentIds) {
    return this.request("/chat/sessions", {
      method: "POST",
      body: { document_ids: documentIds },
    });
  }

  ask(sessionId, question) {
    return this.request("/chat/sessions/" + sessionId + "/messages", {
      method: "POST",
      body: { question },
    });
  }

  /**
   * Streams an answer, invoking handlers as Server-Sent Events arrive.
   * Frames are parsed manually because EventSource cannot issue a POST.
   */
  async askStreaming(sessionId, question, handlers) {
    const response = await fetch(
      BASE + "/chat/sessions/" + sessionId + "/messages/stream",
      {
        method: "POST",
        headers: { "X-API-Key": this.apiKey, "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      }
    );

    if (!response.ok) {
      const payload = await response.json().catch(() => null);
      throw new Error(payload && payload.error ? payload.error.message : "Request failed");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Frames are separated by a blank line; keep any partial tail buffered.
      const frames = buffer.split("\n\n");
      buffer = frames.pop();

      frames.forEach((frame) => {
        const event = {};
        frame.split("\n").forEach((line) => {
          if (line.startsWith("event: ")) event.name = line.slice(7).trim();
          else if (line.startsWith("data: ")) event.data = line.slice(6);
        });
        if (!event.name || !event.data) return;

        let parsed;
        try {
          parsed = JSON.parse(event.data);
        } catch (error) {
          return;
        }
        const handler = handlers[event.name];
        if (handler) handler(parsed);
      });
    }
  }
}

/** Returns the singular or plural form of a noun for the given count. */
function plural(count, noun) {
  return count === 1 ? noun : noun + "s";
}

/** Rendering helpers. Text is inserted as text nodes, never as raw HTML. */
const dom = {
  el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  },

  /** Renders answer text, turning [n] markers into superscript references. */
  answerBody(text) {
    const body = dom.el("div", "body");
    const pattern = /\[(\d+)\]/g;
    let cursor = 0;
    let match = pattern.exec(text);

    while (match !== null) {
      if (match.index > cursor) {
        body.appendChild(document.createTextNode(text.slice(cursor, match.index)));
      }
      body.appendChild(dom.el("span", "cite-ref", match[0]));
      cursor = match.index + match[0].length;
      match = pattern.exec(text);
    }
    if (cursor < text.length) {
      body.appendChild(document.createTextNode(text.slice(cursor)));
    }
    return body;
  },
};

class VaultUI {
  constructor() {
    this.api = new ApiClient();
    this.documents = [];
    this.selected = new Set();
    this.sessionId = null;
    this.pollTimer = null;

    const byId = (id) => document.getElementById(id);
    this.nodes = {
      apiKey: byId("apiKey"),
      dropzone: byId("dropzone"),
      fileInput: byId("fileInput"),
      documents: byId("documents"),
      docCount: byId("docCount"),
      messages: byId("messages"),
      composer: byId("composer"),
      question: byId("question"),
      send: byId("send"),
      scope: byId("scope"),
      health: byId("health"),
    };
  }

  async start() {
    this.nodes.apiKey.value = this.api.apiKey;
    this.bindEvents();
    await this.checkHealth();
    await this.refreshDocuments();
  }

  bindEvents() {
    const { apiKey, dropzone, fileInput, composer, question } = this.nodes;

    apiKey.addEventListener("change", () => {
      this.api.setKey(apiKey.value);
      this.refreshDocuments();
    });

    fileInput.addEventListener("change", (event) => {
      const file = event.target.files[0];
      if (file) this.upload(file);
      event.target.value = "";
    });

    ["dragover", "dragenter"].forEach((name) =>
      dropzone.addEventListener(name, (event) => {
        event.preventDefault();
        dropzone.classList.add("drag");
      })
    );
    ["dragleave", "drop"].forEach((name) =>
      dropzone.addEventListener(name, () => dropzone.classList.remove("drag"))
    );
    dropzone.addEventListener("drop", (event) => {
      event.preventDefault();
      const file = event.dataTransfer.files[0];
      if (file) this.upload(file);
    });

    composer.addEventListener("submit", (event) => {
      event.preventDefault();
      this.ask();
    });

    question.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        this.ask();
      }
    });
  }

  async checkHealth() {
    const node = this.nodes.health;
    try {
      const response = await fetch("/health");
      const body = await response.json();
      node.textContent = body.status === "ok" ? "healthy" : "degraded";
      node.className = "badge " + (body.status === "ok" ? "ok" : "err");
    } catch (error) {
      node.textContent = "unreachable";
      node.className = "badge err";
    }
  }

  async upload(file) {
    try {
      const created = await this.api.upload(file);
      this.toast("Uploaded " + created.filename, false);
      await this.refreshDocuments();
      this.pollUntilReady();
    } catch (error) {
      this.toast(error.message);
    }
  }

  async refreshDocuments() {
    try {
      const page = await this.api.listDocuments();
      this.documents = page.items;
      this.renderDocuments();
    } catch (error) {
      this.nodes.documents.replaceChildren(dom.el("div", "doc-meta", error.message));
    }
  }

  /** Polls while anything is still processing, then stops. */
  pollUntilReady() {
    clearInterval(this.pollTimer);
    this.pollTimer = setInterval(async () => {
      await this.refreshDocuments();
      const pending = this.documents.some(
        (doc) => doc.status === "queued" || doc.status === "processing"
      );
      if (!pending) clearInterval(this.pollTimer);
    }, POLL_INTERVAL_MS);
  }

  renderDocuments() {
    const { documents, docCount } = this.nodes;
    docCount.textContent = this.documents.length ? "(" + this.documents.length + ")" : "";

    if (!this.documents.length) {
      documents.replaceChildren(dom.el("div", "doc-meta", "No documents yet."));
      this.updateScope();
      return;
    }

    const cards = this.documents.map((doc) => {
      const card = dom.el("div", "doc" + (this.selected.has(doc.id) ? " selected" : ""));

      const name = dom.el("div", "doc-name");
      name.appendChild(dom.el("span", null, doc.filename));
      name.appendChild(dom.el("span", "status " + doc.status, doc.status));
      card.appendChild(name);

      const parts = [doc.file_type.toUpperCase(), Math.round(doc.file_size_bytes / 1024) + " KB"];
      if (doc.page_count) parts.push(doc.page_count + "p");
      if (doc.chunk_count) parts.push(doc.chunk_count + " chunks");
      card.appendChild(dom.el("div", "doc-meta", parts.join(" \u00b7 ")));

      if (doc.failure_reason) {
        card.appendChild(dom.el("div", "doc-meta", doc.failure_reason));
      }

      card.addEventListener("click", () => this.toggleDocument(doc));
      return card;
    });

    documents.replaceChildren(...cards);
    this.updateScope();
  }

  toggleDocument(doc) {
    if (doc.status !== "ready") {
      this.toast(doc.filename + " is still " + doc.status + ".");
      return;
    }
    if (this.selected.has(doc.id)) this.selected.delete(doc.id);
    else this.selected.add(doc.id);

    // Changing the document scope starts a fresh conversation.
    this.sessionId = null;
    this.renderDocuments();
  }

  updateScope() {
    const ready = this.documents.filter((doc) => doc.status === "ready").length;
    const node = this.nodes.scope;

    if (this.selected.size) {
      node.textContent = this.selected.size + " " + plural(this.selected.size, "document") + " selected";
    } else if (ready) {
      node.textContent = ready === 1 ? "1 document" : "all " + ready + " documents";
    } else {
      node.textContent = "no documents ready";
    }
  }

  async ask() {
    const text = this.nodes.question.value.trim();
    if (!text) return;

    const ready = this.documents.filter((doc) => doc.status === "ready");
    if (!ready.length) {
      this.toast("Upload a document and wait for it to finish processing.");
      return;
    }

    this.nodes.question.value = "";
    this.appendUser(text);
    this.setBusy(true);

    try {
      if (!this.sessionId) {
        const scope = this.selected.size ? Array.from(this.selected) : null;
        const session = await this.api.createSession(scope);
        this.sessionId = session.id;
      }
      await this.streamAnswer(text);
    } catch (error) {
      this.appendError(error.message);
    } finally {
      this.setBusy(false);
    }
  }

  /** Renders an answer as it streams, then attaches its citations. */
  async streamAnswer(question) {
    const message = dom.el("div", "msg assistant");
    message.appendChild(dom.el("div", "who", "Assistant"));
    const body = dom.el("div", "body");
    message.appendChild(body);
    this.push(message);

    let text = "";
    let citations = [];

    await this.api.askStreaming(this.sessionId, question, {
      token: (event) => {
        text += event.text;
        body.textContent = text;
        this.nodes.messages.scrollTop = this.nodes.messages.scrollHeight;
      },
      citations: (event) => {
        citations = event.citations;
      },
      done: (event) => {
        // The server renumbers citation markers after generation, so the final
        // text can differ from the streamed tokens; prefer it when present.
        const finalText = event.answer || text;
        if (event.grounded) {
          message.replaceChild(dom.answerBody(finalText), body);
        } else {
          body.className = "body ungrounded";
          body.textContent = finalText;
        }
        if (citations.length) {
          message.appendChild(this.renderCitations(citations));
        }
      },
      error: (event) => {
        body.className = "body ungrounded";
        body.textContent = text + "\n\n[" + event.message + "]";
      },
    });
  }

  renderCitations(citations) {
    const list = dom.el("div", "citations");
    citations.forEach((citation) => {
      const item = dom.el("div", "citation");
      const page = citation.page_number ? ", page " + citation.page_number : "";
      item.appendChild(
        dom.el("span", "src", "[" + citation.marker + "] " + citation.document_name + page)
      );
      item.appendChild(dom.el("span", "quote", citation.snippet));
      list.appendChild(item);
    });
    return list;
  }

  setBusy(busy) {
    this.nodes.send.disabled = busy;
    this.nodes.send.textContent = busy ? "" : "Ask";
    if (busy) this.nodes.send.appendChild(dom.el("span", "spinner"));
  }

  clearEmptyState() {
    const empty = this.nodes.messages.querySelector(".empty");
    if (empty) empty.remove();
  }

  appendUser(text) {
    this.clearEmptyState();
    const message = dom.el("div", "msg user");
    message.appendChild(dom.el("div", "who", "You"));
    message.appendChild(dom.el("div", "body", text));
    this.push(message);
  }

  appendAnswer(answer) {
    const message = dom.el("div", "msg assistant");
    message.appendChild(dom.el("div", "who", "Assistant"));

    if (answer.grounded) {
      message.appendChild(dom.answerBody(answer.answer));
    } else {
      message.appendChild(dom.el("div", "body ungrounded", answer.answer));
    }

    if (answer.citations.length) {
      message.appendChild(this.renderCitations(answer.citations));
    }

    if (answer.suggested_questions.length) {
      const chips = dom.el("div", "suggestions");
      answer.suggested_questions.forEach((text) => {
        const chip = dom.el("button", "chip", text);
        chip.type = "button";
        chip.addEventListener("click", () => {
          this.nodes.question.value = text;
          this.ask();
        });
        chips.appendChild(chip);
      });
      message.appendChild(chips);
    }

    this.push(message);
  }

  appendError(text) {
    const message = dom.el("div", "msg assistant");
    message.appendChild(dom.el("div", "who", "Error"));
    message.appendChild(dom.el("div", "body ungrounded", text));
    this.push(message);
  }

  push(node) {
    this.nodes.messages.appendChild(node);
    this.nodes.messages.scrollTop = this.nodes.messages.scrollHeight;
  }

  toast(text, isError) {
    const node = dom.el("div", "toast", text);
    if (isError === false) node.style.borderColor = "var(--ok)";
    document.body.appendChild(node);
    setTimeout(() => node.remove(), 4000);
  }
}

new VaultUI().start();
