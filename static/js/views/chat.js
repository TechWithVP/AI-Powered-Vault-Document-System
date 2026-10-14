/**
 * Conversations: the session list, an opened transcript, and the composer.
 *
 * A session's document scope is fixed when it is created, so the picker only
 * appears for a new chat. Reopening a session shows what it was scoped to
 * instead of offering to change it, which would silently not apply.
 */

import { answerBody, append, el, fmt } from "../dom.js";

export class ChatView {
  constructor(api, toasts) {
    this.api = api;
    this.toasts = toasts;
    this.sessions = [];
    this.documents = [];
    this.scope = new Set();
    this.session = null;
    this.busy = false;

    this.root = el("div", "view chat-view");
    this.build();
  }

  build() {
    this.sessionList = el("div", "session-list");
    this.newChatButton = el("button", "primary block", "New chat");
    this.newChatButton.type = "button";
    this.newChatButton.addEventListener("click", () => this.startNewChat());

    const rail = el("aside", "rail");
    const railHead = el("div", "rail-head");
    append(railHead, el("h2", null, "Conversations"), this.newChatButton);
    append(rail, railHead, this.sessionList);

    this.scopePicker = el("div", "scope-picker");
    this.transcript = el("div", "messages");
    this.questionInput = el("textarea");
    this.questionInput.rows = 1;
    this.questionInput.placeholder = "Ask a question about your documents...";
    this.sendButton = el("button", "primary", "Ask");
    this.sendButton.type = "submit";

    const composer = el("form", "composer");
    append(composer, this.questionInput, this.sendButton);
    composer.addEventListener("submit", (event) => {
      event.preventDefault();
      this.ask();
    });
    this.questionInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        this.ask();
      }
    });

    const footer = el("footer");
    footer.appendChild(composer);

    const pane = el("section", "pane");
    append(pane, this.scopePicker, this.transcript, footer);

    append(this.root, rail, pane);
  }

  async activate() {
    await Promise.all([this.refreshSessions(), this.refreshDocuments()]);
    if (!this.session) this.startNewChat();
  }

  deactivate() {}

  // -- data ---------------------------------------------------------------- //

  async refreshSessions() {
    try {
      const page = await this.api.listSessions();
      this.sessions = page.items;
    } catch (error) {
      this.sessions = [];
      this.sessionList.replaceChildren(el("p", "muted", error.message));
      return;
    }
    this.renderSessions();
  }

  async refreshDocuments() {
    try {
      const page = await this.api.listDocuments();
      this.documents = page.items;
    } catch (error) {
      this.documents = [];
    }
  }

  get readyDocuments() {
    return this.documents.filter((doc) => doc.status === "ready");
  }

  // -- sessions ------------------------------------------------------------ //

  renderSessions() {
    if (!this.sessions.length) {
      this.sessionList.replaceChildren(
        el("p", "muted", "No conversations yet.")
      );
      return;
    }

    const cards = this.sessions.map((session) => {
      const active = this.session && this.session.id === session.id;
      const card = el("div", "session" + (active ? " active" : ""));

      const title = el("div", "session-title", session.title || "Untitled chat");
      const names = session.documents.map((doc) => doc.filename);
      const scope = names.length
        ? names.length === 1
          ? names[0]
          : names.length + " documents"
        : "all documents";
      const meta = el(
        "div",
        "session-meta",
        session.message_count +
          " " +
          fmt.plural(session.message_count, "message") +
          " · " +
          scope
      );
      const when = el("div", "session-meta", fmt.ago(session.updated_at));

      const remove = el("button", "icon-button", "×");
      remove.type = "button";
      remove.title = "Delete conversation";
      remove.addEventListener("click", (event) => {
        event.stopPropagation();
        this.removeSession(session);
      });

      append(card, title, meta, when, remove);
      card.addEventListener("click", () => this.openSession(session));
      return card;
    });

    this.sessionList.replaceChildren(...cards);
  }

  async openSession(session) {
    this.session = session;
    this.scopePicker.replaceChildren();
    this.renderSessions();
    this.transcript.replaceChildren(el("p", "muted", "Loading conversation..."));

    const names = session.documents.map((doc) => doc.filename);
    const banner = el(
      "div",
      "scope-banner",
      names.length ? "Scoped to " + names.join(", ") : "Scoped to all documents"
    );
    this.scopePicker.replaceChildren(banner);

    try {
      const page = await this.api.getHistory(session.id);
      this.renderHistory(page.items);
    } catch (error) {
      this.transcript.replaceChildren(el("p", "error-text", error.message));
    }
  }

  async removeSession(session) {
    const label = session.title || "this conversation";
    if (!window.confirm("Delete " + label + "? This cannot be undone.")) return;

    try {
      await this.api.deleteSession(session.id);
      this.toasts.ok("Conversation deleted");
      if (this.session && this.session.id === session.id) {
        this.session = null;
        this.startNewChat();
      }
      await this.refreshSessions();
    } catch (error) {
      this.toasts.error(error.message);
    }
  }

  startNewChat() {
    this.session = null;
    this.scope = new Set();
    this.renderSessions();
    this.renderScopePicker();
    this.transcript.replaceChildren(
      el(
        "p",
        "muted",
        this.readyDocuments.length
          ? "Pick the documents to search, then ask a question. Answers cite the page they came from."
          : "Upload a document first - the Documents tab has the uploader."
      )
    );
  }

  /** The scope picker only exists before the first question creates a session. */
  renderScopePicker() {
    const ready = this.readyDocuments;
    if (!ready.length) {
      this.scopePicker.replaceChildren();
      return;
    }

    const row = el("div", "chip-row");
    row.appendChild(el("span", "chip-label", "Search in"));

    const all = el("button", "chip" + (this.scope.size ? "" : " on"), "All documents");
    all.type = "button";
    all.addEventListener("click", () => {
      this.scope.clear();
      this.renderScopePicker();
    });
    row.appendChild(all);

    ready.forEach((doc) => {
      const chip = el(
        "button",
        "chip" + (this.scope.has(doc.id) ? " on" : ""),
        doc.filename
      );
      chip.type = "button";
      chip.addEventListener("click", () => {
        if (this.scope.has(doc.id)) this.scope.delete(doc.id);
        else this.scope.add(doc.id);
        this.renderScopePicker();
      });
      row.appendChild(chip);
    });

    this.scopePicker.replaceChildren(row);
  }

  // -- transcript ---------------------------------------------------------- //

  renderHistory(messages) {
    if (!messages.length) {
      this.transcript.replaceChildren(
        el("p", "muted", "No messages in this conversation yet.")
      );
      return;
    }

    const nodes = messages.map((message) =>
      message.role === "user"
        ? this.userMessage(message.content)
        : this.assistantMessage(message)
    );
    this.transcript.replaceChildren(...nodes);
    this.scrollToEnd();
  }

  userMessage(text) {
    const node = el("div", "msg user");
    append(node, el("div", "who", "You"), el("div", "body", text));
    return node;
  }

  /** Renders a stored assistant turn, including its citations. */
  assistantMessage(message) {
    const node = el("div", "msg assistant");
    node.appendChild(el("div", "who", "Assistant"));
    node.appendChild(
      message.grounded
        ? answerBody(message.content)
        : el("div", "body ungrounded", message.content)
    );
    if (message.citations && message.citations.length) {
      node.appendChild(this.citations(message.citations));
    }
    if (message.suggested_questions && message.suggested_questions.length) {
      node.appendChild(this.suggestions(message.suggested_questions));
    }
    return node;
  }

  citations(citations) {
    const list = el("div", "citations");
    citations.forEach((citation) => {
      const item = el("div", "citation");
      const page = citation.page_number ? ", page " + citation.page_number : "";
      const source = el(
        "span",
        "src",
        "[" + citation.marker + "] " + citation.document_name + page
      );

      // Which arm found a passage is on the payload, and is the quickest way
      // to see hybrid retrieval doing something rather than trust that it is.
      const arm =
        citation.lexical_rank > 0 && citation.similarity > 0
          ? "both"
          : citation.lexical_rank > 0
          ? "exact term"
          : "vector";
      append(
        item,
        source,
        el("span", "found-by", arm),
        el("span", "quote", citation.snippet)
      );
      list.appendChild(item);
    });
    return list;
  }

  suggestions(questions) {
    const row = el("div", "suggestions");
    questions.forEach((text) => {
      const chip = el("button", "chip", text);
      chip.type = "button";
      chip.addEventListener("click", () => {
        this.questionInput.value = text;
        this.ask();
      });
      row.appendChild(chip);
    });
    return row;
  }

  // -- asking -------------------------------------------------------------- //

  async ask() {
    const text = this.questionInput.value.trim();
    if (!text || this.busy) return;

    if (!this.session && !this.readyDocuments.length) {
      this.toasts.error("Upload a document and wait for it to finish processing.");
      return;
    }

    this.questionInput.value = "";
    this.clearPlaceholder();
    this.push(this.userMessage(text));
    this.setBusy(true);

    const isFirstTurn = !this.session;
    try {
      if (!this.session) {
        const ids = this.scope.size ? Array.from(this.scope) : null;
        this.session = await this.api.createSession(ids);
        this.scopePicker.replaceChildren();
      }
      await this.streamAnswer(text);
      // The title is generated from the first exchange, so the list only
      // becomes meaningful after the turn completes.
      await this.refreshSessions();
      if (isFirstTurn) this.renderSessions();
    } catch (error) {
      this.push(this.errorMessage(error.message));
    } finally {
      this.setBusy(false);
    }
  }

  /** Renders an answer as it streams, then attaches its citations. */
  async streamAnswer(question) {
    const node = el("div", "msg assistant");
    node.appendChild(el("div", "who", "Assistant"));
    let body = el("div", "body");
    node.appendChild(body);
    this.push(node);

    let text = "";
    let citations = [];

    await this.api.askStreaming(this.session.id, question, {
      token: (event) => {
        text += event.text;
        body.textContent = text;
        this.scrollToEnd();
      },
      citations: (event) => {
        citations = event.citations;
      },
      done: (event) => {
        // Markers are renumbered server-side after generation, so the final
        // text can differ from the streamed tokens; prefer it when present.
        const finalText = event.answer || text;
        if (event.grounded) {
          const rendered = answerBody(finalText);
          node.replaceChild(rendered, body);
          body = rendered;
        } else {
          body.className = "body ungrounded";
          body.textContent = finalText;
        }
        if (event.source && event.source !== "documents") {
          node.appendChild(
            el("div", "source-note", "Answered from the conversation, not the documents")
          );
        }
        if (citations.length) node.appendChild(this.citations(citations));
        this.scrollToEnd();
      },
      error: (event) => {
        body.className = "body ungrounded";
        body.textContent = text + "\n\n[" + event.message + "]";
      },
    });
  }

  errorMessage(text) {
    const node = el("div", "msg assistant");
    append(node, el("div", "who", "Error"), el("div", "body ungrounded", text));
    return node;
  }

  // -- plumbing ------------------------------------------------------------ //

  clearPlaceholder() {
    const placeholder = this.transcript.querySelector(".muted, .error-text");
    if (placeholder) placeholder.remove();
  }

  push(node) {
    this.transcript.appendChild(node);
    this.scrollToEnd();
  }

  scrollToEnd() {
    this.transcript.scrollTop = this.transcript.scrollHeight;
  }

  setBusy(busy) {
    this.busy = busy;
    this.sendButton.disabled = busy;
    this.sendButton.replaceChildren();
    if (busy) this.sendButton.appendChild(el("span", "spinner"));
    else this.sendButton.textContent = "Ask";
  }
}
