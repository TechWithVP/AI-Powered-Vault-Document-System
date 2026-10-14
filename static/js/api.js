/**
 * Typed-ish client for every endpoint the service exposes.
 *
 * One method per endpoint, named after what it does rather than the verb and
 * path, so a view never builds a URL or remembers a payload shape. The error
 * envelope is unwrapped here too: views receive an Error with the server's
 * message, never a raw Response.
 */

const BASE = "/api/v1";
const KEY_STORAGE = "vault.apiKey";
const DEFAULT_KEY = "dev-key-change-me";

export class ApiError extends Error {
  constructor(message, { status = 0, code = "" } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

export class ApiClient {
  constructor() {
    this.apiKey = localStorage.getItem(KEY_STORAGE) || DEFAULT_KEY;
  }

  setKey(key) {
    this.apiKey = key.trim();
    try {
      localStorage.setItem(KEY_STORAGE, this.apiKey);
    } catch (error) {
      // Private browsing can refuse storage; the key still works this session.
    }
  }

  async request(path, options) {
    const { method = "GET", body, isForm = false, base = BASE } = options || {};
    const headers = { "X-API-Key": this.apiKey };
    if (body && !isForm) headers["Content-Type"] = "application/json";

    let response;
    try {
      response = await fetch(base + path, {
        method,
        headers,
        body: isForm ? body : body ? JSON.stringify(body) : undefined,
      });
    } catch (error) {
      throw new ApiError("Cannot reach the server.");
    }

    if (response.status === 204) return null;

    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const envelope = payload && payload.error ? payload.error : null;
      throw new ApiError(
        (envelope && envelope.message) || response.statusText || "Request failed",
        { status: response.status, code: envelope ? envelope.code : "" }
      );
    }
    return payload;
  }

  // -- documents ----------------------------------------------------------- //

  listDocuments({ limit = 50, offset = 0, status = "", search = "" } = {}) {
    const query = new URLSearchParams({ limit, offset });
    if (status) query.set("status", status);
    if (search) query.set("search", search);
    return this.request("/documents?" + query.toString());
  }

  getDocument(id) {
    return this.request("/documents/" + id);
  }

  getInsights(id) {
    return this.request("/documents/" + id + "/insights");
  }

  upload(file) {
    const form = new FormData();
    form.append("file", file);
    return this.request("/documents/upload", {
      method: "POST",
      body: form,
      isForm: true,
    });
  }

  reprocess(id) {
    return this.request("/documents/" + id + "/reprocess", { method: "POST" });
  }

  deleteDocument(id) {
    return this.request("/documents/" + id, { method: "DELETE" });
  }

  // -- chat ---------------------------------------------------------------- //

  listSessions({ limit = 50, offset = 0 } = {}) {
    return this.request(
      "/chat/sessions?" + new URLSearchParams({ limit, offset }).toString()
    );
  }

  getSession(id) {
    return this.request("/chat/sessions/" + id);
  }

  createSession(documentIds, title) {
    return this.request("/chat/sessions", {
      method: "POST",
      body: { document_ids: documentIds, title: title || null },
    });
  }

  deleteSession(id) {
    return this.request("/chat/sessions/" + id, { method: "DELETE" });
  }

  getHistory(id, { limit = 100, offset = 0 } = {}) {
    return this.request(
      "/chat/sessions/" + id + "/messages?" +
        new URLSearchParams({ limit, offset }).toString()
    );
  }

  ask(sessionId, question) {
    return this.request("/chat/sessions/" + sessionId + "/messages", {
      method: "POST",
      body: { question },
    });
  }

  /**
   * Streams an answer, invoking handlers as Server-Sent Events arrive.
   * Frames are parsed by hand because EventSource cannot issue a POST.
   */
  async askStreaming(sessionId, question, handlers) {
    const response = await fetch(
      BASE + "/chat/sessions/" + sessionId + "/messages/stream",
      {
        method: "POST",
        headers: {
          "X-API-Key": this.apiKey,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
      }
    );

    if (!response.ok) {
      const payload = await response.json().catch(() => null);
      const envelope = payload && payload.error ? payload.error : null;
      throw new ApiError(
        (envelope && envelope.message) || "Request failed",
        { status: response.status }
      );
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

  // -- monitoring ---------------------------------------------------------- //

  documentMetrics() {
    return this.request("/metrics/documents");
  }

  processingMetrics() {
    return this.request("/metrics/processing");
  }

  aiUsageMetrics() {
    return this.request("/metrics/ai-usage");
  }

  retrievalMetrics() {
    return this.request("/metrics/retrieval");
  }

  /** Health is unauthenticated and sits outside the versioned prefix. */
  health() {
    return this.request("/health", { base: "" });
  }
}
