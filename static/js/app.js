/**
 * Application shell: the API key, the health badge, and view switching.
 *
 * Views are constructed once and kept, so switching tabs does not refetch or
 * lose a half-typed question. Each one owns its data and timers and is told
 * when it becomes visible, so only the visible view polls.
 */

import { ApiClient } from "./api.js";
import { append, el } from "./dom.js";
import { Toasts } from "./toast.js";
import { ChatView } from "./views/chat.js";
import { DocumentsView } from "./views/documents.js";
import { MetricsView } from "./views/metrics.js";

const HEALTH_INTERVAL_MS = 30000;

class AppShell {
  constructor() {
    this.api = new ApiClient();
    this.toasts = new Toasts();

    this.nodes = {
      nav: document.getElementById("nav"),
      views: document.getElementById("views"),
      apiKey: document.getElementById("apiKey"),
      health: document.getElementById("health"),
    };

    this.chat = new ChatView(this.api, this.toasts);
    this.documents = new DocumentsView(this.api, this.toasts, {
      // The chat view's scope picker lists ready documents, so it has to hear
      // about an upload finishing even while the Documents tab is in front.
      onChange: () => this.chat.refreshDocuments(),
    });
    this.metrics = new MetricsView(this.api, this.toasts);

    this.tabs = [
      { id: "chat", label: "Chat", view: this.chat },
      { id: "documents", label: "Documents", view: this.documents },
      { id: "metrics", label: "Monitoring", view: this.metrics },
    ];
    this.current = null;
  }

  async start() {
    this.nodes.apiKey.value = this.api.apiKey;
    this.nodes.apiKey.addEventListener("change", () => {
      this.api.setKey(this.nodes.apiKey.value);
      this.toasts.ok("API key updated");
      if (this.current) this.current.view.activate();
    });

    this.renderNav();
    this.tabs.forEach((tab) => {
      tab.view.root.hidden = true;
      this.nodes.views.appendChild(tab.view.root);
    });

    this.checkHealth();
    setInterval(() => this.checkHealth(), HEALTH_INTERVAL_MS);

    await this.show(this.tabFromHash() || "chat");
    window.addEventListener("hashchange", () => {
      const id = this.tabFromHash();
      if (id && (!this.current || this.current.id !== id)) this.show(id);
    });
  }

  tabFromHash() {
    const id = window.location.hash.replace("#", "");
    return this.tabs.some((tab) => tab.id === id) ? id : null;
  }

  renderNav() {
    this.navButtons = new Map();
    const buttons = this.tabs.map((tab) => {
      const button = el("button", "tab", tab.label);
      button.type = "button";
      button.addEventListener("click", () => this.show(tab.id));
      this.navButtons.set(tab.id, button);
      return button;
    });
    this.nodes.nav.replaceChildren(...buttons);
  }

  async show(id) {
    const tab = this.tabs.find((candidate) => candidate.id === id);
    if (!tab) return;

    if (this.current) {
      this.current.view.deactivate();
      this.current.view.root.hidden = true;
    }
    this.navButtons.forEach((button, key) =>
      button.classList.toggle("on", key === id)
    );

    tab.view.root.hidden = false;
    this.current = tab;
    if (window.location.hash !== "#" + id) {
      // replaceState rather than assigning hash, so switching tabs does not
      // fill the back button with view changes.
      window.history.replaceState(null, "", "#" + id);
    }

    try {
      await tab.view.activate();
    } catch (error) {
      this.toasts.error(error.message);
    }
  }

  async checkHealth() {
    const node = this.nodes.health;
    try {
      const body = await this.api.health();
      const degraded = body.components.filter(
        (component) => component.status !== "up"
      );
      if (!degraded.length) {
        node.textContent = "healthy";
        node.className = "badge ok";
        node.title = "All dependencies up";
        return;
      }
      // A degraded component still serves, so it must not read as an outage.
      const down = degraded.some((component) => component.status === "down");
      node.textContent = down ? "unhealthy" : "degraded";
      node.className = "badge " + (down ? "err" : "warn");
      node.title = degraded
        .map((component) => component.name + ": " + component.status)
        .join("\n");
    } catch (error) {
      node.textContent = "unreachable";
      node.className = "badge err";
      node.title = "";
    }
  }
}

new AppShell().start();
