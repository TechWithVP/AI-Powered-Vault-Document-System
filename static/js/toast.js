/**
 * Transient notices.
 *
 * Stacked rather than overlapping, because an upload followed by a failure
 * should show both messages instead of the second hiding the first.
 */

import { el } from "./dom.js";

const VISIBLE_MS = 4000;

export class Toasts {
  constructor() {
    this.root = el("div", "toasts");
    document.body.appendChild(this.root);
  }

  show(text, level = "info") {
    const node = el("div", "toast " + level, text);
    this.root.appendChild(node);
    setTimeout(() => node.remove(), VISIBLE_MS);
  }

  ok(text) {
    this.show(text, "ok");
  }

  error(text) {
    this.show(text, "err");
  }
}
