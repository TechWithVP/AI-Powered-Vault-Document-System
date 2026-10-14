/**
 * Documents: upload, browse, inspect AI insights, reprocess, delete.
 *
 * Insights are fetched when a document is opened rather than with the list.
 * The list endpoint deliberately omits them - a summary per row would make
 * browsing fifty documents a fifty-summary download.
 */

import { append, el, field, fmt, table } from "../dom.js";

const POLL_INTERVAL_MS = 2500;
const PENDING = ["queued", "processing"];

export class DocumentsView {
  constructor(api, toasts, { onChange } = {}) {
    this.api = api;
    this.toasts = toasts;
    this.onChange = onChange || (() => {});
    this.documents = [];
    this.openId = null;
    this.pollTimer = null;

    this.root = el("div", "view documents-view");
    this.build();
  }

  build() {
    this.fileInput = el("input");
    this.fileInput.type = "file";
    this.fileInput.accept = ".pdf,.docx,.txt,.md";
    this.fileInput.addEventListener("change", (event) => {
      const file = event.target.files[0];
      if (file) this.upload(file);
      event.target.value = "";
    });

    this.dropzone = el("label", "uploader");
    const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    icon.setAttribute("viewBox", "0 0 24 24");
    icon.setAttribute("class", "uploader-icon");
    icon.setAttribute("aria-hidden", "true");
    ["M12 16V4m0 0L8 8m4-4 4 4", "M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"].forEach(
      (d) => {
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", d);
        icon.appendChild(path);
      }
    );
    append(
      this.dropzone,
      this.fileInput,
      icon,
      el("span", "uploader-title", "Drop a file or click to choose"),
      el("span", "uploader-hint", "PDF, DOCX, TXT, MD · max 25 MB")
    );
    this.bindDropzone();

    this.search = el("input", "search");
    this.search.type = "search";
    this.search.placeholder = "Filter by name or summary";
    this.search.addEventListener("input", () => this.debouncedRefresh());

    this.statusFilter = el("select", "search");
    [
      ["", "Any status"],
      ["ready", "Ready"],
      ["processing", "Processing"],
      ["queued", "Queued"],
      ["failed", "Failed"],
    ].forEach(([value, label]) => {
      const option = el("option", null, label);
      option.value = value;
      this.statusFilter.appendChild(option);
    });
    this.statusFilter.addEventListener("change", () => this.refresh());

    const filters = el("div", "filter-row");
    append(filters, this.search, this.statusFilter);

    this.list = el("div", "doc-list");
    this.detail = el("aside", "detail");

    const left = el("section", "pane pane-pad");
    append(left, this.dropzone, filters, this.list);
    append(this.root, left, this.detail);
  }

  bindDropzone() {
    ["dragover", "dragenter"].forEach((name) =>
      this.dropzone.addEventListener(name, (event) => {
        event.preventDefault();
        this.dropzone.classList.add("drag");
      })
    );
    ["dragleave", "drop"].forEach((name) =>
      this.dropzone.addEventListener(name, () =>
        this.dropzone.classList.remove("drag")
      )
    );
    this.dropzone.addEventListener("drop", (event) => {
      event.preventDefault();
      const file = event.dataTransfer.files[0];
      if (file) this.upload(file);
    });
  }

  async activate() {
    await this.refresh();
    this.detail.replaceChildren(
      el("p", "muted", "Select a document to see its AI insights.")
    );
  }

  deactivate() {
    clearInterval(this.pollTimer);
    clearTimeout(this.searchTimer);
  }

  debouncedRefresh() {
    clearTimeout(this.searchTimer);
    this.searchTimer = setTimeout(() => this.refresh(), 250);
  }

  // -- data ---------------------------------------------------------------- //

  async refresh() {
    try {
      const page = await this.api.listDocuments({
        search: this.search.value.trim(),
        status: this.statusFilter.value,
      });
      this.documents = page.items;
      this.renderList(page.total);
      this.onChange(this.documents);
    } catch (error) {
      this.list.replaceChildren(el("p", "error-text", error.message));
    }
  }

  /** Polls while anything is still processing, then stops. */
  pollUntilReady() {
    clearInterval(this.pollTimer);
    this.pollTimer = setInterval(async () => {
      await this.refresh();
      if (!this.documents.some((doc) => PENDING.includes(doc.status))) {
        clearInterval(this.pollTimer);
      }
    }, POLL_INTERVAL_MS);
  }

  async upload(file) {
    try {
      const created = await this.api.upload(file);
      this.toasts.ok("Uploaded " + created.filename + " - processing");
      await this.refresh();
      this.pollUntilReady();
    } catch (error) {
      this.toasts.error(error.message);
    }
  }

  // -- list ---------------------------------------------------------------- //

  renderList(total) {
    if (!this.documents.length) {
      this.list.replaceChildren(
        el("p", "muted", "No documents match. Upload one above.")
      );
      return;
    }

    const header = el(
      "div",
      "list-count",
      this.documents.length +
        " of " +
        total +
        " " +
        fmt.plural(total, "document")
    );
    const cards = this.documents.map((doc) => this.card(doc));
    this.list.replaceChildren(header, ...cards);
  }

  card(doc) {
    const node = el("div", "doc" + (this.openId === doc.id ? " selected" : ""));

    const name = el("div", "doc-name");
    append(
      name,
      el("span", null, doc.filename),
      el("span", "status " + doc.status, doc.status)
    );

    const parts = [
      doc.file_type.toUpperCase(),
      fmt.bytes(doc.file_size_bytes),
    ];
    if (doc.page_count) parts.push(doc.page_count + "p");
    if (doc.chunk_count) parts.push(doc.chunk_count + " chunks");
    parts.push(fmt.ago(doc.created_at));

    append(node, name, el("div", "doc-meta", parts.join(" · ")));
    if (doc.failure_reason) {
      node.appendChild(el("div", "doc-meta error-text", doc.failure_reason));
    }

    node.addEventListener("click", () => this.open(doc));
    return node;
  }

  // -- detail -------------------------------------------------------------- //

  async open(doc) {
    this.openId = doc.id;
    this.renderList(this.documents.length);
    this.detail.replaceChildren(el("p", "muted", "Loading..."));

    try {
      const detail = await this.api.getDocument(doc.id);
      this.renderDetail(detail);
    } catch (error) {
      this.detail.replaceChildren(el("p", "error-text", error.message));
    }
  }

  renderDetail(doc) {
    const nodes = [];

    const head = el("div", "detail-head");
    append(
      head,
      el("h3", null, doc.filename),
      el("span", "status " + doc.status, doc.status)
    );
    nodes.push(head);

    const facts = el("div", "fields");
    append(
      facts,
      field("Type", doc.file_type.toUpperCase()),
      field("Size", fmt.bytes(doc.file_size_bytes)),
      field("Pages", doc.page_count || "-"),
      field("Chunks", doc.chunk_count || "-"),
      field("Characters", doc.character_count ? fmt.count(doc.character_count) : "-"),
      field("Uploaded", fmt.date(doc.created_at)),
      field("Processed", doc.processed_at ? fmt.date(doc.processed_at) : "-")
    );
    nodes.push(facts);

    if (doc.failure_reason) {
      nodes.push(el("p", "error-text", doc.failure_reason));
    }

    const insight = doc.insight;
    if (insight) {
      nodes.push(el("h4", null, "AI insights"));
      nodes.push(el("p", "summary", insight.summary));

      const tags = el("div", "chip-row");
      append(
        tags,
        el("span", "tag cat", insight.category),
        el("span", "tag", insight.sentiment),
        el("span", "tag", insight.language)
      );
      (insight.tags || []).forEach((tag) => tags.appendChild(el("span", "tag", tag)));
      nodes.push(tags);

      if (insight.key_points && insight.key_points.length) {
        nodes.push(el("h4", null, "Key points"));
        const list = el("ul", "points");
        insight.key_points.forEach((point) =>
          list.appendChild(el("li", null, point))
        );
        nodes.push(list);
      }

      if (insight.entities && insight.entities.length) {
        nodes.push(el("h4", null, "Entities"));
        nodes.push(
          table(
            ["Entity", "Type"],
            insight.entities.map((entity) => [entity.name, entity.type]),
            { numericFrom: 99 }
          )
        );
      }

      if (insight.was_truncated) {
        nodes.push(
          el(
            "p",
            "muted",
            "Insights were generated from the opening portion of a longer document."
          )
        );
      }
    } else if (doc.status === "ready") {
      nodes.push(el("p", "muted", "No insights were produced for this document."));
    }

    const actions = el("div", "detail-actions");
    const reprocess = el("button", "ghost", "Reprocess");
    reprocess.type = "button";
    reprocess.addEventListener("click", () => this.reprocess(doc));
    const remove = el("button", "danger", "Delete");
    remove.type = "button";
    remove.addEventListener("click", () => this.remove(doc));
    append(actions, reprocess, remove);
    nodes.push(actions);

    this.detail.replaceChildren(...nodes);
  }

  async reprocess(doc) {
    try {
      await this.api.reprocess(doc.id);
      this.toasts.ok("Reprocessing " + doc.filename);
      await this.refresh();
      this.pollUntilReady();
    } catch (error) {
      this.toasts.error(error.message);
    }
  }

  async remove(doc) {
    if (
      !window.confirm(
        "Delete " + doc.filename + "? Its chunks and insights go with it."
      )
    ) {
      return;
    }
    try {
      await this.api.deleteDocument(doc.id);
      this.toasts.ok("Deleted " + doc.filename);
      this.openId = null;
      this.detail.replaceChildren(
        el("p", "muted", "Select a document to see its AI insights.")
      );
      await this.refresh();
    } catch (error) {
      this.toasts.error(error.message);
    }
  }
}
