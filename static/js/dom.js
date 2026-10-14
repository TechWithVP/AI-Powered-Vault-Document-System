/**
 * Element construction and value formatting.
 *
 * Everything here builds nodes and sets textContent. No helper accepts or
 * emits an HTML string, so document text and model output cannot become
 * markup no matter what they contain.
 */

/** Creates an element, optionally with a class and text. */
export function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

/** Appends children, skipping nulls so callers can use inline conditionals. */
export function append(parent, ...children) {
  children.forEach((child) => {
    if (child) parent.appendChild(child);
  });
  return parent;
}

/** A labelled value, the building block of the metric panels. */
export function field(label, value) {
  const row = el("div", "field");
  append(row, el("span", "field-label", label), el("span", "field-value", value));
  return row;
}

/** A headline number with its caption. */
export function tile(label, value, hint) {
  const node = el("div", "tile");
  append(
    node,
    el("div", "tile-value", value),
    el("div", "tile-label", label),
    hint ? el("div", "tile-hint", hint) : null
  );
  return node;
}

/**
 * A table from column headers and row arrays.
 *
 * Metric breakdowns are mostly many small categories, which a table reads
 * better than any chart; this keeps them consistent.
 */
export function table(headers, rows, { numericFrom = 1 } = {}) {
  const node = el("table", "data");
  const head = el("thead");
  const headRow = el("tr");
  headers.forEach((header, index) => {
    const cell = el("th", index >= numericFrom ? "num" : null, header);
    headRow.appendChild(cell);
  });
  head.appendChild(headRow);
  node.appendChild(head);

  const body = el("tbody");
  rows.forEach((row) => {
    const tr = el("tr");
    row.forEach((value, index) => {
      tr.appendChild(el("td", index >= numericFrom ? "num" : null, value));
    });
    body.appendChild(tr);
  });
  node.appendChild(body);
  return node;
}

/** A titled panel, the shell every metric block sits in. */
export function panel(title, subtitle) {
  const node = el("section", "panel");
  const header = el("div", "panel-head");
  append(
    header,
    el("h3", null, title),
    subtitle ? el("p", "panel-sub", subtitle) : null
  );
  node.appendChild(header);
  return node;
}

/** Renders answer text, turning [n] markers into superscript references. */
export function answerBody(text, className) {
  const body = el("div", className || "body");
  const pattern = /\[(\d+)\]/g;
  let cursor = 0;
  let match = pattern.exec(text);

  while (match !== null) {
    if (match.index > cursor) {
      body.appendChild(document.createTextNode(text.slice(cursor, match.index)));
    }
    body.appendChild(el("span", "cite-ref", match[0]));
    cursor = match.index + match[0].length;
    match = pattern.exec(text);
  }
  if (cursor < text.length) {
    body.appendChild(document.createTextNode(text.slice(cursor)));
  }
  return body;
}

// -- formatting ------------------------------------------------------------ //

export const fmt = {
  plural(count, noun) {
    return count === 1 ? noun : noun + "s";
  },

  count(value) {
    return Number(value || 0).toLocaleString();
  },

  bytes(value) {
    const units = ["B", "KB", "MB", "GB"];
    let size = Number(value || 0);
    let unit = 0;
    while (size >= 1024 && unit < units.length - 1) {
      size /= 1024;
      unit += 1;
    }
    return (unit === 0 ? size : size.toFixed(1)) + " " + units[unit];
  },

  /** Durations switch to seconds past a second, where ms stop being readable. */
  duration(ms) {
    const value = Number(ms || 0);
    if (value < 1000) return Math.round(value) + " ms";
    return (value / 1000).toFixed(value < 10000 ? 2 : 1) + " s";
  },

  percent(fraction, digits = 0) {
    return (Number(fraction || 0) * 100).toFixed(digits) + "%";
  },

  /** Costs are shown to the cent below a dollar, where rounding hides them. */
  usd(value) {
    const amount = Number(value || 0);
    if (amount === 0) return "$0";
    if (amount < 1) return "$" + amount.toFixed(4);
    return "$" + amount.toFixed(2);
  },

  date(iso) {
    if (!iso) return "-";
    const parsed = new Date(iso);
    if (Number.isNaN(parsed.getTime())) return "-";
    return parsed.toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  },

  /** Relative time, which is what a session list is actually read for. */
  ago(iso) {
    if (!iso) return "";
    const then = new Date(iso).getTime();
    if (Number.isNaN(then)) return "";
    const seconds = Math.max(0, (Date.now() - then) / 1000);
    if (seconds < 60) return "just now";
    const minutes = seconds / 60;
    if (minutes < 60) return Math.floor(minutes) + "m ago";
    const hours = minutes / 60;
    if (hours < 24) return Math.floor(hours) + "h ago";
    const days = hours / 24;
    if (days < 7) return Math.floor(days) + "d ago";
    return fmt.date(iso);
  },
};
