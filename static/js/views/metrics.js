/**
 * Monitoring: corpus, pipeline, cost, retrieval attribution and health.
 *
 * Form choices follow the data's job rather than habit. Headline numbers are
 * stat tiles, not one-bar charts. The many-small-categories breakdowns are
 * tables, which read better than any chart past a handful of classes. Only
 * two things here are genuinely graphical: the semantic/lexical split, which
 * is a part-to-whole of exactly two, and uploads over time.
 *
 * The two series colors are categorical slots 1 and 2, validated against this
 * app's dark surface (#1e222b): worst-pair CVD Delta E 26.8, normal-vision
 * 31.8, both clear of their floors. Segments carry direct labels and a legend
 * as well, so identity never rests on color alone.
 */

import { append, el, fmt, panel, table, tile } from "../dom.js";

export class MetricsView {
  constructor(api, toasts) {
    this.api = api;
    this.toasts = toasts;

    this.root = el("div", "view metrics-view");
    this.body = el("div", "metrics-body");
    this.tooltip = el("div", "viz-tooltip");
    this.tooltip.hidden = true;

    const head = el("div", "metrics-head");
    this.refreshButton = el("button", "ghost", "Refresh");
    this.refreshButton.type = "button";
    this.refreshButton.addEventListener("click", () => this.activate());
    append(head, el("h2", null, "Monitoring"), this.refreshButton);

    append(this.root, head, this.body, this.tooltip);
  }

  async activate() {
    this.body.replaceChildren(el("p", "muted", "Loading metrics..."));
    try {
      const [documents, processing, usage, retrieval, health] = await Promise.all([
        this.api.documentMetrics(),
        this.api.processingMetrics(),
        this.api.aiUsageMetrics(),
        this.api.retrievalMetrics(),
        this.api.health(),
      ]);
      this.render({ documents, processing, usage, retrieval, health });
    } catch (error) {
      this.body.replaceChildren(el("p", "error-text", error.message));
    }
  }

  deactivate() {
    this.hideTooltip();
  }

  render({ documents, processing, usage, retrieval, health }) {
    this.body.replaceChildren(
      this.kpis(documents, usage, retrieval),
      this.retrievalPanel(retrieval),
      this.processingPanel(processing),
      this.usagePanel(usage),
      this.corpusPanel(documents),
      this.healthPanel(health)
    );
  }

  // -- headline numbers ---------------------------------------------------- //

  kpis(documents, usage, retrieval) {
    const totals = documents.totals || {};
    const row = el("div", "tile-row");
    append(
      row,
      tile("Documents", fmt.count(totals.documents)),
      tile("Pages", fmt.count(totals.pages)),
      tile("Chunks", fmt.count(totals.chunks), fmt.count(totals.avg_chunks_per_document) + " per document"),
      tile("Stored", fmt.bytes(totals.storage_bytes)),
      tile("Searches", fmt.count(retrieval.summary.searches)),
      tile("AI spend", fmt.usd(usage.total.cost_usd), fmt.count(usage.total.calls) + " calls")
    );
    return row;
  }

  // -- retrieval attribution ----------------------------------------------- //

  retrievalPanel(retrieval) {
    const node = panel(
      "Retrieval attribution",
      "Share of each search's fused score contributed by vector similarity versus exact-term matching."
    );
    const summary = retrieval.summary;

    if (!summary.searches) {
      node.appendChild(
        el("p", "muted", "No searches recorded yet. Ask a question to populate this.")
      );
      return node;
    }

    node.appendChild(
      this.shareMeter([
        { label: "Vector", value: summary.semantic_share, slot: 1 },
        { label: "Exact term", value: summary.lexical_share, slot: 2 },
      ])
    );

    const chunks = el("div", "fields");
    append(
      chunks,
      this.chunkStat("Found by vector only", summary.chunks_from_semantic_only),
      this.chunkStat("Found by exact term only", summary.chunks_from_lexical_only),
      this.chunkStat("Found by both", summary.chunks_from_both)
    );
    node.appendChild(chunks);

    if (!summary.chunks_from_lexical_only) {
      node.appendChild(
        el(
          "p",
          "muted",
          "No chunk has been found by exact-term search alone. The lexical arm is confirming results rather than contributing any of its own - which is the line between a hybrid and a vector search with a confirmation step."
        )
      );
    }

    if (retrieval.by_question_kind.length) {
      node.appendChild(el("h4", null, "By question kind"));
      node.appendChild(
        table(
          ["Kind", "Searches", "Vector", "Exact term", "Answered"],
          retrieval.by_question_kind.map((row) => [
            row.question_kind,
            fmt.count(row.searches),
            fmt.percent(row.semantic_share),
            fmt.percent(row.lexical_share),
            fmt.percent(row.relevant_rate),
          ])
        )
      );
    }
    return node;
  }

  chunkStat(label, value) {
    const row = el("div", "field");
    append(row, el("span", "field-label", label), el("span", "field-value", fmt.count(value)));
    return row;
  }

  /**
   * A two-segment part-to-whole bar.
   *
   * Segments are separated by a surface-colored gap rather than a border, so
   * the boundary reads at any size, and each carries its own label so the
   * reader never has to match a color to a legend to know which is which.
   */
  shareMeter(segments) {
    const wrap = el("div", "meter-wrap");

    const bar = el("div", "meter");
    segments.forEach((segment) => {
      const fill = el("div", "meter-seg series-" + segment.slot);
      fill.style.flexGrow = String(Math.max(segment.value, 0.001));
      fill.setAttribute("role", "img");
      fill.setAttribute(
        "aria-label", segment.label + " " + fmt.percent(segment.value, 1)
      );
      this.attachTooltip(fill, segment.label + ": " + fmt.percent(segment.value, 1));
      bar.appendChild(fill);
    });

    const legend = el("div", "legend");
    segments.forEach((segment) => {
      const item = el("div", "legend-item");
      append(
        item,
        el("span", "swatch series-" + segment.slot),
        el("span", "legend-label", segment.label),
        el("span", "legend-value", fmt.percent(segment.value, 1))
      );
      legend.appendChild(item);
    });

    append(wrap, bar, legend);
    return wrap;
  }

  // -- pipeline ------------------------------------------------------------ //

  processingPanel(processing) {
    const node = panel(
      "Pipeline",
      "Per-stage timings across every document processed. Stage rows exist so a slow or failing stage is visible directly."
    );

    const row = el("div", "tile-row");
    append(
      row,
      tile("Runs", fmt.count(processing.total_runs)),
      tile("Failures", fmt.count(processing.total_failures)),
      tile("Success", fmt.percent(processing.overall_success_rate, 1)),
      tile("Slowest stage", processing.slowest_stage || "-")
    );
    node.appendChild(row);

    node.appendChild(
      table(
        ["Stage", "Runs", "Failed", "Average", "p95", "Max"],
        processing.stages.map((stage) => [
          stage.stage,
          fmt.count(stage.runs),
          fmt.count(stage.failed),
          fmt.duration(stage.avg_ms),
          fmt.duration(stage.p95_ms),
          fmt.duration(stage.max_ms),
        ])
      )
    );
    return node;
  }

  // -- cost ---------------------------------------------------------------- //

  usagePanel(usage) {
    const node = panel(
      "AI usage",
      "Every provider call is recorded as it happens, so cost is measured rather than estimated."
    );

    const row = el("div", "tile-row");
    append(
      row,
      tile("Calls", fmt.count(usage.total.calls)),
      tile("Tokens", fmt.count(usage.total.total_tokens)),
      tile("Cost", fmt.usd(usage.total.cost_usd))
    );
    node.appendChild(row);

    node.appendChild(
      table(
        ["Operation", "Model", "Calls", "Tokens", "Cost"],
        usage.by_operation.map((item) => [
          item.operation,
          item.model,
          fmt.count(item.calls),
          fmt.count(item.total_tokens),
          fmt.usd(item.cost_usd),
        ]),
        { numericFrom: 2 }
      )
    );
    return node;
  }

  // -- corpus -------------------------------------------------------------- //

  corpusPanel(documents) {
    const node = panel(
      "Corpus",
      "What has been uploaded, how it was classified, and when it arrived."
    );

    const grid = el("div", "breakdown-grid");
    append(
      grid,
      this.countBlock("By status", documents.by_status),
      this.countBlock("By file type", documents.by_file_type),
      this.countBlock("By category", documents.by_category)
    );
    node.appendChild(grid);

    if (documents.top_tags && documents.top_tags.length) {
      node.appendChild(el("h4", null, "Top tags"));
      const tags = el("div", "chip-row");
      documents.top_tags.forEach((item) => {
        tags.appendChild(el("span", "tag", item.tag + " · " + item.count));
      });
      node.appendChild(tags);
    }

    if (documents.uploads_by_day && documents.uploads_by_day.length) {
      node.appendChild(el("h4", null, "Uploads by day"));
      node.appendChild(this.columns(documents.uploads_by_day));
    }
    return node;
  }

  countBlock(title, counts) {
    const entries = Object.entries(counts || {});
    const node = el("div", "breakdown");
    node.appendChild(el("h4", null, title));
    if (!entries.length) {
      node.appendChild(el("p", "muted", "Nothing yet."));
      return node;
    }
    node.appendChild(
      table(
        ["Value", "Count"],
        entries
          .sort((left, right) => right[1] - left[1])
          .map(([label, count]) => [label, fmt.count(count)])
      )
    );
    return node;
  }

  /**
   * Uploads over time as columns.
   *
   * One series, so a single hue and no legend - the heading names it. Only the
   * busiest day is labelled directly; the rest are on hover, because a number
   * over every column is noise rather than information.
   */
  columns(points) {
    const peak = points.reduce(
      (highest, point) => Math.max(highest, point.count),
      0
    );
    const chart = el("div", "columns");

    points.forEach((point) => {
      const slot = el("div", "column");
      const track = el("div", "column-track");
      const bar = el("div", "column-bar");
      bar.style.height = peak ? (point.count / peak) * 100 + "%" : "0%";
      bar.setAttribute("role", "img");
      bar.setAttribute(
        "aria-label",
        point.count + " " + fmt.plural(point.count, "upload") + " on " + point.date
      );
      this.attachTooltip(
        bar,
        point.date + ": " + point.count + " " + fmt.plural(point.count, "upload")
      );

      if (point.count === peak && peak > 0) {
        track.appendChild(el("span", "column-label", point.count));
      }
      track.appendChild(bar);

      const label = point.date.slice(5);
      append(slot, track, el("div", "column-tick", label));
      chart.appendChild(slot);
    });
    return chart;
  }

  // -- health -------------------------------------------------------------- //

  healthPanel(health) {
    const node = panel(
      "Dependencies",
      "Version " + health.version + " · " + health.environment
    );
    node.appendChild(
      table(
        ["Component", "Status", "Latency", "Detail"],
        health.components.map((component) => [
          component.name,
          component.status,
          component.latency_ms === null ? "-" : fmt.duration(component.latency_ms),
          component.detail || "-",
        ]),
        { numericFrom: 2 }
      )
    );
    return node;
  }

  // -- tooltip ------------------------------------------------------------- //

  attachTooltip(node, text) {
    node.addEventListener("mouseenter", () => {
      this.tooltip.textContent = text;
      this.tooltip.hidden = false;
    });
    node.addEventListener("mousemove", (event) => {
      this.tooltip.style.left = event.clientX + "px";
      this.tooltip.style.top = event.clientY + "px";
    });
    node.addEventListener("mouseleave", () => this.hideTooltip());
  }

  hideTooltip() {
    this.tooltip.hidden = true;
  }
}
