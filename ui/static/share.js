/* Public read-only viewer for a shared chat.
 * Fetches /api/share/<id>/data (the unguessable id in the URL is the only
 * credential) and renders messages with the same markdown renderer as the
 * app (md.js). While the share is live, polls every 3s and follows new
 * content; when the owner ends live mode, polling stops. */

"use strict";

const shareId = location.pathname.split("/").pop();
const $s = (id) => document.getElementById(id);
let lastUpdated = 0;

const fmtT = (ts) => ts ? new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "";

function renderShared(data) {
  document.title = `${data.title} · vllm-ui`;
  $s("share-title").textContent = data.title;
  const pill = $s("live-pill");
  pill.hidden = !data.live;
  pill.className = data.live ? "pill pill-ok" : "pill pill-off";
  $s("share-status").hidden = true;

  const box = $s("chat-messages");
  const stick = box.scrollHeight - box.scrollTop - box.clientHeight < 140;
  const open = new Set();
  box.querySelectorAll("details[data-fold]").forEach((d) => { if (d.open) open.add(d.dataset.fold); });
  box.textContent = "";

  for (const m of data.messages || []) {
    const wrap = document.createElement("div");
    wrap.className = `msg ${m.role === "user" ? "user" : "assistant"}`;
    const who = document.createElement("div");
    who.className = "who";
    const name = document.createElement("span");
    name.textContent = m.role === "user" ? "User" : (m.model || "Assistant");
    who.appendChild(name);
    if (m.ts) {
      const t = document.createElement("span");
      t.className = "msg-meta";
      t.textContent = fmtT(m.ts);
      who.appendChild(t);
    }
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    if (m.role === "user") {
      bubble.textContent = m.text || "";
      if (m.files && m.files.length) {
        const files = document.createElement("div");
        files.className = "msg-attachments";
        for (const f of m.files) {
          const chip = document.createElement("span");
          chip.className = "attach-chip";
          if (f.kind === "image" && typeof f.dataurl === "string" &&
              /^data:image\/(png|jpeg|webp|gif);base64,[A-Za-z0-9+/=]+$/.test(f.dataurl)) {
            const img = document.createElement("img");
            img.src = f.dataurl;
            img.alt = f.name || "attachment";
            chip.appendChild(img);
          }
          const nm = document.createElement("span");
          nm.textContent = f.name || "file";
          chip.appendChild(nm);
          files.appendChild(chip);
        }
        wrap.appendChild(files);
      }
    } else {
      // browsing / python activity (debug stripped by the sharer)
      const events = m.browsing || [];
      const starts = events.filter((e) => e.event === "tool_start");
      const results = events.filter((e) => e.event === "tool_result");
      if (starts.length) {
        const act = document.createElement("div");
        act.className = "browse-activity";
        starts.forEach((e, i) => {
          const row = document.createElement("div");
          row.className = "browse-row";
          const what = document.createElement("span");
          what.className = "browse-what";
          what.textContent = e.tool === "web_search"
            ? `Searched: ${e.args?.query ?? ""}`
            : e.tool === "run_python" ? "Ran Python"
            : `Fetched: ${e.args?.url ?? ""}`;
          const status = document.createElement("span");
          status.className = "browse-status";
          const res = results[i];
          status.textContent = res ? String(res.preview || "") : "…";
          row.append(what, status);
          act.appendChild(row);
          if (e.tool === "run_python" && e.args?.code) {
            const det = document.createElement("details");
            det.className = "browse-debug";
            det.dataset.fold = `sc${m.ts || 0}-${i}`;
            const sum = document.createElement("summary");
            sum.textContent = "Code";
            const pre = document.createElement("pre");
            pre.className = "log small";
            pre.textContent = e.args.code;
            det.append(sum, pre);
            act.appendChild(det);
          }
          if (res && Array.isArray(res.images)) {
            const figs = document.createElement("div");
            figs.className = "browse-figures";
            for (const src of res.images) {
              if (typeof src !== "string" ||
                  !/^data:image\/png;base64,[A-Za-z0-9+/=]+$/.test(src)) continue;
              const img = document.createElement("img");
              img.src = src;
              img.alt = "generated figure";
              figs.appendChild(img);
            }
            act.appendChild(figs);
          }
        });
        bubble.appendChild(act);
      }
      if (m.reasoning) {
        const det = document.createElement("details");
        det.className = "reasoning";
        det.dataset.fold = `sr${m.ts || 0}`;
        const sum = document.createElement("summary");
        sum.textContent = "Reasoning";
        const body = document.createElement("div");
        body.className = "r-body";
        body.textContent = m.reasoning;
        det.append(sum, body);
        bubble.appendChild(det);
      }
      const content = document.createElement("div");
      content.innerHTML = renderMarkdown(m.text || "");
      bubble.appendChild(content);
    }
    wrap.append(who, bubble);
    box.appendChild(wrap);
  }
  box.querySelectorAll("details[data-fold]").forEach((d) => {
    if (open.has(d.dataset.fold)) d.open = true;
  });
  if (stick) box.scrollTop = box.scrollHeight;
}

async function poll(first = false) {
  let r;
  try {
    r = await fetch(`/api/share/${encodeURIComponent(shareId)}/data`);
  } catch {
    setTimeout(poll, 5000);
    return;
  }
  if (r.status === 404) {
    $s("share-status").hidden = false;
    $s("share-status").querySelector("h2").textContent = "Share not found";
    $s("share-status").querySelector("p").textContent =
      "This link is invalid or the share was revoked.";
    return;
  }
  const data = await r.json();
  if (first || data.updated !== lastUpdated) {
    lastUpdated = data.updated;
    renderShared(data);
  }
  if (data.live) setTimeout(poll, 3000);
}

poll(true);
