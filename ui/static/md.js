/* Shared between the main app (index.html) and the public share viewer
 * (share.html): minimal markdown renderer, HTML escaping, clipboard helper
 * with a plain-http fallback, and the code-block copy-button delegation. */

"use strict";

async function copyText(text) {
  // navigator.clipboard needs a secure context; this UI is often served over
  // plain HTTP on a LAN/VPN domain, so fall back to the legacy path.
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.cssText = "position:fixed;opacity:0";
    document.body.appendChild(ta);
    ta.select();
    let ok = false;
    try { ok = document.execCommand("copy"); } catch {}
    ta.remove();
    return ok;
  }
}

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;");
}

function renderMarkdown(src) {
  // NUL delimits our code-block placeholders below and is never legitimate
  // chat text — strip it so crafted input can't forge a placeholder.
  src = src.replace(/\u0000/g, "");
  // 1. pull fenced code blocks out so nothing inside them is transformed.
  // Line-based scan: a fence only opens/closes at the start of a line, so
  // adjacent or nested fences in model output (```markdown containing ```py)
  // can't bleed code into the prose the way a lazy regex does.
  const codes = [];
  const kept = [];
  let fence = null;   // {lang, lines} while inside a block
  for (const line of src.split("\n")) {
    const open = fence === null && line.match(/^\s*```(\S*)\s*$/);
    if (open) {
      fence = { lang: open[1], lines: [] };
    } else if (fence !== null && /^\s*```\s*$/.test(line)) {
      codes.push({ lang: fence.lang, code: fence.lines.join("\n") });
      kept.push(`\u0000CODE${codes.length - 1}\u0000`);
      fence = null;
    } else if (fence !== null) {
      fence.lines.push(line);
    } else {
      kept.push(line);
    }
  }
  // Unterminated fence (mid-stream, or a stray trailing ``` from the model):
  // render its content as code, but drop it entirely while still empty so a
  // lone closing fence never leaves an empty box behind.
  if (fence !== null && fence.lines.length) {
    codes.push({ lang: fence.lang, code: fence.lines.join("\n") });
    kept.push(`\u0000CODE${codes.length - 1}\u0000`);
  }
  let html = escapeHtml(kept.join("\n"));

  // tables (| a | b | with a |---| separator row)
  html = html.replace(
    /(^\|.+\|\n\|[\s\-:|]+\|\n(?:\|.+\|\n?)*)/gm,
    (block) => {
      const rows = block.trim().split("\n").map((r) =>
        r.replace(/^\||\|$/g, "").split("|").map((c) => c.trim()));
      const head = rows.shift(); rows.shift(); // header + separator
      const tr = (cells, tag) =>
        `<tr>${cells.map((c) => `<${tag}>${c}</${tag}>`).join("")}</tr>`;
      return `<table>${tr(head, "th")}${rows.map((r) => tr(r, "td")).join("")}</table>`;
    });

  html = html
    .replace(/^###+ (.*)$/gm, "<h3>$1</h3>")
    .replace(/^## (.*)$/gm, "<h2>$1</h2>")
    .replace(/^# (.*)$/gm, "<h1>$1</h1>")
    .replace(/^&gt; ?(.*)$/gm, "<blockquote>$1</blockquote>")
    .replace(/^(?:-{3,}|\*{3,})$/gm, "<hr>")
    .replace(/^[*\-] (.*)$/gm, "<li>$1</li>")
    .replace(/^\d+\. (.*)$/gm, "<li data-ol>$1</li>")
    .replace(/(<li data-ol>[\s\S]*?<\/li>)(?!\n<li data-ol)/g, "<ol>$1</ol>")
    .replace(/(<li>(?:(?!<li data-ol)[\s\S])*?<\/li>)(?!\n<li>)/g, "<ul>$1</ul>")
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
             '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/\n{2,}/g, "</p><p>")
    .replace(/\n/g, "<br>");
  html = `<p>${html}</p>`;

  return html.replace(/\u0000CODE(\d+)\u0000/g, (_, i) => {
    const { lang, code } = codes[+i];
    return `<pre><button class="btn ghost small-btn copy-code">Copy</button>` +
           `<code data-lang="${escapeHtml(lang)}">${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`;
  });
}

document.addEventListener("click", (e) => {
  if (e.target.classList && e.target.classList.contains("copy-code")) {
    copyText(e.target.nextElementSibling.textContent);
    e.target.textContent = "Copied!";
    setTimeout(() => { e.target.textContent = "Copy"; }, 1200);
  }
});
