/**
 * ComfyDL Help-menu semantics (runtime, independent of hashed frontend chunks).
 *
 * Why this exists: the Help link group was baked into a hashed upstream chunk
 * (Chinese-only labels) and our earlier cleanup lived inside the cdlapp poll.
 * Both were easy to miss when the browser reused a cached chunk / the poll
 * path changed.  This module watches the DOM itself and logs every action.
 *
 * - keeps only the ComfyDL-branded links; drops upstream items + What's New
 * - removes the decorative separator hairline below the link group
 * - rewrites the labels to the live Comfy.Locale (zh/en)
 * - NO z-index manipulation: the App dock lives at 1550, below the official
 *   1700 overlay band (see the layer map at the top of cdlapp.css).
 */
(() => {
  "use strict";

  const TAG = "[ComfyDL help-menu]";
  const ZH_TO_EN = {
    "项目文档": "Docs",
    "QQ 交流群": "QQ group",
    "GitHub 仓库": "GitHub repo",
    "Bilibili": "Bilibili",
  };

  function log(...args) {
    console.log(TAG, ...args);
  }

  function locale() {
    try {
      const value = window.comfyAPI?.app?.app?.extensionManager?.setting
        ?.get?.("Comfy.Locale") || "";
      return String(value).split("-")[0] || "en";
    } catch (e) { return "en"; }
  }

  function labelFor(raw) {
    const lang = locale();
    if (lang.startsWith("zh")) return raw;
    let out = raw;
    for (const [zh, en] of Object.entries(ZH_TO_EN)) {
      out = out.split(zh).join(en);
    }
    return out;
  }

  function tidy(popup) {
    const menu = popup.querySelector(".help-center-menu") || popup;
    const before = [...menu.querySelectorAll(".help-menu-item")].map((b) =>
      (b.textContent || "").trim());
    let removed = 0;
    menu.querySelectorAll("section").forEach((el) => { el.remove(); removed++; });
    menu.querySelectorAll("hr, [class*='divider'], [class*='separator'],"
      + " [class*='border-t']").forEach((el) => { el.remove(); removed++; });
    menu.querySelectorAll(".help-menu-item").forEach((btn) => {
      const raw = (btn.textContent || "").trim();
      if (!raw.startsWith("ComfyDL")) {
        btn.remove();
        removed++;
        return;
      }
      const wanted = labelFor(raw);
      for (const node of btn.childNodes) {
        if (node.nodeType === Node.TEXT_NODE && node.nodeValue.includes("ComfyDL")) {
          if (node.nodeValue.trim() !== wanted) node.nodeValue = wanted;
          break;
        }
      }
    });
    const after = [...menu.querySelectorAll(".help-menu-item")].map((b) =>
      (b.textContent || "").trim());
    log("tidy", {
      locale: locale(),
      removed,
      before,
      after,
      popupZ: getComputedStyle(popup).zIndex,
      rect: { w: Math.round(popup.getBoundingClientRect().width),
        h: Math.round(popup.getBoundingClientRect().height) },
    });
  }

  function scan() {
    const popup = document.querySelector(".help-center-popup");
    if (!popup || !popup.offsetWidth) return;
    if (popup.dataset.cdlHelped !== "1") {
      popup.dataset.cdlHelped = "1";
      log("popup detected (class/layer unchanged, semantics only)");
      tidy(popup);
    }
  }

  log("observer armed");
  const observer = new MutationObserver(scan);
  observer.observe(document.documentElement, { childList: true, subtree: true });
  window.setInterval(scan, 400);
})();
