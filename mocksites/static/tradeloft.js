// "Load more" pagination: fetches the next HTML fragment and appends it.
document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("tl-load-more");
  const list = document.getElementById("tl-results");
  if (!btn || !list) return;
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.textContent = "Loading…";
    const offset = btn.dataset.nextOffset;
    const res = await fetch(`/tradeloft/search/more?${list.dataset.query}&offset=${offset}`);
    const html = await res.text();
    const tmp = document.createElement("div");
    tmp.innerHTML = html;
    const state = tmp.querySelector(".tl-more-state");
    tmp.querySelectorAll(".tl-row").forEach((row) => list.appendChild(row));
    btn.dataset.nextOffset = state ? state.dataset.nextOffset : offset;
    btn.hidden = !state || state.dataset.hasMore !== "true";
    btn.disabled = false;
    btn.textContent = "Load more";
  });
});
