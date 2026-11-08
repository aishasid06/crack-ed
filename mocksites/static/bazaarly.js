// Tabs on item pages; the "Next" button navigates with JavaScript (there is no link to follow).
document.addEventListener("DOMContentLoaded", () => {
  const tabs = document.querySelectorAll('[role="tab"]');
  tabs.forEach((tab) => tab.addEventListener("click", () => {
    tabs.forEach((t) => {
      const on = t === tab;
      t.setAttribute("aria-selected", on ? "true" : "false");
      document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
    });
  }));
  document.querySelectorAll("[data-go]").forEach((b) =>
    b.addEventListener("click", () => { window.location.href = b.dataset.go; }));
});
