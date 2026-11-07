// Show/hide conditional fieldsets (data-show-when="field=value|value") and the "add another listing" rows.
document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("step-form");
  if (!form) return;
  const conds = form.querySelectorAll("[data-show-when]");
  const update = () => {
    conds.forEach((el) => {
      const [field, values] = el.dataset.showWhen.split("=");
      const checked = form.querySelector(`[name="${field}"]:checked`);
      const show = !!checked && values.split("|").includes(checked.value);
      el.hidden = !show;
      el.querySelectorAll("input,select,textarea").forEach((i) => (i.disabled = !show));
    });
  };
  form.addEventListener("change", update);
  update();

  const rows = document.getElementById("url-rows");
  const add = document.getElementById("add-url");
  if (rows && add) {
    const renumber = () => rows.querySelectorAll(".pt-url-row legend").forEach((l, i) => (l.textContent = `Listing ${i + 1}`));
    add.addEventListener("click", () => {
      const clone = rows.querySelector(".pt-url-row").cloneNode(true);
      clone.querySelectorAll("input,textarea").forEach((i) => (i.value = ""));
      clone.querySelectorAll("select").forEach((s) => (s.selectedIndex = 0));
      clone.querySelectorAll(".pt-field-error").forEach((e) => e.remove());
      rows.appendChild(clone);
      renumber();
    });
    rows.addEventListener("click", (ev) => {
      if (ev.target.classList.contains("pt-remove-row") && rows.querySelectorAll(".pt-url-row").length > 1) {
        ev.target.closest(".pt-url-row").remove();
        renumber();
      }
    });
  }
});
