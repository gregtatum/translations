document.addEventListener("DOMContentLoaded", async () => {
  document.querySelectorAll("th").forEach((th, index) => {
    th.addEventListener("click", () => sortTable(index));
  });

  const trainingRuns = await loadTrainingRuns();
});

/**
 * Fetches JSON data from a given URL.
 *
 * @param {string} url
 * @returns {Promise<Object>}
 */
async function fetchJSON(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.statusText}`);
  }
  return response.json();
}
const modelsData = [];

/**
 * Fetches and displays the training runs list.
 * @returns {Promise<void>}
 */
async function loadTrainingRuns() {
  const trainingRunListing = await fetchJSON("training-runs-listing.json");
  const promises = trainingRunListing.map(async (filename) => {
    const modelData = await fetchJSON(`training-runs/${filename}`);
    appendModelToTable(modelData);
    return modelData;
  });
  const results = await Promise.allSettled(promises);
  const rejected = results
    .filter(({ status }) => status == "rejected")
    .map(({ reason }) => reason);
  const fulfilled = results
    .filter(({ status }) => status == "fulfilled")
    .map(({ value }) => value);
  if (rejected.length) {
    console.error("Some fetches failed", rejected);
  }
  return fulfilled;
}

/**
 * Appends a model entry to the table.
 * @param {Object} modelData
 */
function appendModelToTable(modelData) {
  const table = document.getElementById("model-table");
  const row = document.createElement("tr");

  row.innerHTML = `
    <td>${modelData.name}</td>
    <td>${modelData.langpair}</td>
    <td>${modelData.date_started}</td>
    <td>${(modelData.comet_flores_comparison || {}).google ?? "N/A"}</td>
    <td>${(modelData.bleu_flores_comparison || {}).google ?? "N/A"}</td>
    <td><a href="${
      modelData.parallel_corpus?.source_url
    }" target="_blank">Source</a></td>
    <td><a href="${
      modelData.parallel_corpus?.target_url
    }" target="_blank">Target</a></td>
  `;

  table.appendChild(row);
}

let prevColumnIndex = -1;
let prevDirection = 1;
/**
 * @param {number} columnIndex
 */
function sortTable(columnIndex) {
  const tableBody = document.getElementById("model-table");
  const rows = Array.from(tableBody.children);
  // Swap the direction on double clicks
  const direction = prevColumnIndex === columnIndex ? -prevDirection : 1;
  prevDirection = direction;
  prevColumnIndex = columnIndex;

  rows.sort((rowA, rowB) => {
    const valueA = rowA.querySelectorAll("td")[columnIndex].innerText;
    const valueB = rowB.querySelectorAll("td")[columnIndex].innerText;
    return String(valueA).localeCompare(String(valueB)) * direction;
  });

  // Re-appending puts this row at the bottom
  rows.forEach((row) => tableBody.appendChild(row));
}
