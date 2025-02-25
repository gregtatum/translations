// @ts-check
import {
  create,
  createTableRow,
  exposeAsGlobal,
  formatBytes,
  getElement,
  isNever,
  jsonToYAML,
  parseSearchQuery,
} from "../utils.mjs";

/**
 * @import {Corpus, TrainingRun, ModelRun, ModelReference} from '../@types/training-run.d.ts'
 */

const elements = {
  table: getElement("table", HTMLTableElement),
  tbody: getElement("table-body"),
  thead: getElement("table-thead"),
  tableContainer: getElement("table-container", HTMLDivElement),
  loading: getElement("loading", HTMLElement),
  error: getElement("error", HTMLElement),
  searchFilter: /** @type {HTMLInputElement} */ (
    getElement("search-filter", HTMLInputElement)
  ),
  overlay: getElement("overlay"),
  overlayCloseButton: getElement("overlay-close-button"),
  overlayContent: getElement("overlay-content"),
  scrollContainer: getElement("scroll-container"),
};

/**
 * @typedef {ReturnType<typeof StateManager.prototype.getURLState>} State
 */

/**
 * The state for the UI, which is serialized to the URL.
 */
class StateManager {
  /**
   * @type {State}
   */
  state = this.getURLState();

  constructor() {
    addEventListener("popstate", (event) => {
      this.state = event.state.state;
      this.updateUI();
    });
  }

  getURLState() {
    const urlParams = new URLSearchParams(window.location.search);

    /** @type {ModelReference | null} */
    let modelReference = null;
    {
      const name = urlParams.get("modelName");
      const langpair = urlParams.get("modelLangpair");
      const modelName = toModelName(urlParams.get("modelModelName"));

      if (name && langpair && modelName) {
        modelReference = {
          name,
          langpair,
          modelName,
        };
      }
    }

    return {
      searchString: urlParams.get("searchString") ?? "",
      showModels: urlParams.get("showModels") == "true" ? true : false,
      showCorpora: urlParams.get("showCorpora") == "true" ? true : false,
      modelReference,
    };
  }

  stateToURLParams() {
    const urlParams = new URLSearchParams();
    urlParams.set("searchString", this.state.searchString);
    if (this.state.showModels) {
      urlParams.set("showModels", "true");
    }
    if (this.state.showCorpora) {
      urlParams.set("showCorpora", "true");
    }
    if (this.state.modelReference) {
      urlParams.set("modelName", this.state.modelReference.name);
      urlParams.set("modelLangpair", this.state.modelReference.langpair);
      urlParams.set("modelModelName", this.state.modelReference.modelName);
    }

    return urlParams;
  }

  /**
   * Updates the state in place, but does not update the history or UI.
   *
   * @param {Partial<State>} partialState
   */
  replaceState(partialState) {
    this.state = {
      ...this.state,
      ...partialState,
    };
  }

  /**
   * Updates the state, URL history, and view.
   *
   * @param {Partial<State>} partialState
   */
  update(partialState) {
    this.replaceState(partialState);
    this.pushHistory();
    this.updateUI();
  }

  pushHistory() {
    const urlParams = this.stateToURLParams();
    const url = new URL(window.location.href);
    const newLocation = `${url.origin}${url.pathname}?${urlParams}`;
    history.pushState(stateManager, "", newLocation);
  }

  updateUI() {
    updateSearchFilterUI(this.state.searchString);
    updateModelView(this.state.modelReference);
    if (this.state.showModels) {
      elements.table.classList.add("show-models");
    } else {
      elements.table.classList.remove("show-models");
    }
    if (this.state.showCorpora) {
      elements.table.classList.add("show-corpora");
    } else {
      elements.table.classList.remove("show-corpora");
    }
    elements.searchFilter.value = this.state.searchString;
  }
}

const stateManager = new StateManager();
exposeAsGlobal("stateManager", stateManager);

/** @type {TrainingRun[] | null} */
let trainingRuns = null;

document.addEventListener("DOMContentLoaded", async () => {
  elements.table.querySelectorAll("th button").forEach((button, index) => {
    button.addEventListener("click", () => sortTable(index));
  });

  trainingRuns = await loadTrainingRuns();
  exposeAsGlobal("trainingRuns", trainingRuns);
  setupSearchFilter();
  setupOverlay();

  stateManager.updateUI();

  elements.tableContainer.style.display = "block";
  elements.loading.style.display = "none";
});

function setupSearchFilter() {
  elements.searchFilter.addEventListener("keyup", () => {
    updateSearchFilterUI(elements.searchFilter.value);
  });
  function pushSearchFilter() {
    stateManager.replaceState({ searchString: elements.searchFilter.value });
    stateManager.pushHistory();
  }
  elements.searchFilter.addEventListener("keyup", (event) => {
    if (event.key === "Enter") {
      pushSearchFilter();
    }
  });
  elements.searchFilter.addEventListener("blur", pushSearchFilter);
}

function setupOverlay() {
  function hideOverlay() {
    stateManager.update({ modelReference: null });
  }
  elements.overlayCloseButton.addEventListener("click", hideOverlay);
  document.body.addEventListener("keyup", (event) => {
    if (event.key === "Escape") {
      hideOverlay();
    }
  });
  elements.overlay.addEventListener("click", (event) => {
    if (event.target === elements.overlay) {
      hideOverlay();
    }
  });
}

/**
 * @param {string} search
 */
function updateSearchFilterUI(search) {
  search = search.trim();
  const trs = Array.from(elements.tbody.querySelectorAll("tr"));

  // Unhide everything.
  for (const tr of trs) {
    tr.style.display = "table-row";
  }

  if (!search.trim()) {
    // Nothing to search.
    return;
  }

  const { filters, terms } = parseSearchQuery(search);

  // Filter terms
  for (const tr of elements.tbody.querySelectorAll("tr")) {
    const rowText = tr.innerText.toLowerCase();
    for (const term of terms) {
      if (!rowText.includes(term)) {
        tr.style.display = "none";
        break;
      }
    }
  }

  for (const filter of filters) {
    // Find the table header
    if (!filter.key.match(/^[a-z-]+$/)) {
      continue;
    }
    const ths = elements.thead.querySelectorAll("th");

    let columnIndex = null;
    for (let i = 0; i < ths.length; i++) {
      if (ths[i].dataset.key === filter.key) {
        columnIndex = i;
        break;
      }
    }
    if (columnIndex === null) {
      continue;
    }

    for (const tr of trs) {
      const td = /** @type {HTMLElement} */ (tr.children[columnIndex]);
      const rowText = td.innerText.toLowerCase();
      if (filter.negated) {
        if (rowText.includes(filter.value)) {
          tr.style.display = "none";
        }
      } else if (!rowText.includes(filter.value)) {
        tr.style.display = "none";
      }
    }
  }
}

/**
 * @param {ModelReference | null} modelReference
 */
function updateModelView(modelReference) {
  if (!modelReference) {
    document.body.classList.remove("overlay-show");
    elements.scrollContainer.removeAttribute("inert");
    return;
  }
  if (!trainingRuns) {
    // The training runs aren't available yet.
    return;
  }

  if (document.body.classList.contains("overlay-show")) {
    // The model is already being shown.
    return;
  }

  const { name, langpair, modelName } = modelReference;

  const trainingRun = trainingRuns.find(
    (trainingRun) =>
      trainingRun.name === name && trainingRun.langpair == langpair
  );
  if (!trainingRun) {
    elements.error.style.display = "block";
    elements.error.innerText = `Could not find the model "${name}" (${langpair})`;
    return;
  }

  const modelRun = trainingRun[modelName];
  if (!modelRun) {
    elements.error.style.display = "block";
    elements.error.innerText = `That model couldn't be found for "${name}" (${langpair})`;
    return;
  }

  elements.overlayContent.innerText = "";

  // Create headers.
  create.h1({
    children: `${name} (${langpair})`,
    parent: elements.overlayContent,
  });
  create.h2({
    children: modelNameToLabel(modelName),
    parent: elements.overlayContent,
  });

  const detailsUL = create.ul({
    parent: elements.overlayContent,
  });

  const floresTbody = create.tbody();

  create.li({
    parent: detailsUL,
    children: [
      "Flores Evaluation",
      create.table({
        className: "flores-table",
        children: [
          create.thead({
            children: [
              create.tr({
                children: [
                  create.th({ children: "Metric" }),
                  create.th({ children: "Value" }),
                ],
              }),
            ],
          }),
          floresTbody,
        ],
      }),
    ],
  });

  /**
   * @param {string} metric
   * @param {string} value
   */
  const createMetricRow = (metric, value) => {
    create.tr({
      parent: floresTbody,
      children: [
        create.td({ children: metric }),
        create.td({ children: value ? value : "-" }),
      ],
    });
  };

  for (const metric of ["chrf", "bleu", "comet"]) {
    const value = modelRun.flores
      ? String(modelRun.flores[metric])
      : "Not available";
    createMetricRow(metric, value);
  }

  const googleFlores = getGoogleFloresCometScore(trainingRun, modelRun);
  if (googleFlores) {
    createMetricRow("comet (vs Google)", googleFlores.difference);
    createMetricRow("comet (Google)", googleFlores.score);
  } else {
    createMetricRow("Google Flores", "Not Available");
  }

  create.li({
    parent: detailsUL,
    children: `Date – ${modelRun.date.slice(0, "2025-01-01".length)}`,
  });

  // TaskGroup Link
  create.li({
    parent: detailsUL,
    children: [
      "TaskGroup – ",
      create.a({
        children: modelRun.task_group_id,
        href: `https://firefox-ci-tc.services.mozilla.com/tasks/groups/${modelRun.task_group_id}`,
      }),
    ],
  });

  create.li({
    parent: detailsUL,
    children: "Artifacts",
  });
  {
    const artifactsUL = create.ul({ parent: detailsUL });

    for (const url of modelRun.artifact_urls) {
      const urlParts = url.split("/");
      const fileName = urlParts[urlParts.length - 1];
      create.li({
        parent: artifactsUL,
        children: ["Artifact – ", create.a({ children: fileName, href: url })],
      });
    }
  }

  let headerGenerated = false;

  /**
   * @param {string} text
   */
  const createTrainingHeader = (text) => {
    if (!headerGenerated) {
      headerGenerated = true;
      create.h2({
        parent: elements.overlayContent,
        children: "Training Continuation",
      });
      create.p({
        parent: elements.overlayContent,
        children: [
          "Re-use this model in another training run. See the ",
          create.a({
            children: "training continuation docs",
            href: "../docs/training/using-pretrained-models/",
          }),
          " for more information.",
        ],
      });
    }

    create.h4({
      parent: elements.overlayContent,
      children: text,
    });
  };

  modelReference;

  switch (modelName) {
    case "backwards":
      createTrainingHeader("Back translation inference");
      create.pre({
        parent: elements.overlayContent,
        children: [
          "experiment:",
          "  pretrained-models:",
          `    # Use the ${langpair} model from the "${name}" training run for back translations.`,
          "    # See: https://mozilla.github.io/translations/docs/training/using-pretrained-models/",
          "    train-backwards:",
          "      urls:",
          "        - " + modelRun.artifact_folder,
          "      mode: use",
          "      type: default",
          "",
        ].join("\n"),
      });
      break;
    case "teacher_1":
    case "teacher_2":
      createTrainingHeader("Teacher distillation inference");
      create.pre({
        parent: elements.overlayContent,
        children: [
          "experiment:",
          "  pretrained-models:",
          `    # Use the existing ${langpair} model from the "${name}" training run.`,
          "    # See: https://mozilla.github.io/translations/docs/training/using-pretrained-models/",
          "    train-teacher:",
          "      urls:",
          "        - " + modelRun.artifact_folder,
          "      mode: use",
          "      type: default",
          "",
        ].join("\n"),
      });

      createTrainingHeader("Fine-tune the teacher");
      create.pre({
        parent: elements.overlayContent,
        children: [
          "experiment:",
          "  pretrained-models:",
          `    # Fine tune the ${langpair} model from the "${name}" training run.`,
          "    # See: https://mozilla.github.io/translations/docs/training/using-pretrained-models/",
          "    train-teacher:",
          "      urls:",
          "        - " + modelRun.artifact_folder,
          "      mode: continue",
          "      type: default",
          "",
        ].join("\n"),
      });
      break;
    case "student":
      createTrainingHeader("Back translation inference");
      create.pre({
        parent: elements.overlayContent,
        children: [
          "experiment:",
          "  pretrained-models:",
          `    # Use the ${langpair} model from the "${name}" training run for back translations.`,
          "    # See: https://mozilla.github.io/translations/docs/training/using-pretrained-models/",
          "    train-backwards:",
          "      urls:",
          "        - " + modelRun.artifact_folder,
          "      mode: use",
          "      type: default",
          "",
        ].join("\n"),
      });

      createTrainingHeader("Fine-tune the student");
      create.pre({
        parent: elements.overlayContent,
        children: [
          "experiment:",
          "  pretrained-models:",
          `    # Fine tune the ${langpair} model from the "${name}" training run.`,
          "    # See: https://mozilla.github.io/translations/docs/training/using-pretrained-models/",
          "    train-student:",
          "      urls:",
          "        - " + modelRun.artifact_folder,
          "      mode: continue",
          "      type: default",
          "",
        ].join("\n"),
      });

      createTrainingHeader("Run evaluations and export");
      create.pre({
        parent: elements.overlayContent,
        children: [
          "experiment:",
          "  pretrained-models:",
          `    # Use the existing ${langpair} model from the "${name}" training run.`,
          "    # See: https://mozilla.github.io/translations/docs/training/using-pretrained-models/",
          "    train-student:",
          "      urls:",
          "        - " + modelRun.artifact_folder,
          "      mode: use",
          "      type: default",
          "",
        ].join("\n"),
      });

    case "student_finetuned":
    case "student_quantized":
    case "student_exported":
      // These don't support training continuation.
      break;
    default:
      // Ensure every type of model is supported.
      isNever(modelName);
  }

  create.h2({
    parent: elements.overlayContent,
    children: "Training Config",
  });

  create.pre({
    parent: elements.overlayContent,
    children: jsonToYAML(modelRun.config),
  });

  elements.scrollContainer.setAttribute("inert", "");
  document.body.classList.add("overlay-show");
}

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

/**
 * Fetches and displays the training runs list.
 * @returns {Promise<TrainingRun[]>}
 */
async function loadTrainingRuns() {
  const trainingRunListing = await fetchJSON("training-runs-listing.json");
  const promises = trainingRunListing.map(async (filename) => {
    /** @type {TrainingRun} */
    const trainingRun = await fetchJSON(`training-runs/${filename}`);
    try {
      buildTrainingRunRow(trainingRun);
    } catch (error) {
      elements.error.style.display = "block";
      elements.error.innerText = "Error building training run row.";
      console.error(error);
    }
    return trainingRun;
  });
  const results = await Promise.allSettled(promises);
  const rejected = results
    .filter(({ status }) => status == "rejected")
    // @ts-expect-error - Not sure why the allSettled disagrees.
    .map(({ reason }) => reason);
  const fulfilled = results
    .filter(({ status }) => status == "fulfilled")
    // @ts-expect-error - Not sure why the allSettled disagrees.
    .map(({ value }) => value);
  if (rejected.length) {
    console.error("Some fetches failed", rejected);
  }
  return fulfilled;
}

const displayName = new Intl.DisplayNames("en", { type: "language" });

/**
 * @param {TrainingRun} trainingRun
 */
function buildTrainingRunRow(trainingRun) {
  const { tr } = createTableRow(elements.tbody);

  const languageTag =
    trainingRun.source_lang === "en"
      ? trainingRun.target_lang
      : trainingRun.source_lang;

  /**
   * @param {string} key
   * @param {string} value
   */
  function createFilterableButton(key, value) {
    create.td({
      parent: tr,
      children: create.button({
        className: "button-text",
        children: value,
        onClick() {
          elements.searchFilter.value = value.includes(" ")
            ? `${key}:"${value}"`
            : `${key}:${value}`;

          updateSearchFilterUI(elements.searchFilter.value);
          stateManager.replaceState({
            searchString: elements.searchFilter.value,
          });
          stateManager.pushHistory();
        },
      }),
    });
  }

  createFilterableButton("name", trainingRun.name);
  createFilterableButton(
    "language",
    displayName.of(languageTag) ?? languageTag
  );
  createFilterableButton("langpair", trainingRun.langpair);
  create.td({
    parent: tr,
    children: (trainingRun.date_started ?? "–").slice(0, "2025-01-01".length),
  });

  {
    // Create the button to show models.
    create.td({
      parent: tr,
      children: create.button({
        onClick() {
          stateManager.update({ showModels: !stateManager.state.showModels });
        },
        children: [
          create.span({
            className: "toggle-models-show",
            children: "Show",
          }),
          create.span({
            className: "toggle-models-hide",
            children: "Hide",
          }),
        ],
      }),
    });
  }

  /**
   * @param {ModelReference["modelName"]} modelName
   */
  function createModelLink(modelName) {
    const div = document.createElement("div");

    const modelRun = trainingRun[modelName];
    const googleFlores = getGoogleFloresCometScore(trainingRun, modelRun);

    /** @type {Partial<CSSStyleDeclaration>} */
    const style = {};
    if (googleFlores && googleFlores.percentage < -5) {
      // Does not meet release criteria.
      style.background = "#ffa537";
    }

    create.td({
      parent: tr,
      className: "models-td",
      style,
      children: create.div({
        children: !trainingRun[modelName]
          ? "–"
          : create.button({
              parent: div,
              children: googleFlores ? googleFlores.difference : "view",
              className: "button-text",
              onClick() {
                stateManager.update({
                  modelReference: {
                    name: trainingRun.name,
                    langpair: trainingRun.langpair,
                    modelName,
                  },
                });
              },
            }),
      }),
    });
  }

  createModelLink("backwards");
  createModelLink("teacher_1");
  createModelLink("teacher_2");
  createModelLink("student");
  createModelLink("student_finetuned");
  createModelLink("student_quantized");
  createModelLink("student_exported");

  {
    // Create the button to show corpora.
    create.td({
      parent: tr,
      children: create.button({
        onClick() {
          stateManager.update({
            showCorpora: !stateManager.state.showCorpora,
          });
        },
        children: [
          create.span({
            className: "toggle-corpora-show",
            children: "Show",
          }),

          create.span({
            className: "toggle-corpora-hide",
            children: "Hide",
          }),
        ],
      }),
    });
  }

  /**
   * @param {Corpus} [corpus]
   */
  function createCorpusLink(corpus) {
    const div = document.createElement("div");
    create.td({
      parent: tr,
      className: "corpus-td",
      children: create.div({
        children: corpus
          ? [
              create.a({
                children: trainingRun.source_lang,
                href: corpus.source_url,
                title: formatBytes(corpus.source_bytes),
                parent: div,
              }),
              create.a({
                children: trainingRun.target_lang,
                href: corpus.target_url,
                title: formatBytes(corpus.target_bytes),
                parent: div,
              }),
            ]
          : "–",
      }),
    });
  }
  createCorpusLink(trainingRun.parallel_corpus_aligned);
  createCorpusLink(trainingRun.backtranslations_corpus_aligned);
  createCorpusLink(trainingRun.distillation_corpus_aligned);
  createCorpusLink(trainingRun.parallel_corpus);
  createCorpusLink(trainingRun.backtranslations_corpus);
  createCorpusLink(trainingRun.distillation_corpus);

  create.td({
    parent: tr,
    children: create.button({
      children: "log",
      title: "View this run in the console.log",
      onClick() {
        alert("View this run in the console.log");
        console.log(trainingRun.name, trainingRun.langpair);
        console.log(trainingRun);
      },
    }),
  });
}

let prevColumnIndex = -1;
let prevDirection = 1;

/**
 * @param {number} columnIndex
 */
function sortTable(columnIndex) {
  const rows = Array.from(elements.tbody.children);
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
  rows.forEach((row) => elements.tbody.appendChild(row));
}

/**
 * @param {string | null | undefined} modelName
 * @returns {ModelReference["modelName"] | null}
 */
function toModelName(modelName) {
  switch (modelName) {
    case "backwards":
    case "teacher_1":
    case "teacher_2":
    case "student":
    case "student_finetuned":
    case "student_quantized":
    case "student_exported":
      return modelName;
    default:
      return null;
  }
}

/**
 * @param {ModelReference["modelName"]} modelName
 */
function modelNameToLabel(modelName) {
  switch (modelName) {
    case "backwards":
      return "Backwards";
    case "teacher_1":
      return "Teacher 1";
    case "teacher_2":
      return "Teacher 2";
    case "student":
      return "Student";
    case "student_finetuned":
      return "Student Finetuned";
    case "student_quantized":
      return "Student Quantized";
    case "student_exported":
      return "Student Exported";
    default:
      isNever(modelName);
      throw new Error("Could not convert model name to label: " + modelName);
  }
}

/**
 * Get Google comet display
 * @param {TrainingRun} trainingRun
 * @param {ModelRun} [modelRun]
 */
function getGoogleFloresCometScore(trainingRun, modelRun) {
  const googleFlores = trainingRun.comet_flores_comparison.google;
  if (!googleFlores || !modelRun?.flores?.comet) {
    return null;
  }
  const percentage = 100 * (1 - googleFlores / (modelRun?.flores.comet / 100));
  const sign = percentage >= 0 ? "+" : "";
  return {
    percentage,
    difference: `${sign}${percentage.toFixed(2)}`,
    score: `${(googleFlores * 100).toFixed(2)}`,
  };
}
