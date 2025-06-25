/**
 * @import { Evaluation, Scores, Analysis, ScoreNumbers, Terminology } from "./llm-evals"
 */

import { getElement, exposeAsGlobal, create } from "../utils.mjs";
import { terminology } from "./terminology.mjs";

const elements = {
  loading: getElement("loading"),
  error: getElement("error"),
  tbody: getElement("tbody"),
};

/**
 * @param {string} message
 */
function showError(message) {
  elements.error.style.display = "block";
  elements.error.innerText = message;
}

document.addEventListener("DOMContentLoaded", async () => {
  /** @type {Evaluation[]} */
  let evals;
  try {
    evals = await getEvals();
  } catch (error) {
    showError("Failed to get the evals. See the console for more information.");
    console.error(error);
    return;
  }
  elements.loading.style.display = "none";

  exposeAsGlobal("evals", evals);
  const analysis = analyzeEvals(evals);
  exposeAsGlobal("analysis", analysis);

  renderSummary(analysis);
});

/**
 * @returns {Promise<Evaluation[]>}
 */
async function getEvals() {
  const response = await fetch("llm-eval.json");
  return response.json();
}

/**
 * Compute mean, median, and histogram for evaluation scores.
 *
 * @param {Evaluation[]} evals
 * @returns {Record<keyof Scores, Analysis>}
 */
function analyzeEvals(evals) {
  /** @type {Record<keyof Scores, ScoreNumbers[]>} */
  const scoresByType = {
    adequacy: [],
    fluency: [],
    terminology: [],
    hallucination: [],
    punctuation: [],
  };

  // Group the scores by type.
  for (const evaluation of evals) {
    if (!evaluation.scores) {
      continue;
    }
    for (const key of /** @type {(keyof Scores)[]} */ (
      Object.keys(evaluation.scores)
    )) {
      const [score] = evaluation.scores[key];
      scoresByType[key].push(score);
    }
  }

  /**
   * @type {Record<keyof Scores, Analysis>}
   */
  return {
    adequacy: summarize(scoresByType.adequacy),
    fluency: summarize(scoresByType.fluency),
    terminology: summarize(scoresByType.terminology),
    hallucination: summarize(scoresByType.hallucination),
    punctuation: summarize(scoresByType.punctuation),
  };
}

/**
 * Summarize a numeric array.
 *
 * @param {ScoreNumbers[]} values
 * @returns {Analysis}
 */
function summarize(values) {
  /** @type {Record<ScoreNumbers, number>} */
  const histogram = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };

  for (const v of values) {
    histogram[v]++;
  }

  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const median =
    sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];

  const mean = values.reduce((sum, v) => sum + v, 0) / values.length;

  return { mean, median, histogram };
}

/**
 * @param {Record<keyof import('./llm-evals.js').Scores, {
 *   mean: number,
 *   median: number,
 *   histogram: Record<number, number>
 * }>} summary
 */
function renderSummary(summary) {
  for (const [scoreType, data] of Object.entries(summary)) {
    const { description, scales } = terminology[scoreType];
    create.tr({
      parent: elements.tbody,
      className: "criteria",
      children: [
        create.td({
          children: [
            //
            create.h3({ children: scoreType }),
            create.p({
              className: "criteraDescription",
              children: description,
            }),
          ],
        }),
        create.td({ children: `${data.mean.toFixed(2)}` }),
        create.td({
          children: [
            //
            data.median,
            " – ",
            scales[data.median],
          ],
        }),
        create.td({ children: createHistogram(data.histogram, scales) }),
      ],
    });
  }
}

/**
 * @param {Record<number, number>} histogram
 * @param {Record<ScoreNumbers, string>} scales - The description of the scales.
 */

function createHistogram(histogram, scales) {
  const labels = [1, 2, 3, 4, 5];
  const values = labels.map((k) => histogram[k]);
  const total = values.reduce((sum, v) => sum + v, 0);

  return create.div({
    className: "histogram",
    children: labels.map((label, index) => {
      const count = values[index];
      const freq = total > 0 ? count / total : 0;
      const score = /** @type {ScoreNumbers} */ (index + 1);
      const scoreDocumentation = scales[score];

      return create.div({
        className: "histogramBucket",
        title: `${score} – ${scoreDocumentation}`,
        children: [
          create.div({
            className: "histogramCount",
            children: count,
          }),
          create.div({
            className: "histogramBar",
            style: { height: `${freq * 100}px` },
          }),
          create.div({
            className: "histogramBucketNumber",
            children: `${label}`,
          }),
        ],
      });
    }),
  });
}
