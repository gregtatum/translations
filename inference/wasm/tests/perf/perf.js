import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import yargs from "yargs";
import { hideBin } from "yargs/helpers";
import { TranslationsEngine } from "../tests/engine/translations-engine";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

main();

function main() {
  const args = parseArgs();
  runPerfTest({
    sourceLanguage: args.src,
    targetLanguage: args.trg,
    taskGroupId: args.taskGroupId,
  });
}

const server = "https://firefox-ci-tc.services.mozilla.com";

export async function fetchTaskGroup(server, taskGroupId, updateStatusMessage) {
  const listUrl = `${server}/api/queue/v1/task-group/${taskGroupId}/list`;
  const response = await fetch(listUrl);

  if (!response.ok) {
    response.json().then((json) => console.error(json));
    return Promise.reject("Could not fetch task.");
  }

  /** @type {TaskGroup} */
  const taskGroup = await response.json();
  /** @type {TaskGroup} */
  let nextTaskGroup = taskGroup;

  // This API is paginated, continue through it.
  const maxPages = 20;
  for (let page = 0; page < maxPages; page++) {
    if (!nextTaskGroup.continuationToken) {
      // There are no more continuations.
      break;
    }
    updateStatusMessage(
      `Fetching Task Group ${taskGroupId} (page ${page + 2})`
    );
    const continuationUrl =
      listUrl + "?continuationToken=" + nextTaskGroup.continuationToken;
    console.log("Fetching next tasks for Task Group", continuationUrl);
    const response = await fetch(continuationUrl);
    if (!response.ok) {
      console.error("Failed to fetch a TaskGroup task continuation");
      break;
    }
    nextTaskGroup = await response.json();
    taskGroup.tasks = [...taskGroup.tasks, ...nextTaskGroup.tasks];
  }

  return taskGroup;
}

function fetchModelFiles(taskGroupId) {
  const taskGroup(taskGroupId);
}

function parseArgs() {
  return yargs(hideBin(process.argv))
    .option("src", {
      type: "string",
      demandOption: true,
      describe: "Source language",
    })
    .option("trg", {
      type: "string",
      demandOption: true,
      describe: "Target language",
    })
    .option("model", {
      type: "task_group_id",
      demandOption: true,
      describe: "Path to the model",
    })
    .help().argv;
}

async function runPerfTest({ sourceLanguage, targetLanguage, model }) {
  if (!(sourceLanguage === "en" || targetLanguage === "en")) {
    throw new Error("Pivot languages are not supported.");
  }

  const sourceFilePath = path.join(
    __dirname,
    "source",
    `${sourceLanguage}.txt`
  );
  const lines = getLines(sourceFilePath);

  const translator = new TranslationsEngine(
    sourceLanguage,
    targetLanguage,
    model
  );

  const translationPromises = [];
  const timer = startTimer();
  for (const line of lines) {
    translationPromises.push(translator.translate(line, false /* isHTML */));
  }

  const translations = await Promise.all(translationPromises);
  const endTimeMS = timer.end();

  const translationsDir = path.join(__dirname, "translations");
  mkdirp(translationsDir);

  const outputFilePath = path.join(translationsDir, `${sourceLanguage}.txt`);
  writeLines(outputFilePath, translations);

  console.log(`Translation completed in ${endTimeMS}ms`);

  translator.terminate();
}

function getLines(filePath) {
  if (!fs.existsSync(filePath)) {
    throw new Error(
      `Please provide a source translation file for that language: ${filePath}`
    );
  }
  return fs.readFileSync(filePath, "utf-8").split("\n").filter(Boolean);
}

function startTimer() {
  const start = Date.now();
  return {
    end: () => Date.now() - start,
  };
}

function mkdirp(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function writeLines(filePath, lines) {
  fs.writeFileSync(filePath, lines.join("\n"), "utf-8");
}

function name() {}
