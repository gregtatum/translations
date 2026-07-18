// Firefox WebDriver harness for the wasm fxtranslate engine (build-order step 9).
//
// Launches HEADLESS Firefox via geckodriver + selenium-webdriver, serves the
// crate + repo over a tiny static HTTP server (fetch() + ES modules do not work
// over file://), loads web/index.html, and drives the in-page `window.fxrun`:
//
//   node webdriver.js parity [corpus]   line-by-line corpus, shortlist ON;
//                                        writes the browser output to stdout so
//                                        it diffs against the native / Node-wasm
//                                        reference (in-browser corpus parity).
//   node webdriver.js perf [corpus] [--runs N] [--warmup N]
//                                        host-timed words/s + TTFT + decode tok/s
//                                        via the SAME phase hook as js/perf.js,
//                                        timed in-page with performance.now();
//                                        shortlist OFF (production baseline).
//
// The wasm module served here is whatever is in ../pkg — build it first with
//   RUSTFLAGS="-C target-feature=+simd128" wasm-pack build --target web --no-default-features
// so backend() reports "wasm-simd128" (the representative browser number).
//
// Everything runs synchronously to completion with the driver quit in a finally,
// so there is no lingering browser/server process.

const fs = require("fs");
const path = require("path");
const http = require("http");
const { Builder } = require("selenium-webdriver");
const firefox = require("selenium-webdriver/firefox");

const CRATE_DIR = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(__dirname, "..", "..", "..", ".."); // /Users/.../translations
const INFERENCE_ROOT = path.resolve(__dirname, "..", "..", ".."); // /Users/.../translations/inference-rs
const PAGE_PATH = "/inference-rs/crates/fxtranslate-wasm/web/index.html";
const DEFAULT_CORPUS = path.join(INFERENCE_ROOT, "corpora", "nllb-en-fr.blocks.txt");

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".wasm": "application/wasm",
  ".spm": "application/octet-stream",
  ".bin": "application/octet-stream",
  ".txt": "text/plain; charset=utf-8",
};

// Serve the repo root read-only. Root-relative page paths (/data/models/enfr/…)
// and the page's own ../pkg/ glue both resolve under it.
function startServer() {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const urlPath = decodeURIComponent(req.url.split("?")[0]);
      const full = path.normalize(path.join(REPO_ROOT, urlPath));
      if (!full.startsWith(REPO_ROOT)) {
        res.writeHead(403).end();
        return;
      }
      fs.stat(full, (err, st) => {
        if (err || !st.isFile()) {
          res.writeHead(404).end();
          return;
        }
        res.writeHead(200, {
          "content-type": MIME[path.extname(full)] || "application/octet-stream",
          "content-length": st.size,
        });
        fs.createReadStream(full).pipe(res);
      });
    });
    server.listen(0, "127.0.0.1", () => resolve(server));
  });
}

function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}
function medIqr(xs) {
  const m = median(xs);
  if (xs.length < 2) return [m, xs[0], xs[0]];
  if (xs.length < 4) return [m, Math.min(...xs), Math.max(...xs)];
  const s = [...xs].sort((a, b) => a - b);
  const q = (p) => {
    const idx = p * (s.length + 1) - 1;
    const lo = Math.floor(idx), hi = Math.ceil(idx);
    if (lo < 0) return s[0];
    if (hi >= s.length) return s[s.length - 1];
    return s[lo] + (s[hi] - s[lo]) * (idx - lo);
  };
  return [m, q(0.25), q(0.75)];
}

async function main() {
  const args = process.argv.slice(2);
  const mode = args[0] || "perf";
  const getOpt = (name, dflt) => {
    const i = args.indexOf(name);
    return i >= 0 && args[i + 1] ? Number(args[i + 1]) : dflt;
  };
  const runs = getOpt("--runs", 5);
  const warmup = getOpt("--warmup", 1);
  const corpusPath =
    args.slice(1).find((a) => !a.startsWith("--") && !String(runs).endsWith(a) && !String(warmup).endsWith(a)) ||
    DEFAULT_CORPUS;
  const corpus = fs.readFileSync(corpusPath, "utf8");

  const server = await startServer();
  const port = server.address().port;
  const url = `http://127.0.0.1:${port}${PAGE_PATH}`;

  const opts = new firefox.Options().addArguments("-headless");
  // Point at the installed Firefox explicitly so geckodriver doesn't have to
  // guess (avoids "Failed to read marionette port" when the default lookup
  // misses). Override with FIREFOX_BIN if the app lives elsewhere.
  const fxBin =
    process.env.FIREFOX_BIN || "/Applications/Firefox.app/Contents/MacOS/firefox";
  if (fs.existsSync(fxBin)) opts.setBinary(fxBin);
  // The corpus can produce a large payload; give the async script room.
  // Drive geckodriver explicitly rather than via Selenium Manager's auto-lookup,
  // which times out on the marionette port here ("Failed to read marionette
  // port"). An explicit ServiceBuilder uses geckodriver from PATH directly.
  const service = new firefox.ServiceBuilder();
  let driver;
  try {
    driver = await new Builder()
      .forBrowser("firefox")
      .setFirefoxOptions(opts)
      .setFirefoxService(service)
      .build();
    const caps = await driver.getCapabilities();
    const fxVersion = caps.get("browserVersion") || caps.get("version") || "unknown";
    process.stderr.write(`Firefox ${fxVersion} (headless) | geckodriver via selenium-webdriver\n`);

    await driver.get(url);
    // Big model buffer + init; the async script itself awaits `ready`, so a
    // generous script timeout is the only safety needed.
    await driver.manage().setTimeouts({ script: 300000 });

    const payload = {
      mode,
      corpus,
      shortlist: true,
      runs,
      warmup,
    };
    // executeAsyncScript: the page's window.fxrun returns a promise; resolve the
    // WebDriver callback (last arg) with its result.
    const result = await driver.executeAsyncScript(
      "const cb = arguments[arguments.length - 1];" +
        "window.fxrun(arguments[0]).then(cb).catch(e => cb({ok:false, error:String(e)}));",
      payload
    );

    if (!result || !result.ok) {
      process.stderr.write(`in-page error: ${result && result.error}\n`);
      process.exitCode = 1;
      return;
    }

    if (mode === "parity") {
      process.stderr.write(`backend: ${result.backend} | lines: ${result.lines.length}\n`);
      process.stdout.write(result.lines.join("\n") + "\n");
    } else {
      const [wm, wlo, whi] = medIqr(result.wps);
      const [tm, tlo, thi] = medIqr(result.ttft);
      const [dm, dlo, dhi] = medIqr(result.tokps);
      process.stderr.write(`Firefox ${fxVersion} | UA: ${result.userAgent}\n`);
      process.stdout.write(
        `\nblocks: ${path.basename(corpusPath)} (${result.blocks} blocks, ${result.srcWords} source words) | ` +
          `1 thread | runs=${runs} warmup=${warmup} | shortlist=off\n` +
          `kernel: ${result.kernel} (wasm) | Firefox ${fxVersion} (headless, WebDriver)\n\n` +
          `words/s        median ${wm.toFixed(0)}   IQR ${wlo.toFixed(0)}-${whi.toFixed(0)}\n` +
          `TTFT (ms)      median ${tm.toFixed(1)}   IQR ${tlo.toFixed(1)}-${thi.toFixed(1)}\n` +
          `decode tok/s   median ${dm.toFixed(0)}   IQR ${dlo.toFixed(0)}-${dhi.toFixed(0)}\n` +
          `linear-memory high-water   ${(result.memHighBytes / 1048576).toFixed(1)} MiB\n`
      );
    }
  } finally {
    if (driver) await driver.quit();
    server.close();
  }
}

main().catch((e) => {
  process.stderr.write(String(e && e.stack ? e.stack : e) + "\n");
  process.exit(1);
});
