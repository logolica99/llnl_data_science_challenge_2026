import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const projectRoot = new URL("../", import.meta.url);

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the orchestration demonstrator", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Part 2 NDE Orchestration Demonstrator<\/title>/i);
  assert.match(html, /Every stage earns access to the next\./);
  assert.match(html, /The orchestration and hash gates below are real\./);
  assert.match(html, /Production control-plane code is running live\./);
  assert.match(html, /Specialist outputs are deterministic fixtures/);
  assert.match(html, /Fixture specialists/);
  assert.match(html, /Live control-plane terminal/);
  assert.match(html, /One check is one legal state transition\./);
  assert.match(html, /Backend stdout mirror/);
  assert.match(html, /The verified handoff chain/);
  assert.match(html, /Labels move through narrow lanes/);
  assert.match(html, /What just happened/);
  assert.match(html, /http:\/\/localhost(?::3000)?\/og\.png/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|SkeletonPreview/);
});

test("ships the live local adapter and removes starter-only assets", async () => {
  const [page, layout, packageJson, viteConfig, server] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
    readFile(new URL("../vite.config.ts", import.meta.url), "utf8"),
    readFile(new URL("../server/demo_server.py", import.meta.url), "utf8"),
  ]);

  assert.match(page, /\/api\/v1\/demo-runs/);
  assert.match(page, /expectedManifestSha256/);
  assert.match(page, /terminalLines/);
  assert.match(page, /manual_review/);
  assert.match(page, /sealedEvaluationConsumed/);
  assert.match(layout, /Part 2 NDE Orchestration Demonstrator/);
  assert.match(packageJson, /"demo"/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
  assert.match(viteConfig, /127\.0\.0\.1:8765/);
  assert.match(server, /ThreadingHTTPServer\(\(HOST, PORT\)/);
  assert.match(server, /Cache-Control/);
  assert.match(server, /expectedManifestSha256/);
  assert.match(server, /terminalLines/);
  assert.match(server, /flush=True/);

  await assert.rejects(access(new URL("../app/_sites-preview", import.meta.url)));
  await assert.rejects(access(new URL("../public/favicon.svg", import.meta.url)));
  await access(new URL("../server/demo_pipeline.py", import.meta.url));
  await access(new URL("../tooling/sites-vite-plugin.ts", import.meta.url));
  await assert.rejects(
    access(new URL("../build/sites-vite-plugin.ts", import.meta.url)),
  );
  await access(projectRoot);
});
