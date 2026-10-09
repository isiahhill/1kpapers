// Selection rules for the rolling tracker. These pin the decisions that are
// easy to regress quietly: what counts as a "new model", how repeat Daily
// Papers rows collapse, and when the writer must refuse to overwrite.

import assert from "node:assert/strict";
import test from "node:test";
import {
  arxivIdsFromTags,
  fineTunedFrom,
  isDerivativeRepack,
  linkPapersToModels,
  mergeModels,
  mergePapers,
  toTrackedModel,
  toTrackedPaper,
  validateTrackerItems,
  type HfModelRow,
} from "./tracker.js";

const now = new Date("2026-10-09T12:00:00Z");

function model(overrides: Partial<HfModelRow> = {}): HfModelRow {
  return {
    id: "acme/Acme-7B",
    author: "acme",
    createdAt: "2026-10-01T00:00:00Z",
    likes: 10,
    downloads: 100,
    trendingScore: 50,
    pipeline_tag: "text-generation",
    tags: ["license:apache-2.0", "arxiv:2510.01234"],
    safetensors: { total: 7_000_000_000 },
    cardData: { license: "apache-2.0" },
    ...overrides,
  };
}

test("keeps a recent original release with its metadata", () => {
  const tracked = toTrackedModel(model(), now, 120);
  assert.ok(tracked);
  assert.equal(tracked.parameters, 7_000_000_000);
  assert.equal(tracked.license, "apache-2.0");
  assert.equal(tracked.derivedFrom, null);
  assert.deepEqual(tracked.arxivIds, ["2510.01234"]);
  assert.equal(tracked.url, "https://huggingface.co/acme/Acme-7B");
});

test("drops quantized, adapter and merge repacks but keeps fine-tunes", () => {
  for (const relation of ["quantized", "adapter", "merge"]) {
    const row = model({ tags: [`base_model:${relation}:acme/Acme-7B`] });
    assert.equal(isDerivativeRepack(row.tags), true);
    assert.equal(toTrackedModel(row, now, 120), null, relation);
  }
  const fineTune = toTrackedModel(model({ tags: ["base_model:finetune:Qwen/Qwen3-8B"] }), now, 120);
  assert.equal(fineTune?.derivedFrom, "Qwen/Qwen3-8B");
  assert.equal(fineTunedFrom(["base_model:Qwen/Qwen3-8B"]), null);
});

test("drops models outside the window, private repos and incomplete rows", () => {
  assert.equal(toTrackedModel(model({ createdAt: "2026-01-01T00:00:00Z" }), now, 120), null);
  assert.equal(toTrackedModel(model({ createdAt: "2026-12-01T00:00:00Z" }), now, 120), null);
  assert.equal(toTrackedModel(model({ private: true }), now, 120), null);
  const noPipeline = model();
  delete noPipeline.pipeline_tag;
  assert.equal(toTrackedModel(noPipeline, now, 120), null);
  assert.equal(toTrackedModel(model({ createdAt: "not a date" }), now, 120), null);
});

test("ignores malformed arXiv tags instead of throwing", () => {
  assert.deepEqual(arxivIdsFromTags(["arxiv:hep-th/9901001", "arxiv:2510.01234v2", "arxiv:2510.01234"]), ["2510.01234"]);
});

test("merges pipeline lists without duplicates, ordered by trending", () => {
  const a = toTrackedModel(model({ id: "a/one", trendingScore: 10 }), now, 120)!;
  const b = toTrackedModel(model({ id: "b/two", trendingScore: 90 }), now, 120)!;
  const merged = mergeModels([[a, b], [b]], 10);
  assert.deepEqual(merged.map((m) => m.id), ["b/two", "a/one"]);
  assert.equal(mergeModels([[a, b]], 1).length, 1);
});

function dailyRow(id: string, upvotes: number, publishedAt = "2026-10-05T00:00:00Z") {
  return { paper: { id, title: `Paper ${id}`, summary: "An abstract about reasoning agents.", publishedAt, upvotes, authors: [{ name: "A. Author" }] } };
}

test("collapses repeat Daily Papers rows to the highest upvote count", () => {
  const start = new Date("2026-09-09T00:00:00Z");
  const rows = [dailyRow("2510.00001", 5), dailyRow("2510.00001", 40), dailyRow("2510.00002", 20)];
  const papers = rows.map((row) => toTrackedPaper(row as never, start, now)).filter((p) => p !== null);
  const merged = mergePapers(papers, 10);
  assert.deepEqual(merged.map((p) => [p.arxivId, p.hfUpvotes]), [["2510.00001", 40], ["2510.00002", 20]]);
  assert.equal(merged[0]!.hfUrl, "https://huggingface.co/papers/2510.00001");
});

test("drops papers outside the window or with unusable ids", () => {
  const start = new Date("2026-09-09T00:00:00Z");
  assert.equal(toTrackedPaper(dailyRow("2401.00001", 9, "2024-01-02T00:00:00Z") as never, start, now), null);
  assert.equal(toTrackedPaper(dailyRow("not-an-id", 9) as never, start, now), null);
});

test("links a paper to tracked models whose cards cite it", () => {
  const start = new Date("2026-09-09T00:00:00Z");
  const paper = toTrackedPaper(dailyRow("2510.01234", 3) as never, start, now)!;
  const tracked = toTrackedModel(model(), now, 120)!;
  assert.deepEqual(linkPapersToModels([paper], [tracked])[0]!.linkedModelIds, ["acme/Acme-7B"]);
});

test("validator rejects thin or duplicated results so the writer keeps old data", () => {
  assert.deepEqual(validateTrackerItems(["a", "b", "c"], 2, (x) => x), []);
  assert.match(validateTrackerItems(["a"], 2, (x) => x).join(), /only 1 items/);
  assert.match(validateTrackerItems(["a", "a", "b"], 2, (x) => x).join(), /duplicate keys: a/);
});
