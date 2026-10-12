import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import type { AddressInfo } from "node:net";
import { inferEmotionFromTyping } from "../src/services/emotionEngine.js";
import {
  evaluateTree,
  extractTypingFeatures,
  inferEmotion,
  softmax,
  TYPING_FEATURE_NAMES,
  validateModelArtifact,
  XGBoostEmotionClassifier,
  type XGBoostModelArtifact,
  type XGBTree,
} from "../src/services/xgbEmotionClassifier.js";

// Temporary isolated database for integration testing
process.env.NODE_ENV = "test";
const tempDbDir = fs.mkdtempSync(path.join(os.tmpdir(), "thinkflow-xgb-test-"));
const tempDbPath = path.join(tempDbDir, "thinkflow-xgb-test-db.json");
process.env.THINKFLOW_DB_PATH = process.env.THINKFLOW_DB_PATH ?? tempDbPath;

const { createApp } = await import("../src/server.js");
const { db } = await import("../src/storage/db.js");

test("XGBoost Emotion Classifier Suite", async (t) => {
  await t.test("1. Feature mapping & ratio math extracts all 10 features with safe edge-case handling", () => {
    const standardInput = {
      charsTyped: 300,
      wordsTyped: 60,
      backspaces: 15,
      errorCount: 3,
      pauseMs: 15000,
      durationMs: 60000,
    };

    const { features, values } = extractTypingFeatures(standardInput);

    assert.equal(values.length, 10);
    assert.equal(values.length, TYPING_FEATURE_NAMES.length);

    assert.equal(features.typingSpeedWpm, 60.0);
    assert.equal(features.backspaceRatio, 0.05); // 15 / 300
    assert.equal(features.errorRate, 0.05); // 3 / 60
    assert.equal(features.pauseRatio, 0.25); // 15000 / 60000
    assert.equal(features.charsTyped, 300);
    assert.equal(features.wordsTyped, 60);
    assert.equal(features.backspaces, 15);
    assert.equal(features.errorCount, 3);
    assert.equal(features.pauseMs, 15000);
    assert.equal(features.durationMs, 60000);

    const zeroInput = {
      charsTyped: 0,
      wordsTyped: 0,
      backspaces: 0,
      errorCount: 0,
      pauseMs: 0,
      durationMs: 0,
    };
    const zeroFeats = extractTypingFeatures(zeroInput);
    assert.equal(Number.isNaN(zeroFeats.features.typingSpeedWpm), false);
    assert.equal(Number.isNaN(zeroFeats.features.backspaceRatio), false);
    assert.equal(Number.isNaN(zeroFeats.features.errorRate), false);
    assert.equal(Number.isNaN(zeroFeats.features.pauseRatio), false);
    assert.equal(zeroFeats.features.backspaceRatio, 0);
    assert.equal(zeroFeats.features.errorRate, 0);
  });

  await t.test("2. Model artifact validation rejects malformed structures and accepts valid artifacts", () => {
    assert.equal(validateModelArtifact(null), null);
    assert.equal(validateModelArtifact("not an object"), null);

    assert.equal(
      validateModelArtifact({
        modelType: "logistic-regression",
        classes: ["frustration", "confidence", "neutral"],
      }),
      null
    );

    assert.equal(
      validateModelArtifact({
        modelType: "xgboost-multiclass",
        classes: ["frustration", "confidence"],
        featureNames: TYPING_FEATURE_NAMES,
        trees: [],
      }),
      null
    );

    assert.equal(
      validateModelArtifact({
        modelType: "xgboost-multiclass",
        classes: ["frustration", "confidence", "neutral"],
        featureNames: ["typingSpeedWpm"],
        trees: [],
      }),
      null
    );

    const validArtifact: XGBoostModelArtifact = {
      modelType: "xgboost-multiclass",
      version: "1.0.0",
      classes: ["frustration", "confidence", "neutral"],
      featureNames: [...TYPING_FEATURE_NAMES],
      baseScores: [0, 0, 0],
      trees: [
        {
          treeId: 0,
          classIndex: 0,
          nodes: [{ id: 0, leafValue: 0.5 }],
        },
      ],
    };
    const validated = validateModelArtifact(validArtifact);
    assert.notEqual(validated, null);
    assert.equal(validated?.version, "1.0.0");
  });

  await t.test("3. Tree evaluation and numerically stable softmax probability computation", () => {
    const testTree: XGBTree = {
      treeId: 0,
      classIndex: 0,
      nodes: [
        {
          id: 0,
          featureIndex: 1, // backspaceRatio
          splitValue: 0.15,
          leftId: 1,
          rightId: 2,
        },
        { id: 1, leafValue: -1.5 },
        { id: 2, leafValue: 2.0 },
      ],
    };

    const lowBackspaceVector = [50, 0.05, 0.01, 0.1, 200, 40, 2, 1, 5000, 60000];
    assert.equal(evaluateTree(testTree, lowBackspaceVector), -1.5);

    const highBackspaceVector = [25, 0.25, 0.08, 0.3, 100, 20, 25, 8, 20000, 60000];
    assert.equal(evaluateTree(testTree, highBackspaceVector), 2.0);

    const probs = softmax([2.0, -1.0, 0.0]);
    assert.equal(probs.length, 3);
    const sum = probs.reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(sum - 1.0) < 0.01, `Softmax probs should sum to ~1.0: got ${sum}`);
    assert.ok(probs[0] > probs[1] && probs[0] > probs[2], "Highest margin must yield highest probability");
  });

  await t.test("4. Model loading from disk and graceful handling of missing or corrupted files", () => {
    const classifier = new XGBoostEmotionClassifier();
    assert.equal(classifier.isModelLoaded(), false);

    const loadedNonExistent = classifier.loadModel(path.join(os.tmpdir(), "non-existent-xgb.json"));
    assert.equal(loadedNonExistent, false);
    assert.equal(classifier.isModelLoaded(), false);

    const tempCorruptedPath = path.join(os.tmpdir(), `corrupt-xgb-${Date.now()}.json`);
    fs.writeFileSync(tempCorruptedPath, "{ invalid json data ", "utf8");
    try {
      const loadedCorrupt = classifier.loadModel(tempCorruptedPath);
      assert.equal(loadedCorrupt, false);
      assert.equal(classifier.isModelLoaded(), false);
    } finally {
      fs.unlinkSync(tempCorruptedPath);
    }
  });

  await t.test("5. Fallback behavior preserves existing rule-based engine when model is not loaded", () => {
    const uninitialisedClassifier = new XGBoostEmotionClassifier();
    assert.equal(uninitialisedClassifier.isModelLoaded(), false);

    const testMetrics = {
      charsTyped: 500,
      wordsTyped: 90,
      backspaces: 4,
      errorCount: 1,
      pauseMs: 8000,
      durationMs: 70000,
    };

    const fallbackResult = uninitialisedClassifier.predictWithFallback(testMetrics);
    const baselineResult = inferEmotionFromTyping(testMetrics);

    assert.equal(fallbackResult.source, "rule-based-fallback");
    assert.equal(fallbackResult.emotionSignal, baselineResult.emotionSignal);
    assert.equal(fallbackResult.vibeScore, baselineResult.vibeScore);
    assert.equal(fallbackResult.typingSpeedWpm, baselineResult.typingSpeedWpm);
  });

  await t.test("6. Actual trained model artifact loads cleanly and generates multiclass predictions", () => {
    const trainedModelPath = path.resolve(process.cwd(), "models/xgboost-emotion.json");
    if (!fs.existsSync(trainedModelPath)) {
      t.skip("Trained model artifact not present at models/xgboost-emotion.json");
      return;
    }

    const classifier = new XGBoostEmotionClassifier(trainedModelPath);
    assert.equal(classifier.isModelLoaded(), true);

    const meta = classifier.getModelMetadata();
    assert.notEqual(meta, null);
    assert.equal(meta?.sampleCount, 196);
    assert.equal(meta?.classes?.length, 3);

    const sample = {
      charsTyped: 420,
      wordsTyped: 75,
      backspaces: 5,
      errorCount: 1,
      pauseMs: 6000,
      durationMs: 60000,
    };

    const pred = classifier.predict(sample);
    assert.notEqual(pred, null);
    assert.equal(pred?.source, "xgboost");
    assert.ok(["frustration", "confidence", "neutral"].includes(pred!.emotionSignal));
    assert.ok(pred!.vibeScore >= 0 && pred!.vibeScore <= 100);
    assert.ok(pred!.confidence >= 0 && pred!.confidence <= 1);
    assert.ok(pred!.probabilities !== undefined);
  });

  await t.test("7. End-to-end HTTP integration smoke test for typing endpoints", async () => {
    const app = createApp();
    const server = app.listen(0, "127.0.0.1");
    await new Promise<void>((resolve) => server.once("listening", () => resolve()));
    const port = (server.address() as AddressInfo).port;
    const baseUrl = `http://127.0.0.1:${port}`;

    try {
      // 1. POST /api/v1/telemetry/typing
      const telemetryRes = await fetch(`${baseUrl}/api/v1/telemetry/typing`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          userId: "test-smoke-user",
          charsTyped: 350,
          wordsTyped: 65,
          backspaces: 6,
          errorCount: 2,
          pauseMs: 8000,
          durationMs: 60000,
        }),
      });

      assert.equal(telemetryRes.status, 201);
      const teleBody = (await telemetryRes.json()) as {
        emotionSignal: string;
        vibeScore: number;
        typingSpeedWpm: number;
      };
      assert.ok(["frustration", "confidence", "neutral"].includes(teleBody.emotionSignal));
      assert.ok(teleBody.vibeScore >= 0 && teleBody.vibeScore <= 100);

      // 2. POST /api/v1/extension/events/typing
      const extRes = await fetch(`${baseUrl}/api/v1/extension/events/typing`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          userId: "test-smoke-user",
          charsTyped: 400,
          wordsTyped: 70,
          backspaces: 4,
          errorCount: 1,
          pauseMs: 7000,
          durationMs: 60000,
        }),
      });

      assert.equal(extRes.status, 201);
      const extBody = (await extRes.json()) as {
        snapshot: { emotionSignal: string; vibeScore: number; typingSpeedWpm: number };
        adaptiveMode: string;
        vibeScore: number;
      };
      assert.ok(["frustration", "confidence", "neutral"].includes(extBody.snapshot.emotionSignal));
      assert.ok(extBody.vibeScore >= 0 && extBody.vibeScore <= 100);
      assert.ok(typeof extBody.adaptiveMode === "string");

      // Verify AI mood entry was recorded
      const userMoods = db.data.moodEntries.filter((m) => m.userId === "test-smoke-user" && m.source === "ai");
      assert.ok(userMoods.length >= 1, "Expected AI mood entry to be recorded from extension event");
    } finally {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
      try {
        fs.rmSync(tempDbDir, { recursive: true, force: true });
      } catch {
        // cleanup best-effort
      }
    }
  });

  await t.test("8. Candidate model artifact loads cleanly and performs rate-based inference", () => {
    const candidatePath = path.resolve(process.cwd(), "models/xgboost-emotion-candidate.json");
    if (!fs.existsSync(candidatePath)) {
      return;
    }

    const candidateClassifier = new XGBoostEmotionClassifier(candidatePath);
    assert.equal(candidateClassifier.isModelLoaded(), true);

    const meta = candidateClassifier.getModelMetadata();
    assert.ok(meta);
    assert.equal(meta.classes?.length, 3);

    const pred = candidateClassifier.predict({
      charsTyped: 300,
      wordsTyped: 60,
      backspaces: 4,
      errorCount: 1,
      pauseMs: 4000,
      durationMs: 60000,
    });

    assert.ok(pred);
    assert.ok(["frustration", "confidence", "neutral"].includes(pred.emotionSignal));
    assert.equal(pred.source, "xgboost");
    assert.ok(pred.vibeScore >= 0 && pred.vibeScore <= 100);
  });

  await t.test("9. Robust handling of non-finite, negative, and extreme inputs", () => {
    const classifier = new XGBoostEmotionClassifier();
    classifier.loadModel(path.resolve(process.cwd(), "models/xgboost-emotion.json"));

    const extremeInput = {
      charsTyped: Number.NaN,
      wordsTyped: Number.POSITIVE_INFINITY,
      backspaces: -50,
      errorCount: Number.NaN,
      pauseMs: -1000,
      durationMs: 0,
    };

    const result = classifier.predictWithFallback(extremeInput);
    assert.ok(["frustration", "confidence", "neutral"].includes(result.emotionSignal));
    assert.ok(Number.isFinite(result.vibeScore));
    assert.ok(Number.isFinite(result.confidence));
    assert.ok(result.vibeScore >= 0 && result.vibeScore <= 100);
  });

  await t.test("10. Reversible model promotion and rollback verification", () => {
    const prodPath = path.resolve(process.cwd(), "models/xgboost-emotion.json");
    const backupPath = path.resolve(process.cwd(), "models/xgboost-emotion.json.bak");

    assert.ok(fs.existsSync(prodPath), "Production model must exist");
    assert.ok(fs.existsSync(backupPath), "Backup model must exist for rollback safety");

    const prodClassifier = new XGBoostEmotionClassifier(prodPath);
    const backupClassifier = new XGBoostEmotionClassifier(backupPath);

    assert.equal(prodClassifier.isModelLoaded(), true);
    assert.equal(backupClassifier.isModelLoaded(), true);

    const input = {
      charsTyped: 250,
      wordsTyped: 50,
      backspaces: 10,
      errorCount: 2,
      pauseMs: 8000,
      durationMs: 60000,
    };

    const prodPred = prodClassifier.predict(input);
    const backupPred = backupClassifier.predict(input);

    assert.ok(prodPred);
    assert.ok(backupPred);
    // Verifies backup is a valid rollback artifact with matching output
    assert.equal(prodPred.emotionSignal, backupPred.emotionSignal);
  });
});
