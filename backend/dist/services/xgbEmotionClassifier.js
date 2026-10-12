import fs from "node:fs";
import path from "node:path";
import { clamp } from "../utils.js";
import { inferEmotionFromTyping } from "./emotionEngine.js";
export const TYPING_FEATURE_NAMES = [
    "typingSpeedWpm",
    "backspaceRatio",
    "errorRate",
    "pauseRatio",
    "charsTyped",
    "wordsTyped",
    "backspaces",
    "errorCount",
    "pauseMs",
    "durationMs",
];
export const EMOTION_CLASSES = ["frustration", "confidence", "neutral"];
/**
 * Extracts normalized numerical typing features compatible with
 * researchDataService and XGBoost emotion classification.
 */
export function extractTypingFeatures(metrics) {
    const safeChars = Math.max(0, Number.isFinite(metrics.charsTyped) ? metrics.charsTyped : 0);
    const safeWords = Math.max(0, Number.isFinite(metrics.wordsTyped) ? metrics.wordsTyped : 0);
    const safeBackspaces = Math.max(0, Number.isFinite(metrics.backspaces) ? metrics.backspaces : 0);
    const safeErrors = Math.max(0, Number.isFinite(metrics.errorCount) ? metrics.errorCount : 0);
    const safePauseMs = Math.max(0, Number.isFinite(metrics.pauseMs) ? metrics.pauseMs : 0);
    const rawDuration = Number.isFinite(metrics.durationMs) ? metrics.durationMs : 1000;
    const safeDurationMs = Math.max(1000, rawDuration);
    const minutes = Math.max(safeDurationMs / 60000, 0.1);
    const typingSpeedWpm = Number((safeWords / minutes).toFixed(1));
    const backspaceRatio = Number((safeBackspaces / Math.max(safeChars, 1)).toFixed(4));
    const errorRate = Number((safeErrors / Math.max(safeWords, 1)).toFixed(4));
    const pauseRatio = Number((safePauseMs / safeDurationMs).toFixed(4));
    const features = {
        typingSpeedWpm: Number.isFinite(typingSpeedWpm) ? typingSpeedWpm : 0,
        backspaceRatio: Number.isFinite(backspaceRatio) ? backspaceRatio : 0,
        errorRate: Number.isFinite(errorRate) ? errorRate : 0,
        pauseRatio: Number.isFinite(pauseRatio) ? pauseRatio : 0,
        charsTyped: safeChars,
        wordsTyped: safeWords,
        backspaces: safeBackspaces,
        errorCount: safeErrors,
        pauseMs: safePauseMs,
        durationMs: safeDurationMs,
    };
    const values = TYPING_FEATURE_NAMES.map((name) => features[name]);
    return { features, values };
}
/**
 * Validates whether an object matches the XGBoost model artifact specification.
 */
export function validateModelArtifact(data) {
    if (typeof data !== "object" || data === null) {
        return null;
    }
    const obj = data;
    if (obj.modelType !== "xgboost-multiclass") {
        return null;
    }
    if (!Array.isArray(obj.classes) || obj.classes.length !== 3) {
        return null;
    }
    if (!Array.isArray(obj.featureNames) || obj.featureNames.length !== TYPING_FEATURE_NAMES.length) {
        return null;
    }
    if (!Array.isArray(obj.trees)) {
        return null;
    }
    for (const tree of obj.trees) {
        if (typeof tree.classIndex !== "number" || tree.classIndex < 0 || tree.classIndex > 2) {
            return null;
        }
        if (!Array.isArray(tree.nodes) || tree.nodes.length === 0) {
            return null;
        }
    }
    return obj;
}
/**
 * Evaluates a single decision tree on the given feature vector.
 */
export function evaluateTree(tree, featureValues) {
    if (!tree || !Array.isArray(tree.nodes) || tree.nodes.length === 0) {
        return 0;
    }
    const nodeMap = new Map();
    for (const node of tree.nodes) {
        nodeMap.set(node.id, node);
    }
    let current = nodeMap.get(0);
    if (!current) {
        return 0;
    }
    let steps = 0;
    const maxSteps = tree.nodes.length + 5;
    while (current && current.leafValue === undefined && steps < maxSteps) {
        steps += 1;
        const featureIdx = current.featureIndex ?? 0;
        const splitVal = Number.isFinite(current.splitValue) ? current.splitValue : 0;
        const rawVal = featureValues[featureIdx];
        const featureVal = Number.isFinite(rawVal) ? rawVal : 0;
        const nextId = featureVal < splitVal ? current.leftId : current.rightId;
        if (nextId === undefined) {
            break;
        }
        current = nodeMap.get(nextId);
    }
    return Number.isFinite(current?.leafValue) ? current.leafValue : 0;
}
/**
 * Numerically stable multiclass softmax.
 */
export function softmax(margins) {
    if (!Array.isArray(margins) || margins.length === 0) {
        return [];
    }
    const safeMargins = margins.map((m) => (Number.isFinite(m) ? m : 0));
    const maxMargin = Math.max(...safeMargins);
    const exps = safeMargins.map((m) => Math.exp(m - maxMargin));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    if (sumExps <= 0 || !Number.isFinite(sumExps)) {
        return safeMargins.map(() => Number((1 / safeMargins.length).toFixed(4)));
    }
    return exps.map((e) => Number((e / sumExps).toFixed(4)));
}
/**
 * XGBoost multiclass emotion classifier with integrated rule-based fallback.
 */
export class XGBoostEmotionClassifier {
    model = null;
    modelPath = null;
    constructor(modelPath) {
        if (modelPath) {
            this.loadModel(modelPath);
        }
    }
    loadModel(source) {
        try {
            let artifact;
            if (typeof source === "string") {
                if (!fs.existsSync(source)) {
                    return false;
                }
                const text = fs.readFileSync(source, "utf8");
                artifact = JSON.parse(text);
                this.modelPath = source;
            }
            else {
                artifact = source;
                this.modelPath = null;
            }
            const validated = validateModelArtifact(artifact);
            if (!validated) {
                this.model = null;
                return false;
            }
            this.model = validated;
            return true;
        }
        catch {
            this.model = null;
            return false;
        }
    }
    isModelLoaded() {
        return this.model !== null;
    }
    getModelMetadata() {
        return this.model?.metadata ?? null;
    }
    unloadModel() {
        this.model = null;
        this.modelPath = null;
    }
    predict(metrics) {
        if (!this.model) {
            return null;
        }
        try {
            const { features, values } = extractTypingFeatures(metrics);
            // Initialize raw margins from base scores
            const margins = [...(this.model.baseScores ?? [0, 0, 0])];
            // Accumulate leaf values across all gradient boosted trees
            for (const tree of this.model.trees) {
                const leafValue = evaluateTree(tree, values);
                margins[tree.classIndex] += leafValue;
            }
            const probs = softmax(margins);
            // Map probabilities to classes: 0: frustration, 1: confidence, 2: neutral
            const probFrustration = probs[0] ?? 0;
            const probConfidence = probs[1] ?? 0;
            const probNeutral = probs[2] ?? 0;
            let predictedClass = "neutral";
            let maxProb = probNeutral;
            if (probFrustration > maxProb) {
                predictedClass = "frustration";
                maxProb = probFrustration;
            }
            if (probConfidence > maxProb) {
                predictedClass = "confidence";
                maxProb = probConfidence;
            }
            // Derive continuous vibeScore consistent with ThinkFlow range (0-100)
            // High confidence -> ~85-100, Neutral -> ~50-65, Frustration -> ~10-35
            const vibeScore = Math.round(clamp(probConfidence * 90 + probNeutral * 55 + probFrustration * 15, 0, 100));
            return {
                emotionSignal: predictedClass,
                vibeScore,
                typingSpeedWpm: features.typingSpeedWpm,
                confidence: maxProb,
                probabilities: {
                    frustration: probFrustration,
                    confidence: probConfidence,
                    neutral: probNeutral,
                },
                source: "xgboost",
                modelVersion: this.model.version,
                diagnostic: {
                    treesEvaluated: this.model.trees.length,
                },
            };
        }
        catch {
            return null;
        }
    }
    predictWithFallback(metrics) {
        if (this.isModelLoaded()) {
            const pred = this.predict(metrics);
            if (pred) {
                return pred;
            }
        }
        // Rule-based engine fallback
        const fallback = inferEmotionFromTyping(metrics);
        return {
            emotionSignal: fallback.emotionSignal,
            vibeScore: fallback.vibeScore,
            typingSpeedWpm: fallback.typingSpeedWpm,
            confidence: fallback.emotionSignal === "neutral" ? 0.6 : 0.85,
            source: "rule-based-fallback",
        };
    }
}
export const defaultClassifier = new XGBoostEmotionClassifier();
// Try loading default model if present at standard location
const defaultModelPath = process.env.THINKFLOW_MODEL_PATH ?? path.resolve(process.cwd(), "models/xgboost-emotion.json");
if (fs.existsSync(defaultModelPath)) {
    defaultClassifier.loadModel(defaultModelPath);
}
/**
 * Primary emotion inference entry point.
 * Transparently uses XGBoost when available and loaded,
 * smoothly falling back to the rule-based engine otherwise.
 */
export function inferEmotion(metrics) {
    return defaultClassifier.predictWithFallback(metrics);
}
