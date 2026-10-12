import crypto from "node:crypto";
import { nanoid } from "nanoid";
import { db } from "../storage/db.js";
const MIN_SALT_LENGTH = 32;
const MAX_SELF_REPORT_LATENCY_SEC = 300; // 5 minutes
export const ISO_8601_REGEX = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})$/;
export function isValidIso8601(timestamp) {
    if (typeof timestamp !== "string" || !ISO_8601_REGEX.test(timestamp)) {
        return false;
    }
    const parsed = Date.parse(timestamp);
    return !Number.isNaN(parsed) && Number.isFinite(parsed);
}
/**
 * Retrieves the research salt from the environment.
 * FAILS CLOSED if THINKFLOW_RESEARCH_SALT is missing, empty, or shorter than 32 characters.
 * NEVER logs, exposes, or echoes the secret.
 */
export function getResearchSalt() {
    const salt = process.env.THINKFLOW_RESEARCH_SALT;
    if (!salt || typeof salt !== "string" || salt.trim().length < MIN_SALT_LENGTH) {
        throw new Error("Pseudonymization unavailable: THINKFLOW_RESEARCH_SALT is missing or has insufficient entropy (must be at least 32 characters).");
    }
    return salt.trim();
}
/**
 * Derives a pseudonymized participant ID from a user ID.
 *
 * NOTE ON DATA PRIVACY & GDPR ART. 4(5):
 * This identifier is PSEUDONYMIZED, not anonymous. Holding the secret salt and the
 * user ID allows re-identification, which is required to support voluntary
 * consent withdrawal and data deletion (Right to Erasure).
 *
 * Fails closed if the research salt is missing or weak.
 */
export function pseudonymizeParticipantId(userId) {
    if (!userId || typeof userId !== "string" || !userId.trim()) {
        throw new Error("Invalid userId provided for pseudonymization");
    }
    const salt = getResearchSalt();
    return crypto
        .createHmac("sha256", salt)
        .update(userId.trim())
        .digest("hex");
}
/**
 * Maps voluntary emotion self-reports to ML target classes without
 * making unsupported assumptions.
 *
 * Ground-truth mapping rationale:
 * - "frustrated" -> "frustration": direct 1:1 affective match (goal obstacle).
 * - "confident"   -> "confidence": explicit voluntary flow/confidence self-report.
 * - "neutral"     -> "neutral": baseline emotional state.
 * - "calm"        -> "neutral": steady, low-arousal baseline.
 * - "energetic"   -> null: EXCLUDED. High energy is not assumed to mean confidence.
 * - "fatigued"    -> null: EXCLUDED. Fatigue is not assumed to mean frustration.
 */
export function mapSelfReportToTargetClass(emotion, _energy) {
    switch (emotion) {
        case "frustrated":
            return "frustration";
        case "confident":
            return "confidence";
        case "neutral":
        case "calm":
            return "neutral";
        case "energetic":
        case "fatigued":
        default:
            return null;
    }
}
/**
 * Calculates aggregate numerical typing ratios from raw window metrics.
 * Operates purely on counts and durations; never ingests raw text or keystrokes.
 */
export function computeAggregateFeatures(metrics) {
    const minutes = Math.max(metrics.durationMs / 60000, 0.1);
    const typingSpeedWpm = Number((metrics.wordsTyped / minutes).toFixed(1));
    const backspaceRatio = Number((metrics.backspaces / Math.max(metrics.charsTyped, 1)).toFixed(4));
    const errorRate = Number((metrics.errorCount / Math.max(metrics.wordsTyped, 1)).toFixed(4));
    const pauseRatio = Number((metrics.pauseMs / Math.max(metrics.durationMs, 1)).toFixed(4));
    return {
        charsTyped: metrics.charsTyped,
        wordsTyped: metrics.wordsTyped,
        backspaces: metrics.backspaces,
        errorCount: metrics.errorCount,
        pauseMs: metrics.pauseMs,
        durationMs: metrics.durationMs,
        typingSpeedWpm,
        backspaceRatio,
        errorRate,
        pauseRatio,
    };
}
/**
 * TIMESTAMP SEMANTICS SPECIFICATION:
 * `window.timestamp` explicitly represents the WINDOW END TIMESTAMP (ISO 8601),
 * marking the instant the observation window concluded and was captured.
 *
 * The window start time is derived as:
 *   windowStartTime = windowEndTime - window.durationMs
 *
 * Temporal validity rules:
 * 1. Both window.timestamp and reportTimestamp must be valid, parseable ISO-8601 dates.
 * 2. The self-report cannot precede the typing window end time
 *    (reportTimestamp must be >= windowEndTime).
 * 3. The self-report cannot be stale (reportTimestamp - windowEndTime <= 300 seconds).
 */
export async function recordResearchSample(params) {
    const user = db.data.users.find((u) => u.id === params.userId);
    if (!user) {
        return { accepted: false, reason: "User not found" };
    }
    // Gate 1: Explicit opt-in verification (default disabled)
    if (!user.researchConsent?.optedIn) {
        return {
            accepted: false,
            reason: "Research telemetry consent is not granted by user",
        };
    }
    // Gate 2: Metric sanity, non-negative integer counts, and consistency checks
    const { window, selfReport } = params;
    // Count fields must be non-negative integers
    const countFields = [
        ["charsTyped", window.charsTyped],
        ["wordsTyped", window.wordsTyped],
        ["backspaces", window.backspaces],
        ["errorCount", window.errorCount],
    ];
    for (const [name, val] of countFields) {
        if (typeof val !== "number" || !Number.isFinite(val) || Number.isNaN(val) || !Number.isInteger(val)) {
            return { accepted: false, reason: `Invalid metric value for '${name}': must be an integer` };
        }
        if (val < 0) {
            return { accepted: false, reason: `Invalid negative metric value for '${name}'` };
        }
    }
    // Duration/timing fields must be finite non-negative numbers
    const durationFields = [
        ["pauseMs", window.pauseMs],
        ["durationMs", window.durationMs],
    ];
    for (const [name, val] of durationFields) {
        if (typeof val !== "number" || !Number.isFinite(val) || Number.isNaN(val)) {
            return { accepted: false, reason: `Invalid metric value for '${name}': must be a finite number` };
        }
        if (val < 0) {
            return { accepted: false, reason: `Invalid negative metric value for '${name}'` };
        }
    }
    if (window.durationMs < 1000) {
        return { accepted: false, reason: "Duration too short (< 1000ms)" };
    }
    if (window.charsTyped === 0 && window.backspaces === 0) {
        return { accepted: false, reason: "No typing activity detected in window" };
    }
    // Metric consistency checks
    if (window.wordsTyped > window.charsTyped) {
        return {
            accepted: false,
            reason: "Inconsistent metrics: wordsTyped cannot exceed charsTyped",
        };
    }
    if (window.pauseMs > window.durationMs) {
        return {
            accepted: false,
            reason: "Inconsistent metrics: pauseMs cannot exceed durationMs",
        };
    }
    if (window.errorCount > window.charsTyped + window.backspaces) {
        return {
            accepted: false,
            reason: "Inconsistent metrics: errorCount cannot exceed total keystrokes (charsTyped + backspaces)",
        };
    }
    if (selfReport.energy !== undefined) {
        if (typeof selfReport.energy !== "number" ||
            !Number.isFinite(selfReport.energy) ||
            !Number.isInteger(selfReport.energy) ||
            selfReport.energy < 1 ||
            selfReport.energy > 10) {
            return {
                accepted: false,
                reason: "Invalid energy: must be an integer between 1 and 10",
            };
        }
    }
    // Gate 3: Explicit ISO-8601 timestamp parsing, temporal ordering, and freshness
    if (!isValidIso8601(window.timestamp)) {
        return { accepted: false, reason: "Invalid window timestamp format: must be valid ISO-8601 date" };
    }
    if (!isValidIso8601(selfReport.reportTimestamp)) {
        return { accepted: false, reason: "Invalid self-report timestamp format: must be valid ISO-8601 date" };
    }
    const windowEndTime = Date.parse(window.timestamp);
    const reportTime = Date.parse(selfReport.reportTimestamp);
    // Self-report cannot precede the typing window end time (no tolerance)
    if (reportTime < windowEndTime) {
        return {
            accepted: false,
            reason: "Self-report timestamp precedes the typing window end time",
        };
    }
    // Self-report must not exceed the freshness latency threshold (5 minutes / 300s)
    const latencySec = Math.floor((reportTime - windowEndTime) / 1000);
    if (latencySec > MAX_SELF_REPORT_LATENCY_SEC) {
        return {
            accepted: false,
            reason: `Stale self-report: report occurred ${latencySec}s after typing window ended (maximum ${MAX_SELF_REPORT_LATENCY_SEC}s allowed)`,
        };
    }
    // Gate 4: Defensible label mapping
    const targetClass = mapSelfReportToTargetClass(selfReport.emotion, selfReport.energy);
    if (!targetClass) {
        return {
            accepted: false,
            reason: `Emotion label '${selfReport.emotion}' cannot be mapped defensibly to target classes`,
        };
    }
    // Gate 5: Pseudonymization with fail-closed secret protection
    let participantId;
    try {
        participantId = pseudonymizeParticipantId(params.userId);
    }
    catch {
        // Fail closed safely without exposing error details or secret names
        return {
            accepted: false,
            reason: "Pseudonymization unavailable: secure salt configuration missing or insufficient",
        };
    }
    const features = computeAggregateFeatures(window);
    const sample = {
        id: nanoid(),
        participantId,
        sessionId: params.sessionId,
        timestamp: window.timestamp,
        features,
        label: {
            selfReportedEmotion: selfReport.emotion,
            energy: selfReport.energy,
            targetClass,
            reportLatencySec: Math.max(0, latencySec),
        },
    };
    if (!db.data.researchSamples) {
        db.data.researchSamples = [];
    }
    db.data.researchSamples.push(sample);
    await db.write();
    return { accepted: true, sample };
}
/**
 * Updates a user's explicit research opt-in setting.
 */
export async function updateUserResearchConsent(userId, optedIn) {
    const user = db.data.users.find((u) => u.id === userId);
    if (!user) {
        return null;
    }
    const consent = {
        optedIn,
        updatedAt: new Date().toISOString(),
        version: "1.0",
    };
    user.researchConsent = consent;
    await db.write();
    return consent;
}
/**
 * Right to Erasure / Consent Withdrawal:
 * Deletes all research samples associated with a user's pseudonymized ID
 * and sets their consent flag to false.
 * Fails closed if pseudonymization secret is unavailable.
 */
export async function deleteUserResearchData(userId) {
    const participantId = pseudonymizeParticipantId(userId);
    const beforeCount = db.data.researchSamples?.length ?? 0;
    db.data.researchSamples = (db.data.researchSamples ?? []).filter((s) => s.participantId !== participantId);
    const deletedCount = beforeCount - db.data.researchSamples.length;
    const user = db.data.users.find((u) => u.id === userId);
    if (user && user.researchConsent) {
        user.researchConsent.optedIn = false;
        user.researchConsent.updatedAt = new Date().toISOString();
    }
    await db.write();
    return { deletedCount, participantId };
}
/**
 * Returns isolated summary statistics on the research dataset
 * to assess collection progress and class balance.
 */
export function getResearchDatasetStats() {
    const samples = db.data.researchSamples ?? [];
    const participants = new Set(samples.map((s) => s.participantId));
    const classDistribution = {
        frustration: 0,
        confidence: 0,
        neutral: 0,
    };
    for (const sample of samples) {
        if (sample.label.targetClass in classDistribution) {
            classDistribution[sample.label.targetClass] += 1;
        }
    }
    return {
        totalSamples: samples.length,
        participantCount: participants.size,
        classDistribution,
    };
}
