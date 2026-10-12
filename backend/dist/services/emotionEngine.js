import { safeAvg, clamp } from "../utils.js";
export function inferEmotionFromTyping(metrics) {
    const minutes = Math.max(metrics.durationMs / 60000, 0.1);
    const typingSpeedWpm = metrics.wordsTyped / minutes;
    const backspaceRatio = metrics.backspaces / Math.max(metrics.charsTyped, 1);
    const errorRate = metrics.errorCount / Math.max(metrics.wordsTyped, 1);
    const pausePenalty = metrics.pauseMs / Math.max(metrics.durationMs, 1);
    const frustrationIndex = backspaceRatio * 45 +
        errorRate * 30 +
        pausePenalty * 25;
    const confidenceIndex = clamp(typingSpeedWpm / 70, 0, 1) * 40 +
        clamp(1 - backspaceRatio, 0, 1) * 35 +
        clamp(1 - errorRate, 0, 1) * 25;
    const rawVibe = 55 + confidenceIndex - frustrationIndex;
    const vibeScore = Math.round(clamp(rawVibe, 0, 100));
    if (frustrationIndex > 28 || vibeScore < 40) {
        return { emotionSignal: "frustration", vibeScore, typingSpeedWpm: Number(typingSpeedWpm.toFixed(1)) };
    }
    if (confidenceIndex > 65 || vibeScore >= 70) {
        return { emotionSignal: "confidence", vibeScore, typingSpeedWpm: Number(typingSpeedWpm.toFixed(1)) };
    }
    return { emotionSignal: "neutral", vibeScore, typingSpeedWpm: Number(typingSpeedWpm.toFixed(1)) };
}
export function fuseMoodSignals(params) {
    const { manualMood, manualEnergy, aiSignal, latestVibeScore } = params;
    const aiMood = aiSignal === "frustration"
        ? "frustrated"
        : aiSignal === "confidence"
            ? "energetic"
            : "neutral";
    if (!manualMood) {
        return {
            fusedMood: aiMood,
            confidence: clamp(0.55 + latestVibeScore / 250, 0.5, 0.92),
        };
    }
    const blendedConfidence = manualMood === aiMood || (manualMood === "confident" && aiMood === "energetic")
        ? 0.93
        : clamp(safeAvg([
            manualEnergy ? manualEnergy / 10 : 0.5,
            latestVibeScore / 100,
            0.6,
        ]), 0.55, 0.89);
    if (manualMood === "fatigued" || manualMood === "frustrated") {
        return {
            fusedMood: manualMood,
            confidence: blendedConfidence,
        };
    }
    if (manualMood !== aiMood && latestVibeScore < 45) {
        return {
            fusedMood: "frustrated",
            confidence: blendedConfidence,
        };
    }
    return {
        fusedMood: manualMood,
        confidence: blendedConfidence,
    };
}
export function computeSessionVibeAverage(snapshots) {
    if (snapshots.length === 0) {
        return 50;
    }
    return Number(safeAvg(snapshots.map((s) => s.vibeScore)).toFixed(1));
}
