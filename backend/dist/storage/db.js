import { JSONFilePreset } from "lowdb/node";
import { config } from "../config.js";
const defaultData = {
    users: [],
    sleepLogs: [],
    moodEntries: [],
    typingSnapshots: [],
    codingSessions: [],
    habitLogs: [],
    manualCircadianLogs: [],
    emotionBridge: {
        typingMood: "Neutral",
        faceMood: "Neutral",
        updatedAt: new Date().toISOString(),
    },
    extensionLinks: [],
    breakRecommendations: [],
    quizAttempts: [],
    rewardEvents: [],
    researchSamples: [],
};
export const db = await JSONFilePreset(config.dbPath, defaultData);
if (!db.data.researchSamples) {
    db.data.researchSamples = [];
}
