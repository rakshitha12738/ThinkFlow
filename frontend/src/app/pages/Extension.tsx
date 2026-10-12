import { useEffect, useState } from "react";
import { Link } from "react-router";
import { ArrowLeft, ExternalLink } from "lucide-react";
import { motion } from "motion/react";
import EmotionCamera from "./EmotionCamera";

const bridgeBaseUrl = "http://localhost:8000";

export default function ExtensionPage() {
  const [typingMood, setTypingMood] = useState("Neutral");
  const [faceMood, setFaceMood] = useState("Neutral");
  const [finalMood, setFinalMood] = useState("Neutral");

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        let data: { typingMood?: string; faceMood?: string } | null = null;
        try {
          const res = await fetch(`${bridgeBaseUrl}/emotion`);
          if (res.ok) {
            data = (await res.json()) as { typingMood?: string; faceMood?: string };
          }
        } catch {
          // Bridge on 8000 not reachable; fallback to main Express backend on 8080
        }

        if (!data) {
          try {
            const fallbackRes = await fetch("http://localhost:8080/api/v1/emotion");
            if (fallbackRes.ok) {
              data = (await fallbackRes.json()) as { typingMood?: string; faceMood?: string };
            }
          } catch {
            // Main backend also not reachable
          }
        }

        if (data) {
          setTypingMood(data.typingMood ?? "Neutral");
          setFaceMood(data.faceMood ?? "Neutral");

          const isFrustrated =
            (data.typingMood ?? "").toLowerCase().includes("frustrated") ||
            ["sad", "angry"].includes((data.faceMood ?? "").toLowerCase());

          setFinalMood(isFrustrated ? "Frustrated 😓" : "Confident 🔥");
        }
      } catch {
        // Keep polling non-blocking.
      }
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        <div className="flex items-center gap-4">
          <Link to="/dashboard" className="p-2 hover:bg-muted rounded-lg transition-colors">
            <ArrowLeft className="w-6 h-6 text-muted-foreground" />
          </Link>
          <div>
            <h1 className="text-4xl font-bold text-foreground mb-2">Extension Bridge</h1>
            <p className="text-muted-foreground">IDE extension features: typing telemetry, webcam emotion monitor, adaptive prompts.</p>
          </div>
        </div>

        <div className="bg-card border border-border rounded-2xl p-6 shadow-lg">
          <h3 className="text-xl font-bold text-foreground mb-2">Open VS Code Extension</h3>
          <p className="text-muted-foreground mb-4">
            Use this button to open the ThinkFlow extension directly in VS Code.
          </p>
          <a
            href="vscode:extension/thinkflow.thinkflow-vscode-link"
            className="inline-flex items-center gap-2 rounded-lg bg-gradient-to-r from-[#800020] to-[#DCAE96] px-4 py-3 text-white font-semibold"
          >
            Open Extension in VS Code
            <ExternalLink className="w-4 h-4" />
          </a>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {[{ label: "Typing", value: typingMood }, { label: "Face", value: faceMood }, { label: "Final", value: finalMood }].map((card) => (
            <motion.div
              key={card.label}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="rounded-2xl border border-border bg-card p-5 shadow"
            >
              <div className="text-xs uppercase tracking-wide text-muted-foreground mb-2">{card.label} Mood</div>
              <div className="text-2xl font-bold text-foreground">{card.value}</div>
            </motion.div>
          ))}
        </div>

        <EmotionCamera />
      </div>
    </div>
  );
}
