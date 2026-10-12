import express from "express";
import cors from "cors";
import { config } from "./config.js";
import { apiRouter } from "./routes/api.js";
export function createApp() {
    const app = express();
    app.use(cors({
        origin: (origin, callback) => {
            // Allow non-browser clients (VS Code extension Node.js client, curl, etc.) with no Origin header
            if (!origin) {
                return callback(null, true);
            }
            if (config.corsOrigins.includes(origin)) {
                return callback(null, true);
            }
            return callback(new Error(`CORS blocked: Origin '${origin}' is not allowed`));
        },
        methods: ["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allowedHeaders: ["Content-Type", "Authorization", "X-User-Id"],
        credentials: true,
    }));
    app.use(express.json({ limit: "1mb" }));
    app.get("/", (_req, res) => {
        res.json({
            service: "ThinkFlow backend",
            version: "1.0.0",
            docs: "/api/v1/health",
        });
    });
    app.use("/api/v1", apiRouter);
    app.use((err, _req, res, _next) => {
        const message = err instanceof Error ? err.message : "Unknown server error";
        res.status(500).json({ error: message });
    });
    return app;
}
export const app = createApp();
if (process.env.NODE_ENV !== "test") {
    app.listen(config.port, config.host, () => {
        console.log(`ThinkFlow backend running on http://${config.host}:${config.port}`);
    });
}
