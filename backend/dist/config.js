import path from "node:path";
import os from "node:os";
const defaultDbPath = process.env.NODE_ENV === "test"
    ? path.join(os.tmpdir(), "thinkflow-test-db.json")
    : "./data/thinkflow-db.json";
export const config = {
    port: Number(process.env.PORT ?? 8080),
    host: process.env.HOST ?? "127.0.0.1",
    dbPath: path.resolve(process.cwd(), process.env.THINKFLOW_DB_PATH ?? defaultDbPath),
    corsOrigins: [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        ...(process.env.FRONTEND_URL ? [process.env.FRONTEND_URL] : []),
    ],
};
