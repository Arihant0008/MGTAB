const express = require("express");
const helmet = require("helmet");
const mongoSanitize = require("express-mongo-sanitize");
const morgan = require("morgan");

const { PORT, NODE_ENV } = require("./config/env");
const corsMiddleware = require("./config/cors");
const connectDB = require("./config/db");
const errorHandler = require("./middleware/errorHandler");
const { generalLimiter } = require("./middleware/rateLimiter");
const logger = require("./utils/logger");
const { success } = require("./utils/responseHelper");
const inferenceService = require("./services/inferenceService");

// Route imports
const authRoutes = require("./routes/authRoutes");
const predictRoutes = require("./routes/predictRoutes");
const analyticsRoutes = require("./routes/analyticsRoutes");

const app = express();

// ── Global Middleware ────────────────────────────────────────────────────────
app.use(helmet());
app.use(corsMiddleware);
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));
app.use(mongoSanitize());
app.use(generalLimiter);

// HTTP logging
if (NODE_ENV === "development") {
  app.use(morgan("dev"));
}

// ── Routes ───────────────────────────────────────────────────────────────────
app.use("/api/auth", authRoutes);
app.use("/api/predict", predictRoutes);
app.use("/api/analytics", analyticsRoutes);

// ── Health Check ─────────────────────────────────────────────────────────────
app.get("/api/health", async (req, res) => {
  const pythonHealth = await inferenceService.checkHealth();
  return success(res, {
    server: "ok",
    environment: NODE_ENV,
    pythonService: pythonHealth,
    timestamp: new Date().toISOString(),
  });
});

// ── 404 Handler ──────────────────────────────────────────────────────────────
app.use((req, res) => {
  res.status(404).json({
    success: false,
    message: `Route ${req.method} ${req.path} not found`,
    statusCode: 404,
  });
});

// ── Error Handler ────────────────────────────────────────────────────────────
app.use(errorHandler);

// ── Start Server ─────────────────────────────────────────────────────────────
const startServer = async () => {
  await connectDB();
  app.listen(PORT, () => {
    logger.info(`🚀 MGTAB API Server running on port ${PORT} (${NODE_ENV})`);
    logger.info(`   Health: http://localhost:${PORT}/api/health`);
  });
};

startServer();

module.exports = app;
