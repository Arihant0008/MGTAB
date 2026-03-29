const express = require("express");
const router = express.Router();
const predictController = require("../controllers/predictController");
const authMiddleware = require("../middleware/authMiddleware");
const { predictSingleRules, validate, paginationRules } = require("../middleware/validator");
const { predictLimiter } = require("../middleware/rateLimiter");

// POST /api/predict/single
router.post("/single", authMiddleware, predictLimiter, predictSingleRules, validate, predictController.predictSingle);

// POST /api/predict/batch
router.post("/batch", authMiddleware, predictController.predictBatch);

// GET /api/predict/history
router.get("/history", authMiddleware, paginationRules, validate, predictController.getHistory);

// GET /api/predict/history/:id
router.get("/history/:id", authMiddleware, predictController.getHistoryById);

// DELETE /api/predict/history/:id
router.delete("/history/:id", authMiddleware, predictController.deleteHistory);

// GET /api/predict/batch/:id
router.get("/batch/:id", authMiddleware, predictController.getBatchJob);

module.exports = router;
