const express = require("express");
const router = express.Router();
const analyticsController = require("../controllers/analyticsController");
const authMiddleware = require("../middleware/authMiddleware");
const { modelParamRules, validate } = require("../middleware/validator");

// GET /api/analytics/models  (public)
router.get("/models", analyticsController.getModels);

// GET /api/analytics/training-logs/:model  (public)
router.get("/training-logs/:model", modelParamRules, validate, analyticsController.getTrainingLogs);

// GET /api/analytics/stats  (auth required)
router.get("/stats", authMiddleware, analyticsController.getUserStats);

// GET /api/analytics/global-stats  (public)
router.get("/global-stats", analyticsController.getGlobalStats);

module.exports = router;
