const analyticsService = require("../services/analyticsService");
const inferenceService = require("../services/inferenceService");
const { success, error } = require("../utils/responseHelper");

/**
 * GET /api/analytics/models
 */
const getModels = async (req, res, next) => {
  try {
    const models = analyticsService.getModelComparison();
    return success(res, { models });
  } catch (err) {
    next(err);
  }
};

/**
 * GET /api/analytics/training-logs/:model
 */
const getTrainingLogs = async (req, res, next) => {
  try {
    const { model } = req.params;
    const logs = analyticsService.getTrainingLog(model);
    if (!logs) {
      return error(res, `Training logs not found for model: ${model}`, 404);
    }
    return success(res, { model, logs });
  } catch (err) {
    next(err);
  }
};

/**
 * GET /api/analytics/stats
 */
const getUserStats = async (req, res, next) => {
  try {
    const stats = await analyticsService.getUserStats(req.user.userId);
    return success(res, { stats });
  } catch (err) {
    next(err);
  }
};

/**
 * GET /api/analytics/global-stats
 */
const getGlobalStats = async (req, res, next) => {
  try {
    const stats = await analyticsService.getGlobalStats();

    // Also get dataset stats from Python service
    const datasetStats = await inferenceService.getStats();

    return success(res, { stats, datasetStats });
  } catch (err) {
    next(err);
  }
};

module.exports = { getModels, getTrainingLogs, getUserStats, getGlobalStats };
