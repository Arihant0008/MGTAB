const Prediction = require("../models/Prediction");
const User = require("../models/User");
const BatchJob = require("../models/BatchJob");
const fs = require("fs");
const path = require("path");

// Path to the training logs and model results (existing research data)
const DATA_DIR = path.join(__dirname, "..", "..", "Datasets and precrosessing");

/**
 * Get model comparison data (all 4 models).
 */
const getModelComparison = () => {
  const csvPath = path.join(DATA_DIR, "final_model_results.csv");
  if (!fs.existsSync(csvPath)) return [];

  const content = fs.readFileSync(csvPath, "utf-8").trim();
  const lines = content.split("\n");
  const headers = lines[0].split(",").map((h) => h.trim());

  return lines.slice(1).filter(Boolean).map((line) => {
    const values = line.split(",").map((v) => v.trim());
    const obj = {};
    headers.forEach((h, i) => {
      obj[h] = isNaN(values[i]) ? values[i] : parseFloat(values[i]);
    });
    return obj;
  });
};

/**
 * Get training log data for a specific model.
 */
const getTrainingLog = (modelName) => {
  const filename = `${modelName}_training_log.csv`;
  const csvPath = path.join(DATA_DIR, "logs", filename);

  if (!fs.existsSync(csvPath)) return null;

  const content = fs.readFileSync(csvPath, "utf-8").trim();
  const lines = content.split("\n");
  const headers = lines[0].split(",").map((h) => h.trim());

  return lines.slice(1).filter(Boolean).map((line) => {
    const values = line.split(",").map((v) => v.trim());
    const obj = {};
    headers.forEach((h, i) => {
      obj[h] = isNaN(values[i]) ? values[i] : parseFloat(values[i]);
    });
    return obj;
  });
};

/**
 * Get personal stats for a user.
 */
const getUserStats = async (userId) => {
  const totalPredictions = await Prediction.countDocuments({ userId });
  const botsDetected = await Prediction.countDocuments({ userId, prediction: "bot" });
  const humansDetected = await Prediction.countDocuments({ userId, prediction: "human" });
  const lastPrediction = await Prediction.findOne({ userId }).sort({ createdAt: -1 });

  return {
    totalPredictions,
    botsDetected,
    humansDetected,
    lastQueryAt: lastPrediction ? lastPrediction.createdAt : null,
  };
};

/**
 * Get global platform stats.
 */
const getGlobalStats = async () => {
  const totalUsers = await User.countDocuments();
  const totalPredictions = await Prediction.countDocuments();
  const botsDetected = await Prediction.countDocuments({ prediction: "bot" });
  const humansDetected = await Prediction.countDocuments({ prediction: "human" });
  const totalBatchJobs = await BatchJob.countDocuments();

  return {
    totalUsers,
    totalPredictions,
    botsDetected,
    humansDetected,
    totalBatchJobs,
  };
};

module.exports = {
  getModelComparison,
  getTrainingLog,
  getUserStats,
  getGlobalStats,
};
