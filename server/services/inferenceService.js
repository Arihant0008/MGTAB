const axios = require("axios");
const { PYTHON_SERVICE_URL } = require("../config/env");
const logger = require("../utils/logger");

const pythonClient = axios.create({
  baseURL: PYTHON_SERVICE_URL,
  timeout: 30000,
  headers: { "Content-Type": "application/json" },
});

/**
 * Call the Python inference microservice for a single username.
 */
const predictSingle = async (username) => {
  try {
    const { data } = await pythonClient.post("/predict", { username });
    return data;
  } catch (err) {
    if (err.response) {
      logger.error(`Python service returned ${err.response.status}: ${JSON.stringify(err.response.data)}`);
      const error = new Error(err.response.data.error || "Inference failed");
      error.statusCode = err.response.status;
      throw error;
    }
    logger.error(`Python service unreachable: ${err.message}`);
    const error = new Error("Inference service unavailable");
    error.statusCode = 503;
    throw error;
  }
};

/**
 * Call the Python inference microservice for a batch of usernames.
 */
const predictBatch = async (usernames) => {
  try {
    const { data } = await pythonClient.post("/predict/batch", { usernames });
    return data;
  } catch (err) {
    if (err.response) {
      logger.error(`Python batch service error: ${JSON.stringify(err.response.data)}`);
      const error = new Error(err.response.data.error || "Batch inference failed");
      error.statusCode = err.response.status;
      throw error;
    }
    logger.error(`Python service unreachable: ${err.message}`);
    const error = new Error("Inference service unavailable");
    error.statusCode = 503;
    throw error;
  }
};

/**
 * Health check for the Python service.
 */
const checkHealth = async () => {
  try {
    const { data } = await pythonClient.get("/health");
    return { status: "ok", ...data };
  } catch (err) {
    return { status: "down", error: err.message };
  }
};

/**
 * Get dataset stats from the Python service.
 */
const getStats = async () => {
  try {
    const { data } = await pythonClient.get("/stats");
    return data;
  } catch (err) {
    logger.error(`Failed to get stats: ${err.message}`);
    return null;
  }
};

module.exports = {
  predictSingle,
  predictBatch,
  checkHealth,
  getStats,
};
