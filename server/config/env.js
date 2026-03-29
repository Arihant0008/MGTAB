const dotenv = require("dotenv");
const path = require("path");

dotenv.config({ path: path.join(__dirname, "..", ".env") });

module.exports = {
  PORT: process.env.PORT || 4000,
  NODE_ENV: process.env.NODE_ENV || "development",
  MONGO_URI: process.env.MONGO_URI || "mongodb://localhost:27017/mgtab",
  JWT_SECRET: process.env.JWT_SECRET || "fallback_secret_change_me",
  JWT_EXPIRES_IN: process.env.JWT_EXPIRES_IN || "24h",
  PYTHON_SERVICE_URL: process.env.PYTHON_SERVICE_URL || "http://localhost:5000",
  RATE_LIMIT_WINDOW_MS: parseInt(process.env.RATE_LIMIT_WINDOW_MS) || 900000,
  RATE_LIMIT_MAX_AUTH: parseInt(process.env.RATE_LIMIT_MAX_AUTH) || 10,
  RATE_LIMIT_MAX_PREDICT: parseInt(process.env.RATE_LIMIT_MAX_PREDICT) || 30,
  CLIENT_URL: process.env.CLIENT_URL || "http://localhost:5173",
};
