const rateLimit = require("express-rate-limit");
const { RATE_LIMIT_WINDOW_MS, RATE_LIMIT_MAX_AUTH, RATE_LIMIT_MAX_PREDICT } = require("../config/env");

const authLimiter = rateLimit({
  windowMs: RATE_LIMIT_WINDOW_MS,
  max: RATE_LIMIT_MAX_AUTH,
  message: {
    success: false,
    message: "Too many authentication attempts, please try again later",
    statusCode: 429,
  },
  standardHeaders: true,
  legacyHeaders: false,
});

const predictLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: RATE_LIMIT_MAX_PREDICT,
  message: {
    success: false,
    message: "Too many prediction requests, please try again later",
    statusCode: 429,
  },
  standardHeaders: true,
  legacyHeaders: false,
});

const generalLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 200,
  message: {
    success: false,
    message: "Too many requests, please try again later",
    statusCode: 429,
  },
  standardHeaders: true,
  legacyHeaders: false,
});

module.exports = { authLimiter, predictLimiter, generalLimiter };
