const { body, param, query, validationResult } = require("express-validator");
const { error } = require("../utils/responseHelper");

// ── Run validation and return errors ─────────────────────────────────────────
const validate = (req, res, next) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    const formatted = errors.array().map((e) => ({
      field: e.path,
      message: e.msg,
    }));
    return error(res, "Validation failed", 400, formatted);
  }
  next();
};

// ── Auth Validations ─────────────────────────────────────────────────────────
const registerRules = [
  body("name")
    .trim()
    .notEmpty().withMessage("Name is required")
    .isLength({ min: 2, max: 50 }).withMessage("Name must be 2–50 characters")
    .matches(/^[a-zA-Z\s]+$/).withMessage("Name can only contain letters and spaces"),
  body("email")
    .trim()
    .notEmpty().withMessage("Email is required")
    .isEmail().withMessage("Invalid email format")
    .normalizeEmail(),
  body("password")
    .notEmpty().withMessage("Password is required")
    .isLength({ min: 8 }).withMessage("Password must be at least 8 characters")
    .matches(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*])/)
    .withMessage("Password must contain 1 uppercase, 1 lowercase, 1 digit, 1 special char (!@#$%^&*)"),
  body("confirmPassword")
    .notEmpty().withMessage("Confirm password is required")
    .custom((value, { req }) => {
      if (value !== req.body.password) throw new Error("Passwords do not match");
      return true;
    }),
];

const loginRules = [
  body("email")
    .trim()
    .notEmpty().withMessage("Email is required")
    .isEmail().withMessage("Invalid email format")
    .normalizeEmail(),
  body("password")
    .notEmpty().withMessage("Password is required"),
];

// ── Predict Validations ──────────────────────────────────────────────────────
const predictSingleRules = [
  body("username")
    .trim()
    .notEmpty().withMessage("Username is required")
    .isLength({ min: 1, max: 50 }).withMessage("Username must be 1–50 characters")
    .matches(/^[a-zA-Z0-9_]+$/).withMessage("Username can only contain letters, numbers, and underscores")
    .escape(),
];

// ── Analytics Validations ────────────────────────────────────────────────────
const modelParamRules = [
  param("model")
    .isIn(["gcn", "gat", "graphsage", "rgcn"])
    .withMessage("Model must be one of: gcn, gat, graphsage, rgcn"),
];

const paginationRules = [
  query("page")
    .optional()
    .isInt({ min: 1 }).withMessage("Page must be a positive integer")
    .toInt(),
  query("limit")
    .optional()
    .isInt({ min: 1, max: 100 }).withMessage("Limit must be between 1 and 100")
    .toInt(),
];

module.exports = {
  validate,
  registerRules,
  loginRules,
  predictSingleRules,
  modelParamRules,
  paginationRules,
};
