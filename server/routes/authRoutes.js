const express = require("express");
const router = express.Router();
const authController = require("../controllers/authController");
const authMiddleware = require("../middleware/authMiddleware");
const { registerRules, loginRules, validate } = require("../middleware/validator");
const { authLimiter } = require("../middleware/rateLimiter");

// POST /api/auth/register
router.post("/register", authLimiter, registerRules, validate, authController.register);

// POST /api/auth/login
router.post("/login", authLimiter, loginRules, validate, authController.login);

// GET /api/auth/me
router.get("/me", authMiddleware, authController.getMe);

// PUT /api/auth/update
router.put("/update", authMiddleware, authController.updateProfile);

module.exports = router;
