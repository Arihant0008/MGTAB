const authService = require("../services/authService");
const { success, created, error } = require("../utils/responseHelper");
const logger = require("../utils/logger");

/**
 * POST /api/auth/register
 */
const register = async (req, res, next) => {
  try {
    const { name, email, password } = req.body;
    const result = await authService.registerUser({ name, email, password });
    logger.info(`User registered: ${email}`);
    return created(res, result, "Registration successful");
  } catch (err) {
    if (err.statusCode) {
      return error(res, err.message, err.statusCode);
    }
    next(err);
  }
};

/**
 * POST /api/auth/login
 */
const login = async (req, res, next) => {
  try {
    const { email, password } = req.body;
    const result = await authService.loginUser({ email, password });
    logger.info(`User logged in: ${email}`);
    return success(res, result, "Login successful");
  } catch (err) {
    if (err.statusCode) {
      return error(res, err.message, err.statusCode);
    }
    next(err);
  }
};

/**
 * GET /api/auth/me
 */
const getMe = async (req, res, next) => {
  try {
    const user = await authService.getUserById(req.user.userId);
    if (!user) return error(res, "User not found", 404);
    return success(res, { user });
  } catch (err) {
    next(err);
  }
};

/**
 * PUT /api/auth/update
 */
const updateProfile = async (req, res, next) => {
  try {
    const user = await authService.updateUser(req.user.userId, req.body);
    if (!user) return error(res, "User not found", 404);
    return success(res, { user }, "Profile updated");
  } catch (err) {
    next(err);
  }
};

module.exports = { register, login, getMe, updateProfile };
