const jwt = require("jsonwebtoken");
const { JWT_SECRET } = require("../config/env");
const { error } = require("../utils/responseHelper");

const authMiddleware = (req, res, next) => {
  const authHeader = req.headers.authorization;

  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return error(res, "Authentication required", 401);
  }

  const token = authHeader.split(" ")[1];

  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    req.user = { userId: decoded.userId, role: decoded.role };
    next();
  } catch (err) {
    if (err.name === "TokenExpiredError") {
      return error(res, "Token expired, please login again", 401);
    }
    return error(res, "Invalid authentication token", 401);
  }
};

module.exports = authMiddleware;
