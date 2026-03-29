const cors = require("cors");
const { CLIENT_URL } = require("./env");

const corsOptions = {
  origin: [CLIENT_URL, "http://localhost:5173", "http://localhost:3000"],
  credentials: true,
  methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization"],
};

module.exports = cors(corsOptions);
