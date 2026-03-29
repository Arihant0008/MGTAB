const inferenceService = require("../services/inferenceService");
const Prediction = require("../models/Prediction");
const BatchJob = require("../models/BatchJob");
const User = require("../models/User");
const { success, accepted, error } = require("../utils/responseHelper");
const logger = require("../utils/logger");
const multer = require("multer");
const csvParser = require("csv-parser");
const fs = require("fs");
const path = require("path");

// ── Multer config for CSV uploads ────────────────────────────────────────────
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const dir = path.join(__dirname, "..", "uploads");
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    cb(null, dir);
  },
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 5 * 1024 * 1024 }, // 5MB
  fileFilter: (req, file, cb) => {
    if (file.mimetype === "text/csv" || file.originalname.endsWith(".csv")) {
      cb(null, true);
    } else {
      cb(new Error("Only CSV files are allowed"), false);
    }
  },
}).single("file");

/**
 * Determine risk level from confidence.
 */
const getRiskLevel = (confidence, prediction) => {
  // For bot predictions, higher confidence = higher risk
  // For human predictions, lower confidence = higher risk (uncertain)
  if (prediction === "bot") {
    if (confidence >= 0.8) return "high";
    if (confidence >= 0.5) return "medium";
    return "low";
  } else {
    if (confidence >= 0.8) return "low";
    if (confidence >= 0.5) return "medium";
    return "high";
  }
};

/**
 * POST /api/predict/single
 */
const predictSingle = async (req, res, next) => {
  try {
    const { username } = req.body;
    const startTime = Date.now();

    // Call Python inference service
    const inferenceResult = await inferenceService.predictSingle(username);

    const riskLevel = getRiskLevel(inferenceResult.confidence, inferenceResult.prediction);
    const inferenceTimeMs = inferenceResult.inferenceTimeMs || (Date.now() - startTime);

    // Save to MongoDB
    const prediction = await Prediction.create({
      userId: req.user.userId,
      inputUsername: username,
      nodeIndex: inferenceResult.nodeIndex || null,
      prediction: inferenceResult.prediction,
      confidence: inferenceResult.confidence,
      riskLevel,
      modelUsed: "RGCN",
      inferenceTimeMs,
    });

    // Increment user's total queries
    await User.findByIdAndUpdate(req.user.userId, { $inc: { totalQueries: 1 } });

    logger.info(`Prediction: ${username} → ${inferenceResult.prediction} (${inferenceResult.confidence})`);

    return success(res, {
      predictionId: prediction._id,
      username,
      prediction: inferenceResult.prediction,
      confidence: inferenceResult.confidence,
      riskLevel,
      modelUsed: "RGCN",
      inferenceTimeMs,
      timestamp: prediction.createdAt,
    });
  } catch (err) {
    if (err.statusCode) return error(res, err.message, err.statusCode);
    next(err);
  }
};

/**
 * POST /api/predict/batch
 */
const predictBatch = async (req, res, next) => {
  upload(req, res, async (uploadErr) => {
    if (uploadErr) {
      return error(res, uploadErr.message, 400);
    }

    if (!req.file) {
      return error(res, "CSV file is required", 400);
    }

    try {
      const usernames = [];
      const filePath = req.file.path;

      // Parse CSV
      await new Promise((resolve, reject) => {
        fs.createReadStream(filePath)
          .pipe(csvParser())
          .on("data", (row) => {
            const username = row.username || row.Username || row.handle || Object.values(row)[0];
            if (username && username.trim()) {
              usernames.push(username.trim());
            }
          })
          .on("end", resolve)
          .on("error", reject);
      });

      // Clean up uploaded file
      fs.unlinkSync(filePath);

      if (usernames.length === 0) {
        return error(res, "CSV must contain a 'username' column with at least one entry", 400);
      }

      if (usernames.length > 500) {
        return error(res, "Maximum 500 usernames per batch", 400);
      }

      // Create batch job
      const batchJob = await BatchJob.create({
        userId: req.user.userId,
        totalAccounts: usernames.length,
        status: "processing",
      });

      // Process batch in background (non-blocking)
      processBatchAsync(batchJob._id, usernames, req.user.userId);

      return accepted(res, {
        jobId: batchJob._id,
        totalAccounts: usernames.length,
        status: "processing",
      }, "Batch job started");
    } catch (err) {
      next(err);
    }
  });
};

/**
 * Process batch predictions asynchronously.
 */
const processBatchAsync = async (jobId, usernames, userId) => {
  try {
    const batchResult = await inferenceService.predictBatch(usernames);
    const results = [];
    let botsFound = 0;
    let humansFound = 0;

    for (const r of batchResult.results) {
      const riskLevel = getRiskLevel(r.confidence, r.prediction);
      results.push({
        username: r.username,
        prediction: r.prediction,
        confidence: r.confidence,
        riskLevel,
      });
      if (r.prediction === "bot") botsFound++;
      else humansFound++;
    }

    await BatchJob.findByIdAndUpdate(jobId, {
      status: "completed",
      processedCount: results.length,
      botsFound,
      humansFound,
      results,
      completedAt: new Date(),
    });

    await User.findByIdAndUpdate(userId, { $inc: { totalQueries: results.length } });

    logger.info(`Batch job ${jobId} completed: ${results.length} predictions`);
  } catch (err) {
    await BatchJob.findByIdAndUpdate(jobId, { status: "failed" });
    logger.error(`Batch job ${jobId} failed: ${err.message}`);
  }
};

/**
 * GET /api/predict/history
 */
const getHistory = async (req, res, next) => {
  try {
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 20;
    const skip = (page - 1) * limit;

    const [predictions, total] = await Promise.all([
      Prediction.find({ userId: req.user.userId })
        .sort({ createdAt: -1 })
        .skip(skip)
        .limit(limit),
      Prediction.countDocuments({ userId: req.user.userId }),
    ]);

    return success(res, {
      predictions,
      pagination: {
        page,
        limit,
        total,
        pages: Math.ceil(total / limit),
      },
    });
  } catch (err) {
    next(err);
  }
};

/**
 * GET /api/predict/history/:id
 */
const getHistoryById = async (req, res, next) => {
  try {
    const prediction = await Prediction.findOne({
      _id: req.params.id,
      userId: req.user.userId,
    });

    if (!prediction) return error(res, "Prediction not found", 404);
    return success(res, { prediction });
  } catch (err) {
    next(err);
  }
};

/**
 * DELETE /api/predict/history/:id
 */
const deleteHistory = async (req, res, next) => {
  try {
    const prediction = await Prediction.findOneAndDelete({
      _id: req.params.id,
      userId: req.user.userId,
    });

    if (!prediction) return error(res, "Prediction not found", 404);
    return success(res, null, "Prediction deleted");
  } catch (err) {
    next(err);
  }
};

/**
 * GET /api/predict/batch/:id
 */
const getBatchJob = async (req, res, next) => {
  try {
    const job = await BatchJob.findOne({
      _id: req.params.id,
      userId: req.user.userId,
    });

    if (!job) return error(res, "Batch job not found", 404);
    return success(res, { job });
  } catch (err) {
    next(err);
  }
};

module.exports = {
  predictSingle,
  predictBatch,
  getHistory,
  getHistoryById,
  deleteHistory,
  getBatchJob,
};
