const mongoose = require("mongoose");

const batchResultSchema = new mongoose.Schema(
  {
    username: String,
    prediction: String,
    confidence: Number,
    riskLevel: String,
  },
  { _id: false }
);

const batchJobSchema = new mongoose.Schema(
  {
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },
    status: {
      type: String,
      enum: ["pending", "processing", "completed", "failed"],
      default: "pending",
    },
    totalAccounts: {
      type: Number,
      required: true,
    },
    processedCount: {
      type: Number,
      default: 0,
    },
    botsFound: {
      type: Number,
      default: 0,
    },
    humansFound: {
      type: Number,
      default: 0,
    },
    results: [batchResultSchema],
    uploadedAt: {
      type: Date,
      default: Date.now,
    },
    completedAt: {
      type: Date,
      default: null,
    },
  },
  {
    timestamps: true,
  }
);

// Indexes
batchJobSchema.index({ userId: 1, status: 1 });

module.exports = mongoose.model("BatchJob", batchJobSchema);
