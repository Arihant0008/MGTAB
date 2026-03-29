const mongoose = require("mongoose");

const predictionSchema = new mongoose.Schema(
  {
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },
    inputUsername: {
      type: String,
      required: true,
      trim: true,
    },
    nodeIndex: {
      type: Number,
      default: null,
    },
    prediction: {
      type: String,
      enum: ["bot", "human"],
      required: true,
    },
    confidence: {
      type: Number,
      min: 0,
      max: 1,
      required: true,
    },
    riskLevel: {
      type: String,
      enum: ["low", "medium", "high"],
      required: true,
    },
    modelUsed: {
      type: String,
      default: "RGCN",
    },
    inferenceTimeMs: {
      type: Number,
    },
  },
  {
    timestamps: true,
  }
);

// Indexes
predictionSchema.index({ userId: 1, createdAt: -1 });
predictionSchema.index({ inputUsername: 1 });

module.exports = mongoose.model("Prediction", predictionSchema);
