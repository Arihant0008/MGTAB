import { motion } from "framer-motion";
import ConfidenceMeter from "./ConfidenceMeter";
import { HiShieldCheck, HiShieldExclamation, HiClock, HiChip } from "react-icons/hi";
import "./PredictionCard.css";

const PredictionCard = ({ result }) => {
  if (!result) return null;

  const isBot = result.prediction === "bot";

  return (
    <motion.div
      className={`prediction-card glass-card ${isBot ? "card-bot" : "card-human"}`}
      initial={{ opacity: 0, y: 30, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
    >
      <div className="prediction-header">
        <div className={`prediction-icon ${isBot ? "icon-bot" : "icon-human"}`}>
          {isBot ? <HiShieldExclamation /> : <HiShieldCheck />}
        </div>
        <div>
          <h3 className="prediction-username">@{result.username}</h3>
          <span className={`badge ${isBot ? "badge-bot" : "badge-human"}`}>
            {isBot ? "🤖 BOT" : "✅ HUMAN"}
          </span>
        </div>
      </div>

      <div className="prediction-body">
        <ConfidenceMeter confidence={result.confidence} />

        <div className="prediction-meta">
          <div className="meta-item">
            <span className="meta-label">Risk Level</span>
            <span className={`badge badge-${result.riskLevel}`}>
              {result.riskLevel?.toUpperCase()}
            </span>
          </div>
          <div className="meta-item">
            <HiChip className="meta-icon" />
            <span className="meta-label">Model</span>
            <span className="meta-value">{result.modelUsed || "RGCN"}</span>
          </div>
          <div className="meta-item">
            <HiClock className="meta-icon" />
            <span className="meta-label">Inference</span>
            <span className="meta-value">{result.inferenceTimeMs}ms</span>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default PredictionCard;
