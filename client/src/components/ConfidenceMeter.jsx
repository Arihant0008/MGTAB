import { useEffect, useState } from "react";
import "./ConfidenceMeter.css";

const ConfidenceMeter = ({ confidence = 0, size = 140 }) => {
  const [animatedValue, setAnimatedValue] = useState(0);
  const percent = Math.round(confidence * 100);

  useEffect(() => {
    const timer = setTimeout(() => setAnimatedValue(percent), 100);
    return () => clearTimeout(timer);
  }, [percent]);

  const radius = (size - 16) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (animatedValue / 100) * circumference;

  const getColor = () => {
    if (percent >= 80) return "#ef4444";
    if (percent >= 50) return "#f59e0b";
    return "#22c55e";
  };

  return (
    <div className="confidence-meter" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="rgba(100, 116, 139, 0.15)"
          strokeWidth="8"
        />
        {/* Progress circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={getColor()}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{
            transition: "stroke-dashoffset 1s ease-out, stroke 0.3s ease",
            filter: `drop-shadow(0 0 6px ${getColor()}50)`,
          }}
        />
      </svg>
      <div className="confidence-value">
        <span className="confidence-number" style={{ color: getColor() }}>
          {animatedValue}
        </span>
        <span className="confidence-percent">%</span>
      </div>
    </div>
  );
};

export default ConfidenceMeter;
