import { useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload) return null;
  return (
    <div style={{
      background: "rgba(17, 24, 39, 0.95)",
      border: "1px solid rgba(99, 102, 241, 0.2)",
      borderRadius: 12,
      padding: "12px 16px",
      backdropFilter: "blur(10px)"
    }}>
      <p style={{ fontWeight: 700, marginBottom: 8, color: "#f1f5f9" }}>Epoch {label}</p>
      {payload.map((entry, i) => (
        <p key={i} style={{ color: entry.color, fontSize: "0.85rem", margin: "4px 0" }}>
          {entry.name}: {typeof entry.value === "number" ? entry.value.toFixed(4) : entry.value}
        </p>
      ))}
    </div>
  );
};

const TrainingCurveChart = ({ data = [], models = ["rgcn", "gcn", "gat", "graphsage"], onModelChange, currentModel = "rgcn" }) => {
  if (!data.length) return <p style={{ color: "var(--text-muted)" }}>No training log data</p>;

  // Downsample for performance if needed
  const chartData = data.length > 200 ? data.filter((_, i) => i % 2 === 0) : data;

  return (
    <div className="chart-wrapper">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <h3 style={{ fontSize: "1.1rem", fontWeight: 600 }}>Training Curves</h3>
        <select
          value={currentModel}
          onChange={(e) => onModelChange && onModelChange(e.target.value)}
          style={{
            padding: "6px 12px",
            background: "var(--bg-tertiary)",
            border: "1px solid var(--border-color)",
            borderRadius: "var(--radius-sm)",
            color: "var(--text-primary)",
            fontSize: "0.85rem",
            cursor: "pointer",
            fontFamily: "var(--font-sans)",
          }}
        >
          {models.map((m) => (
            <option key={m} value={m}>{m.toUpperCase()}</option>
          ))}
        </select>
      </div>
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,116,139,0.1)" />
          <XAxis
            dataKey="Epoch"
            tick={{ fill: "#94a3b8", fontSize: 12 }}
            label={{ value: "Epoch", position: "insideBottom", offset: -5, fill: "#64748b" }}
          />
          <YAxis tick={{ fill: "#94a3b8", fontSize: 12 }} domain={[0, "auto"]} />
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ paddingTop: 12, fontSize: "0.85rem" }} iconType="circle" />
          <Line type="monotone" dataKey="Loss" stroke="#ef4444" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="Train_Acc" stroke="#6366f1" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="Val_Acc" stroke="#06b6d4" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default TrainingCurveChart;
