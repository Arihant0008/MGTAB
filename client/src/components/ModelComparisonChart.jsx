import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from "recharts";

const COLORS = {
  "Test Accuracy": "#6366f1",
  "Bot Recall": "#06b6d4",
  "Train Accuracy": "#8b5cf6",
};

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
      <p style={{ fontWeight: 700, marginBottom: 8, color: "#f1f5f9" }}>{label}</p>
      {payload.map((entry, i) => (
        <p key={i} style={{ color: entry.color, fontSize: "0.85rem", margin: "4px 0" }}>
          {entry.name}: {(entry.value * 100).toFixed(1)}%
        </p>
      ))}
    </div>
  );
};

const ModelComparisonChart = ({ data = [] }) => {
  if (!data.length) return <p style={{ color: "var(--text-muted)" }}>No model data available</p>;

  const chartData = data.map((m) => ({
    name: m.Model,
    "Test Accuracy": m["Test Accuracy"],
    "Bot Recall": m["Bot Recall"],
    "Train Accuracy": m["Train Accuracy"],
  }));

  return (
    <div className="chart-wrapper">
      <h3 style={{ marginBottom: 16, fontSize: "1.1rem", fontWeight: 600 }}>Model Performance Comparison</h3>
      <ResponsiveContainer width="100%" height={320}>
        <BarChart data={chartData} barGap={4} barCategoryGap="20%">
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,116,139,0.1)" />
          <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 13 }} />
          <YAxis
            domain={[0.6, 1]}
            tick={{ fill: "#94a3b8", fontSize: 12 }}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ paddingTop: 12, fontSize: "0.85rem" }}
            iconType="circle"
          />
          <Bar dataKey="Test Accuracy" fill={COLORS["Test Accuracy"]} radius={[6, 6, 0, 0]} />
          <Bar dataKey="Bot Recall" fill={COLORS["Bot Recall"]} radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default ModelComparisonChart;
