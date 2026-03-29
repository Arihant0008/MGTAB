const ConfusionMatrixHeatmap = () => {
  // RGCN confusion matrix values from research
  const matrix = [
    { actual: "Human", predicted: "Human", value: 850, label: "TN" },
    { actual: "Human", predicted: "Bot", value: 78, label: "FP" },
    { actual: "Bot", predicted: "Human", value: 35, label: "FN" },
    { actual: "Bot", predicted: "Bot", value: 327, label: "TP" },
  ];

  const getColor = (value) => {
    const maxVal = 850;
    const intensity = Math.floor((value / maxVal) * 255);
    if (value > 300) return `rgba(99, 102, 241, ${0.3 + (value / maxVal) * 0.7})`;
    if (value > 50) return `rgba(245, 158, 11, ${0.3 + (value / maxVal) * 0.5})`;
    return `rgba(34, 197, 94, ${0.2 + (value / maxVal) * 0.8})`;
  };

  return (
    <div className="chart-wrapper">
      <h3 style={{ marginBottom: 16, fontSize: "1.1rem", fontWeight: 600 }}>
        RGCN Confusion Matrix
      </h3>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
        <div style={{ display: "flex", gap: 4, fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: 4 }}>
          <span style={{ width: 100 }}></span>
          <span style={{ width: 120, textAlign: "center", fontWeight: 600 }}>Pred: Human</span>
          <span style={{ width: 120, textAlign: "center", fontWeight: 600 }}>Pred: Bot</span>
        </div>
        {["Human", "Bot"].map((actual) => (
          <div key={actual} style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 100, fontSize: "0.8rem", color: "var(--text-muted)", textAlign: "right", paddingRight: 8, fontWeight: 600 }}>
              Actual: {actual}
            </span>
            {["Human", "Bot"].map((predicted) => {
              const cell = matrix.find((m) => m.actual === actual && m.predicted === predicted);
              return (
                <div
                  key={predicted}
                  style={{
                    width: 120,
                    height: 80,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    background: getColor(cell.value),
                    borderRadius: 8,
                    border: "1px solid rgba(100,116,139,0.1)",
                  }}
                >
                  <span style={{ fontSize: "1.5rem", fontWeight: 800, fontFamily: "var(--font-mono)" }}>
                    {cell.value}
                  </span>
                  <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontWeight: 600 }}>
                    {cell.label}
                  </span>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
};

export default ConfusionMatrixHeatmap;
