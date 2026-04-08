import './ResultCard.css';

export default function ResultCard({ result, loading }) {
  if (loading) {
    return (
      <div className="result-card glass-card result-loading">
        <div className="spinner spinner-lg" />
        <p className="text-muted mt-2">Running RGCN inference...</p>
        <p className="text-muted" style={{ fontSize: '12px' }}>
          Computing features → Building graph → Forward pass
        </p>
      </div>
    );
  }

  if (!result) return null;

  const isBot = result.label_pred === 'bot';
  const confidence = Math.round(result.confidence * 100);
  const probBot = Math.round(result.prob_bot * 100);
  const probHuman = Math.round(result.prob_human * 100);

  return (
    <div className={`result-card glass-card animate-slide-up ${isBot ? 'result-bot' : 'result-human'}`}>
      {/* Main result */}
      <div className="result-main">
        <div className={`result-icon ${isBot ? 'result-icon-bot' : 'result-icon-human'}`}>
          {isBot ? '🤖' : '👤'}
        </div>
        <div className="result-label">
          <span className={`result-label-text ${isBot ? 'text-bot' : 'text-human'}`}>
            {isBot ? 'Bot Detected' : 'Human Account'}
          </span>
          <span className="result-confidence">
            {confidence}% confidence
          </span>
        </div>
      </div>

      {/* Probability bars */}
      <div className="result-probs">
        <div className="result-prob">
          <div className="result-prob-header">
            <span className="result-prob-label">👤 Human</span>
            <span className="result-prob-value">{probHuman}%</span>
          </div>
          <div className="result-prob-bar">
            <div
              className="result-prob-fill result-prob-fill-human"
              style={{ width: `${probHuman}%` }}
            />
          </div>
        </div>

        <div className="result-prob">
          <div className="result-prob-header">
            <span className="result-prob-label">🤖 Bot</span>
            <span className="result-prob-value">{probBot}%</span>
          </div>
          <div className="result-prob-bar">
            <div
              className="result-prob-fill result-prob-fill-bot"
              style={{ width: `${probBot}%` }}
            />
          </div>
        </div>
      </div>

      {/* Graph info */}
      {result.graph_info && (
        <div className="result-meta">
          <div className="result-meta-item">
            <span className="result-meta-label">Graph Nodes</span>
            <span className="result-meta-value">{result.graph_info.num_nodes}</span>
          </div>
          <div className="result-meta-item">
            <span className="result-meta-label">Graph Edges</span>
            <span className="result-meta-value">{result.graph_info.num_edges}</span>
          </div>
          <div className="result-meta-item">
            <span className="result-meta-label">Model</span>
            <span className="result-meta-value">RGCN</span>
          </div>
          <div className="result-meta-item">
            <span className="result-meta-label">Features</span>
            <span className="result-meta-value">788-dim</span>
          </div>
        </div>
      )}
    </div>
  );
}
