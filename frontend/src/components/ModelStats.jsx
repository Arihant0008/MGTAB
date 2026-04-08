import './ModelStats.css';

const models = [
  { name: 'GCN', accuracy: 79.21, recall: 68.70, color: '#8b5cf6' },
  { name: 'GAT', accuracy: 81.67, recall: 84.53, color: '#f59e0b' },
  { name: 'GraphSAGE', accuracy: 87.16, recall: 88.85, color: '#06b6d4' },
  { name: 'RGCN', accuracy: 88.23, recall: 90.29, color: '#3b82f6', best: true },
];

const pipeline = [
  { icon: '👤', title: 'Profile Data', desc: '20 features extracted' },
  { icon: '💬', title: 'Tweet Text', desc: '768-dim LaBSE embedding' },
  { icon: '🔗', title: 'Relations', desc: '7 edge types in graph' },
  { icon: '🧠', title: 'RGCN', desc: 'Graph neural network' },
  { icon: '✅', title: 'Result', desc: 'Bot or Human prediction' },
];

export default function ModelStats() {
  return (
    <section className="model-stats-section">
      <div className="container">
        {/* Pipeline */}
        <div className="pipeline-header">
          <h2 className="section-title">How It Works</h2>
          <p className="text-muted" style={{ fontSize: '14px' }}>
            From raw profile data to bot detection in 5 steps
          </p>
        </div>

        <div className="pipeline">
          {pipeline.map((step, i) => (
            <div key={i} className="pipeline-step glass-card animate-fade-in" style={{ animationDelay: `${i * 0.1}s` }}>
              <div className="pipeline-icon">{step.icon}</div>
              <div className="pipeline-title">{step.title}</div>
              <div className="pipeline-desc">{step.desc}</div>
              {i < pipeline.length - 1 && <div className="pipeline-arrow">→</div>}
            </div>
          ))}
        </div>

        {/* Model comparison */}
        <div className="models-header mt-4">
          <h2 className="section-title">Model Performance</h2>
          <p className="text-muted" style={{ fontSize: '14px' }}>
            Benchmarked on MGTAB — 10,199 expert-annotated accounts
          </p>
        </div>

        <div className="models-grid">
          {models.map((m) => (
            <div key={m.name} className={`model-card glass-card ${m.best ? 'model-card-best' : ''}`}>
              {m.best && <div className="model-best-badge">⭐ Best Model</div>}
              <div className="model-name" style={{ color: m.color }}>{m.name}</div>

              <div className="model-metric">
                <div className="model-metric-label">Test Accuracy</div>
                <div className="model-metric-value">{m.accuracy}%</div>
                <div className="model-bar">
                  <div
                    className="model-bar-fill"
                    style={{ width: `${m.accuracy}%`, background: m.color }}
                  />
                </div>
              </div>

              <div className="model-metric">
                <div className="model-metric-label">Bot Recall</div>
                <div className="model-metric-value">{m.recall}%</div>
                <div className="model-bar">
                  <div
                    className="model-bar-fill"
                    style={{ width: `${m.recall}%`, background: m.color, opacity: 0.7 }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
