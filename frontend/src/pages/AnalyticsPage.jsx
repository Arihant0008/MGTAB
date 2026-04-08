import './AnalyticsPage.css';

const models = [
  { name: 'GCN', type: 'Homogeneous', trainAcc: 78.31, testAcc: 79.21, recall: 68.70, color: '#8b5cf6' },
  { name: 'GAT', type: 'Homogeneous', trainAcc: 79.54, testAcc: 81.67, recall: 84.53, color: '#f59e0b' },
  { name: 'GraphSAGE', type: 'Homogeneous', trainAcc: 88.14, testAcc: 87.16, recall: 88.85, color: '#06b6d4' },
  { name: 'RGCN', type: 'Heterogeneous', trainAcc: 89.50, testAcc: 88.23, recall: 90.29, color: '#3b82f6' },
];

const relations = [
  { id: 0, name: 'Follower', direction: '→', desc: 'User A is followed by User B' },
  { id: 1, name: 'Friend', direction: '→', desc: 'User A follows User B' },
  { id: 2, name: 'Mention', direction: '→', desc: 'User A mentions User B in tweets' },
  { id: 3, name: 'Reply', direction: '→', desc: 'User A replies to User B' },
  { id: 4, name: 'Quoted', direction: '→', desc: 'User A quotes User B' },
  { id: 5, name: 'URL', direction: '↔', desc: 'Co-occurrence via shared URLs' },
  { id: 6, name: 'Hashtag', direction: '↔', desc: 'Co-occurrence via shared hashtags' },
];

const datasetStats = [
  { label: 'Total Users', value: '410,199', icon: '👥' },
  { label: 'Expert Annotated', value: '10,199', icon: '🏷️' },
  { label: 'Human Accounts', value: '7,451', icon: '👤' },
  { label: 'Bot Accounts', value: '2,748', icon: '🤖' },
  { label: 'Relation Types', value: '7', icon: '🔗' },
  { label: 'Feature Dim', value: '788', icon: '📐' },
  { label: 'Profile Features', value: '20', icon: '📋' },
  { label: 'Tweet Encoder', value: 'LaBSE', icon: '🧠' },
];

const top5BotFeatures = [
  { name: 'has_url', ig: 0.064248, desc: 'Most bots have empty URL' },
  { name: 'default_profile', ig: 0.025997, desc: 'Bots tend to use default profile' },
  { name: 'default_profile_image', ig: 0.025402, desc: 'Default avatar indicates bot' },
  { name: 'followers_friends_ratio', ig: 0.391857, desc: 'Bots have low follower/friend ratio (numerical)' },
  { name: 'listed_count', ig: 0.333101, desc: 'Bots appear in more/fewer public lists' },
];

export default function AnalyticsPage() {
  return (
    <div className="page analytics-page">
      <div className="container">
        <div className="analytics-header animate-fade-in">
          <h1 className="analytics-title">Analytics & Research</h1>
          <p className="text-secondary" style={{ fontSize: '15px', maxWidth: 600, margin: '0 auto' }}>
            MGTAB benchmark dataset analysis, model comparisons, and feature importance
          </p>
        </div>

        {/* Dataset Stats */}
        <section className="analytics-section animate-slide-up">
          <h2 className="section-title mb-3">📊 MGTAB Dataset Overview</h2>
          <div className="stats-grid">
            {datasetStats.map(s => (
              <div key={s.label} className="stat-card glass-card">
                <div className="stat-icon">{s.icon}</div>
                <div className="stat-value">{s.value}</div>
                <div className="stat-label">{s.label}</div>
              </div>
            ))}
          </div>
        </section>

        {/* Model Comparison Table */}
        <section className="analytics-section animate-slide-up" style={{ animationDelay: '0.1s' }}>
          <h2 className="section-title mb-3">🏆 Model Comparison</h2>
          <div className="table-wrapper glass-card">
            <table className="analytics-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Type</th>
                  <th>Train Acc</th>
                  <th>Test Acc</th>
                  <th>Bot Recall</th>
                  <th>Performance</th>
                </tr>
              </thead>
              <tbody>
                {models.map(m => (
                  <tr key={m.name} className={m.name === 'RGCN' ? 'table-best' : ''}>
                    <td>
                      <span className="model-dot" style={{ background: m.color }} />
                      <strong>{m.name}</strong>
                      {m.name === 'RGCN' && <span className="badge badge-human" style={{ marginLeft: 8, fontSize: 10 }}>Best</span>}
                    </td>
                    <td><span className="text-muted">{m.type}</span></td>
                    <td>{m.trainAcc}%</td>
                    <td><strong>{m.testAcc}%</strong></td>
                    <td>{m.recall}%</td>
                    <td>
                      <div className="mini-bar">
                        <div className="mini-bar-fill" style={{ width: `${m.testAcc}%`, background: m.color }} />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* Relations */}
        <section className="analytics-section animate-slide-up" style={{ animationDelay: '0.2s' }}>
          <h2 className="section-title mb-3">🔗 7 Relation Types</h2>
          <div className="relations-grid">
            {relations.map(r => (
              <div key={r.id} className="relation-card glass-card">
                <div className="relation-id">R{r.id}</div>
                <div className="relation-name">{r.name}</div>
                <div className="relation-dir">{r.direction}</div>
                <div className="relation-desc">{r.desc}</div>
              </div>
            ))}
          </div>
        </section>

        {/* Top features */}
        <section className="analytics-section animate-slide-up" style={{ animationDelay: '0.3s' }}>
          <h2 className="section-title mb-3">🔬 Top Bot Detection Features (by Information Gain)</h2>
          <div className="features-list">
            {top5BotFeatures.map((f, i) => (
              <div key={f.name} className="feature-item glass-card">
                <div className="feature-rank">#{i + 1}</div>
                <div className="feature-info">
                  <div className="feature-name font-mono">{f.name}</div>
                  <div className="feature-desc text-muted">{f.desc}</div>
                </div>
                <div className="feature-ig">
                  <div className="feature-ig-value">{f.ig.toFixed(4)}</div>
                  <div className="feature-ig-label">IG Score</div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Paper reference */}
        <section className="analytics-section animate-slide-up" style={{ animationDelay: '0.4s' }}>
          <div className="paper-card glass-card">
            <div className="paper-icon">📄</div>
            <div className="paper-info">
              <div className="paper-title">MGTAB: A Multi-Relational Graph-Based Twitter Account Detection Benchmark</div>
              <div className="paper-authors text-muted">
                Shuhao Shi, Kai Qiao, Jian Chen, Shuai Yang, Jie Yang, Baojie Song, Linyuan Wang, Bin Yan
              </div>
              <div className="paper-venue text-muted" style={{ fontSize: '12px', marginTop: 4 }}>
                arXiv:2301.01123 · 2023
              </div>
            </div>
            <a
              href="https://github.com/GraphDetec/MGTAB"
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-sm btn-secondary"
            >
              GitHub ↗
            </a>
          </div>
        </section>
      </div>
    </div>
  );
}
