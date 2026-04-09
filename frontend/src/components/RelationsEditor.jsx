import { useEffect } from 'react';

const EXPLICIT_RELATIONS = [
  { key: 'follower', label: 'Follower', placeholder: 'follower_user', desc: 'User A follows User B' },
  { key: 'friend', label: 'Friend', placeholder: 'friend_user', desc: 'User A is followed back by User B' },
  { key: 'mention', label: 'Mention', placeholder: 'mentioned_user', desc: 'User A mentions User B in a tweet' },
  { key: 'reply', label: 'Reply', placeholder: 'replied_user', desc: 'User A replies directly to User B' },
  { key: 'quoted', label: 'Quote', placeholder: 'quoted_user', desc: 'User A quotes a tweet from User B' },
];

const IMPLICIT_RELATIONS = [
  { key: 'hashtag', label: 'Shared Hashtags', placeholder: 'user_with_same_hashtag', desc: 'Two users using the same hashtags, indicating topic alignment' },
  { key: 'url', label: 'Shared URLs', placeholder: 'user_with_same_url', desc: 'Two users sharing the same links, often used in coordinated campaigns' },
];

const ALL_RELATIONS = [...EXPLICIT_RELATIONS, ...IMPLICIT_RELATIONS];

export default function RelationsEditor({ relations, onChange }) {

  useEffect(() => {
    if (relations.length !== 7) {
      onChange(ALL_RELATIONS.map(rt => ({
        source: 'target',
        target: '',
        relation: rt.key
      })));
    }
  }, []);

  const updateTarget = (index, value) => {
    const updated = [...relations];
    updated[index] = { ...updated[index], target: value };
    onChange(updated);
  };

  const loadDemoRelations = () => {
    const demo = [
      { source: 'target', target: 'user_alice', relation: 'follower' },
      { source: 'target', target: 'user_bob', relation: 'friend' },
      { source: 'target', target: 'user_carol', relation: 'mention' },
      { source: 'target', target: 'user_dave', relation: 'reply' },
      { source: 'target', target: 'user_eve', relation: 'quoted' },
      { source: 'target', target: 'user_frank', relation: 'hashtag' },
      { source: 'target', target: 'user_grace', relation: 'url' }
    ];
    onChange(demo);
  };

  return (
    <div className="relations-editor">
      <div className="section-header">
        <div className="section-icon">🔗</div>
        <div>
          <div className="section-title">Relations <span className="badge badge-human" style={{ fontSize: '11px', marginLeft: 8 }}>Mandatory</span></div>
          <div className="section-subtitle">
            Provide the 7 required explicit and implicit graph edges
          </div>
        </div>
      </div>

      <div className="demo-buttons mb-2">
        <button type="button" className="btn btn-sm btn-secondary" onClick={loadDemoRelations}>
          🔗 Load Demo Relations
        </button>
      </div>

      {/* ── Explicit Relationships ── */}
      <div className="relations-group">
        <div className="relations-group-header">
          <span className="relations-group-badge explicit">Explicit</span>
          <span className="relations-group-title">Direct Interactions</span>
        </div>
        <p className="relations-group-desc">
          These are <strong>direct interactions</strong> between users that clearly show how they connect with each other.
        </p>

        <div className="relations-table">
          <div className="relations-header" style={{ gridTemplateColumns: 'minmax(100px, 140px) 1fr' }}>
            <span>Relation Type</span>
            <span>Target User Identifier</span>
          </div>
          {relations.length === 7 && EXPLICIT_RELATIONS.map((rt, i) => (
            <div key={i} className="relation-row" style={{ gridTemplateColumns: 'minmax(100px, 140px) 1fr' }}>
              <div className="relation-label-group">
                <span className="relation-label-name">{rt.label}</span>
                <span className="relation-label-desc">{rt.desc}</span>
              </div>
              <input
                className="form-input"
                type="text"
                value={relations[i].target}
                onChange={e => updateTarget(i, e.target.value)}
                placeholder={rt.placeholder}
              />
            </div>
          ))}
        </div>
      </div>

      {/* ── Implicit Relationships ── */}
      <div className="relations-group" style={{ marginTop: '20px' }}>
        <div className="relations-group-header">
          <span className="relations-group-badge implicit">Implicit</span>
          <span className="relations-group-title">Hidden Connections</span>
        </div>
        <p className="relations-group-desc">
          These are <strong>indirect or hidden connections</strong>, which aren't obvious at first but show shared behavior.
        </p>

        <div className="relations-table">
          <div className="relations-header" style={{ gridTemplateColumns: 'minmax(100px, 140px) 1fr' }}>
            <span>Relation Type</span>
            <span>Target User Identifier</span>
          </div>
          {relations.length === 7 && IMPLICIT_RELATIONS.map((rt, idx) => {
            const i = idx + 5; // offset by explicit count
            return (
              <div key={i} className="relation-row" style={{ gridTemplateColumns: 'minmax(100px, 140px) 1fr' }}>
                <div className="relation-label-group">
                  <span className="relation-label-name">{rt.label}</span>
                  <span className="relation-label-desc">{rt.desc}</span>
                </div>
                <input
                  className="form-input"
                  type="text"
                  value={relations[i].target}
                  onChange={e => updateTarget(i, e.target.value)}
                  placeholder={rt.placeholder}
                />
              </div>
            );
          })}
        </div>
      </div>

      <p className="text-muted mt-2" style={{ fontSize: '12px' }}>
        <strong>Tip:</strong> Provide the target username or identifier for each of the 7 relationship types. If a relation is absent, leave it blank.
      </p>
    </div>
  );
}
