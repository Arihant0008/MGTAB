import { useEffect } from 'react';

const RELATION_TYPES = [
  { key: 'follower', label: 'Follower', placeholder: 'follower_user' },
  { key: 'friend', label: 'Friend', placeholder: 'friend_user' },
  { key: 'mention', label: 'Mention', placeholder: 'mentioned_user' },
  { key: 'reply', label: 'Reply', placeholder: 'replied_user' },
  { key: 'quoted', label: 'Quote', placeholder: 'quoted_user' },
  { key: 'hashtag', label: 'Shared Hashtags', placeholder: 'user_with_same_hashtag' },
  { key: 'url', label: 'Shared URLs', placeholder: 'user_with_same_url' },
];

export default function RelationsEditor({ relations, onChange }) {

  useEffect(() => {
    if (relations.length !== 7) {
      onChange(RELATION_TYPES.map(rt => ({
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

      <div className="relations-table">
        <div className="relations-header" style={{ gridTemplateColumns: 'minmax(120px, 150px) 1fr' }}>
          <span>Relation Type</span>
          <span>Target User Identifier</span>
        </div>
        {relations.length === 7 && relations.map((rel, i) => (
          <div key={i} className="relation-row" style={{ gridTemplateColumns: 'minmax(120px, 150px) 1fr' }}>
            <span style={{ fontSize: '14px', fontWeight: 500, color: 'var(--text-primary)' }}>
              {RELATION_TYPES[i].label}
            </span>
            <input
              className="form-input"
              type="text"
              value={rel.target}
              onChange={e => updateTarget(i, e.target.value)}
              placeholder={RELATION_TYPES[i].placeholder}
            />
          </div>
        ))}
      </div>

      <p className="text-muted mt-2" style={{ fontSize: '12px' }}>
        <strong>Tip:</strong> Provide the target username or identifier for each of the 7 relationship types. If a relation is absent, leave it blank.
      </p>
    </div>
  );
}
