const RELATION_OPTIONS = [
  'follower', 'friend', 'mention', 'reply', 'quoted', 'url', 'hashtag'
];

export default function RelationsEditor({ relations, onChange }) {

  const addRelation = () => {
    onChange([...relations, { source: 'target', target: '', relation: 'follower' }]);
  };

  const updateRelation = (index, key, value) => {
    const updated = [...relations];
    updated[index] = { ...updated[index], [key]: value };
    onChange(updated);
  };

  const removeRelation = (index) => {
    onChange(relations.filter((_, i) => i !== index));
  };

  const loadDemoRelations = () => {
    onChange([
      { source: 'target', target: 'user_alice', relation: 'friend' },
      { source: 'user_alice', target: 'target', relation: 'follower' },
      { source: 'target', target: 'user_bob', relation: 'mention' },
      { source: 'target', target: 'user_carol', relation: 'reply' },
    ]);
  };

  return (
    <div className="relations-editor">
      <div className="section-header">
        <div className="section-icon">🔗</div>
        <div>
          <div className="section-title">Relations <span className="badge badge-human" style={{ fontSize: '11px', marginLeft: 8 }}>Optional</span></div>
          <div className="section-subtitle">
            Define graph edges — more relations = better accuracy
          </div>
        </div>
      </div>

      <div className="demo-buttons mb-2">
        <button type="button" className="btn btn-sm btn-secondary" onClick={loadDemoRelations}>
          🔗 Load Demo Relations
        </button>
      </div>

      {relations.length === 0 && (
        <div className="tweet-empty glass-card" onClick={addRelation}>
          <span style={{ fontSize: '24px' }}>🔗</span>
          <span className="text-muted">No relations. The model will still work using features only.</span>
        </div>
      )}

      {relations.length > 0 && (
        <div className="relations-table">
          <div className="relations-header">
            <span>Source</span>
            <span>Relation</span>
            <span>Target</span>
            <span></span>
          </div>
          {relations.map((rel, i) => (
            <div key={i} className="relation-row">
              <input
                className="form-input"
                type="text"
                value={rel.source}
                onChange={e => updateRelation(i, 'source', e.target.value)}
                placeholder="source_user"
              />
              <select
                className="form-input"
                value={rel.relation}
                onChange={e => updateRelation(i, 'relation', e.target.value)}
              >
                {RELATION_OPTIONS.map(r => (
                  <option key={r} value={r}>{r}</option>
                ))}
              </select>
              <input
                className="form-input"
                type="text"
                value={rel.target}
                onChange={e => updateRelation(i, 'target', e.target.value)}
                placeholder="target_user"
              />
              <button
                type="button"
                className="btn btn-sm btn-danger"
                onClick={() => removeRelation(i)}
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}

      {relations.length > 0 && (
        <button type="button" className="btn btn-sm btn-secondary mt-2" onClick={addRelation}>
          + Add Relation
        </button>
      )}

      <p className="text-muted mt-1" style={{ fontSize: '12px' }}>
        <strong>Tip:</strong> Use "target" as the source or target to refer to the account being analyzed.
        Relation types: follower, friend, mention, reply, quoted, url, hashtag.
      </p>
    </div>
  );
}
