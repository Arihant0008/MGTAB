import { useEffect } from 'react';

/**
 * Paper Table 4 — Relation types with correct direction semantics.
 *
 * Explicit (directed):
 *   follower: User B follows User A (target) → edge: B → A
 *   friend:   User A (target) follows User B → edge: A → B
 *   mention:  User A (target) mentions User B → edge: A → B
 *   reply:    User A (target) replies to User B → edge: A → B
 *   quoted:   User A (target) quotes User B → edge: A → B
 *
 * Implicit (undirected):
 *   hashtag: Users A and B share hashtags → edges: A↔B
 *   url:     Users A and B share URLs → edges: A↔B
 */

const EXPLICIT_RELATIONS = [
  {
    key: 'follower',
    label: 'Follower',
    placeholder: 'e.g. follower_user_id',
    desc: 'Someone who follows this account',
    dirHint: 'Neighbor → Target',
    // For follower, the neighbor is the SOURCE (they follow the target)
    sourceIsTarget: false,
  },
  {
    key: 'friend',
    label: 'Friend',
    placeholder: 'e.g. friend_user_id',
    desc: 'Someone this account follows',
    dirHint: 'Target → Neighbor',
    sourceIsTarget: true,
  },
  {
    key: 'mention',
    label: 'Mention',
    placeholder: 'e.g. mentioned_user_id',
    desc: 'Someone this account mentions in tweets',
    dirHint: 'Target → Neighbor',
    sourceIsTarget: true,
  },
  {
    key: 'reply',
    label: 'Reply',
    placeholder: 'e.g. replied_user_id',
    desc: 'Someone this account replies to',
    dirHint: 'Target → Neighbor',
    sourceIsTarget: true,
  },
  {
    key: 'quoted',
    label: 'Quote',
    placeholder: 'e.g. quoted_user_id',
    desc: 'Someone whose tweets this account quotes',
    dirHint: 'Target → Neighbor',
    sourceIsTarget: true,
  },
];

const IMPLICIT_RELATIONS = [
  {
    key: 'hashtag',
    label: 'Shared Hashtags',
    placeholder: 'e.g. user_with_same_hashtag',
    desc: 'Someone using the same hashtags (co-occurrence)',
    dirHint: 'Undirected ↔',
    sourceIsTarget: true, // backend adds reverse edge
  },
  {
    key: 'url',
    label: 'Shared URLs',
    placeholder: 'e.g. user_with_same_url',
    desc: 'Someone sharing the same links (co-occurrence)',
    dirHint: 'Undirected ↔',
    sourceIsTarget: true, // backend adds reverse edge
  },
];

const ALL_RELATIONS = [...EXPLICIT_RELATIONS, ...IMPLICIT_RELATIONS];

export default function RelationsEditor({ relations, onChange }) {

  useEffect(() => {
    if (relations.length !== 7) {
      onChange(ALL_RELATIONS.map(rt => ({
        source: rt.sourceIsTarget ? 'target' : '',
        target: '',
        relation: rt.key,
      })));
    }
  }, []);

  const updateTarget = (index, value) => {
    const rt = ALL_RELATIONS[index];
    const updated = [...relations];

    if (rt.sourceIsTarget) {
      // target user is the source (friend, mention, reply, quoted, hashtag, url)
      updated[index] = { ...updated[index], source: 'target', target: value };
    } else {
      // neighbor is the source (follower: neighbor follows target)
      updated[index] = { ...updated[index], source: value, target: 'target' };
    }

    onChange(updated);
  };

  const getNeighborValue = (index) => {
    if (!relations[index]) return '';
    const rt = ALL_RELATIONS[index];
    // Return the non-target side of the relation
    return rt.sourceIsTarget ? relations[index].target : relations[index].source;
  };

  const loadDemoRelations = () => {
    const demo = [
      { source: 'user_alice', target: 'target', relation: 'follower' },   // alice follows target
      { source: 'target', target: 'user_bob', relation: 'friend' },       // target follows bob
      { source: 'target', target: 'user_carol', relation: 'mention' },    // target mentions carol
      { source: 'target', target: 'user_dave', relation: 'reply' },       // target replies to dave
      { source: 'target', target: 'user_eve', relation: 'quoted' },       // target quotes eve
      { source: 'target', target: 'user_frank', relation: 'hashtag' },    // shared hashtags with frank
      { source: 'target', target: 'user_grace', relation: 'url' },        // shared urls with grace
    ];
    onChange(demo);
  };

  return (
    <div className="relations-editor">
      <div className="section-header">
        <div className="section-icon">🔗</div>
        <div>
          <div className="section-title">Relations <span className="badge badge-human" style={{ fontSize: '11px', marginLeft: 8 }}>7 Types</span></div>
          <div className="section-subtitle">
            Define the 7 graph edge types (per Paper Table 4). Edges to neighbors <strong>without profile data</strong> are auto-skipped.
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
          <span className="relations-group-title">Direct Interactions (Directed)</span>
        </div>
        <p className="relations-group-desc">
          These are <strong>directed edges</strong> between users. Edge direction follows the paper's Table 4.
        </p>

        <div className="relations-table">
          <div className="relations-header" style={{ gridTemplateColumns: 'minmax(100px, 140px) 1fr' }}>
            <span>Relation Type</span>
            <span>Neighbor User Identifier</span>
          </div>
          {relations.length === 7 && EXPLICIT_RELATIONS.map((rt, i) => (
            <div key={i} className="relation-row" style={{ gridTemplateColumns: 'minmax(100px, 140px) 1fr' }}>
              <div className="relation-label-group">
                <span className="relation-label-name">{rt.label} <span style={{ fontSize: '10px', opacity: 0.6 }}>{rt.dirHint}</span></span>
                <span className="relation-label-desc">{rt.desc}</span>
              </div>
              <input
                className="form-input"
                type="text"
                value={getNeighborValue(i)}
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
          <span className="relations-group-title">Hidden Connections (Undirected)</span>
        </div>
        <p className="relations-group-desc">
          These are <strong>undirected co-occurrence edges</strong>. The backend automatically creates edges in both directions.
        </p>

        <div className="relations-table">
          <div className="relations-header" style={{ gridTemplateColumns: 'minmax(100px, 140px) 1fr' }}>
            <span>Relation Type</span>
            <span>Neighbor User Identifier</span>
          </div>
          {relations.length === 7 && IMPLICIT_RELATIONS.map((rt, idx) => {
            const i = idx + 5; // offset by explicit count
            return (
              <div key={i} className="relation-row" style={{ gridTemplateColumns: 'minmax(100px, 140px) 1fr' }}>
                <div className="relation-label-group">
                  <span className="relation-label-name">{rt.label} <span style={{ fontSize: '10px', opacity: 0.6 }}>{rt.dirHint}</span></span>
                  <span className="relation-label-desc">{rt.desc}</span>
                </div>
                <input
                  className="form-input"
                  type="text"
                  value={getNeighborValue(i)}
                  onChange={e => updateTarget(i, e.target.value)}
                  placeholder={rt.placeholder}
                />
              </div>
            );
          })}
        </div>
      </div>

      <p className="text-muted mt-2" style={{ fontSize: '12px' }}>
        <strong>Note:</strong> Relations only affect predictions when the neighbor also has profile/tweet data provided. 
        Without neighbor data, the model uses features-only mode (self-loop).
      </p>
    </div>
  );
}
