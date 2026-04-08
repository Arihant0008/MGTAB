import { useState } from 'react';

const DEMO_HUMAN = {
  followers_count: 843,
  friends_count: 312,
  listed_count: 15,
  statuses_count: 4520,
  favourites_count: 8900,
  name: 'Sarah Chen',
  screen_name: 'sarahchen_dev',
  description: 'Full-stack developer | Open source contributor | Coffee enthusiast ☕',
  created_at: '2015-03-22T00:00:00Z',
  default_profile: false,
  default_profile_image: false,
  verified: false,
  has_url: true,
  geo_enabled: true,
  profile_use_background_image: true,
  default_profile_background_color: false,
  default_profile_sidebar_fill_color: false,
  default_profile_sidebar_border_color: false,
  profile_background_image_url: false,
};

const DEMO_BOT = {
  followers_count: 12,
  friends_count: 4800,
  listed_count: 0,
  statuses_count: 35000,
  favourites_count: 2,
  name: 'News Bot 38291',
  screen_name: 'xnews_bot38291',
  description: '',
  created_at: '2023-11-01T00:00:00Z',
  default_profile: true,
  default_profile_image: true,
  verified: false,
  has_url: false,
  geo_enabled: false,
  profile_use_background_image: false,
  default_profile_background_color: true,
  default_profile_sidebar_fill_color: true,
  default_profile_sidebar_border_color: true,
  profile_background_image_url: false,
};

const BOOLEAN_FIELDS = [
  { key: 'default_profile', label: 'Default profile (not customized)' },
  { key: 'default_profile_image', label: 'Default profile image' },
  { key: 'verified', label: 'Verified account' },
  { key: 'has_url', label: 'Profile has URL' },
  { key: 'geo_enabled', label: 'Geolocation enabled' },
  { key: 'profile_use_background_image', label: 'Has background image' },
  { key: 'default_profile_background_color', label: 'Default background color' },
  { key: 'default_profile_sidebar_fill_color', label: 'Default sidebar fill color' },
  { key: 'default_profile_sidebar_border_color', label: 'Default sidebar border color' },
  { key: 'profile_background_image_url', label: 'Background image has URL' },
];

export default function ProfileForm({ profile, onChange }) {
  const [activeDemo, setActiveDemo] = useState(null);

  const update = (key, value) => {
    onChange({ ...profile, [key]: value });
  };

  const loadDemo = (type) => {
    setActiveDemo(type);
    onChange(type === 'human' ? { ...DEMO_HUMAN } : { ...DEMO_BOT });
  };

  return (
    <div className="profile-form">
      <div className="section-header">
        <div className="section-icon">👤</div>
        <div>
          <div className="section-title">Profile Information</div>
          <div className="section-subtitle">Enter the account's profile data</div>
        </div>
      </div>

      {/* Demo buttons */}
      <div className="demo-buttons mb-3">
        <button
          type="button"
          className={`btn btn-sm btn-secondary ${activeDemo === 'human' ? 'demo-active-human' : ''}`}
          onClick={() => loadDemo('human')}
        >
          👤 Load Human Demo
        </button>
        <button
          type="button"
          className={`btn btn-sm btn-secondary ${activeDemo === 'bot' ? 'demo-active-bot' : ''}`}
          onClick={() => loadDemo('bot')}
        >
          🤖 Load Bot Demo
        </button>
      </div>

      {/* Text fields */}
      <div className="grid-3 mb-3">
        <div className="form-group">
          <label className="form-label">Name</label>
          <input
            className="form-input"
            type="text"
            value={profile.name || ''}
            onChange={e => update('name', e.target.value)}
            placeholder="Display name"
          />
        </div>
        <div className="form-group">
          <label className="form-label">Screen Name (@handle)</label>
          <input
            className="form-input"
            type="text"
            value={profile.screen_name || ''}
            onChange={e => update('screen_name', e.target.value)}
            placeholder="username"
          />
        </div>
        <div className="form-group">
          <label className="form-label">Account Created</label>
          <input
            className="form-input"
            type="date"
            value={profile.created_at ? profile.created_at.split('T')[0] : ''}
            onChange={e => update('created_at', e.target.value + 'T00:00:00Z')}
          />
        </div>
      </div>

      <div className="form-group mb-3">
        <label className="form-label">Description / Bio</label>
        <input
          className="form-input"
          type="text"
          value={profile.description || ''}
          onChange={e => update('description', e.target.value)}
          placeholder="Account bio"
        />
      </div>

      {/* Numeric fields */}
      <div className="grid-5-compact mb-3">
        {[
          { key: 'followers_count', label: 'Followers' },
          { key: 'friends_count', label: 'Following' },
          { key: 'listed_count', label: 'Listed' },
          { key: 'statuses_count', label: 'Tweets' },
          { key: 'favourites_count', label: 'Likes' },
        ].map(f => (
          <div className="form-group" key={f.key}>
            <label className="form-label">{f.label}</label>
            <input
              className="form-input"
              type="number"
              min="0"
              value={profile[f.key] || 0}
              onChange={e => update(f.key, parseInt(e.target.value) || 0)}
            />
          </div>
        ))}
      </div>

      {/* Boolean toggles */}
      <div className="toggles-section">
        <div className="form-label mb-2" style={{ fontWeight: 600, fontSize: '14px' }}>
          Profile Flags
        </div>
        <div className="toggles-grid">
          {BOOLEAN_FIELDS.map(f => (
            <div className="toggle-group" key={f.key}>
              <span className="toggle-label">{f.label}</span>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={!!profile[f.key]}
                  onChange={e => update(f.key, e.target.checked)}
                />
                <span className="toggle-slider" />
              </label>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
