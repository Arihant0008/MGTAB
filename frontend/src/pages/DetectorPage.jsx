import { useState } from 'react';
import ProfileForm from '../components/ProfileForm';
import TweetInput from '../components/TweetInput';
import RelationsEditor from '../components/RelationsEditor';
import ResultCard from '../components/ResultCard';
import { predictUser } from '../api/predict';
import './DetectorPage.css';

const INITIAL_PROFILE = {
  followers_count: 0,
  friends_count: 0,
  listed_count: 0,
  statuses_count: 0,
  favourites_count: 0,
  name: '',
  screen_name: '',
  description: '',
  created_at: '',
  default_profile: false,
  default_profile_image: false,
  verified: false,
  has_url: false,
  geo_enabled: false,
  profile_use_background_image: true,
  default_profile_background_color: false,
  default_profile_sidebar_fill_color: false,
  default_profile_sidebar_border_color: false,
  profile_background_image_url: false,
};

export default function DetectorPage() {
  const [profile, setProfile] = useState({ ...INITIAL_PROFILE });
  const [tweets, setTweets] = useState([]);
  const [relations, setRelations] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const requestBody = {
        target: {
          profile,
          tweets: tweets.filter(t => t.trim()),
        },
        neighbors: [],
        relations,
      };

      const response = await predictUser(requestBody);
      setResult(response);
    } catch (err) {
      setError(err.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setProfile({ ...INITIAL_PROFILE });
    setTweets([]);
    setRelations([]);
    setResult(null);
    setError(null);
  };

  return (
    <div className="page detector-page">
      <div className="container">
        <div className="detector-header animate-fade-in">
          <h1 className="detector-title">Bot Detector</h1>
          <p className="detector-subtitle">
            Enter profile data, tweets, and relations to classify an account using the RGCN model.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="detector-form">
          <div className="detector-sections">
            {/* Profile */}
            <div className="detector-section glass-card animate-slide-up">
              <ProfileForm profile={profile} onChange={setProfile} />
            </div>

            {/* Tweets */}
            <div className="detector-section glass-card animate-slide-up" style={{ animationDelay: '0.1s' }}>
              <TweetInput tweets={tweets} onChange={setTweets} />
            </div>

            {/* Relations */}
            <div className="detector-section glass-card animate-slide-up" style={{ animationDelay: '0.2s' }}>
              <RelationsEditor relations={relations} onChange={setRelations} />
            </div>
          </div>

          {/* Action buttons */}
          <div className="detector-actions animate-slide-up" style={{ animationDelay: '0.3s' }}>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={loading}
              style={{ padding: '14px 40px', fontSize: '15px' }}
            >
              {loading ? (
                <>
                  <span className="spinner" /> Analyzing...
                </>
              ) : (
                '🔍 Analyze Account'
              )}
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={handleReset}
            >
              ↺ Reset
            </button>
          </div>

          {/* Error */}
          {error && (
            <div className="detector-error glass-card animate-fade-in">
              <span style={{ fontSize: '18px' }}>⚠️</span>
              <div>
                <strong>Error:</strong> {error}
              </div>
            </div>
          )}

          {/* Result */}
          <ResultCard result={result} loading={loading} />
        </form>
      </div>
    </div>
  );
}
