import { useState, useRef, useCallback } from 'react';
import ProfileForm from '../components/ProfileForm';
import TweetInput from '../components/TweetInput';
import RelationsEditor from '../components/RelationsEditor';
import ResultCard from '../components/ResultCard';
import { predictUser, submitPrediction, getJobStatus } from '../api/predict';
import './DetectorPage.css';

// ── Step definitions for the progress stepper ────────────────────

const PIPELINE_STEPS = [
  { id: 1, label: 'Scraping Profile',     icon: '👤', status_key: 'scraping_profile' },
  { id: 2, label: 'Fetching Network',     icon: '🌐', status_key: 'fetching_network' },
  { id: 3, label: 'Enriching Neighbors',  icon: '📊', status_key: 'enriching_neighbors' },
  { id: 4, label: 'Building Graph',       icon: '🧮', status_key: 'building_graph' },
  { id: 5, label: 'Running RGCN',         icon: '🤖', status_key: 'running_rgcn' },
];

// ── Manual mode initial state ────────────────────────────────────

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

// ── Time formatting helper ───────────────────────────────────────

function formatCountdown(totalSeconds) {
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  if (hours > 0) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
}


export default function DetectorPage() {
  // ── Mode toggle ────────────────────────────────────────────────
  const [mode, setMode] = useState('auto'); // 'auto' | 'manual'

  // ── One-Click mode state ───────────────────────────────────────
  const [username, setUsername] = useState('');
  const [currentStep, setCurrentStep] = useState(0);
  const [stepMessage, setStepMessage] = useState('');
  const [scrapeMeta, setScrapeMeta] = useState(null);
  const [autoResult, setAutoResult] = useState(null);
  const [autoLoading, setAutoLoading] = useState(false);
  const [autoError, setAutoError] = useState(null);
  const abortRef = useRef(null);

  // ── Manual mode state ──────────────────────────────────────────
  const [profile, setProfile] = useState({ ...INITIAL_PROFILE });
  const [tweets, setTweets] = useState([]);
  const [relations, setRelations] = useState([]);
  const [manualResult, setManualResult] = useState(null);
  const [manualLoading, setManualLoading] = useState(false);
  const [manualError, setManualError] = useState(null);

  // ── One-Click: Submit → Poll Architecture ──────────────────────

  const handleAutoSubmit = useCallback(async (e) => {
    e.preventDefault();
    if (!username.trim() || autoLoading) return;

    // Reset state
    setAutoLoading(true);
    setAutoError(null);
    setAutoResult(null);
    setScrapeMeta(null);
    setCurrentStep(1);
    setStepMessage('Submitting job...');

    // Cancel any existing poll
    if (abortRef.current) abortRef.current();

    try {
      // Submit the job
      const job = await submitPrediction(username);

      // If cache hit → instant completion
      if (job.status === 'completed') {
        const status = await getJobStatus(job.job_id);
        if (status.result) {
          setAutoResult(status.result);
          setScrapeMeta(status.result.graph_info?.scrape_meta || null);
        }
        setCurrentStep(6);
        setAutoLoading(false);
        return;
      }

      // Start polling every 2 seconds
      let cancelled = false;
      const pollInterval = setInterval(async () => {
        if (cancelled) return;

        try {
          const status = await getJobStatus(job.job_id);

          // Update stepper UI from progress
          if (status.progress) {
            setCurrentStep(status.progress.step || 1);
            setStepMessage(status.progress.message || 'Processing...');
          }

          if (status.status === 'completed') {
            clearInterval(pollInterval);
            if (status.result) {
              setAutoResult(status.result);
              setScrapeMeta(status.result.graph_info?.scrape_meta || null);
            }
            setCurrentStep(6); // All done
            setAutoLoading(false);
          }

          if (status.status === 'failed') {
            clearInterval(pollInterval);
            setAutoError(status.error || 'Analysis failed');
            setCurrentStep(-1); // Error state
            setAutoLoading(false);
          }
        } catch (pollErr) {
          // Don't kill the loop on transient network errors
          console.warn('Poll error:', pollErr);
        }
      }, 2000);

      // Store cancel function for cleanup
      abortRef.current = () => {
        cancelled = true;
        clearInterval(pollInterval);
      };

    } catch (err) {
      if (err.code === 'RATE_LIMITED') {
        setAutoError(
          `Rate limit reached — you can analyze one account every 24 hours. ` +
          `Try again in ${formatCountdown(err.retryAfterSeconds)}.`
        );
      } else {
        setAutoError(err.message || 'Submission failed');
      }
      setCurrentStep(-1);
      setAutoLoading(false);
    }
  }, [username, autoLoading]);

  const handleAutoReset = useCallback(() => {
    if (abortRef.current) abortRef.current();
    setUsername('');
    setCurrentStep(0);
    setStepMessage('');
    setScrapeMeta(null);
    setAutoResult(null);
    setAutoLoading(false);
    setAutoError(null);
  }, []);

  // ── Manual mode submission ─────────────────────────────────────

  const handleManualSubmit = async (e) => {
    e.preventDefault();
    setManualLoading(true);
    setManualError(null);
    setManualResult(null);

    try {
      const validRelations = relations.filter((r) => {
        const targetAliases = ['target', '__target__', 'this_user', 'self'];
        const srcIsTarget = targetAliases.includes((r.source || '').toLowerCase());
        const neighborId = srcIsTarget ? r.target : r.source;
        return neighborId && neighborId.trim() !== '';
      });

      const requestBody = {
        target: {
          profile,
          tweets: tweets.filter((t) => t.trim()),
        },
        neighbors: [],
        relations: validRelations,
      };

      const response = await predictUser(requestBody);
      setManualResult(response);
    } catch (err) {
      setManualError(err.message || 'Prediction failed');
    } finally {
      setManualLoading(false);
    }
  };

  const handleManualReset = () => {
    setProfile({ ...INITIAL_PROFILE });
    setTweets([]);
    setRelations([]);
    setManualResult(null);
    setManualError(null);
  };

  // ── Render helpers ─────────────────────────────────────────────

  const getStepState = (stepId) => {
    if (currentStep === -1) return 'error';
    if (stepId < currentStep) return 'done';
    if (stepId === currentStep) return 'active';
    return 'pending';
  };

  // ── Render ─────────────────────────────────────────────────────

  return (
    <div className="page detector-page">
      <div className="container">
        {/* Header */}
        <div className="detector-header animate-fade-in">
          <h1 className="detector-title">Bot Detector</h1>
          <p className="detector-subtitle">
            Analyze any Twitter/X account using our multi-relational graph neural network.
          </p>
        </div>

        {/* Mode Tabs */}
        <div className="mode-tabs animate-fade-in">
          <button
            id="tab-auto"
            className={`mode-tab${mode === 'auto' ? ' active' : ''}`}
            onClick={() => setMode('auto')}
          >
            <span className="mode-tab-icon">⚡</span>
            One-Click Analysis
          </button>
          <button
            id="tab-manual"
            className={`mode-tab${mode === 'manual' ? ' active' : ''}`}
            onClick={() => setMode('manual')}
          >
            <span className="mode-tab-icon">🔧</span>
            Manual Mode
          </button>
        </div>

        {/* ═══════════════════════════════════════════════════════════
            AUTO MODE — Queue-Based Analysis with Polling
           ═══════════════════════════════════════════════════════════ */}
        {mode === 'auto' && (
          <>
            {/* Search Bar */}
            <div className="search-container animate-slide-up">
              <form onSubmit={handleAutoSubmit}>
                <div className="search-card">
                  <label className="search-label" htmlFor="username-input">
                    Twitter / X Handle
                  </label>
                  <div className="search-input-wrapper">
                    <span className="search-at-symbol">@</span>
                    <input
                      id="username-input"
                      className="search-input"
                      type="text"
                      value={username}
                      onChange={(e) => setUsername(e.target.value)}
                      placeholder="elonmusk"
                      disabled={autoLoading}
                      autoComplete="off"
                      spellCheck="false"
                      maxLength={15}
                    />
                  </div>
                  <div className="search-actions">
                    <span className="search-hint">
                      {autoLoading
                        ? 'Analyzing... this may take a few minutes.'
                        : 'Enter a public Twitter handle and hit Analyze.'}
                    </span>
                    <div style={{ display: 'flex', gap: '8px' }}>
                      {(autoResult || autoError || currentStep > 0) && (
                        <button
                          type="button"
                          className="btn btn-secondary btn-sm"
                          onClick={handleAutoReset}
                        >
                          ↺ Reset
                        </button>
                      )}
                      <button
                        type="submit"
                        className="search-btn"
                        disabled={autoLoading || !username.trim()}
                        id="analyze-btn"
                      >
                        {autoLoading ? (
                          <>
                            <span className="spinner" /> Analyzing...
                          </>
                        ) : (
                          <>🔍 Analyze</>
                        )}
                      </button>
                    </div>
                  </div>
                </div>
              </form>
            </div>

            {/* Progress Stepper */}
            {currentStep > 0 && currentStep <= 5 && (
              <div className="progress-container animate-fade-in">
                <div className="progress-card">
                  <div className="progress-title">
                    🛡️ Detection Pipeline
                  </div>
                  <div className="progress-steps">
                    {PIPELINE_STEPS.map((step) => {
                      const state = getStepState(step.id);
                      return (
                        <div key={step.id} className={`step-item ${state}`}>
                          <div className={`step-icon-wrap ${state}`}>
                            {state === 'done' ? '✓' : step.icon}
                          </div>
                          <div className="step-content">
                            <div className="step-label">{step.label}</div>
                            {state === 'active' && stepMessage && (
                              <div className="step-message">{stepMessage}</div>
                            )}
                          </div>
                          {step.id < 5 && <div className="step-line" />}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}

            {/* Scrape Summary */}
            {scrapeMeta && (
              <div className="scrape-summary animate-slide-up">
                <div className="scrape-summary-card">
                  <div className="scrape-summary-header">
                    <div className="scrape-summary-avatar">👤</div>
                    <div className="scrape-summary-user">
                      <div className="scrape-summary-name">
                        {scrapeMeta.display_name || scrapeMeta.username}
                      </div>
                      <div className="scrape-summary-handle">
                        @{scrapeMeta.username}
                      </div>
                    </div>
                  </div>
                  <div className="scrape-stats">
                    <div className="scrape-stat">
                      <div className="scrape-stat-value">{scrapeMeta.tweets_scraped}</div>
                      <div className="scrape-stat-label">Tweets</div>
                    </div>
                    <div className="scrape-stat">
                      <div className="scrape-stat-value">{scrapeMeta.neighbors_found}</div>
                      <div className="scrape-stat-label">Neighbors</div>
                    </div>
                    <div className="scrape-stat">
                      <div className="scrape-stat-value">{scrapeMeta.total_relations}</div>
                      <div className="scrape-stat-label">Edges</div>
                    </div>
                  </div>
                  {scrapeMeta.relation_breakdown && (
                    <div className="scrape-relations">
                      {Object.entries(scrapeMeta.relation_breakdown).map(([rel, count]) => (
                        <span key={rel} className="scrape-relation-badge">
                          {rel} <span className="count">×{count}</span>
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
            {/* Error */}
            {autoError && (
              <div className="detector-error glass-card animate-fade-in">
                <span style={{ fontSize: '18px' }}>⚠️</span>
                <div style={{ flex: 1 }}>
                  <strong>Analysis Unavailable:</strong> {autoError}
                  
                  {/* Only show the scraper breakdown warning if it's NOT a rate limit error */}
                  {!autoError.toLowerCase().includes("rate limit") && (
                    <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '8px', marginBottom: '12px' }}>
                      Twitter/X frequently changes their internal API, which can temporarily break automated scraping.
                      Use <strong>Manual Mode</strong> to enter profile data directly and still run the full RGCN analysis.
                    </p>
                  )}
                  
                  <button
                    className="btn btn-secondary btn-sm"
                    onClick={() => { setMode('manual'); setAutoError(null); }}
                    style={{ fontSize: '13px', marginTop: autoError.toLowerCase().includes("rate limit") ? '8px' : '0' }}
                  >
                    🔧 Switch to Manual Mode
                  </button>
                </div>
              </div>
            )}

            {/* Result */}
            <div className="result-wrapper">
              <ResultCard result={autoResult} loading={false} />
            </div>
          </>
        )}

        {/* ═══════════════════════════════════════════════════════════
            MANUAL MODE — Fallback / Developer Mode
           ═══════════════════════════════════════════════════════════ */}
        {mode === 'manual' && (
          <>
            <div className="manual-mode-info animate-fade-in">
              <span>🔧</span>
              <span>
                Manual mode — enter profile data directly. Use this if automated scraping is unavailable.
              </span>
            </div>

            <form onSubmit={handleManualSubmit} className="detector-form">
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
                  disabled={manualLoading}
                  style={{ padding: '14px 40px', fontSize: '15px' }}
                  id="manual-analyze-btn"
                >
                  {manualLoading ? (
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
                  onClick={handleManualReset}
                >
                  ↺ Reset
                </button>
              </div>

              {/* Error */}
              {manualError && (
                <div className="detector-error glass-card animate-fade-in">
                  <span style={{ fontSize: '18px' }}>⚠️</span>
                  <div>
                    <strong>Error:</strong> {manualError}
                  </div>
                </div>
              )}

              {/* Result */}
              <ResultCard result={manualResult} loading={manualLoading} />
            </form>
          </>
        )}
      </div>
    </div>
  );
}
