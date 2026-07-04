/**
 * LoginPage — Shock Glitch Authentication Interface
 *
 * Full-viewport dark login with:
 * - Glitched "MGTAB" header (CSS clip-path pseudo-element animation)
 * - Google Sign-In with click-flash artifact
 * - Deep-link redirect after successful auth
 */
import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import './LoginPage.css';

/* Inline Google "G" SVG — no external assets */
function GoogleIcon() {
  return (
    <svg className="google-btn-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
      <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z" fill="#4285F4"/>
      <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
      <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18A10.96 10.96 0 0 0 1 12c0 1.77.42 3.45 1.18 4.93l3.66-2.84z" fill="#FBBC05"/>
      <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
    </svg>
  );
}

export default function LoginPage() {
  const { loginWithGoogle, user } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showFlash, setShowFlash] = useState(false);
  const [showRgbSplit, setShowRgbSplit] = useState(true);

  // Determine redirect target from route state (deep-link preservation)
  const from = location.state?.from?.pathname || '/';

  // If user is already authenticated, redirect immediately
  useEffect(() => {
    if (user) {
      navigate(from, { replace: true });
    }
  }, [user, navigate, from]);

  // Remove RGB split entry class after animation completes
  useEffect(() => {
    const timer = setTimeout(() => setShowRgbSplit(false), 500);
    return () => clearTimeout(timer);
  }, []);

  const handleSignIn = async () => {
    setError(null);
    setShowFlash(true);

    // Allow flash artifact to render before firing popup
    setTimeout(async () => {
      setIsLoading(true);
      try {
        await loginWithGoogle();
        // onAuthStateChanged will handle navigation via the useEffect above
      } catch (err) {
        // Don't show error for user-cancelled popups
        if (err.code !== 'auth/popup-closed-by-user' &&
            err.code !== 'auth/cancelled-popup-request') {
          setError(getErrorMessage(err));
        }
      } finally {
        setIsLoading(false);
        setShowFlash(false);
      }
    }, 150);
  };

  return (
    <div className="login-page">
      <div className={`login-card${showRgbSplit ? ' rgb-split-entry' : ''}`}>

        {/* Shield icon */}
        <div className="login-shield">🛡️</div>

        {/* Glitched title — data-text attr drives pseudo-element content */}
        <h1 className="glitch-title" data-text="MGTAB-Live">MGTAB-Live</h1>

        <p className="login-subtitle">Bot Detection System&ensp;•&ensp;Authenticate to Continue</p>

        <hr className="login-divider" />

        {/* Error alert */}
        {error && (
          <div className="login-error" role="alert">
            {error}
          </div>
        )}

        {/* Google Sign-In */}
        <button
          className="google-btn"
          onClick={handleSignIn}
          disabled={isLoading}
          id="google-sign-in-btn"
        >
          {showFlash && <span className="click-flash" />}
          <GoogleIcon />
          <span className="google-btn-text">
            {isLoading ? 'Authenticating…' : 'Sign in with Google'}
          </span>
        </button>


      </div>
    </div>
  );
}


/* ── Error Message Helper ──────────────────────────────────────── */
function getErrorMessage(err) {
  switch (err.code) {
    case 'auth/network-request-failed':
      return 'Network error — check your internet connection and try again.';
    case 'auth/too-many-requests':
      return 'Too many attempts — please wait a moment before trying again.';
    case 'auth/user-disabled':
      return 'This account has been disabled. Contact support.';
    case 'auth/operation-not-allowed':
      return 'Google Sign-In is not enabled. Contact the administrator.';
    case 'auth/internal-error':
      return 'An internal error occurred. Please try again.';
    default:
      return err.message || 'Authentication failed. Please try again.';
  }
}
