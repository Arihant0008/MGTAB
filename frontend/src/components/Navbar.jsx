import { Link, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function Navbar() {
  const { pathname } = useLocation();
  const { user, logout } = useAuth();

  const handleSignOut = async () => {
    try {
      await logout();
    } catch (err) {
      console.error('Sign-out failed:', err);
    }
  };

  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <Link to="/" className="navbar-brand">
          <span className="navbar-brand-icon">🛡️</span>
          MGTAB Detector
        </Link>

        <div className="navbar-links">
          <Link
            to="/"
            className={`navbar-link${pathname === '/' ? ' active' : ''}`}
          >
            Home
          </Link>
          <Link
            to="/detect"
            className={`navbar-link${pathname === '/detect' ? ' active' : ''}`}
          >
            Detector
          </Link>
          <Link
            to="/analytics"
            className={`navbar-link${pathname === '/analytics' ? ' active' : ''}`}
          >
            Analytics
          </Link>

          {/* ── User Avatar & Sign Out ── */}
          {user && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              marginLeft: '12px',
              paddingLeft: '12px',
              borderLeft: '1px solid rgba(56, 189, 248, 0.15)',
            }}>
              {user.photoURL && (
                <img
                  src={user.photoURL}
                  alt={user.displayName || 'User'}
                  referrerPolicy="no-referrer"
                  style={{
                    width: 30,
                    height: 30,
                    borderRadius: '50%',
                    border: '1px solid rgba(56, 189, 248, 0.25)',
                    objectFit: 'cover',
                  }}
                />
              )}
              <button
                onClick={handleSignOut}
                style={{
                  padding: '6px 14px',
                  fontSize: '13px',
                  fontWeight: 500,
                  fontFamily: 'inherit',
                  color: '#94a3b8',
                  background: 'rgba(255, 255, 255, 0.04)',
                  border: '1px solid rgba(255, 255, 255, 0.08)',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                  whiteSpace: 'nowrap',
                }}
                onMouseEnter={(e) => {
                  e.target.style.color = '#e2e8f0';
                  e.target.style.borderColor = 'rgba(239, 68, 68, 0.3)';
                  e.target.style.background = 'rgba(239, 68, 68, 0.08)';
                }}
                onMouseLeave={(e) => {
                  e.target.style.color = '#94a3b8';
                  e.target.style.borderColor = 'rgba(255, 255, 255, 0.08)';
                  e.target.style.background = 'rgba(255, 255, 255, 0.04)';
                }}
              >
                Sign Out
              </button>
            </div>
          )}
        </div>
      </div>
    </nav>
  );
}
