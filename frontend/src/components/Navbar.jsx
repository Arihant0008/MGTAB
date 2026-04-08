import { Link, useLocation } from 'react-router-dom';

export default function Navbar() {
  const { pathname } = useLocation();

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
        </div>
      </div>
    </nav>
  );
}
