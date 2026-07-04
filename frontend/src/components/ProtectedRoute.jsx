/**
 * ProtectedRoute — Layout route guard.
 * - loading  → dark spinner
 * - !user    → redirect to /login (preserves deep link)
 * - user ✓   → render child routes via <Outlet />
 */
import { Navigate, Outlet, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function ProtectedRoute() {
  const { user, loading } = useAuth();
  const location = useLocation();

  if (loading) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        background: '#04070d',
      }}>
        <div style={{
          width: 40,
          height: 40,
          border: '3px solid rgba(56, 189, 248, 0.15)',
          borderTopColor: '#3b82f6',
          borderRadius: '50%',
          animation: 'spin 0.6s linear infinite',
        }} />
      </div>
    );
  }

  if (!user) {
    // Preserve attempted deep-link for post-login redirect
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return <Outlet />;
}
