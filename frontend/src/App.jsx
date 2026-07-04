import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import ShootingStars from './components/ShootingStars';
import ProtectedRoute from './components/ProtectedRoute';
import AppLayout from './components/AppLayout';
import HomePage from './pages/HomePage';
import DetectorPage from './pages/DetectorPage';
import AnalyticsPage from './pages/AnalyticsPage';
import LoginPage from './pages/LoginPage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public — login page sits OUTSIDE the guard */}
        <Route path="/login" element={<LoginPage />} />

        {/* Protected — everything behind auth */}
        <Route element={<ProtectedRoute />}>
          <Route element={<AppLayout />}>
            <Route path="/"         element={<HomePage />} />
            <Route path="/detect"   element={<DetectorPage />} />
            <Route path="/analytics" element={<AnalyticsPage />} />
          </Route>
        </Route>

        {/* Catch-all — redirect to root (auth guard handles the rest) */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
