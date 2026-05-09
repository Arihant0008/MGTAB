import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import ShootingStars from './components/ShootingStars';
import HomePage from './pages/HomePage';
import DetectorPage from './pages/DetectorPage';
import AnalyticsPage from './pages/AnalyticsPage';

export default function App() {
  return (
    <BrowserRouter>
      {/* Full-page shooting stars canvas — z-index 0, pointer-events none */}
      <ShootingStars />

      <Navbar />

      <Routes>
        <Route path="/"        element={<HomePage />} />
        <Route path="/detect"  element={<DetectorPage />} />
        <Route path="/analytics" element={<AnalyticsPage />} />
      </Routes>

      <Footer />
    </BrowserRouter>
  );
}
