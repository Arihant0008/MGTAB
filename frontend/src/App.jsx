import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import HomePage from './pages/HomePage';
import DetectorPage from './pages/DetectorPage';
import AnalyticsPage from './pages/AnalyticsPage';

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/detect" element={<DetectorPage />} />
        <Route path="/analytics" element={<AnalyticsPage />} />
      </Routes>
    </BrowserRouter>
  );
}
