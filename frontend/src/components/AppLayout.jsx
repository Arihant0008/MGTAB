/**
 * AppLayout — Authenticated application shell.
 * Renders the shared chrome (ShootingStars, Navbar, Footer)
 * around whatever child route is active via <Outlet />.
 */
import { Outlet } from 'react-router-dom';
import Navbar from './Navbar';
import Footer from './Footer';
import ShootingStars from './ShootingStars';

export default function AppLayout() {
  return (
    <>
      {/* Full-page shooting stars canvas — z-index 0, pointer-events none */}
      <ShootingStars />
      <Navbar />
      <Outlet />
      <Footer />
    </>
  );
}
