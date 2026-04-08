import { Link } from 'react-router-dom';
import './Hero.css';

export default function Hero() {
  return (
    <section className="hero">
      <div className="hero-bg-grid" />
      <div className="hero-glow hero-glow-1" />
      <div className="hero-glow hero-glow-2" />

      <div className="container hero-content">
        <div className="hero-badge animate-fade-in">
          <span className="hero-badge-dot" />
          Powered by RGCN · 88.23% Accuracy
        </div>

        <h1 className="hero-title animate-slide-up">
          Multi-Relational Graph
          <span className="hero-title-accent"> Bot Detection</span>
        </h1>

        <p className="hero-subtitle animate-slide-up" style={{ animationDelay: '0.1s' }}>
          Detect Twitter/X bot accounts using Relational Graph Convolutional Networks
          trained on the MGTAB benchmark — 10,199 expert-annotated users,
          7 relationship types, 788-dimensional feature vectors.
        </p>

        <div className="hero-actions animate-slide-up" style={{ animationDelay: '0.2s' }}>
          <Link to="/detect" className="btn btn-primary btn-hero">
            🔍 Analyze Account
          </Link>
          <Link to="/analytics" className="btn btn-secondary btn-hero">
            📊 View Analytics
          </Link>
        </div>

        <div className="hero-stats animate-slide-up" style={{ animationDelay: '0.3s' }}>
          <div className="hero-stat">
            <span className="hero-stat-value">88.23%</span>
            <span className="hero-stat-label">Test Accuracy</span>
          </div>
          <div className="hero-stat-divider" />
          <div className="hero-stat">
            <span className="hero-stat-value">90.29%</span>
            <span className="hero-stat-label">Bot Recall</span>
          </div>
          <div className="hero-stat-divider" />
          <div className="hero-stat">
            <span className="hero-stat-value">7</span>
            <span className="hero-stat-label">Relation Types</span>
          </div>
          <div className="hero-stat-divider" />
          <div className="hero-stat">
            <span className="hero-stat-value">788</span>
            <span className="hero-stat-label">Feature Dim</span>
          </div>
        </div>
      </div>
    </section>
  );
}
