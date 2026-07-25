import './Footer.css';

export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-inner">

        {/* Top divider glow line */}
        <div className="footer-glow-line" />

        <div className="footer-content">
          {/* Left — brand */}
          <div className="footer-brand">
            <div className="footer-brand-icon">🛡️</div>
            <div>
              <div className="footer-brand-name">MGTAB</div>
              <div className="footer-brand-sub">
                Multi-relational Graph Twitter Account Bot Detector
              </div>
            </div>
          </div>

          {/* Center — tech stack */}
          <div className="footer-stack">
            <span className="footer-stack-label">Powered by</span>
            <div className="footer-badges">
              <span className="footer-badge">RGCN</span>
              <span className="footer-badge">LaBSE</span>
              <span className="footer-badge">PyTorch</span>
              <span className="footer-badge">FastAPI</span>
              <span className="footer-badge">React</span>
            </div>
          </div>

          {/* Right — accuracy stat */}
          <div className="footer-stat">
            <div className="footer-stat-value">88.2%</div>
            <div className="footer-stat-label">Detection Accuracy</div>
          </div>
        </div>

        {/* Bottom copyright */}
        <div className="footer-bottom">
          <div className="footer-copyright">
            <span className="footer-copy-icon">©</span>
            <span>2026</span>
            <span className="footer-divider">·</span>
            <span>Made with</span>
            <span className="footer-heart">♥</span>
            <span>by</span>
            <span className="footer-authors">
              <span className="footer-author">Arihant Jain</span>
              <span className="footer-comma">,</span>
              <span className="footer-author">Pratham</span>
              <span className="footer-comma">,</span>
              <span className="footer-author">Frank</span>
              <span className="footer-comma">&</span>
              <span className="footer-author">Aayush</span>
            </span>
          </div>
          <div className="footer-authors">
            MGTAB-Live: A Production-Ready Multi-Relational Graph-Based Framework for Real-Time Twitter Bot Detection Using Relational Graph Convolutional Networks (Published in Cureus Journal)
          </div>
          <div className="footer-rights">
            All rights reserved 
          </div>
        </div>

      </div>
    </footer>
  );
}
