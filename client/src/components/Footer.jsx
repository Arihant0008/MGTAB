import { HiHeart } from "react-icons/hi";
import { FaGithub, FaPython, FaReact, FaNodeJs } from "react-icons/fa";
import { SiMongodb, SiPytorch } from "react-icons/si";
import "./Footer.css";

const Footer = () => {
  return (
    <footer className="footer">
      <div className="container footer-inner">
        <div className="footer-brand">
          <h3 className="gradient-text">MGTAB</h3>
          <p>Multi-relational Graph-based Twitter Account Bot Detection</p>
        </div>

        <div className="footer-tech">
          <span className="tech-badge"><FaReact /> React</span>
          <span className="tech-badge"><FaNodeJs /> Node.js</span>
          <span className="tech-badge"><SiMongodb /> MongoDB</span>
          <span className="tech-badge"><FaPython /> Python</span>
          <span className="tech-badge"><SiPytorch /> PyTorch</span>
        </div>

        <div className="footer-bottom">
          <p>
            Built with <HiHeart className="heart-icon" /> by Arihant
          </p>
          <a href="https://github.com/Arihant0008" target="_blank" rel="noopener noreferrer" className="github-link">
            <FaGithub /> GitHub
          </a>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
