import { motion } from "framer-motion";
import "./Loader.css";

const Loader = ({ text = "Loading..." }) => {
  return (
    <div className="loader-container">
      <div className="loader-rings">
        <motion.div
          className="ring ring-outer"
          animate={{ rotate: 360 }}
          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
        />
        <motion.div
          className="ring ring-inner"
          animate={{ rotate: -360 }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
        />
        <div className="ring-dot" />
      </div>
      <p className="loader-text">{text}</p>
    </div>
  );
};

export default Loader;
