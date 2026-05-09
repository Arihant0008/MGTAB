import { useEffect, useRef } from 'react';

/**
 * ShootingStars — full-page canvas overlay with continuously spawning
 * shooting stars (comets) streaking diagonally across the background.
 * Runs at 60fps using requestAnimationFrame, zero dependencies.
 */
export default function ShootingStars() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    // ── Resize canvas to fill viewport ──────────────────────────────
    const resize = () => {
      canvas.width  = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    // ── Star class ───────────────────────────────────────────────────
    class Star {
      constructor() { this.reset(true); }

      reset(randomY = false) {
        const w = canvas.width;
        const h = canvas.height;

        // Spawn anywhere along the top edge or left edge
        if (Math.random() < 0.6) {
          // Top edge
          this.x = Math.random() * w * 1.5;
          this.y = -20;
        } else {
          // Left edge (for variety)
          this.x = -20;
          this.y = Math.random() * h * 0.5;
        }

        if (randomY) {
          // On init, spread across the whole screen so it doesn't start empty
          this.x = Math.random() * w * 1.5;
          this.y = Math.random() * h;
        }

        // Speed: slow = 2, fast = 12 — mix of fast bolts and slow drifters
        this.speed   = 2 + Math.random() * 10;
        // Angle: ~30–50° downward diagonal (like real shooting stars)
        this.angle   = (Math.PI / 180) * (30 + Math.random() * 20);
        this.vx      = Math.cos(this.angle) * this.speed;
        this.vy      = Math.sin(this.angle) * this.speed;
        // Tail length proportional to speed
        this.length  = 80 + this.speed * 12;
        // Width: thin to medium
        this.width   = 0.5 + Math.random() * 1.5;
        // Opacity peak
        this.alpha   = 0.4 + Math.random() * 0.6;
        // Colour — mostly white/cyan, occasional gold accent
        const r = Math.random();
        if (r < 0.6)       this.colour = '255,255,255';      // white
        else if (r < 0.85) this.colour = '100,220,255';      // cyan
        else               this.colour = '255,210,100';      // gold accent
      }

      update() {
        this.x += this.vx;
        this.y += this.vy;
      }

      isOffscreen() {
        return this.x > canvas.width + 200 || this.y > canvas.height + 200;
      }

      draw() {
        // Tail: gradient from transparent tail-end to bright head
        const tailX = this.x - Math.cos(this.angle) * this.length;
        const tailY = this.y - Math.sin(this.angle) * this.length;

        const grad = ctx.createLinearGradient(tailX, tailY, this.x, this.y);
        grad.addColorStop(0, `rgba(${this.colour},0)`);
        grad.addColorStop(0.6, `rgba(${this.colour},${this.alpha * 0.3})`);
        grad.addColorStop(1, `rgba(${this.colour},${this.alpha})`);

        ctx.beginPath();
        ctx.moveTo(tailX, tailY);
        ctx.lineTo(this.x, this.y);
        ctx.strokeStyle = grad;
        ctx.lineWidth   = this.width;
        ctx.lineCap     = 'round';
        ctx.stroke();

        // Bright head glow
        const glow = ctx.createRadialGradient(
          this.x, this.y, 0,
          this.x, this.y, this.width * 3
        );
        glow.addColorStop(0, `rgba(${this.colour},${this.alpha})`);
        glow.addColorStop(1, `rgba(${this.colour},0)`);
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.width * 3, 0, Math.PI * 2);
        ctx.fillStyle = glow;
        ctx.fill();
      }
    }

    // ── Static background twinkle stars ─────────────────────────────
    class Twinkle {
      constructor() {
        this.x     = Math.random() * canvas.width;
        this.y     = Math.random() * canvas.height;
        this.r     = 0.3 + Math.random() * 1.2;
        this.base  = 0.1 + Math.random() * 0.5;
        this.phase = Math.random() * Math.PI * 2;
        this.freq  = 0.005 + Math.random() * 0.015;
      }
      draw(t) {
        const a = this.base + Math.sin(this.phase + t * this.freq) * this.base * 0.6;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(200,220,255,${a})`;
        ctx.fill();
      }
    }

    // ── Initialise pools ─────────────────────────────────────────────
    const MAX_STARS    = 12;   // max simultaneous shooting stars
    const NUM_TWINKLES = 180;  // background twinkle dots
    const SPAWN_EVERY  = 800;  // ms between new shooting star spawns

    const stars    = Array.from({ length: 4 }, () => new Star(true));
    const twinkles = Array.from({ length: NUM_TWINKLES }, () => new Twinkle());

    let lastSpawn = 0;
    let frame     = 0;
    let rafId;

    // ── Animation loop ───────────────────────────────────────────────
    const animate = (timestamp) => {
      // Clear with a very slight fade so tails naturally decay
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw background twinkles
      for (const tw of twinkles) tw.draw(frame);

      // Spawn new shooting stars on a timer
      if (timestamp - lastSpawn > SPAWN_EVERY && stars.length < MAX_STARS) {
        stars.push(new Star());
        lastSpawn = timestamp;
      }

      // Update + draw shooting stars
      for (let i = stars.length - 1; i >= 0; i--) {
        stars[i].update();
        stars[i].draw();
        if (stars[i].isOffscreen()) {
          stars.splice(i, 1);
        }
      }

      frame++;
      rafId = requestAnimationFrame(animate);
    };

    rafId = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',   // never blocks clicks
        zIndex: 0,               // behind everything
      }}
      aria-hidden="true"
    />
  );
}
