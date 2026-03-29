export const validateEmail = (email) => {
  const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return re.test(email);
};

export const validatePassword = (password) => {
  const errors = [];
  if (password.length < 8) errors.push("Must be at least 8 characters");
  if (!/[A-Z]/.test(password)) errors.push("Must contain an uppercase letter");
  if (!/[a-z]/.test(password)) errors.push("Must contain a lowercase letter");
  if (!/[0-9]/.test(password)) errors.push("Must contain a digit");
  if (!/[!@#$%^&*]/.test(password)) errors.push("Must contain a special character (!@#$%^&*)");
  return errors;
};

export const getPasswordStrength = (password) => {
  let score = 0;
  if (password.length >= 8) score++;
  if (password.length >= 12) score++;
  if (/[A-Z]/.test(password) && /[a-z]/.test(password)) score++;
  if (/[0-9]/.test(password)) score++;
  if (/[!@#$%^&*]/.test(password)) score++;

  if (score <= 2) return { label: "Weak", color: "#ef4444", percent: 33 };
  if (score <= 3) return { label: "Medium", color: "#f59e0b", percent: 66 };
  return { label: "Strong", color: "#22c55e", percent: 100 };
};

export const validateUsername = (username) => {
  const cleaned = username.replace(/^@/, "").trim();
  if (!cleaned) return { valid: false, error: "Username is required", cleaned };
  if (cleaned.length > 50) return { valid: false, error: "Max 50 characters", cleaned };
  if (!/^[a-zA-Z0-9_]+$/.test(cleaned))
    return { valid: false, error: "Only letters, numbers, and underscores", cleaned };
  return { valid: true, error: null, cleaned };
};
