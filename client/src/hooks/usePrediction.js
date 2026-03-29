import { useState, useCallback } from "react";
import { predictSingle } from "../services/predictService";

export const usePrediction = () => {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const predict = useCallback(async (username) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await predictSingle(username);
      setResult(res.data);
      return res.data;
    } catch (err) {
      const msg =
        err.response?.data?.message || err.response?.data?.error || "Prediction failed";
      setError(msg);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
    setLoading(false);
  }, []);

  return { result, loading, error, predict, reset };
};
