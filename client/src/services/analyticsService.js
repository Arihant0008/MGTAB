import api from "./api";

export const getModels = async () => {
  const res = await api.get("/analytics/models");
  return res.data;
};

export const getTrainingLogs = async (model) => {
  const res = await api.get(`/analytics/training-logs/${model}`);
  return res.data;
};

export const getUserStats = async () => {
  const res = await api.get("/analytics/stats");
  return res.data;
};

export const getGlobalStats = async () => {
  const res = await api.get("/analytics/global-stats");
  return res.data;
};
