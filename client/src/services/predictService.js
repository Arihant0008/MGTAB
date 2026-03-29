import api from "./api";

export const predictSingle = async (username) => {
  const res = await api.post("/predict/single", { username });
  return res.data;
};

export const predictBatch = async (file) => {
  const formData = new FormData();
  formData.append("file", file);
  const res = await api.post("/predict/batch", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
};

export const getHistory = async (page = 1, limit = 20) => {
  const res = await api.get(`/predict/history?page=${page}&limit=${limit}`);
  return res.data;
};

export const getHistoryById = async (id) => {
  const res = await api.get(`/predict/history/${id}`);
  return res.data;
};

export const deleteHistory = async (id) => {
  const res = await api.delete(`/predict/history/${id}`);
  return res.data;
};

export const getBatchJob = async (id) => {
  const res = await api.get(`/predict/batch/${id}`);
  return res.data;
};
