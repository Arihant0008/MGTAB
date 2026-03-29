import { createContext, useState, useEffect, useCallback } from "react";
import { getMe } from "../services/authService";

export const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem("mgtab_token"));
  const [loading, setLoading] = useState(true);

  const fetchUser = useCallback(async () => {
    if (!token) {
      setLoading(false);
      return;
    }
    try {
      const res = await getMe();
      setUser(res.data.user);
    } catch {
      // Token invalid/expired
      localStorage.removeItem("mgtab_token");
      localStorage.removeItem("mgtab_user");
      setToken(null);
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    fetchUser();
  }, [fetchUser]);

  const loginUser = (tokenStr, userData) => {
    localStorage.setItem("mgtab_token", tokenStr);
    localStorage.setItem("mgtab_user", JSON.stringify(userData));
    setToken(tokenStr);
    setUser(userData);
  };

  const logoutUser = () => {
    localStorage.removeItem("mgtab_token");
    localStorage.removeItem("mgtab_user");
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider
      value={{ user, token, loading, loginUser, logoutUser, isAuthenticated: !!token && !!user }}
    >
      {children}
    </AuthContext.Provider>
  );
};
