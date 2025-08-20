import React, { createContext, useContext, useEffect, useMemo, useState, useCallback } from 'react';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => localStorage.getItem('auth_token') || null);
  const [role, setRole] = useState(() => localStorage.getItem('auth_role') || null);
  const [user, setUser] = useState(() => {
    const raw = localStorage.getItem('auth_user');
    try { return raw ? JSON.parse(raw) : null; } catch { return null; }
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const apiBase = 'http://localhost:9090/api';

  useEffect(() => {
    if (token) {
      localStorage.setItem('auth_token', token);
    } else {
      localStorage.removeItem('auth_token');
    }
  }, [token]);

  useEffect(() => {
    if (role) localStorage.setItem('auth_role', role); else localStorage.removeItem('auth_role');
  }, [role]);

  useEffect(() => {
    if (user) localStorage.setItem('auth_user', JSON.stringify(user)); else localStorage.removeItem('auth_user');
  }, [user]);

  const login = React.useCallback(async (username, password) => {
    setLoading(true); setError(null);
    try {
      const res = await fetch(`${apiBase}/auth/login`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });
      const data = await res.json();
      if (!res.ok || data.status === 'error') throw new Error(data.message || 'Login failed');
      setToken(data.token);
      setRole(data.role);

      try {
        const meRes = await fetch(`${apiBase}/me`, { headers: { Authorization: `Bearer ${data.token}` } });
        const me = await meRes.json();
        if (me && me.user) setUser(me.user);
      } catch { /* ignore */ }
      return true;
    } catch (e) {
      setError(e.message);
      return false;
    } finally { setLoading(false); }
  }, [apiBase]);

  const signup = React.useCallback(async (username, password, role = 'cashier') => {
    setLoading(true); setError(null);
    try {
      const res = await fetch(`${apiBase}/auth/signup`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password, role })
      });
      const data = await res.json();
      if (!res.ok || data.status === 'error') throw new Error(data.message || 'Signup failed');
      return true;
    } catch (e) {
      setError(e.message);
      return false;
    } finally { setLoading(false); }
  }, [apiBase]);

  const logout = React.useCallback(async () => {
    try {
      if (token) {
        await fetch(`${apiBase}/auth/logout`, { method: 'POST', headers: { Authorization: `Bearer ${token}` } });
      }
    } catch { /* ignore */ }
    setToken(null); setRole(null); setUser(null);
  }, [token, apiBase]);

  const authFetch = useCallback((url, options = {}) => {
    const headers = new Headers(options.headers || {});
    if (token) headers.set('Authorization', `Bearer ${token}`);
    return fetch(url, { ...options, headers });
  }, [token]);

  const demoLogin = React.useCallback(() => {
    const demoToken = 'demo-token';
    const demoUser = { id: 'demo', name: 'Demo User', username: 'demo', role: 'cashier' };
    setToken(demoToken);
    setRole('cashier');
    setUser(demoUser);
    return true;
  }, []);

  const value = useMemo(
    () => ({ token, role, user, login, signup, logout, loading, error, setError, authFetch, demoLogin }),
    [token, role, user, loading, error, authFetch, login, signup, logout, demoLogin]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() { return useContext(AuthContext); }
