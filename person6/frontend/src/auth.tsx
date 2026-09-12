import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { authApi, clearToken, getToken, setToken } from "./api";

type AuthContextValue = { token: string | null; signIn: (email: string, password: string, register?: boolean) => Promise<void>; signOut: () => void; };
const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setAuthToken] = useState(getToken());
  useEffect(() => { const expire = () => setAuthToken(null); window.addEventListener("satquery:unauthorized", expire); return () => window.removeEventListener("satquery:unauthorized", expire); }, []);
  const signIn = async (email: string, password: string, register = false) => {
    const response = register ? await authApi.register(email, password) : await authApi.login(email, password);
    setToken(response.access_token); setAuthToken(response.access_token);
  };
  const signOut = () => { clearToken(); sessionStorage.removeItem("satquery.uploads"); setAuthToken(null); };
  return <AuthContext.Provider value={{ token, signIn, signOut }}>{children}</AuthContext.Provider>;
}
export function useAuth(): AuthContextValue {
  const value = useContext(AuthContext);
  if (!value) throw new Error("useAuth must be used inside AuthProvider");
  return value;
}
