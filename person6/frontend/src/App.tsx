import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { AuthProvider } from "./auth";
import { ThemeProvider } from "./theme";
import { AuthPage, ComparePage, DashboardPage, DataPage, HistoryPage, OpticalSarPage, ResultPage, SettingsPage, WorkspacePage } from "./pages";
import { MapsPage } from "./components/MapsPage";
import "./styles.css";

export function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <BrowserRouter>
        <Routes>
          <Route path="/login" element={<AuthPage mode="login" />} />
          <Route path="/register" element={<AuthPage mode="register" />} />
          <Route element={<AppShell />}>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/workspace" element={<WorkspacePage />} />
            <Route path="/data" element={<DataPage />} />
            <Route path="/analysis" element={<HistoryPage />} />
            <Route path="/analysis/:id" element={<ResultPage />} />
            <Route path="/compare" element={<ComparePage />} />
            <Route path="/optical-sar" element={<OpticalSarPage />} />
            <Route path="/maps" element={<MapsPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
    </ThemeProvider>
  );
}
