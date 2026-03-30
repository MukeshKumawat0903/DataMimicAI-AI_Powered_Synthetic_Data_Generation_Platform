import axios, { AxiosError, AxiosInstance } from "axios";
import type { AppError } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

let reqCounter = 0;

function createApiClient(): AxiosInstance {
  const instance = axios.create({
    baseURL: API_URL,
    timeout: 120_000,
    headers: { "Content-Type": "application/json" },
  });

  // Inject a unique request ID on every outgoing request
  instance.interceptors.request.use((config) => {
    config.headers["X-Request-ID"] = `dmx-${Date.now()}-${++reqCounter}`;
    return config;
  });

  // Normalize errors to AppError shape
  instance.interceptors.response.use(
    (res) => res,
    (err: AxiosError) => {
      const status = err.response?.status ?? 0;
      const data = err.response?.data as Record<string, unknown> | undefined;
      const message =
        (data?.detail as string) ||
        (data?.message as string) ||
        err.message ||
        "An unexpected error occurred.";

      const appError: AppError = { status, message };
      return Promise.reject(appError);
    }
  );

  return instance;
}

export const apiClient = createApiClient();
