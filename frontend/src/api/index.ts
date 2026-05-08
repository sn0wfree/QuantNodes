import axios from 'axios'
import type { AxiosInstance, AxiosRequestConfig } from 'axios'

const api: AxiosInstance = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

api.interceptors.request.use(
  (config) => config,
  (error) => Promise.reject(error)
)

api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    console.error('API Error:', error.response?.data || error.message)
    return Promise.reject(error)
  }
)

export default api

export const get = <T>(url: string, config?: AxiosRequestConfig): Promise<T> =>
  api.get<T>(url, config) as unknown as Promise<T>

export const post = <T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> =>
  api.post<T>(url, data, config) as unknown as Promise<T>

export const put = <T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> =>
  api.put<T>(url, data, config) as unknown as Promise<T>

export const del = <T>(url: string, config?: AxiosRequestConfig): Promise<T> =>
  api.delete<T>(url, config) as unknown as Promise<T>
