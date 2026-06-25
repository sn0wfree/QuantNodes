import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')

  return {
    plugins: [vue()],
    resolve: {
      alias: {
        '@': resolve(__dirname, 'src'),
      },
    },
    server: {
      port: env.PORT ? parseInt(env.PORT) : 5173,
      host: env.HOST || '0.0.0.0',
      proxy: {
        '/api': {
          target: `http://localhost:${env.API_PORT || 8000}`,
          changeOrigin: true,
          ws: true,
        },
        // v3.0.0: proxy nanobot gateway APIs (session/settings/mcp/workspace)
        // so the Vue frontend can call /gateway/api/* without CORS issues.
        '/gateway': {
          target: `http://localhost:${env.GATEWAY_PORT || 18090}`,
          changeOrigin: true,
          pathRewrite: { '^/gateway': '' },
        },
      },
    },
    build: {
      outDir: 'dist',
      sourcemap: true,
    },
  }
})
