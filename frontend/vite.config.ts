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
      port: env.PORT ? parseInt(env.PORT) : 3000,
      host: env.HOST || '0.0.0.0',
      proxy: {
        '/api': {
          target: `http://localhost:${env.API_PORT || 8000}`,
          changeOrigin: true,
        },
        '/ws': {
          target: `ws://localhost:${env.API_PORT || 8000}`,
          ws: true,
        },
      },
    },
    build: {
      outDir: 'dist',
      sourcemap: true,
    },
  }
})
