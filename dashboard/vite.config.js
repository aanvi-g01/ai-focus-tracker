import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev: `npm run dev` on :5173 proxies /api to the FastAPI app on :8765.
// Prod: `npm run build` -> dist/, served by `focus-app` itself.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { '/api': 'http://127.0.0.1:8765' } },
})
