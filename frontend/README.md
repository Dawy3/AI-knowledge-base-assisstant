# RAG Frontend

Modern React frontend for the AI Knowledge Assistant RAG system.

## Features

- 🎨 Beautiful, modern UI with Tailwind CSS
- 🔍 Query interface with real-time results
- 📄 Document upload (single and batch)
- 📊 System statistics and metrics dashboard
- ⚡ Fast and responsive
- 🎯 TypeScript for type safety

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Axios** - HTTP client
- **React Router** - Navigation
- **Lucide React** - Icons

## Development

### Prerequisites

- Node.js 18+ and npm

### Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Create `.env` file:
   ```bash
   cp .env.example .env
   ```

3. Update `.env` with your backend URL:
   ```
   VITE_API_URL=http://localhost:8000
   ```

4. Start development server:
   ```bash
   npm run dev
   ```

5. Open http://localhost:3000

## Build

Build for production:

```bash
npm run build
```

The built files will be in the `dist/` directory.

## Docker

See the main README-DOCKER.md for Docker setup instructions.

## Project Structure

```
frontend/
├── src/
│   ├── api/          # API client and types
│   ├── components/   # React components
│   ├── App.tsx       # Main app component
│   ├── main.tsx      # Entry point
│   └── index.css     # Global styles
├── public/           # Static assets
├── index.html        # HTML template
└── package.json      # Dependencies
```

## Components

- **QueryInterface**: Main query interface with results display
- **DocumentUpload**: Document upload (single and batch)
- **SystemStats**: System statistics and metrics dashboard

## API Integration

The frontend communicates with the backend API at the URL specified in `VITE_API_URL`.

Key endpoints:
- `POST /query` - Query the RAG system
- `POST /documents` - Add single document
- `POST /documents/batch` - Add multiple documents
- `GET /stats` - Get system statistics
- `GET /metrics` - Get system metrics

See `src/api/client.ts` for the full API client implementation.

