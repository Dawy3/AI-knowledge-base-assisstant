/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.BACKEND_URL || 'http://localhost:8000'}/api/:path*`,
      },
      {
        source: '/health/:path*',
        destination: `${process.env.BACKEND_URL || 'http://localhost:8000'}/health/:path*`,
      },
      {
        source: '/health',
        destination: `${process.env.BACKEND_URL || 'http://localhost:8000'}/health`,
      },
    ]
  },
}

module.exports = nextConfig
