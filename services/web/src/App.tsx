import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Layout } from './components/Layout'
import { DialogProvider } from './components/Dialog'
import Videos from './pages/Videos'
import VideoDetail from './pages/VideoDetail'
import LocalSources from './pages/LocalSources'
import Search from './pages/Search'

export default function App() {
  return (
    <BrowserRouter>
      <DialogProvider>
      <Layout>
        <Routes>
          <Route path="/" element={<Navigate to="/videos" replace />} />
          <Route path="/videos" element={<Videos />} />
          <Route path="/videos/:id" element={<VideoDetail />} />
          <Route path="/search" element={<Search />} />
          <Route path="/local-sources" element={<LocalSources />} />
          <Route path="/settings" element={<LocalSources />} />
        </Routes>
      </Layout>
      </DialogProvider>
    </BrowserRouter>
  )
}
