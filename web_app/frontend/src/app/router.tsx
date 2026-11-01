import React from 'react';
import { createBrowserRouter, RouterProvider, Outlet } from 'react-router-dom';
import Navbar from '../components/layout/Navbar';
import Footer from '../components/layout/Footer';

// Lazy loaded pages
const Home = React.lazy(() => import('../pages/Home'));
const Diagnosis = React.lazy(() => import('../pages/Diagnosis'));
const Dashboard = React.lazy(() => import('../pages/Dashboard'));
const ModelInsights = React.lazy(() => import('../pages/ModelInsights'));
const About = React.lazy(() => import('../pages/About'));

const AppLayout = () => (
  <div className="flex flex-col min-h-screen">
    <Navbar />
    <main className="flex-grow">
      <React.Suspense fallback={<div className="flex justify-center items-center min-h-[50vh]">Loading...</div>}>
        <Outlet />
      </React.Suspense>
    </main>
    <Footer />
  </div>
);

const router = createBrowserRouter([
  {
    path: '/',
    element: <AppLayout />,
    children: [
      { path: '/', element: <Home /> },
      { path: '/diagnosis', element: <Diagnosis /> },
      { path: '/dashboard', element: <Dashboard /> },
      { path: '/model-insights', element: <ModelInsights /> },
      { path: '/about', element: <About /> },
    ],
  },
]);

export const AppRouter = () => <RouterProvider router={router} />;
