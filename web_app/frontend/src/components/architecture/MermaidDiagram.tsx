import React, { useEffect, useRef, useState } from 'react';
import mermaid from 'mermaid';

interface MermaidDiagramProps {
  chart: string;
  className?: string;
}

mermaid.initialize({
  startOnLoad: false,
  theme: 'neutral',
  securityLevel: 'loose',
  fontFamily: 'Inter, system-ui, sans-serif',
  themeVariables: {
    primaryColor: '#eff6ff',
    primaryBorderColor: '#3b82f6',
    primaryTextColor: '#1e293b',
    lineColor: '#64748b',
    secondaryColor: '#f8fafc',
    tertiaryColor: '#f1f5f9',
    edgeLabelBackground: '#ffffff',
  },
});

export const MermaidDiagram: React.FC<MermaidDiagramProps> = ({ chart, className = '' }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [svgContent, setSvgContent] = useState<string>('');
  const [renderError, setRenderError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    const uniqueId = `mermaid-${Math.random().toString(36).substring(2, 9)}`;

    const renderChart = async () => {
      try {
        setRenderError(null);
        const { svg } = await mermaid.render(uniqueId, chart);
        if (isMounted) {
          setSvgContent(svg);
        }
      } catch (err: any) {
        console.error('Mermaid render failure:', err);
        if (isMounted) {
          setRenderError(err?.message || 'Failed to render architecture diagram');
        }
      }
    };

    renderChart();

    return () => {
      isMounted = false;
    };
  }, [chart]);

  if (renderError) {
    return (
      <div className="p-4 bg-red-50 text-red-700 text-xs rounded-xl border border-red-200">
        <p className="font-semibold">Diagram rendering issue:</p>
        <p className="font-mono mt-1">{renderError}</p>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={`mermaid-container w-full overflow-x-auto flex justify-center py-4 ${className}`}
      dangerouslySetInnerHTML={{ __html: svgContent }}
    />
  );
};

export default MermaidDiagram;

