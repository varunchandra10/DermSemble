import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import MermaidDiagram from '../components/architecture/MermaidDiagram';
import DualStream3DArchitecture from '../components/architecture/DualStream3DArchitecture';
import { 
  Cpu, 
  Layers, 
  ShieldCheck, 
  Zap, 
  Activity, 
  CheckCircle2, 
  AlertTriangle,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Sliders,
  Box,
  GitBranch,
  ArrowRight,
  Stethoscope,
  Scale,
  FileText
} from 'lucide-react';

const fullPipelineChart = `
graph TD
    classDef input fill:#eff6ff,stroke:#3b82f6,stroke-width:2px,color:#1e3a8a
    classDef prep fill:#f0fdf4,stroke:#22c55e,stroke-width:2px,color:#14532d
    classDef model fill:#faf5ff,stroke:#a855f7,stroke-width:2px,color:#581c87
    classDef fusion fill:#fff7ed,stroke:#f97316,stroke-width:2px,color:#7c2d12
    classDef safety fill:#fef2f2,stroke:#ef4444,stroke-width:2px,color:#7f1d1d
    classDef output fill:#f0fdfa,stroke:#14b8a6,stroke-width:2px,color:#134e4a

    subgraph STAGE1 ["1. Clinical Ingestion & Optical Standardization"]
        A["Patient Skin Photograph<br/>(RGB Raw Ingestion)"]:::input --> B["DullRazor Hair Inpainting<br/>(4-Dir Kernels + Telea)"]:::prep
        B --> C["Minkowski p=6 Shades-of-Gray<br/>(Color Constancy Norm)"]:::prep
        C --> D["Standardized Tensor<br/>(384 x 384 x 3 Native)"]:::prep
    end

    subgraph STAGE2 ["2. Dual INT8 ONNX Deep Learning Engines"]
        D -->|High-Frequency Texture| E["ConvNeXt-Tiny INT8<br/>(28.5 MB, 7x7 Depthwise CNN)"]:::model
        D -->|Global Plaque Asymmetry| F["Swin-Transformer INT8<br/>(28.7 MB, Shifted Window ViT)"]:::model
        E --> G["Probability Vector P_conv<br/>(25 Classes, Calibrated)"]:::model
        F --> H["Probability Vector P_swin<br/>(25 Classes, Calibrated)"]:::model
    end

    subgraph STAGE3 ["3. Decision-Theoretic Calibration Engine"]
        G --> I["Calibrated Soft Voting<br/>P_ens = 0.42 * P_conv + 0.58 * P_swin"]:::fusion
        H --> I
        I --> J["Melanoma Acuity Multiplier<br/>alpha = 1.85 on Tier 1 Cancer"]:::fusion
        J --> K{"Uncertainty Gate<br/>max(P) >= 0.50 ?"}:::safety
    end

    subgraph STAGE4 ["4. Clinical Triage & Visual Attribution"]
        K -->|Conf >= 0.50| L["Automated Triage<br/>Tier 1 Urgent vs Tier 2 Routine"]:::output
        K -->|Conf < 0.50| M["Clinical Abstention<br/>Biopsy & Dermoscopy Advised"]:::safety
        E -.->|Conv Layer 4| N["Dual-Stream Grad-CAM<br/>Cross-Attribution Service"]:::output
        F -.->|Norm Layer| N
        N --> O["Transparent Saliency Heatmap<br/>Spatial Evidence Overlay"]:::output
    end
`;

const dualBackboneChart = `
graph LR
    classDef conv fill:#eff6ff,stroke:#2563eb,stroke-width:2px,color:#1e40af
    classDef swin fill:#faf5ff,stroke:#9333ea,stroke-width:2px,color:#6b21a8
    classDef fusion fill:#ecfdf5,stroke:#059669,stroke-width:2px,color:#065f46

    subgraph CONVNEXT ["ConvNeXt-Tiny (CNN Backbone)"]
        C1["Input 384x384"]:::conv --> C2["Stem: 4x4 Patchify"]:::conv
        C2 --> C3["4 ResNet-style Stages<br/>7x7 Depthwise Convolutions"]:::conv
        C3 --> C4["Inverted Bottleneck + GELU"]:::conv
        C4 --> C5["Local Margin Texture<br/>Follicular Plugs & Networks"]:::conv
    end

    subgraph SWIN ["Swin-Transformer Tiny (ViT Backbone)"]
        S1["Input 384x384"]:::swin --> S2["Patch Partition: 4x4"]:::swin
        S2 --> S3["Hierarchical Transformer Stages"]:::swin
        S3 --> S4["Shifted Window Self-Attention<br/>W=7x7 Local + Cross-Window"]:::swin
        S4 --> S5["Global Lesion Asymmetry<br/>Erythematous Halos & Plaque Spread"]:::swin
    end

    C5 --> M["Soft-Voting Ensemble<br/>0.42 * CNN + 0.58 * ViT<br/>Top-1: 75.52% | Top-3: 89.28%"]:::fusion
    S5 --> M
`;

const decisionPolicyChart = `
graph TD
    classDef start fill:#eff6ff,stroke:#3b82f6,stroke-width:2px,color:#1e3a8a
    classDef math fill:#fefce8,stroke:#ca8a04,stroke-width:2px,color:#713f12
    classDef tier1 fill:#fee2e2,stroke:#ef4444,stroke-width:2px,color:#991b1b
    classDef tier2 fill:#e0e7ff,stroke:#6366f1,stroke-width:2px,color:#3730a3
    classDef abstain fill:#ffedd5,stroke:#f97316,stroke-width:2px,color:#9a3412

    A["Raw Ensemble Softmax Distribution P"]:::start --> B["Check Tier 1 Oncology Classes<br/>(Melanoma, Basal Cell, Squamous Cell)"]:::math
    B --> C["Apply Acuity Multiplier: P_mel = P_mel * 1.85<br/>Re-normalize Probability Simplex"]:::math
    C --> D{"Evaluate Confidence: max(P) >= 0.50 ?"}:::math

    D -->|Above Threshold: max >= 0.50| E{"Is Predicted Class Tier 1 Oncology?"}:::math
    E -->|Yes| F["RED FLAG: Tier 1 Oncology<br/>Immediate Biopsy & Dermoscopy Protocol"]:::tier1
    E -->|No| G["BLUE FLAG: Tier 2 Outpatient<br/>Routine Clinical Management"]:::tier2

    D -->|Below Threshold: max < 0.50| H["AMBER FLAG: Clinical Abstention<br/>Uncertain Diagnostic Boundary<br/>Automated Abstention Triggered (Acc: 85.11%)"]:::abstain
`;

const ModelInsights: React.FC = () => {
  const [displayMode, setDisplayMode] = useState<'3d' | '2d'>('3d');
  const [activeTab, setActiveTab] = useState<'pipeline' | 'backbones' | 'decision'>('pipeline');
  const [zoomScale, setZoomScale] = useState<number>(1);

  const handleZoomIn = () => setZoomScale(prev => Math.min(prev + 0.15, 1.6));
  const handleZoomOut = () => setZoomScale(prev => Math.max(prev - 0.15, 0.7));
  const handleResetZoom = () => setZoomScale(1);

  return (
    <div className="bg-slate-50/70 min-h-screen py-10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-10">
        
        {/* Header */}
        <div className="text-center max-w-3xl mx-auto space-y-3">
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-blue-50 border border-blue-200/80 text-blue-700 text-xs font-semibold">
            <Cpu className="h-3.5 w-3.5 text-blue-600" />
            <span>Dual-Stream 3D Digital Twin & Clinical Schematics</span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
            Dermsemble Deep Learning Architecture
          </h1>
          <p className="text-slate-600 text-sm sm:text-base leading-relaxed">
            A comprehensive, interactive dual visualizer of our INT8 quantized ensemble, DullRazor optical conditioning, 
            acuity-aware safety multipliers, and Grad-CAM spatial attribution.
          </p>

          {/* Primary View Switcher Pills */}
          <div className="pt-2 flex items-center justify-center">
            <div className="inline-flex items-center gap-2 bg-slate-200/70 p-1.5 rounded-2xl border border-slate-300/80 shadow-xs">
              <button
                onClick={() => setDisplayMode('3d')}
                className={`inline-flex items-center gap-2 px-5 py-2 rounded-xl text-xs font-bold transition-all ${
                  displayMode === '3d'
                    ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-md shadow-blue-500/20'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-300/50'
                }`}
              >
                <Box className="h-4 w-4" />
                <span>3D Interactive Neural Twin (WebGL)</span>
              </button>
              <button
                onClick={() => setDisplayMode('2d')}
                className={`inline-flex items-center gap-2 px-5 py-2 rounded-xl text-xs font-bold transition-all ${
                  displayMode === '2d'
                    ? 'bg-white text-blue-700 shadow-sm font-extrabold'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-300/50'
                }`}
              >
                <GitBranch className="h-4 w-4" />
                <span>2D System Flowcharts (Mermaid.js)</span>
              </button>
            </div>
          </div>
        </div>

        {/* Top Metric Highlights Banner */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-xs">
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-500 uppercase">
              <Zap className="h-4 w-4 text-amber-500" />
              <span>Production Footprint</span>
            </div>
            <div className="text-2xl sm:text-3xl font-extrabold text-slate-900 mt-1">57.2 MB</div>
            <p className="text-xs text-slate-500 mt-0.5">INT8 Quantized (11.3x compression)</p>
          </div>

          <div className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-xs">
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-500 uppercase">
              <Activity className="h-4 w-4 text-blue-600" />
              <span>Inference Speed</span>
            </div>
            <div className="text-2xl sm:text-3xl font-extrabold text-slate-900 mt-1">7.2 ms</div>
            <p className="text-xs text-slate-500 mt-0.5">NVIDIA RTX 5050 GPU Execution</p>
          </div>

          <div className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-xs">
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-500 uppercase">
              <ShieldCheck className="h-4 w-4 text-emerald-600" />
              <span>Demographic Equity</span>
            </div>
            <div className="text-2xl sm:text-3xl font-extrabold text-slate-900 mt-1">89.4% DIR</div>
            <p className="text-xs text-slate-500 mt-0.5">EEOC 80% Rule Compliant (I–VI)</p>
          </div>

          <div className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-xs">
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-500 uppercase">
              <Sliders className="h-4 w-4 text-indigo-600" />
              <span>Melanoma Recall</span>
            </div>
            <div className="text-2xl sm:text-3xl font-extrabold text-slate-900 mt-1">86.2%</div>
            <p className="text-xs text-slate-500 mt-0.5">Acuity Multiplier (α = 1.85)</p>
          </div>
        </div>

        {/* 3D View vs 2D View Workspace */}
        {displayMode === '3d' ? (
          <div className="space-y-3">
            <div className="flex items-center justify-between text-xs text-slate-500 px-1">
              <span className="font-semibold text-slate-700 flex items-center gap-1.5">
                <Box className="h-4 w-4 text-blue-600" />
                <span>Real-Time WebGL 3D Architectural Digital Twin</span>
              </span>
              <span className="font-mono text-[11px] text-slate-400">
                Click tensor modules to inspect • Drag to orbit
              </span>
            </div>
            <DualStream3DArchitecture />
          </div>
        ) : (
          <div className="bg-white rounded-3xl border border-slate-200/90 shadow-sm overflow-hidden">
            
            {/* 2D Visualizer Controls Bar */}
            <div className="px-6 py-4 bg-slate-50/80 border-b border-slate-200/80 flex flex-wrap items-center justify-between gap-4">
              
              {/* View Selector Tabs */}
              <div className="flex items-center gap-2 bg-slate-200/60 p-1 rounded-xl text-xs font-semibold">
                <button
                  onClick={() => setActiveTab('pipeline')}
                  className={`px-3.5 py-1.5 rounded-lg transition-all ${
                    activeTab === 'pipeline'
                      ? 'bg-white text-blue-700 shadow-xs font-bold'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  1. Full End-to-End Pipeline
                </button>
                <button
                  onClick={() => setActiveTab('backbones')}
                  className={`px-3.5 py-1.5 rounded-lg transition-all ${
                    activeTab === 'backbones'
                      ? 'bg-white text-blue-700 shadow-xs font-bold'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  2. Dual Backbone Synergy
                </button>
                <button
                  onClick={() => setActiveTab('decision')}
                  className={`px-3.5 py-1.5 rounded-lg transition-all ${
                    activeTab === 'decision'
                      ? 'bg-white text-blue-700 shadow-xs font-bold'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  3. Decision Safety Policies
                </button>
              </div>

              {/* Interactive Zoom Controls */}
              <div className="flex items-center gap-1.5 text-slate-600 text-xs font-medium">
                <span className="text-slate-400 mr-1 text-[11px]">Zoom: {Math.round(zoomScale * 100)}%</span>
                <button
                  onClick={handleZoomIn}
                  className="p-1.5 rounded-lg hover:bg-slate-200/70 border border-slate-200 text-slate-700"
                  title="Zoom In"
                >
                  <ZoomIn className="h-4 w-4" />
                </button>
                <button
                  onClick={handleZoomOut}
                  className="p-1.5 rounded-lg hover:bg-slate-200/70 border border-slate-200 text-slate-700"
                  title="Zoom Out"
                >
                  <ZoomOut className="h-4 w-4" />
                </button>
                <button
                  onClick={handleResetZoom}
                  className="p-1.5 rounded-lg hover:bg-slate-200/70 border border-slate-200 text-slate-700"
                  title="Reset Zoom"
                >
                  <RotateCcw className="h-4 w-4" />
                </button>
              </div>

            </div>

            {/* Mermaid Diagram Canvas Container */}
            <div className="p-6 sm:p-10 min-h-[460px] flex items-center justify-center bg-white overflow-hidden relative">
              <div 
                style={{ 
                  transform: `scale(${zoomScale})`, 
                  transformOrigin: 'top center',
                  transition: 'transform 0.15s ease-out',
                  width: '100%'
                }}
              >
                {activeTab === 'pipeline' && (
                  <MermaidDiagram chart={fullPipelineChart} />
                )}
                {activeTab === 'backbones' && (
                  <MermaidDiagram chart={dualBackboneChart} />
                )}
                {activeTab === 'decision' && (
                  <MermaidDiagram chart={decisionPolicyChart} />
                )}
              </div>
            </div>

            {/* Diagram Legend / Architectural Notes Footer */}
            <div className="px-6 py-4 bg-slate-50/60 border-t border-slate-200/80 text-xs text-slate-500 flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-4 flex-wrap">
                <span className="flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-full bg-blue-500"></span>
                  <span>Ingestion & Normalization</span>
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-full bg-purple-500"></span>
                  <span>Dual INT8 ONNX Engines</span>
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-full bg-amber-500"></span>
                  <span>Soft-Voting Fusion</span>
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-full bg-red-500"></span>
                  <span>Clinical Safety & Abstention</span>
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-full bg-teal-500"></span>
                  <span>Grad-CAM Attribution</span>
                </span>
              </div>
              <span className="font-mono text-[11px] text-slate-400">
                Interactive SVG Graph • Native 384×384
              </span>
            </div>

          </div>
        )}

        {/* Detailed Specification Deep Dives */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          
          <div className="bg-white rounded-2xl p-6 border border-slate-200/80 shadow-xs space-y-3">
            <div className="flex items-center gap-2 text-blue-600 font-bold text-sm">
              <Layers className="h-4 w-4" />
              <h3>1. DullRazor & Color Constancy</h3>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              Clinical images are conditioned with 4-directional morphological line kernels at 0°, 45°, 90°, and 135° to detect and inpaint obstructing hairs using the Telea method without blurring lesion margins. Minkowski p=6 Shades-of-Gray illumination normalization neutralizes lighting color casts across all 6 Fitzpatrick phototypes.
            </p>
            <div className="pt-2 text-[11px] font-semibold text-blue-700 flex items-center gap-1">
              <CheckCircle2 className="h-3.5 w-3.5 text-blue-600" />
              <span>Native 384×384 Pixel Tensor Conditioning</span>
            </div>
          </div>

          <div className="bg-white rounded-2xl p-6 border border-slate-200/80 shadow-xs space-y-3">
            <div className="flex items-center gap-2 text-indigo-600 font-bold text-sm">
              <Cpu className="h-4 w-4" />
              <h3>2. Modern Dual Ensembling</h3>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              Rather than a single CNN, Dermsemble pairs <strong>ConvNeXt-Tiny</strong> (42% weight, 7×7 depthwise convs capturing micro-textures and margin borders) with <strong>Swin-Transformer Tiny</strong> (58% weight, shifted window self-attention capturing global plaque asymmetry and lesion halos), yielding 75.52% Top-1 accuracy and 89.28% Top-3 differential recall.
            </p>
            <div className="pt-2 text-[11px] font-semibold text-indigo-700 flex items-center gap-1">
              <CheckCircle2 className="h-3.5 w-3.5 text-indigo-600" />
              <span>Calibrated Soft-Voting (0.42 / 0.58)</span>
            </div>
          </div>

          <div className="bg-white rounded-2xl p-6 border border-slate-200/80 shadow-xs space-y-3">
            <div className="flex items-center gap-2 text-red-600 font-bold text-sm">
              <AlertTriangle className="h-4 w-4" />
              <h3>3. Acuity Tuning & Grad-CAM</h3>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              To prioritize clinical safety, a Melanoma Acuity Multiplier (α = 1.85) boosts malignant melanoma recall to 86.17% (+24 lethal cancers saved). A selective uncertainty gate (θ = 0.50) abstains on ambiguous cases (lifting covered diagnostic accuracy to 85.11%), while dual Grad-CAM generates spatial heatmap evidence verifying feature localization.
            </p>
            <div className="pt-2 text-[11px] font-semibold text-red-700 flex items-center gap-1">
              <CheckCircle2 className="h-3.5 w-3.5 text-red-600" />
              <span>Supervised Clinical Safety Protocol</span>
            </div>
          </div>

        </div>

        {/* Dual-Backbone Specifications & Hyperparameters Matrix */}
        <div className="bg-white rounded-3xl border border-slate-200/90 shadow-xs overflow-hidden">
          <div className="px-6 py-5 bg-gradient-to-r from-slate-950 via-slate-900 to-indigo-950 text-white flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="inline-flex items-center gap-2 px-2.5 py-0.5 rounded-full bg-blue-500/20 text-blue-300 text-[10px] font-mono font-semibold border border-blue-500/30">
                <Scale className="h-3 w-3" />
                <span>Structural Parity & Synergy Matrix</span>
              </div>
              <h2 className="text-lg sm:text-xl font-bold tracking-tight mt-1 text-white">
                ConvNeXt-Tiny (CNN) vs. Swin-Transformer Tiny (ViT)
              </h2>
              <p className="text-xs text-slate-300 mt-0.5">
                Side-by-side architectural specifications, receptive fields, and inductive bias complementarity.
              </p>
            </div>
            <div className="flex items-center gap-3">
              <span className="px-3 py-1.5 rounded-xl bg-emerald-500/20 text-xs font-mono font-bold text-emerald-300 border border-emerald-500/30 shadow-xs">
                Soft-Voting Top-1: 75.52% (+2.05% Synergy Gain)
              </span>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50/80 text-slate-500 font-mono text-[11px] uppercase tracking-wider">
                  <th className="py-3.5 px-6 font-semibold">Architectural Dimension</th>
                  <th className="py-3.5 px-6 font-semibold text-blue-700 bg-blue-50/50">ConvNeXt-Tiny (42% Weight)</th>
                  <th className="py-3.5 px-6 font-semibold text-purple-700 bg-purple-50/50">Swin-Transformer Tiny (58% Weight)</th>
                  <th className="py-3.5 px-6 font-semibold text-emerald-700 bg-emerald-50/50">Clinical Ensemble Synergy</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-sans">
                <tr className="hover:bg-slate-50/50 transition">
                  <td className="py-3.5 px-6 font-semibold text-slate-900">Network Paradigm</td>
                  <td className="py-3.5 px-6 text-slate-700 bg-blue-50/20 font-mono">Modern Depthwise CNN</td>
                  <td className="py-3.5 px-6 text-slate-700 bg-purple-50/20 font-mono">Shifted Window Hierarchical ViT</td>
                  <td className="py-3.5 px-6 text-emerald-700 bg-emerald-50/20 font-medium">Dual Inductive Hybridization</td>
                </tr>
                <tr className="hover:bg-slate-50/50 transition">
                  <td className="py-3.5 px-6 font-semibold text-slate-900">Native Input Resolution</td>
                  <td className="py-3.5 px-6 text-slate-700 bg-blue-50/20 font-mono">384 × 384 × 3 RGB</td>
                  <td className="py-3.5 px-6 text-slate-700 bg-purple-50/20 font-mono">384 × 384 × 3 RGB</td>
                  <td className="py-3.5 px-6 text-emerald-700 bg-emerald-50/20 font-medium">Synchronized Dual-Stream Ingestion</td>
                </tr>
                <tr className="hover:bg-slate-50/50 transition">
                  <td className="py-3.5 px-6 font-semibold text-slate-900">Stem / Tokenization</td>
                  <td className="py-3.5 px-6 text-slate-700 bg-blue-50/20 font-mono">4×4 Conv (stride 4, 96-D)</td>
                  <td className="py-3.5 px-6 text-slate-700 bg-purple-50/20 font-mono">4×4 Patch Partition (96-D linear)</td>
                  <td className="py-3.5 px-6 text-emerald-700 bg-emerald-50/20 font-medium">Equal 96×96 token spatial grid</td>
                </tr>
                <tr className="hover:bg-slate-50/50 transition">
                  <td className="py-3.5 px-6 font-semibold text-slate-900">Hierarchical Depths</td>
                  <td className="py-3.5 px-6 text-slate-700 bg-blue-50/20 font-mono">[3, 3, 9, 3] = 18 blocks</td>
                  <td className="py-3.5 px-6 text-slate-700 bg-purple-50/20 font-mono">[2, 2, 6, 2] = 12 layers</td>
                  <td className="py-3.5 px-6 text-emerald-700 bg-emerald-50/20 font-medium">Aligned 4-stage resolution decay</td>
                </tr>
                <tr className="hover:bg-slate-50/50 transition">
                  <td className="py-3.5 px-6 font-semibold text-slate-900">Feature Dimensions</td>
                  <td className="py-3.5 px-6 text-slate-700 bg-blue-50/20 font-mono">C = [96, 192, 384, 768]</td>
                  <td className="py-3.5 px-6 text-slate-700 bg-purple-50/20 font-mono">C = [96, 192, 384, 768]</td>
                  <td className="py-3.5 px-6 text-emerald-700 bg-emerald-50/20 font-medium">Matched 768-D latent logit space</td>
                </tr>
                <tr className="hover:bg-slate-50/50 transition">
                  <td className="py-3.5 px-6 font-semibold text-slate-900">Spatial Attention Mechanism</td>
                  <td className="py-3.5 px-6 text-slate-700 bg-blue-50/20 font-mono">7×7 Depthwise Convolutions</td>
                  <td className="py-3.5 px-6 text-slate-700 bg-purple-50/20 font-mono">W=7×7 Shifted Window MSA</td>
                  <td className="py-3.5 px-6 text-emerald-700 bg-emerald-50/20 font-medium">Local translation equivariance + Long-range self-attention</td>
                </tr>
                <tr className="hover:bg-slate-50/50 transition">
                  <td className="py-3.5 px-6 font-semibold text-slate-900">INT8 Model Size & Footprint</td>
                  <td className="py-3.5 px-6 text-slate-700 bg-blue-50/20 font-mono">28.5 MB (ONNX INT8)</td>
                  <td className="py-3.5 px-6 text-slate-700 bg-purple-50/20 font-mono">28.7 MB (ONNX INT8)</td>
                  <td className="py-3.5 px-6 text-emerald-700 bg-emerald-50/20 font-medium">57.2 MB Total Ensemble Footprint</td>
                </tr>
                <tr className="hover:bg-slate-50/50 transition">
                  <td className="py-3.5 px-6 font-semibold text-slate-900">Diagnostic Specialty</td>
                  <td className="py-3.5 px-6 text-blue-700 bg-blue-50/20 font-medium">Micro-margin texture, follicular plugs, pigment globules</td>
                  <td className="py-3.5 px-6 text-purple-700 bg-purple-50/20 font-medium">Global plaque asymmetry, border irregularity, diffuse halos</td>
                  <td className="py-3.5 px-6 text-emerald-700 bg-emerald-50/20 font-bold">Holistic ABCDE Dermatological Coverage</td>
                </tr>
                <tr className="hover:bg-slate-50/50 transition bg-slate-50/40">
                  <td className="py-3.5 px-6 font-extrabold text-slate-900">Top-1 Accuracy / Top-3 Recall</td>
                  <td className="py-3.5 px-6 text-blue-700 bg-blue-50/30 font-bold font-mono">73.47% / 87.61%</td>
                  <td className="py-3.5 px-6 text-purple-700 bg-purple-50/30 font-bold font-mono">72.86% / 87.82%</td>
                  <td className="py-3.5 px-6 text-emerald-700 bg-emerald-50/30 font-extrabold font-mono text-sm">75.52% / 89.28% Champion</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Supervised 3-Tier Clinical Triage Engine */}
        <div className="bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 rounded-3xl p-6 sm:p-8 text-white border border-indigo-900/40 shadow-xl space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-300 text-xs font-semibold border border-emerald-500/30">
                <Stethoscope className="h-3.5 w-3.5" />
                <span>Decision-Theoretic Clinical Protocol</span>
              </div>
              <h2 className="text-xl sm:text-2xl font-bold tracking-tight mt-2 text-white">
                Supervised 3-Tier Clinical Triage Protocol
              </h2>
              <p className="text-xs sm:text-sm text-slate-300 max-w-2xl mt-1">
                Dermsemble avoids black-box predictions by routing all 25 disease classes through an acuity-weighted risk matrix with automated clinical abstention.
              </p>
            </div>
            <div className="flex items-center gap-3">
              <Link
                to="/diagnosis"
                className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-bold shadow-lg shadow-blue-600/30 transition-all hover:scale-[1.02]"
              >
                <Stethoscope className="h-4 w-4" />
                <span>Launch Diagnosis</span>
                <ArrowRight className="h-3.5 w-3.5" />
              </Link>
              <Link
                to="/protocol"
                className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-white/10 hover:bg-white/15 text-slate-200 text-xs font-semibold border border-white/10 transition-all"
              >
                <FileText className="h-4 w-4" />
                <span>Clinical Protocol</span>
              </Link>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
            
            {/* Tier 1 Urgent Oncology */}
            <div className="bg-red-950/30 rounded-2xl p-5 border border-red-500/30 space-y-3">
              <div className="flex items-center justify-between">
                <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold bg-red-500/20 text-red-300 border border-red-500/40">
                  TIER 1 • URGENT ONCOLOGY
                </span>
                <AlertTriangle className="h-4 w-4 text-red-400" />
              </div>
              <h3 className="text-base font-bold text-white">Malignant Neoplasms</h3>
              <p className="text-xs text-slate-300 leading-relaxed">
                Malignant Melanoma, Basal Cell Carcinoma, and Squamous Cell Carcinoma. Calibrated with an Acuity Multiplier (α = 1.85) to achieve 86.2% melanoma recall.
              </p>
              <div className="pt-2 text-[11px] font-mono text-red-300 flex items-center gap-1.5 border-t border-red-500/20">
                <span className="h-1.5 w-1.5 rounded-full bg-red-400"></span>
                <span>Mandatory Urgent 48-Hour Biopsy Protocol</span>
              </div>
            </div>

            {/* Tier 2 Routine Outpatient */}
            <div className="bg-blue-950/30 rounded-2xl p-5 border border-blue-500/30 space-y-3">
              <div className="flex items-center justify-between">
                <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold bg-blue-500/20 text-blue-300 border border-blue-500/40">
                  TIER 2 • ROUTINE OUTPATIENT
                </span>
                <Activity className="h-4 w-4 text-blue-400" />
              </div>
              <h3 className="text-base font-bold text-white">Benign & Inflammatory</h3>
              <p className="text-xs text-slate-300 leading-relaxed">
                Eczema, Psoriasis, Seborrheic Keratoses, Acne, Tinea, and Viral exanthems. High confidence (&ge; 50%) automated routing for standard outpatient treatment.
              </p>
              <div className="pt-2 text-[11px] font-mono text-blue-300 flex items-center gap-1.5 border-t border-blue-500/20">
                <span className="h-1.5 w-1.5 rounded-full bg-blue-400"></span>
                <span>Primary Care Outpatient Management</span>
              </div>
            </div>

            {/* Tier 3 Clinical Abstention */}
            <div className="bg-amber-950/30 rounded-2xl p-5 border border-amber-500/30 space-y-3">
              <div className="flex items-center justify-between">
                <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40">
                  TIER 3 • CLINICAL ABSTENTION
                </span>
                <ShieldCheck className="h-4 w-4 text-amber-400" />
              </div>
              <h3 className="text-base font-bold text-white">Uncertain Diagnostic Boundaries</h3>
              <p className="text-xs text-slate-300 leading-relaxed">
                When top predicted confidence falls below threshold (&lt; 0.50), the system actively abstains, protecting patients and lifting covered accuracy to 85.11%.
              </p>
              <div className="pt-2 text-[11px] font-mono text-amber-300 flex items-center gap-1.5 border-t border-amber-500/20">
                <span className="h-1.5 w-1.5 rounded-full bg-amber-400"></span>
                <span>Requires High-Resolution Re-Imaging</span>
              </div>
            </div>

          </div>
        </div>

      </div>
    </div>
  );
};

export default ModelInsights;
