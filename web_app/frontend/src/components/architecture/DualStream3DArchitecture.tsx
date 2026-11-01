import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { 
  Camera, 
  Layers, 
  RefreshCw, 
  X, 
  Cpu, 
  Microscope, 
  Trophy, 
  BarChart3, 
  ShieldCheck, 
  Gauge, 
  Move3d,
  Zap,
  Maximize2,
  Minimize2,
  Volume2,
  VolumeX
} from 'lucide-react';

interface StageSpec {
  label: string;
  val: string;
}

interface StageMetadata {
  title: string;
  subtitle: string;
  category: string;
  color: string;
  desc: string;
  specs: StageSpec[];
  clinical: string;
}

const STAGE_DATA: Record<string, StageMetadata> = {
  'input': {
    title: "C1 / S1: Dual-Stream Input 384×384×3",
    subtitle: "3-Channel RGB Dermoscopic Image Tensor Slabs",
    category: "Input Preprocessing",
    color: "#94a3b8",
    desc: "High-resolution 384×384×3 RGB dermoscopic image tensors decomposed into floating Red, Green, and Blue spectral glass planes. Both ConvNeXt-Tiny (C1) and Swin-Tiny (S1) receive synchronized 384×384 inputs with 4×4 spatial tokenization grids.",
    specs: [
      { label: "Tensor Shape", val: "384 × 384 × 3" },
      { label: "Spectral Planes", val: "R, G, B Decomposed" },
      { label: "Grid Overlay", val: "4 × 4 Patch Matrix" },
      { label: "Color Space", val: "sRGB Normalized" }
    ],
    clinical: "Red channel penetrates deeper into the dermis to highlight vascular blush and erythema, while Green and Blue channels accentuate superficial melanin pigment networks and follicular plugs."
  },
  'conv_stem': {
    title: "C2: Stem 4×4 Patchify Volume",
    subtitle: "96 × 96 × 96 Volumetric Feature Map Stack",
    category: "ConvNeXt-Tiny (42% Weight)",
    color: "#3b82f6",
    desc: "A 4×4 stride-4 non-overlapping convolution downsamples the 384×384×3 RGB input into a dense 96×96×96 3D feature tensor composed of stacked activation map slices normalized via LayerNorm.",
    specs: [
      { label: "Tensor Shape", val: "96 × 96 × 96" },
      { label: "Kernel / Stride", val: "4×4 Conv, s=4" },
      { label: "Channel Depth", val: "C = 96 Slices" },
      { label: "Normalization", val: "2D LayerNorm" }
    ],
    clinical: "Extracts foundational edge gradients and micro-border transitions across 96 parallel filter banks without losing spatial topology."
  },
  'conv_stages': {
    title: "C3: 4 ResNet-Style Stages (7×7 Depthwise)",
    subtitle: "Multi-Stage Feature Pyramid + Sliding 7×7 Kernel Frustum",
    category: "ConvNeXt-Tiny (42% Weight)",
    color: "#2563eb",
    desc: "4 hierarchical tensor stages progressively compressing spatial resolution (96² → 48² → 24² → 12²) while expanding channel depth (96 → 192 → 384 → 768). Features an active 7×7 Depthwise Convolutional Kernel sliding across the feature slab projecting a 3D receptive field pyramid.",
    specs: [
      { label: "Stage Shapes", val: "96³→48²×192→24²×384→12²×768" },
      { label: "Block Ratios", val: "[3, 3, 9, 3] Blocks" },
      { label: "Kernel Window", val: "7×7 Depthwise Conv" },
      { label: "Receptive Field", val: "Pyramidal Frustum" }
    ],
    clinical: "The large 7×7 depthwise kernel sweeps across skin textures to capture both fine pigment globules and broader reticular pattern transitions."
  },
  'conv_bottleneck': {
    title: "C4: Inverted Bottleneck + GELU",
    subtitle: "1×1 Narrow → 4× Wide Channel Expansion → 1×1 Projection",
    category: "ConvNeXt-Tiny (42% Weight)",
    color: "#60a5fa",
    desc: "Visualizes the 3D expansion-compression tensor sandwich: a compact input tensor (dim C) expands 4× in channel depth via 1×1 pointwise convolution (dim 4C) with glowing GELU activation pathways inside, before projecting back to dim C.",
    specs: [
      { label: "Input Slab", val: "H × W × C (1×1)" },
      { label: "Expanded Core", val: "H × W × 4C (3072-D)" },
      { label: "Activation", val: "GELU Non-Linearity" },
      { label: "Output Slab", val: "H × W × C Residual" }
    ],
    clinical: "Expanding channels 4× in latent space disentangles overlapping dermatological features such as hyperkeratotic scales vs. true melanocytic streaks."
  },
  'conv_feature': {
    title: "C5: Local Margin Texture & Follicular Plugs",
    subtitle: "12 × 12 × 768 High-Frequency Activation Relief Cube",
    category: "ConvNeXt-Tiny Final Head",
    color: "#93c5fd",
    desc: "High-resolution 3D activation map volume with 3D spatial elevation peaks representing intense filter responses to fine local textures, follicular plugs, and pigment networks. Contributes 42% weight to the ensemble.",
    specs: [
      { label: "Output Tensor", val: "12 × 12 × 768" },
      { label: "Pooled Vector", val: "768-D CNN Logits" },
      { label: "Ensemble Weight", val: "0.42 (42%)" },
      { label: "Specialty", val: "High-Freq Micro-Texture" }
    ],
    clinical: "Pinpoints comedo-like follicular plugs, milia-like cysts, and crisp peripheral pigment network terminations."
  },
  'swin_patch': {
    title: "S2: Patch Partition 4×4 Exploded Grid",
    subtitle: "3D Exploded Image Tiles + Linear Token Embeddings",
    category: "Swin-Transformer Tiny (58% Weight)",
    color: "#a855f7",
    desc: "Explodes the 384×384 dermoscopic lesion into a 3D matrix of non-overlapping image patch tiles floating in space. Beneath each spatial tile sits a glowing 96-D linear embedding bar representing its projected ViT token vector.",
    specs: [
      { label: "Token Grid", val: "96 × 96 = 9,216 Tokens" },
      { label: "Patch Size", val: "4 × 4 × 3 (48-D raw)" },
      { label: "Linear Embed", val: "C = 96 Dimension" },
      { label: "3D Structure", val: "Exploded Patch + Token Bar" }
    ],
    clinical: "Converts continuous skin morphology into discrete regional tokens so every micro-zone of the lesion can be compared against all others."
  },
  'swin_stages': {
    title: "S3: Hierarchical Transformer Stages",
    subtitle: "2×2 Patch Merging Multi-Scale Token Hierarchy",
    category: "Swin-Transformer Tiny (58% Weight)",
    color: "#9333ea",
    desc: "Demonstrates hierarchical 2×2 patch merging across 4 stages: neighboring 2×2 token columns merge via golden neural conduits into deeper dimensional pillars (96 → 192 → 384 → 768 channels) while halving spatial resolution.",
    specs: [
      { label: "Hierarchy", val: "96² → 48² → 24² → 12²" },
      { label: "Channel Depth", val: "96 → 192 → 384 → 768" },
      { label: "Stage Depths", val: "[2, 2, 6, 2] Layers" },
      { label: "Operation", val: "2×2 Token Concatenation" }
    ],
    clinical: "Builds a multi-scale anatomical hierarchy from microscopic pigment dots up to macroscopic lesion quadrants."
  },
  'swin_attention': {
    title: "S4: Shifted Window Self-Attention (W=7×7)",
    subtitle: "Animated W-MSA / SW-MSA Bounding Windows + Attention Arcs",
    category: "Swin-Transformer Tiny (58% Weight)",
    color: "#c084fc",
    desc: "3D token grid segmented by glowing W=7×7 local attention window frames that dynamically shift by (3,3) tokens every cycle to compute cross-window self-attention, visualized via curved 3D glowing attention arc beams.",
    specs: [
      { label: "Window Size", val: "W = 7 × 7 Tokens" },
      { label: "Shift Offset", val: "Δ = (3, 3) Cyclic Shift" },
      { label: "Attention Heads", val: "[3, 6, 12, 24] Heads" },
      { label: "Arcs", val: "Local + Cross-Window" }
    ],
    clinical: "Shifted windows allow border tokens on one side of a skin lesion to exchange contextual information with distant margins."
  },
  'swin_feature': {
    title: "S5: Global Lesion Asymmetry & Halos",
    subtitle: "Whole-Lesion Self-Attention Heatmap + Long-Range Arcs",
    category: "Swin-Transformer Tiny Final Head",
    color: "#e9d5ff",
    desc: "3D Global Attention Heatmap Slab crowned by sweeping long-range dependency arcs that measure symmetry, radial plaque spread, and peripheral erythematous halos across the entire lesion span. Contributes 58% weight.",
    specs: [
      { label: "Output Tensor", val: "12 × 12 × 768" },
      { label: "Pooled Vector", val: "768-D ViT Logits" },
      { label: "Ensemble Weight", val: "0.58 (58%)" },
      { label: "Specialty", val: "Global Asymmetry & Halos" }
    ],
    clinical: "Captures global ABCDE melanoma criteria: macroscopic Asymmetry, Border irregularity, and peripheral Erythematous halo spread."
  },
  'fusion_core': {
    title: "M: Soft-Voting Ensemble Multiplexer",
    subtitle: "0.42 × CNN + 0.58 × ViT | Top-1: 75.52% | Top-3: 89.28%",
    category: "Optical Probability Prism Chamber",
    color: "#10b981",
    desc: "Inside a physically-based glass optical prism chamber, 3D probability bar arrays for ConvNeXt (blue), Swin-Tiny (purple), and the Soft-Voting Ensemble (emerald) dynamically compute weighted diagnostic logits in real time.",
    specs: [
      { label: "Fusion Equation", val: "0.42·P_CNN + 0.58·P_ViT" },
      { label: "Top-1 Accuracy", val: "75.52% (State-of-Art)" },
      { label: "Top-3 Accuracy", val: "89.28% Clinical Recall" },
      { label: "Architecture", val: "3D Logit Bar Multiplexer" }
    ],
    clinical: "Combines CNN's sharp local follicular texture precision (42%) with Swin's global asymmetry and halo awareness (58%) for superior diagnostic reliability."
  }
};

export const DualStream3DArchitecture: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const patchCanvasRef = useRef<HTMLCanvasElement>(null);

  // UI Interactive States
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [isMetricsOpen, setIsMetricsOpen] = useState(false);
  const [showAttentionHeatmap, setShowAttentionHeatmap] = useState(false);
  const [explodeFactor, setExplodeFactor] = useState(1.0);
  const [flowSpeed, setFlowSpeed] = useState(1.0);
  const [cnnWeight, setCnnWeight] = useState(42);
  const [lesionSeed, setLesionSeed] = useState(42);

  // Fullscreen, Audio & Camera State
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isAudioActive, setIsAudioActive] = useState(false);
  const [activePreset, setActivePreset] = useState<'overview' | 'conv' | 'swin' | 'fusion'>('overview');

  const audioCtxRef = useRef<AudioContext | null>(null);
  const droneNodesRef = useRef<{ osc1: OscillatorNode; osc2: OscillatorNode; filter: BiquadFilterNode; masterGain: GainNode } | null>(null);

  // Fullscreen handler
  const toggleFullscreen = () => {
    if (!wrapperRef.current) return;
    if (!document.fullscreenElement) {
      wrapperRef.current.requestFullscreen().then(() => setIsFullscreen(true)).catch(console.warn);
    } else {
      document.exitFullscreen().then(() => setIsFullscreen(false)).catch(console.warn);
    }
  };

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  // Ambient Audio Synthesizer
  const toggleAudio = () => {
    if (isAudioActive) {
      if (audioCtxRef.current && audioCtxRef.current.state === 'running') {
        audioCtxRef.current.suspend();
      }
      setIsAudioActive(false);
    } else {
      try {
        const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
        if (!audioCtxRef.current) {
          audioCtxRef.current = new AudioCtx();
        }
        if (audioCtxRef.current.state === 'suspended') {
          audioCtxRef.current.resume();
        }
        const ctx = audioCtxRef.current;
        if (!droneNodesRef.current) {
          const osc1 = ctx.createOscillator();
          const osc2 = ctx.createOscillator();
          const filter = ctx.createBiquadFilter();
          const masterGain = ctx.createGain();

          osc1.type = 'sine';
          osc1.frequency.value = 108; // 108Hz harmonic foundation
          osc2.type = 'triangle';
          osc2.frequency.value = 162; // 162Hz resonant fifth

          filter.type = 'lowpass';
          filter.frequency.value = 320;

          masterGain.gain.setValueAtTime(0.035, ctx.currentTime);

          osc1.connect(filter);
          osc2.connect(filter);
          filter.connect(masterGain);
          masterGain.connect(ctx.destination);

          osc1.start();
          osc2.start();
          droneNodesRef.current = { osc1, osc2, filter, masterGain };
        }
        setIsAudioActive(true);
      } catch (err) {
        console.warn("AudioContext initialization warning", err);
      }
    }
  };

  const playChime = () => {
    if (!isAudioActive || !audioCtxRef.current) return;
    try {
      const ctx = audioCtxRef.current;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(880, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(1320, ctx.currentTime + 0.12);
      gain.gain.setValueAtTime(0.06, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.38);
    } catch {
      // Audio chime fallback
    }
  };

  useEffect(() => {
    return () => {
      if (audioCtxRef.current && audioCtxRef.current.state !== 'closed') {
        audioCtxRef.current.close().catch(() => {});
      }
    };
  }, []);

  // Refs for Three.js instances to avoid re-renders
  const threeRef = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    nodeObjects: THREE.Group[];
    explodableGroups: { slices: THREE.Mesh[]; axis: string }[];
    animatedSubcomponents: Record<string, any>;
    particleStreams: any[];
    dynamicTextures: any[];
    cameraTargetPos: THREE.Vector3;
    cameraLookAtPos: THREE.Vector3;
    currentLookAt: THREE.Vector3;
  } | null>(null);

  // 1. Procedural Lesion Preview Renderer
  const renderProceduralLesion = () => {
    const canvas = patchCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Skin tone background
    ctx.fillStyle = '#dbb29b';
    ctx.fillRect(0, 0, w, h);

    // Vascular diffuse blush
    const blush = ctx.createRadialGradient(w * 0.45, h * 0.5, 10, w * 0.5, h * 0.5, 60);
    blush.addColorStop(0, 'rgba(215, 80, 80, 0.4)');
    blush.addColorStop(1, 'rgba(215, 80, 80, 0)');
    ctx.fillStyle = blush;
    ctx.fillRect(0, 0, w, h);

    // Melanocytic pigment globules
    const numGlobules = 85;
    for (let i = 0; i < numGlobules; i++) {
      const angle = (i / numGlobules) * Math.PI * 2 + lesionSeed;
      const radius = (Math.sin(angle * 3.5 + lesionSeed) * 0.3 + 0.6) * 38;
      const x = w / 2 + Math.cos(angle) * radius + (Math.sin(i * 13) * 6);
      const y = h / 2 + Math.sin(angle) * radius + (Math.cos(i * 7) * 6);

      ctx.beginPath();
      const rSize = (Math.sin(i + lesionSeed) * 0.5 + 1.2) * 3;
      ctx.arc(x, y, rSize, 0, Math.PI * 2);
      ctx.fillStyle = i % 3 === 0 ? 'rgba(45, 20, 15, 0.85)' : 'rgba(92, 45, 25, 0.75)';
      ctx.fill();
    }

    // Micro-reticular border lines
    ctx.strokeStyle = 'rgba(70, 32, 18, 0.45)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let j = 0; j < 12; j++) {
      ctx.moveTo(w / 2 + Math.sin(j * 2) * 20, h / 2 + Math.cos(j * 2) * 20);
      ctx.lineTo(w / 2 + Math.sin(j * 2) * 45, h / 2 + Math.cos(j * 2) * 45);
    }
    ctx.stroke();

    // Attention Heatmap Overlay (Swin Shifted-Window Grid visualization)
    if (showAttentionHeatmap) {
      ctx.strokeStyle = 'rgba(168, 85, 247, 0.5)';
      ctx.lineWidth = 1.2;
      const step = w / 4;
      for (let x = 0; x <= w; x += step) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      }
      for (let y = 0; y <= h; y += step) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }

      const heat = ctx.createRadialGradient(w * 0.42, h * 0.48, 4, w * 0.45, h * 0.5, 36);
      heat.addColorStop(0, 'rgba(239, 68, 68, 0.65)');
      heat.addColorStop(0.5, 'rgba(245, 158, 11, 0.45)');
      heat.addColorStop(1, 'rgba(59, 130, 246, 0)');
      ctx.fillStyle = heat;
      ctx.fillRect(0, 0, w, h);
    }
  };

  useEffect(() => {
    renderProceduralLesion();
  }, [lesionSeed, showAttentionHeatmap]);

  // 2. Initialize Three.js 3D WebGL Canvas
  useEffect(() => {
    if (!containerRef.current) return;

    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight || 650;

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x030712, 0.014);

    const camera = new THREE.PerspectiveCamera(46, width / height, 0.1, 1000);
    camera.position.set(0, 3.5, 34);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    containerRef.current.replaceChildren(renderer.domElement);

    // Physically-Based Lighting
    const ambientLight = new THREE.AmbientLight(0xe2e8f0, 0.75);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.4);
    keyLight.position.set(12, 28, 24);
    scene.add(keyLight);

    const cnnLight = new THREE.PointLight(0x38bdf8, 3.5, 32);
    cnnLight.position.set(-2, 9, 6);
    scene.add(cnnLight);

    const swinLight = new THREE.PointLight(0xc084fc, 3.5, 32);
    swinLight.position.set(-2, -9, 6);
    scene.add(swinLight);

    const fusionLight = new THREE.PointLight(0x10b981, 4.5, 28);
    fusionLight.position.set(17, 0, 6);
    scene.add(fusionLight);

    const grid = new THREE.GridHelper(90, 90, 0x1e293b, 0x0f172a);
    grid.position.y = -10.5;
    scene.add(grid);

    const nodeObjects: THREE.Group[] = [];
    const explodableGroups: { slices: THREE.Mesh[]; axis: string }[] = [];
    const animatedSubcomponents: Record<string, any> = {};
    const particleStreams: any[] = [];
    const dynamicTextures: any[] = [];

    const cameraTargetPos = new THREE.Vector3(0, 3.5, 34);
    const cameraLookAtPos = new THREE.Vector3(0, 0, 0);
    const currentLookAt = new THREE.Vector3(0, 0, 0);

    // Helpers
    function drawTensorTexture(ctx: CanvasRenderingContext2D, w: number, h: number, mode: string, offset: number) {
      const seed = lesionSeed + offset;
      ctx.clearRect(0, 0, w, h);

      if (mode === 'rgb_full' || mode === 'r_chan' || mode === 'g_chan' || mode === 'b_chan') {
        if (mode === 'r_chan') ctx.fillStyle = '#450a0a';
        else if (mode === 'g_chan') ctx.fillStyle = '#064e3b';
        else if (mode === 'b_chan') ctx.fillStyle = '#1e3a8a';
        else ctx.fillStyle = '#dfb59c';
        ctx.fillRect(0, 0, w, h);

        const halo = ctx.createRadialGradient(w * 0.5, h * 0.5, w * 0.08, w * 0.5, h * 0.5, w * 0.44);
        if (mode === 'r_chan') {
          halo.addColorStop(0, 'rgba(255, 90, 90, 0.9)');
          halo.addColorStop(0.6, 'rgba(220, 38, 38, 0.5)');
          halo.addColorStop(1, 'rgba(0, 0, 0, 0)');
        } else if (mode === 'g_chan') {
          halo.addColorStop(0, 'rgba(52, 211, 153, 0.7)');
          halo.addColorStop(1, 'rgba(0, 0, 0, 0)');
        } else if (mode === 'b_chan') {
          halo.addColorStop(0, 'rgba(96, 165, 250, 0.75)');
          halo.addColorStop(1, 'rgba(0, 0, 0, 0)');
        } else {
          halo.addColorStop(0, 'rgba(185, 65, 55, 0.65)');
          halo.addColorStop(0.65, 'rgba(225, 100, 95, 0.38)');
          halo.addColorStop(1, 'rgba(223, 181, 156, 0)');
        }
        ctx.fillStyle = halo;
        ctx.fillRect(0, 0, w, h);

        for (let i = 0; i < 110; i++) {
          const angle = (i / 110) * Math.PI * 2 + seed;
          const asym = Math.sin(angle * 3 + seed) * 0.28 + 0.62;
          const rad = asym * (w * 0.32) * ((i % 7) / 7 * 0.6 + 0.4);
          const x = w * 0.5 + Math.cos(angle) * rad + Math.sin(i * 11 + seed) * 8;
          const y = h * 0.5 + Math.sin(angle) * rad + Math.cos(i * 7 + seed) * 8;
          const rSize = (Math.sin(i + seed) * 0.5 + 1.2) * 4.5;

          ctx.beginPath();
          ctx.arc(x, y, rSize, 0, Math.PI * 2);
          if (mode === 'r_chan') ctx.fillStyle = 'rgba(254, 202, 202, 0.65)';
          else if (mode === 'g_chan') ctx.fillStyle = 'rgba(167, 243, 208, 0.75)';
          else if (mode === 'b_chan') ctx.fillStyle = 'rgba(191, 219, 254, 0.8)';
          else ctx.fillStyle = i % 4 === 0 ? 'rgba(38, 16, 10, 0.88)' : 'rgba(88, 42, 22, 0.72)';
          ctx.fill();
        }
      } else if (mode === 'cnn_activation') {
        ctx.fillStyle = '#07132c';
        ctx.fillRect(0, 0, w, h);
        const cells = 12;
        const cSize = w / cells;
        for (let ix = 0; ix < cells; ix++) {
          for (let iy = 0; iy < cells; iy++) {
            const dx = (ix - cells / 2) / (cells / 2);
            const dy = (iy - cells / 2) / (cells / 2);
            const dist = Math.sqrt(dx * dx + dy * dy);
            const val = Math.max(0, Math.sin(dist * 7 - seed + offset) * 0.5 + 0.5) * (1.1 - dist * 0.6);

            if (val > 0.65) ctx.fillStyle = `rgba(56, 189, 248, ${val})`;
            else if (val > 0.35) ctx.fillStyle = `rgba(37, 99, 235, ${val * 0.85})`;
            else ctx.fillStyle = `rgba(15, 23, 42, 0.9)`;

            ctx.fillRect(ix * cSize + 1, iy * cSize + 1, cSize - 2, cSize - 2);
          }
        }
      } else if (mode === 'vit_attention') {
        ctx.fillStyle = '#16072c';
        ctx.fillRect(0, 0, w, h);
        const cells = 14;
        const cSize = w / cells;
        for (let ix = 0; ix < cells; ix++) {
          for (let iy = 0; iy < cells; iy++) {
            const dx = (ix - cells * 0.45) / (cells * 0.5);
            const dy = (iy - cells * 0.52) / (cells * 0.5);
            const dist = Math.sqrt(dx * dx * 1.3 + dy * dy * 0.8);
            const att = Math.max(0, 1 - dist) * (0.7 + 0.3 * Math.cos(ix * 1.5 + iy + seed));

            if (att > 0.68) ctx.fillStyle = `rgba(244, 63, 94, ${att})`;
            else if (att > 0.42) ctx.fillStyle = `rgba(192, 132, 252, ${att})`;
            else ctx.fillStyle = `rgba(88, 28, 135, ${att * 0.6 + 0.15})`;

            ctx.fillRect(ix * cSize + 1, iy * cSize + 1, cSize - 2, cSize - 2);
          }
        }
      }
    }

    function createTensorTexture(mode: string, offset = 0) {
      const canvas = document.createElement('canvas');
      canvas.width = 256;
      canvas.height = 256;
      const ctx = canvas.getContext('2d')!;
      drawTensorTexture(ctx, 256, 256, mode, offset);
      const tex = new THREE.CanvasTexture(canvas);
      tex.minFilter = THREE.LinearFilter;
      tex.magFilter = THREE.LinearFilter;
      dynamicTextures.push({ tex, canvas, ctx, mode, offset });
      return tex;
    }

    function createMultiSliceTensor(width: number, height: number, depth: number, numSlices: number, mode: string, edgeColor: number, spacing = 0.14) {
      const holder = new THREE.Group();
      const slices: THREE.Mesh[] = [];
      const sliceThickness = Math.max(0.035, (depth / numSlices) * 0.45);
      const sliceGeo = new THREE.BoxGeometry(sliceThickness, height, width);
      const edgesGeo = new THREE.EdgesGeometry(sliceGeo);
      const edgeMat = new THREE.LineBasicMaterial({ color: edgeColor, transparent: true, opacity: 0.55 });

      for (let i = 0; i < numSlices; i++) {
        const tex = createTensorTexture(mode, i * 1.7);
        const mat = new THREE.MeshPhysicalMaterial({
          map: tex,
          color: 0xffffff,
          transparent: true,
          opacity: 0.88,
          roughness: 0.15,
          metalness: 0.1,
          clearcoat: 0.9,
          emissive: edgeColor,
          emissiveMap: tex,
          emissiveIntensity: 0.28
        });

        const mesh = new THREE.Mesh(sliceGeo, mat);
        const baseOffset = (i - (numSlices - 1) / 2) * spacing;
        mesh.position.x = baseOffset;
        mesh.userData.baseX = baseOffset;

        const wire = new THREE.LineSegments(edgesGeo, edgeMat);
        mesh.add(wire);
        holder.add(mesh);
        slices.push(mesh);
      }

      explodableGroups.push({ slices, axis: 'x' });
      return holder;
    }

    // 1. Input RGB Slab
    function buildInputRGB() {
      const group = new THREE.Group();
      const channels = ['r_chan', 'g_chan', 'b_chan', 'rgb_full'];
      const colors = [0xef4444, 0x10b981, 0x3b82f6, 0xffffff];
      const slices: THREE.Mesh[] = [];
      const planeGeo = new THREE.BoxGeometry(0.06, 2.7, 2.7);
      const edgeGeo = new THREE.EdgesGeometry(planeGeo);

      channels.forEach((mode, idx) => {
        const tex = createTensorTexture(mode, 0);
        const mat = new THREE.MeshPhysicalMaterial({
          map: tex,
          transparent: true,
          opacity: idx === 3 ? 0.96 : 0.72,
          roughness: 0.12,
          metalness: 0.1,
          clearcoat: 1.0,
          emissive: colors[idx],
          emissiveIntensity: 0.15
        });
        const slab = new THREE.Mesh(planeGeo, mat);
        const baseX = (idx - 1.5) * 0.28;
        slab.position.x = baseX;
        slab.userData.baseX = baseX;
        slab.rotation.y = 0.22;

        const wire = new THREE.LineSegments(edgeGeo, new THREE.LineBasicMaterial({ color: colors[idx], transparent: true, opacity: 0.75 }));
        slab.add(wire);
        group.add(slab);
        slices.push(slab);
      });

      const laserGeo = new THREE.BoxGeometry(1.1, 0.05, 2.9);
      const laserMat = new THREE.MeshBasicMaterial({ color: 0x34d399, transparent: true, opacity: 0.85 });
      const laserBar = new THREE.Mesh(laserGeo, laserMat);
      laserBar.rotation.y = 0.22;
      group.add(laserBar);
      animatedSubcomponents.inputLaser = laserBar;

      explodableGroups.push({ slices, axis: 'x' });
      return group;
    }

    // 2. Conv Stem
    function buildConvStem() {
      const group = new THREE.Group();
      const slab = createMultiSliceTensor(2.1, 2.1, 1.4, 6, 'cnn_activation', 0x38bdf8, 0.18);
      group.add(slab);
      const cageGeo = new THREE.BoxGeometry(1.35, 2.25, 2.25);
      group.add(new THREE.LineSegments(
        new THREE.EdgesGeometry(cageGeo),
        new THREE.LineBasicMaterial({ color: 0x60a5fa, transparent: true, opacity: 0.45 })
      ));
      return group;
    }

    // 3. Conv Stages
    function buildConvStages() {
      const group = new THREE.Group();
      const stageSpecs = [
        { x: -1.35, h: 2.1, w: 2.1, d: 0.28, slices: 3, color: 0x60a5fa },
        { x: -0.50, h: 1.65, w: 1.65, d: 0.42, slices: 4, color: 0x3b82f6 },
        { x:  0.45, h: 1.25, w: 1.25, d: 0.62, slices: 5, color: 0x2563eb },
        { x:  1.50, h: 0.90, w: 0.90, d: 0.85, slices: 6, color: 0x1d4ed8 }
      ];

      stageSpecs.forEach((st) => {
        const subSlab = createMultiSliceTensor(st.w, st.h, st.d, st.slices, 'cnn_activation', st.color, 0.11);
        subSlab.position.x = st.x;
        group.add(subSlab);
      });

      const rfGroup = new THREE.Group();
      const kernelGeo = new THREE.BoxGeometry(0.12, 0.62, 0.62);
      const kernelMesh = new THREE.Mesh(kernelGeo, new THREE.MeshBasicMaterial({ color: 0xfacc15, transparent: true, opacity: 0.28 }));
      kernelMesh.add(new THREE.LineSegments(new THREE.EdgesGeometry(kernelGeo), new THREE.LineBasicMaterial({ color: 0xfacc15 })));
      kernelMesh.position.x = -1.35;
      rfGroup.add(kernelMesh);

      const frustumGeo = new THREE.ConeGeometry(0.45, 0.95, 4);
      frustumGeo.rotateZ(Math.PI / 2);
      frustumGeo.rotateX(Math.PI / 4);
      const frustumMesh = new THREE.Mesh(frustumGeo, new THREE.MeshBasicMaterial({ color: 0x38bdf8, wireframe: true, transparent: true, opacity: 0.65 }));
      frustumMesh.position.x = -0.88;
      rfGroup.add(frustumMesh);

      group.add(rfGroup);
      animatedSubcomponents.convKernel = rfGroup;
      return group;
    }

    // 4. Inverted Bottleneck
    function buildInvertedBottleneck() {
      const group = new THREE.Group();
      const inSlab = createMultiSliceTensor(1.2, 1.2, 0.25, 2, 'cnn_activation', 0x93c5fd, 0.12);
      inSlab.position.x = -1.15;
      group.add(inSlab);

      const expandedSlab = createMultiSliceTensor(1.85, 1.85, 1.1, 7, 'cnn_activation', 0x38bdf8, 0.14);
      expandedSlab.position.x = 0;
      group.add(expandedSlab);

      const outSlab = createMultiSliceTensor(1.2, 1.2, 0.25, 2, 'cnn_activation', 0x93c5fd, 0.12);
      outSlab.position.x = 1.15;
      group.add(outSlab);

      return group;
    }

    // 5. Conv Feature
    function buildConvFeature() {
      const group = new THREE.Group();
      const baseSlab = createMultiSliceTensor(1.5, 1.5, 0.9, 5, 'cnn_activation', 0x38bdf8, 0.14);
      group.add(baseSlab);

      const reliefGroup = new THREE.Group();
      reliefGroup.position.x = 0.55;
      const barGeo = new THREE.BoxGeometry(1, 0.16, 0.16);
      const bars: THREE.Mesh[] = [];
      for (let r = 0; r < 5; r++) {
        for (let c = 0; c < 5; c++) {
          const isHot = (r + c) % 2 === 0;
          const mat = new THREE.MeshPhysicalMaterial({
            color: isHot ? 0x38bdf8 : 0x1d4ed8,
            emissive: isHot ? 0x0ea5e9 : 0x1e3a8a,
            emissiveIntensity: 0.6,
            clearcoat: 1.0
          });
          const bar = new THREE.Mesh(barGeo, mat);
          bar.position.set(0.15, (r - 2) * 0.25, (c - 2) * 0.25);
          bar.userData.phase = r * 1.3 + c * 0.9;
          reliefGroup.add(bar);
          bars.push(bar);
        }
      }
      group.add(reliefGroup);
      animatedSubcomponents.convReliefBars = bars;
      return group;
    }

    // 6. Swin Patch Partition
    function buildSwinPatch() {
      const group = new THREE.Group();
      const tex = createTensorTexture('rgb_full', 0);
      const tileGeo = new THREE.BoxGeometry(0.08, 0.46, 0.46);
      const tokenBarGeo = new THREE.BoxGeometry(0.45, 0.12, 0.12);
      const tiles: THREE.Group[] = [];

      for (let row = 0; row < 4; row++) {
        for (let col = 0; col < 4; col++) {
          const tileGroup = new THREE.Group();
          const baseY = (row - 1.5) * 0.58;
          const baseZ = (col - 1.5) * 0.58;
          tileGroup.position.set(0, baseY, baseZ);
          tileGroup.userData = { baseY, baseZ, phase: row * 4 + col };

          const tileMesh = new THREE.Mesh(tileGeo, new THREE.MeshPhysicalMaterial({
            map: tex, roughness: 0.15, clearcoat: 1.0, emissive: 0xa855f7, emissiveIntensity: 0.18
          }));
          tileMesh.add(new THREE.LineSegments(new THREE.EdgesGeometry(tileGeo), new THREE.LineBasicMaterial({ color: 0xd8b4fe })));
          tileGroup.add(tileMesh);

          const tokenBar = new THREE.Mesh(tokenBarGeo, new THREE.MeshPhysicalMaterial({
            color: 0xa855f7, emissive: 0x9333ea, emissiveIntensity: 0.5, transparent: true, opacity: 0.85
          }));
          tokenBar.position.x = 0.32;
          tileGroup.add(tokenBar);

          group.add(tileGroup);
          tiles.push(tileGroup);
        }
      }
      group.rotation.y = 0.25;
      animatedSubcomponents.swinPatchTiles = tiles;
      return group;
    }

    // 7. Swin Stages
    function buildSwinStages() {
      const group = new THREE.Group();
      const st1 = new THREE.Group();
      st1.position.x = -1.2;
      const tGeo1 = new THREE.BoxGeometry(0.28, 0.28, 0.28);
      for (let r = 0; r < 4; r++) {
        for (let c = 0; c < 4; c++) {
          const m = new THREE.Mesh(tGeo1, new THREE.MeshPhysicalMaterial({
            color: 0xc084fc, emissive: 0x9333ea, emissiveIntensity: 0.35, clearcoat: 0.9
          }));
          m.position.set(0, (r - 1.5) * 0.42, (c - 1.5) * 0.42);
          st1.add(m);
        }
      }
      group.add(st1);

      const st2 = new THREE.Group();
      st2.position.x = 0.15;
      const tGeo2 = new THREE.BoxGeometry(0.65, 0.48, 0.48);
      const mergeMat = new THREE.LineBasicMaterial({ color: 0xf0abfc, transparent: true, opacity: 0.55 });

      for (let r = 0; r < 2; r++) {
        for (let c = 0; c < 2; c++) {
          const targetY = (r - 0.5) * 0.72;
          const targetZ = (c - 0.5) * 0.72;
          const m = new THREE.Mesh(tGeo2, new THREE.MeshPhysicalMaterial({
            color: 0x9333ea, emissive: 0x7e22ce, emissiveIntensity: 0.45, clearcoat: 1.0
          }));
          m.position.set(0, targetY, targetZ);
          m.add(new THREE.LineSegments(new THREE.EdgesGeometry(tGeo2), mergeMat));
          st2.add(m);
        }
      }
      group.add(st2);

      const st3 = createMultiSliceTensor(1.1, 1.1, 0.85, 5, 'vit_attention', 0xd8b4fe, 0.14);
      st3.position.x = 1.35;
      group.add(st3);
      return group;
    }

    // 8. Swin Shifted Window
    function buildSwinShiftedWindow() {
      const group = new THREE.Group();
      const tokenSlab = createMultiSliceTensor(2.15, 2.15, 0.45, 3, 'vit_attention', 0xa855f7, 0.15);
      group.add(tokenSlab);

      const winGroup = new THREE.Group();
      winGroup.position.x = 0.28;
      const winGeo = new THREE.BoxGeometry(0.14, 0.95, 0.95);
      const winEdges = new THREE.EdgesGeometry(winGeo);
      const windows: THREE.LineSegments[] = [];

      const offsets = [[-0.52, -0.52], [-0.52, 0.52], [0.52, -0.52], [0.52, 0.52]];
      offsets.forEach(([wy, wz], idx) => {
        const wFrame = new THREE.LineSegments(
          winEdges,
          new THREE.LineBasicMaterial({ color: idx % 2 === 0 ? 0xf43f5e : 0xe879f9 })
        );
        wFrame.position.set(0, wy, wz);
        wFrame.userData = { baseY: wy, baseZ: wz, idx };
        winGroup.add(wFrame);
        windows.push(wFrame);
      });
      group.add(winGroup);
      animatedSubcomponents.swinWindows = windows;

      const arcGroup = new THREE.Group();
      arcGroup.position.x = 0.35;
      const arcPairs = [
        [new THREE.Vector3(0, -0.6, -0.6), new THREE.Vector3(0.55, 0, 0), new THREE.Vector3(0, 0.6, 0.6)],
        [new THREE.Vector3(0, 0.6, -0.5), new THREE.Vector3(0.48, 0.2, 0), new THREE.Vector3(0, -0.5, 0.6)]
      ];
      arcPairs.forEach((pts, i) => {
        const curve = new THREE.QuadraticBezierCurve3(pts[0], pts[1], pts[2]);
        const tubeGeo = new THREE.TubeGeometry(curve, 20, 0.028, 6, false);
        const tubeMat = new THREE.MeshBasicMaterial({ color: i === 0 ? 0xf43f5e : 0xd8b4fe, transparent: true, opacity: 0.85 });
        arcGroup.add(new THREE.Mesh(tubeGeo, tubeMat));
      });
      group.add(arcGroup);
      return group;
    }

    // 9. Swin Feature
    function buildSwinFeature() {
      const group = new THREE.Group();
      const slab = createMultiSliceTensor(1.65, 1.65, 0.85, 5, 'vit_attention', 0xe879f9, 0.15);
      group.add(slab);

      const haloGeo = new THREE.RingGeometry(0.65, 0.82, 32);
      const haloMat = new THREE.MeshBasicMaterial({ color: 0xf43f5e, side: THREE.DoubleSide, transparent: true, opacity: 0.7 });
      const haloRing = new THREE.Mesh(haloGeo, haloMat);
      haloRing.rotation.y = Math.PI / 2;
      haloRing.position.x = 0.48;
      group.add(haloRing);
      animatedSubcomponents.swinHaloRing = haloRing;
      return group;
    }

    // 10. Soft-Voting Fusion Core
    function buildFusionCore() {
      const group = new THREE.Group();
      const chamberGeo = new THREE.CylinderGeometry(2.15, 2.15, 3.1, 6);
      const chamberMat = new THREE.MeshPhysicalMaterial({
        color: 0xecfdf5,
        roughness: 0.08,
        metalness: 0.1,
        transmission: 0.82,
        transparent: true,
        opacity: 0.42,
        clearcoat: 1.0,
        emissive: 0x10b981,
        emissiveIntensity: 0.12
      });
      const chamber = new THREE.Mesh(chamberGeo, chamberMat);
      chamber.add(new THREE.LineSegments(
        new THREE.EdgesGeometry(chamberGeo),
        new THREE.LineBasicMaterial({ color: 0x34d399, transparent: true, opacity: 0.65 })
      ));
      group.add(chamber);

      const barsGroup = new THREE.Group();
      barsGroup.position.y = -1.2;
      const barCollections: { cnn: THREE.Mesh[]; vit: THREE.Mesh[]; fused: THREE.Mesh[] } = { cnn: [], vit: [], fused: [] };
      const barGeo = new THREE.BoxGeometry(0.24, 1.0, 0.24);
      barGeo.translate(0, 0.5, 0);

      const cnnProbs = [0.71, 0.18, 0.11];
      const vitProbs = [0.78, 0.14, 0.08];

      for (let cls = 0; cls < 3; cls++) {
        const zPos = (cls - 1) * 0.72;

        const cnnBar = new THREE.Mesh(barGeo, new THREE.MeshPhysicalMaterial({
          color: 0x3b82f6, emissive: 0x2563eb, emissiveIntensity: 0.5
        }));
        cnnBar.position.set(-0.55, 0, zPos);
        cnnBar.scale.y = cnnProbs[cls] * 2.1;
        cnnBar.userData.baseProb = cnnProbs[cls];
        barsGroup.add(cnnBar);
        barCollections.cnn.push(cnnBar);

        const vitBar = new THREE.Mesh(barGeo, new THREE.MeshPhysicalMaterial({
          color: 0xa855f7, emissive: 0x9333ea, emissiveIntensity: 0.5
        }));
        vitBar.position.set(0, 0, zPos);
        vitBar.scale.y = vitProbs[cls] * 2.1;
        vitBar.userData.baseProb = vitProbs[cls];
        barsGroup.add(vitBar);
        barCollections.vit.push(vitBar);

        const fusedProb = 0.42 * cnnProbs[cls] + 0.58 * vitProbs[cls];
        const fusedBar = new THREE.Mesh(barGeo, new THREE.MeshPhysicalMaterial({
          color: 0x10b981, emissive: 0x34d399, emissiveIntensity: 0.8
        }));
        fusedBar.position.set(0.55, 0, zPos);
        fusedBar.scale.y = fusedProb * 2.4;
        barsGroup.add(fusedBar);
        barCollections.fused.push(fusedBar);
      }

      group.add(barsGroup);
      animatedSubcomponents.fusionBars = barCollections;

      const ringGeo = new THREE.TorusGeometry(2.35, 0.035, 12, 64);
      const ring = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({ color: 0x34d399, transparent: true, opacity: 0.7 }));
      ring.rotation.x = Math.PI / 2.2;
      group.userData.ring = ring;
      group.add(ring);
      return group;
    }

    function createDualLineSprite(title: string, subtitle: string, hexColor: number) {
      const canvas = document.createElement('canvas');
      canvas.width = 512;
      canvas.height = 140;
      const ctx = canvas.getContext('2d')!;

      ctx.fillStyle = 'rgba(3, 7, 18, 0.78)';
      ctx.strokeStyle = '#' + hexColor.toString(16).padStart(6, '0');
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.roundRect(16, 14, 480, 112, 22);
      ctx.fill();
      ctx.stroke();

      ctx.font = 'bold 24px monospace';
      ctx.textAlign = 'center';
      ctx.fillStyle = '#ffffff';
      ctx.fillText(title, 256, 60);

      ctx.font = '500 20px monospace';
      ctx.fillStyle = '#' + hexColor.toString(16).padStart(6, '0');
      ctx.fillText(subtitle, 256, 98);

      const texture = new THREE.CanvasTexture(canvas);
      texture.minFilter = THREE.LinearFilter;
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false }));
      sprite.scale.set(5.2, 1.42, 1);
      return sprite;
    }

    const configs = [
      { id: 'input', key: 'input', name: 'Input RGB 384×384×3', sub: '3-Ch Dermoscopic Slab', x: -16.5, y: 0, z: 0, color: 0x94a3b8, builder: buildInputRGB },
      { id: 'conv_stem', key: 'conv_stem', name: 'Stem: 4×4 Patchify', sub: 'Tensor 96×96×96', x: -9.5, y: 4.8, z: 0, color: 0x3b82f6, builder: buildConvStem },
      { id: 'conv_stages', key: 'conv_stages', name: '4 ResNet-Style Stages', sub: '7×7 DW Conv Pyramid', x: -2.2, y: 4.8, z: 0, color: 0x2563eb, builder: buildConvStages },
      { id: 'conv_bottleneck', key: 'conv_bottleneck', name: 'Inverted Bottleneck', sub: '1×1 → 4× GELU → 1×1', x: 5.2, y: 4.8, z: 0, color: 0x60a5fa, builder: buildInvertedBottleneck },
      { id: 'conv_feature', key: 'conv_feature', name: 'Local Margin Texture', sub: '12×12×768 Plugs & Nets', x: 11.4, y: 3.6, z: 0, color: 0x93c5fd, builder: buildConvFeature },
      { id: 'swin_patch', key: 'swin_patch', name: 'Patch Partition 4×4', sub: 'Exploded Tiles + Tokens', x: -9.5, y: -4.8, z: 0, color: 0xa855f7, builder: buildSwinPatch },
      { id: 'swin_stages', key: 'swin_stages', name: 'Hierarchical Stages', sub: '2×2 Patch Merging', x: -2.2, y: -4.8, z: 0, color: 0x9333ea, builder: buildSwinStages },
      { id: 'swin_attention', key: 'swin_attention', name: 'Shifted Window MSA', sub: 'W=7×7 Local + Shift Arcs', x: 5.2, y: -4.8, z: 0, color: 0xc084fc, builder: buildSwinShiftedWindow },
      { id: 'swin_feature', key: 'swin_feature', name: 'Global Asymmetry Map', sub: '12×12×768 Halo Spread', x: 11.4, y: -3.6, z: 0, color: 0xe9d5ff, builder: buildSwinFeature },
      { id: 'fusion_core', key: 'fusion_core', name: 'Soft-Voting Ensemble', sub: '0.42·CNN + 0.58·ViT (75.52%)', x: 17.2, y: 0, z: 0, color: 0x10b981, builder: buildFusionCore }
    ];

    configs.forEach(cfg => {
      const grp = new THREE.Group();
      grp.position.set(cfg.x, cfg.y, cfg.z);
      grp.userData = { id: cfg.id, key: cfg.key, name: cfg.name, originalPos: new THREE.Vector3(cfg.x, cfg.y, cfg.z) };
      grp.add(cfg.builder());

      const label = createDualLineSprite(cfg.name, cfg.sub, cfg.color);
      label.position.set(0, -2.25, 0);
      grp.add(label);

      scene.add(grp);
      nodeObjects.push(grp);
    });

    // Connections
    const connections = [
      { from: 'input', to: 'conv_stem', color: 0x3b82f6 },
      { from: 'input', to: 'swin_patch', color: 0xa855f7 },
      { from: 'conv_stem', to: 'conv_stages', color: 0x3b82f6 },
      { from: 'conv_stages', to: 'conv_bottleneck', color: 0x3b82f6 },
      { from: 'conv_bottleneck', to: 'conv_feature', color: 0x3b82f6 },
      { from: 'conv_feature', to: 'fusion_core', color: 0x38bdf8 },
      { from: 'swin_patch', to: 'swin_stages', color: 0xa855f7 },
      { from: 'swin_stages', to: 'swin_attention', color: 0xa855f7 },
      { from: 'swin_attention', to: 'swin_feature', color: 0xa855f7 },
      { from: 'swin_feature', to: 'fusion_core', color: 0xc084fc }
    ];

    connections.forEach(conn => {
      const fromNode = nodeObjects.find(n => n.userData.id === conn.from);
      const toNode = nodeObjects.find(n => n.userData.id === conn.to);
      if (fromNode && toNode) {
        const midPoint = new THREE.Vector3().addVectors(fromNode.position, toNode.position).multiplyScalar(0.5);
        midPoint.z += 1.0;
        const curve = new THREE.QuadraticBezierCurve3(fromNode.position, midPoint, toNode.position);

        const tubeGeo = new THREE.TubeGeometry(curve, 32, 0.14, 8, false);
        const tubeMesh = new THREE.Mesh(tubeGeo, new THREE.MeshBasicMaterial({ color: conn.color, transparent: true, opacity: 0.18, wireframe: true }));
        scene.add(tubeMesh);

        const particleCount = 28;
        const positions = new Float32Array(particleCount * 3);
        const progress = new Float32Array(particleCount);

        for (let i = 0; i < particleCount; i++) {
          progress[i] = i / particleCount;
          const pt = curve.getPoint(progress[i]);
          positions[i * 3] = pt.x;
          positions[i * 3 + 1] = pt.y;
          positions[i * 3 + 2] = pt.z;
        }

        const particleGeo = new THREE.BufferGeometry();
        particleGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        const pMat = new THREE.PointsMaterial({
          color: conn.color,
          size: 0.45,
          transparent: true,
          blending: THREE.AdditiveBlending,
          depthWrite: false
        });
        const particleSystem = new THREE.Points(particleGeo, pMat);
        scene.add(particleSystem);
        particleStreams.push({ system: particleSystem, curve, progress, count: particleCount });
      }
    });

    // Orbit & Drag Controls
    let isDragging = false;
    let prevMouse = { x: 0, y: 0 };
    const dom = renderer.domElement;

    const onMouseDown = (e: MouseEvent) => {
      isDragging = true;
      prevMouse = { x: e.clientX, y: e.clientY };
    };

    const onMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        const deltaX = e.clientX - prevMouse.x;
        const deltaY = e.clientY - prevMouse.y;

        const spherical = new THREE.Spherical();
        const offset = new THREE.Vector3().subVectors(camera.position, currentLookAt);
        spherical.setFromVector3(offset);
        spherical.theta -= deltaX * 0.005;
        spherical.phi -= deltaY * 0.005;
        spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.2, spherical.phi));
        offset.setFromSpherical(spherical);
        cameraTargetPos.copy(currentLookAt).add(offset);
        prevMouse = { x: e.clientX, y: e.clientY };
      }
    };

    const onMouseUp = () => { isDragging = false; };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const dir = new THREE.Vector3().subVectors(camera.position, currentLookAt).normalize();
      const dist = camera.position.distanceTo(currentLookAt);
      const newDist = Math.max(8, Math.min(65, dist + e.deltaY * 0.03));
      cameraTargetPos.copy(currentLookAt).add(dir.multiplyScalar(newDist));
    };

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const onClick = (e: MouseEvent) => {
      if (Math.abs(e.clientX - prevMouse.x) > 3 || Math.abs(e.clientY - prevMouse.y) > 3) return;

      const rect = dom.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const clickables: THREE.Mesh[] = [];
      nodeObjects.forEach(grp => {
        grp.traverse(ch => {
          if ((ch as THREE.Mesh).isMesh && !(ch as any).isLineSegments) {
            clickables.push(ch as THREE.Mesh);
          }
        });
      });

      const intersects = raycaster.intersectObjects(clickables);
      if (intersects.length > 0) {
        let target = intersects[0].object as THREE.Object3D;
        while (target.parent && target.parent !== scene) {
          target = target.parent;
        }
        if (target && target.userData && target.userData.id) {
          setSelectedNodeId(target.userData.id);
        }
      }
    };

    dom.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    dom.addEventListener('wheel', onWheel, { passive: false });
    dom.addEventListener('click', onClick);

    threeRef.current = {
      scene,
      camera,
      renderer,
      nodeObjects,
      explodableGroups,
      animatedSubcomponents,
      particleStreams,
      dynamicTextures,
      cameraTargetPos,
      cameraLookAtPos,
      currentLookAt
    };

    // Render loop
    const clock = new THREE.Clock();
    let animId: number;

    const animate = () => {
      animId = requestAnimationFrame(animate);
      const delta = clock.getDelta();
      const time = clock.getElapsedTime();

      camera.position.lerp(cameraTargetPos, 0.06);
      currentLookAt.lerp(cameraLookAtPos, 0.06);
      camera.lookAt(currentLookAt);

      // Subtle float
      nodeObjects.forEach((node, idx) => {
        const orig = node.userData.originalPos;
        node.position.y = orig.y + Math.sin(time * 1.4 + idx * 0.7) * 0.12;
        if (node.userData.ring) {
          node.userData.ring.rotation.z = time * 0.65;
        }
      });

      // Laser
      if (animatedSubcomponents.inputLaser) {
        animatedSubcomponents.inputLaser.position.y = Math.sin(time * 2.4) * 1.25;
      }

      // Conv kernel
      if (animatedSubcomponents.convKernel) {
        animatedSubcomponents.convKernel.position.y = Math.sin(time * 2.0) * 0.55;
        animatedSubcomponents.convKernel.position.z = Math.cos(time * 1.5) * 0.55;
      }

      // Conv relief bars
      if (animatedSubcomponents.convReliefBars) {
        animatedSubcomponents.convReliefBars.forEach((bar: THREE.Mesh) => {
          const s = 0.25 + (Math.sin(time * 3.2 + bar.userData.phase) * 0.5 + 0.5) * 0.55;
          bar.scale.x = s;
        });
      }

      // Swin patch tiles
      if (animatedSubcomponents.swinPatchTiles) {
        const spread = 1.0 + Math.sin(time * 1.8) * 0.12 * explodeFactor;
        animatedSubcomponents.swinPatchTiles.forEach((tile: THREE.Group) => {
          tile.position.y = tile.userData.baseY * spread;
          tile.position.z = tile.userData.baseZ * spread;
        });
      }

      // Swin windows
      if (animatedSubcomponents.swinWindows) {
        const shiftCycle = Math.sin(time * 1.6) > 0 ? 0.22 : 0.0;
        animatedSubcomponents.swinWindows.forEach((win: THREE.LineSegments) => {
          win.position.y = THREE.MathUtils.lerp(win.position.y, win.userData.baseY + shiftCycle, 0.08);
          win.position.z = THREE.MathUtils.lerp(win.position.z, win.userData.baseZ + shiftCycle, 0.08);
        });
      }

      // Swin halo
      if (animatedSubcomponents.swinHaloRing) {
        const hScale = 1.0 + Math.sin(time * 2.8) * 0.1;
        animatedSubcomponents.swinHaloRing.scale.set(hScale, hScale, 1);
      }

      // Particles
      particleStreams.forEach(stream => {
        const positions = stream.system.geometry.attributes.position.array;
        const count = stream.count;
        for (let i = 0; i < count; i++) {
          stream.progress[i] += delta * 0.45 * flowSpeed;
          if (stream.progress[i] > 1.0) stream.progress[i] = 0.0;
          const pt = stream.curve.getPoint(stream.progress[i]);
          positions[i * 3] = pt.x;
          positions[i * 3 + 1] = pt.y;
          positions[i * 3 + 2] = pt.z;
        }
        stream.system.geometry.attributes.position.needsUpdate = true;
      });

      renderer.render(scene, camera);
    };

    animate();

    const handleResize = () => {
      if (!containerRef.current) return;
      const w = containerRef.current.clientWidth;
      const h = containerRef.current.clientHeight || 650;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animId);
      dom.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
      dom.removeEventListener('wheel', onWheel);
      dom.removeEventListener('click', onClick);
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
    };
  }, []);

  // Update exploded slices
  useEffect(() => {
    if (!threeRef.current) return;
    threeRef.current.explodableGroups.forEach(grp => {
      grp.slices.forEach(slice => {
        slice.position.x = slice.userData.baseX * explodeFactor;
      });
    });
  }, [explodeFactor]);

  // Update dynamic 3D probability bars on weight change
  useEffect(() => {
    if (!threeRef.current || !threeRef.current.animatedSubcomponents.fusionBars) return;
    const { cnn, vit, fused } = threeRef.current.animatedSubcomponents.fusionBars;
    const cRatio = cnnWeight / 100;
    const vRatio = 1 - cRatio;

    for (let i = 0; i < 3; i++) {
      const cProb = cnn[i].userData.baseProb;
      const vProb = vit[i].userData.baseProb;
      cnn[i].scale.y = Math.max(0.15, cProb * (0.5 + cRatio) * 2.1);
      vit[i].scale.y = Math.max(0.15, vProb * (0.5 + vRatio) * 2.1);
      const fProb = cRatio * cProb + vRatio * vProb;
      fused[i].scale.y = Math.max(0.2, fProb * 2.45);
    }
  }, [cnnWeight]);

  // Camera presets
  const setCameraPreset = (preset: 'overview' | 'conv' | 'swin' | 'fusion') => {
    setActivePreset(preset);
    playChime();
    if (!threeRef.current) return;
    const { cameraTargetPos, cameraLookAtPos } = threeRef.current;
    if (preset === 'overview') {
      cameraTargetPos.set(0, 3, 34);
      cameraLookAtPos.set(0, 0, 0);
    } else if (preset === 'conv') {
      cameraTargetPos.set(0, 8, 18);
      cameraLookAtPos.set(0, 4.5, 0);
    } else if (preset === 'swin') {
      cameraTargetPos.set(0, -8, 18);
      cameraLookAtPos.set(0, -4.5, 0);
    } else if (preset === 'fusion') {
      cameraTargetPos.set(16, 2, 10);
      cameraLookAtPos.set(17, 0, 0);
    }
  };

  const focusOnNode = (id: string) => {
    playChime();
    if (id === 'input') setActivePreset('overview');
    else if (id.startsWith('conv')) setActivePreset('conv');
    else if (id.startsWith('swin')) setActivePreset('swin');
    else if (id === 'fusion_core') setActivePreset('fusion');

    if (!threeRef.current) return;
    let target = threeRef.current.nodeObjects.find(n => n.userData.id === id);
    if (!target) {
      if (id === 'conv') target = threeRef.current.nodeObjects.find(n => n.userData.id === 'conv_stages');
      else if (id === 'swin') target = threeRef.current.nodeObjects.find(n => n.userData.id === 'swin_attention');
      else if (id === 'fusion') target = threeRef.current.nodeObjects.find(n => n.userData.id === 'fusion_core');
      else if (id === 'input') target = threeRef.current.nodeObjects.find(n => n.userData.id === 'input');
    }
    if (target) {
      const pos = target.position;
      threeRef.current.cameraLookAtPos.copy(pos);
      threeRef.current.cameraTargetPos.set(pos.x, pos.y + 1, pos.z + 8.5);
      setSelectedNodeId(target.userData.id);
    }
  };

  // Simulated accuracy curve
  const deltaFromOptimal = Math.abs(cnnWeight - 42);
  const penalty = (deltaFromOptimal / 42) * 2.8;
  const simulatedAcc = (75.52 - (penalty * penalty * 0.35)).toFixed(2);

  const selectedData = selectedNodeId ? STAGE_DATA[selectedNodeId] : null;

  return (
    <div 
      ref={wrapperRef}
      className={`relative w-full ${
        isFullscreen ? 'fixed inset-0 z-50 h-screen w-screen rounded-none' : 'h-[720px] rounded-3xl'
      } overflow-hidden bg-gray-950 border border-slate-800 shadow-2xl text-slate-100 select-none transition-all`}
    >
      
      {/* 3D WebGL Canvas */}
      <div 
        ref={containerRef} 
        className="absolute inset-0 z-0 cursor-grab active:cursor-grabbing"
      />

      {/* Cinematic Overlay Gradient */}
      <div className="pointer-events-none absolute inset-0 z-1 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-indigo-950/20 via-transparent to-black/80" />

      {/* Top Floating Control Bar */}
      <div className="absolute top-4 left-4 right-4 z-20 flex flex-wrap items-center justify-between gap-3 bg-gray-950/80 backdrop-blur-md px-4 py-3 rounded-2xl border border-white/10">
        
        {/* Title & Pill */}
        <div className="flex items-center space-x-3">
          <div className="h-8 w-8 rounded-lg bg-gradient-to-tr from-blue-600 via-purple-600 to-emerald-400 p-[1.5px] flex items-center justify-center shadow-md shadow-purple-900/30">
            <div className="h-full w-full bg-gray-950 rounded-[7px] flex items-center justify-center">
              <Zap className="w-4 h-4 text-emerald-400" />
            </div>
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h3 className="text-xs font-bold tracking-wider uppercase text-white font-mono">Dermsemble 3D Digital Twin</h3>
              <span className="px-2 py-0.5 text-[9px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 rounded-full font-mono">
                Interactive WebGL
              </span>
            </div>
            <p className="text-[10px] text-slate-400 hidden sm:block">ConvNeXt-Tiny (42%) + Swin-Transformer Tiny (58%) Soft-Voting Architecture</p>
          </div>
        </div>

        {/* Camera Presets & Controls */}
        <div className="flex items-center gap-2">
          
          <div className="hidden md:flex items-center bg-gray-900/90 p-1 rounded-xl border border-white/10 text-xs">
            <button 
              onClick={() => setCameraPreset('overview')}
              className={`px-2.5 py-1 rounded-lg transition-all font-mono font-medium text-[11px] ${
                activePreset === 'overview'
                  ? 'bg-white/20 text-white shadow-xs ring-1 ring-white/30 font-bold'
                  : 'text-slate-400 hover:text-white hover:bg-white/5'
              }`}
            >
              Overview
            </button>
            <button 
              onClick={() => setCameraPreset('conv')}
              className={`px-2.5 py-1 rounded-lg transition-all font-mono font-medium text-[11px] ${
                activePreset === 'conv'
                  ? 'bg-blue-500/25 text-blue-300 ring-1 ring-blue-500/50 shadow-[0_0_12px_rgba(59,130,246,0.35)] font-bold'
                  : 'text-blue-400/80 hover:text-blue-300 hover:bg-blue-500/10'
              }`}
            >
              ConvNeXt (CNN)
            </button>
            <button 
              onClick={() => setCameraPreset('swin')}
              className={`px-2.5 py-1 rounded-lg transition-all font-mono font-medium text-[11px] ${
                activePreset === 'swin'
                  ? 'bg-purple-500/25 text-purple-300 ring-1 ring-purple-500/50 shadow-[0_0_12px_rgba(168,85,247,0.35)] font-bold'
                  : 'text-purple-400/80 hover:text-purple-300 hover:bg-purple-500/10'
              }`}
            >
              Swin-T (ViT)
            </button>
            <button 
              onClick={() => setCameraPreset('fusion')}
              className={`px-2.5 py-1 rounded-lg transition-all font-mono font-medium text-[11px] ${
                activePreset === 'fusion'
                  ? 'bg-emerald-500/25 text-emerald-300 ring-1 ring-emerald-500/50 shadow-[0_0_12px_rgba(16,185,129,0.35)] font-bold'
                  : 'text-emerald-400/80 hover:text-emerald-300 hover:bg-emerald-500/10'
              }`}
            >
              Fusion Core
            </button>
          </div>

          {/* Explode Tensors Slider */}
          <div className="hidden xl:flex items-center space-x-2 bg-gray-900/90 px-3 py-1.5 rounded-xl border border-white/10 text-xs">
            <Layers className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-[10px] text-slate-300 font-mono">Explode</span>
            <input 
              type="range" 
              min="1" 
              max="2.5" 
              step="0.05" 
              value={explodeFactor} 
              onChange={(e) => setExplodeFactor(parseFloat(e.target.value))}
              className="w-16 accent-cyan-400 cursor-pointer" 
            />
          </div>

          {/* Flow Speed Slider */}
          <div className="hidden xl:flex items-center space-x-2 bg-gray-900/90 px-3 py-1.5 rounded-xl border border-white/10 text-xs">
            <Zap className="w-3.5 h-3.5 text-amber-400" />
            <span className="text-[10px] text-slate-300 font-mono">Speed</span>
            <input 
              type="range" 
              min="0.2" 
              max="3" 
              step="0.2" 
              value={flowSpeed} 
              onChange={(e) => setFlowSpeed(parseFloat(e.target.value))}
              className="w-14 accent-amber-400 cursor-pointer" 
            />
          </div>

          {/* Audio Synthesizer Toggle */}
          <button 
            onClick={toggleAudio}
            title={isAudioActive ? "Mute Ambient Medical Synth" : "Enable Ambient Medical Synth"}
            className={`p-2 rounded-xl border transition-all text-xs flex items-center justify-center ${
              isAudioActive 
                ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40 shadow-[0_0_10px_rgba(16,185,129,0.3)]' 
                : 'bg-gray-900/90 text-slate-400 border-white/10 hover:text-slate-200 hover:bg-gray-800'
            }`}
          >
            {isAudioActive ? <Volume2 className="w-3.5 h-3.5" /> : <VolumeX className="w-3.5 h-3.5" />}
          </button>

          {/* Fullscreen Toggle */}
          <button 
            onClick={toggleFullscreen}
            title={isFullscreen ? "Exit Fullscreen" : "Enter Fullscreen View"}
            className="p-2 rounded-xl bg-gray-900/90 border border-white/10 text-slate-400 hover:text-slate-200 hover:bg-gray-800 transition-all text-xs flex items-center justify-center"
          >
            {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
          </button>

          {/* Benchmark Drawer Trigger */}
          <button 
            onClick={() => setIsMetricsOpen(true)}
            className="flex items-center space-x-1.5 px-3 py-1.5 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white rounded-xl text-xs font-semibold shadow-md shadow-purple-900/30 transition-all"
          >
            <Gauge className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Weights & Benchmarks</span>
          </button>

        </div>
      </div>



      {/* Bottom Floating Interactive HUD / Simulator */}
      <div className="absolute bottom-4 left-4 right-4 z-20 max-w-4xl mx-auto pointer-events-none">
        <div className="bg-gray-950/85 backdrop-blur-md rounded-2xl p-3 border border-white/10 pointer-events-auto flex flex-col sm:flex-row items-center justify-between gap-3 shadow-xl">
          
          <div className="flex items-center space-x-3 w-full sm:w-auto">
            <div className="relative w-14 h-14 rounded-xl overflow-hidden border border-white/20 bg-black flex-shrink-0">
              <canvas ref={patchCanvasRef} width={128} height={128} className="w-full h-full object-cover" />
              <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent pointer-events-none" />
              <span className="absolute bottom-0.5 right-1 text-[7px] font-mono text-white/80 bg-black/60 px-1 rounded">384px</span>
            </div>
            <div>
              <div className="flex items-center space-x-1.5">
                <span className="text-[11px] font-bold text-white uppercase tracking-wider">Lesion Simulation</span>
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
              </div>
              <p className="text-[10px] text-slate-400 max-w-xs mt-0.5 leading-snug">
                Procedural dermoscopic synthesis demonstrating micro-reticular pigment vs. halo erythema.
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2 w-full sm:w-auto justify-end">
            <button 
              onClick={() => setLesionSeed(Math.random() * 100)}
              className="px-2.5 py-1.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-[11px] font-mono text-slate-200 flex items-center space-x-1.5 transition"
            >
              <RefreshCw className="w-3 h-3 text-blue-400" />
              <span>Mutate Lesion</span>
            </button>
            <button 
              onClick={() => setShowAttentionHeatmap(!showAttentionHeatmap)}
              className="px-2.5 py-1.5 rounded-xl bg-purple-500/10 hover:bg-purple-500/20 border border-purple-500/30 text-[11px] font-mono text-purple-300 flex items-center space-x-1.5 transition"
            >
              <Layers className="w-3 h-3 text-purple-400" />
              <span>{showAttentionHeatmap ? "Hide Attention" : "Show Attention"}</span>
            </button>
          </div>

        </div>
      </div>

      {/* Dim Backdrop Scrim for Drawers */}
      {(selectedData || isMetricsOpen) && (
        <div 
          onClick={() => {
            setSelectedNodeId(null);
            setIsMetricsOpen(false);
          }}
          className="fixed inset-0 bg-black/60 backdrop-blur-xs z-30 animate-in fade-in duration-200 cursor-pointer"
        />
      )}

      {/* Detail Inspection Modal / Drawer */}
      {selectedData && (
        <div className="fixed inset-y-0 right-0 z-40 w-full sm:w-[440px] bg-gray-950/95 border-l border-white/10 p-5 backdrop-blur-2xl flex flex-col justify-between overflow-y-auto shadow-2xl animate-in slide-in-from-right duration-300">
          <div className="space-y-5">
            
            <div className="flex items-start justify-between border-b border-white/10 pb-3">
              <div className="flex items-center space-x-3">
                <div className="p-2 rounded-xl bg-blue-500/20 text-blue-400 border border-blue-500/30">
                  <Cpu className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-white tracking-wide">{selectedData.title}</h3>
                  <p className="text-[11px] text-blue-400 font-mono">{selectedData.subtitle}</p>
                </div>
              </div>
              <button 
                onClick={() => setSelectedNodeId(null)}
                className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-white/5 transition"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-4 text-xs text-slate-300 leading-relaxed">
              <div className="p-3 rounded-xl bg-white/5 border border-white/5">
                <p>{selectedData.desc}</p>
              </div>

              <div>
                <h4 className="text-[10px] font-mono uppercase text-slate-400 font-semibold mb-2 tracking-wider">Algorithmic Specifications</h4>
                <div className="grid grid-cols-2 gap-2 font-mono text-[10px]">
                  {selectedData.specs.map((item, idx) => (
                    <div key={idx} className="p-2 rounded bg-black/40 border border-white/5">
                      <span className="text-slate-400 block text-[8.5px]">{item.label}</span>
                      <span className="text-white font-bold">{item.val}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="p-3 rounded-xl bg-emerald-950/30 border border-emerald-500/20">
                <div className="flex items-center space-x-1.5 text-emerald-400 font-mono text-xs font-semibold mb-1">
                  <Microscope className="w-4 h-4" />
                  <span>Dermatopathology Interpretation</span>
                </div>
                <p className="text-slate-300 text-[11px] leading-normal">{selectedData.clinical}</p>
              </div>

              {/* Interactive Soft-Voting Weight Sandbox (For Fusion node) */}
              {selectedNodeId === 'fusion_core' && (
                <div className="space-y-3 p-3.5 rounded-xl bg-white/5 border border-white/10">
                  <div className="flex justify-between items-center text-xs">
                    <span className="font-bold text-white">Weight Distribution Simulator</span>
                    <span className="font-mono text-emerald-400 font-bold">{cnnWeight}% CNN / {100 - cnnWeight}% ViT</span>
                  </div>
                  <input 
                    type="range" 
                    min="0" 
                    max="100" 
                    value={cnnWeight} 
                    onChange={(e) => setCnnWeight(parseInt(e.target.value))}
                    className="w-full accent-emerald-500 cursor-pointer" 
                  />
                  <div className="flex justify-between text-[10px] font-mono text-slate-400">
                    <span>ConvNeXt: <span className="text-blue-400 font-bold">{(cnnWeight / 100).toFixed(2)}</span></span>
                    <span>Swin-T: <span className="text-purple-400 font-bold">{((100 - cnnWeight) / 100).toFixed(2)}</span></span>
                  </div>
                  <div className="pt-2 border-t border-white/5 flex justify-between items-center text-xs">
                    <span className="text-slate-300">Simulated Top-1 Accuracy:</span>
                    <span className={`font-mono font-bold ${cnnWeight === 42 ? 'text-emerald-400' : 'text-amber-400'}`}>
                      {simulatedAcc}% {cnnWeight === 42 ? '(Champion Ensemble)' : ''}
                    </span>
                  </div>
                </div>
              )}

            </div>
          </div>

          <div className="pt-4 border-t border-white/10 flex space-x-2">
            <button 
              onClick={() => selectedNodeId && focusOnNode(selectedNodeId)}
              className="flex-1 py-2 px-3 rounded-xl bg-white/10 hover:bg-white/15 text-slate-200 text-xs font-medium transition flex items-center justify-center space-x-1.5"
            >
              <Camera className="w-3.5 h-3.5" />
              <span>Center Camera</span>
            </button>
            <button 
              onClick={() => setSelectedNodeId(null)}
              className="py-2 px-4 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium transition"
            >
              Close
            </button>
          </div>
        </div>
      )}

      {/* Benchmarks & Weights Drawer */}
      {isMetricsOpen && (
        <div className="fixed inset-y-0 left-0 z-40 w-full sm:w-[420px] bg-gray-950/95 border-r border-white/10 p-5 backdrop-blur-2xl flex flex-col justify-between overflow-y-auto shadow-2xl animate-in slide-in-from-left duration-300">
          <div className="space-y-4">
            
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <div className="flex items-center space-x-2">
                <BarChart3 className="w-5 h-5 text-emerald-400" />
                <h3 className="text-sm font-bold text-white tracking-wide">Validation Metrics & Fusion</h3>
              </div>
              <button 
                onClick={() => setIsMetricsOpen(false)}
                className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-white/5 transition"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-3">
              <div className="p-3 rounded-xl bg-blue-950/20 border border-blue-500/20 text-xs">
                <div className="flex justify-between items-center mb-1">
                  <span className="font-bold text-blue-300">ConvNeXt-Tiny Standalone</span>
                  <span className="font-mono text-blue-400 font-bold">Weight: 42%</span>
                </div>
                <div className="grid grid-cols-2 gap-2 mt-2 font-mono text-[10px]">
                  <div className="bg-black/30 p-1.5 rounded">Top-1 Acc: <span className="text-white font-semibold">73.47%</span></div>
                  <div className="bg-black/30 p-1.5 rounded">Top-3 Acc: <span className="text-white font-semibold">87.61%</span></div>
                </div>
                <p className="text-[10px] text-slate-400 mt-2">Specializes in fine spatial boundaries, pigment networks, and follicular plugs.</p>
              </div>

              <div className="p-3 rounded-xl bg-purple-950/20 border border-purple-500/20 text-xs">
                <div className="flex justify-between items-center mb-1">
                  <span className="font-bold text-purple-300">Swin-Transformer Tiny</span>
                  <span className="font-mono text-purple-400 font-bold">Weight: 58%</span>
                </div>
                <div className="grid grid-cols-2 gap-2 mt-2 font-mono text-[10px]">
                  <div className="bg-black/30 p-1.5 rounded">Top-1 Acc: <span className="text-white font-semibold">72.86%</span></div>
                  <div className="bg-black/30 p-1.5 rounded">Top-3 Acc: <span className="text-white font-semibold">87.82%</span></div>
                </div>
                <p className="text-[10px] text-slate-400 mt-2">Excels at long-range structural coherence, global lesion asymmetry, and diffuse redness.</p>
              </div>

              <div className="p-3.5 rounded-xl bg-gradient-to-br from-emerald-950/40 via-gray-900 to-emerald-950/30 border border-emerald-500/40 shadow-lg">
                <div className="flex justify-between items-center mb-2">
                  <div className="flex items-center space-x-1.5">
                    <Trophy className="w-4 h-4 text-emerald-400" />
                    <span className="text-xs font-bold text-emerald-300 uppercase tracking-wide">Soft-Voting Champion</span>
                  </div>
                  <span className="px-1.5 py-0.5 rounded text-[9px] font-mono font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">+2.05% Boost</span>
                </div>
                <div className="grid grid-cols-2 gap-2 font-mono">
                  <div className="bg-black/40 p-2 rounded-lg border border-emerald-500/20 text-center">
                    <span className="text-[9px] text-slate-400 block">TOP-1 ACCURACY</span>
                    <span className="text-base font-extrabold text-emerald-400">75.52%</span>
                  </div>
                  <div className="bg-black/40 p-2 rounded-lg border border-emerald-500/20 text-center">
                    <span className="text-[9px] text-slate-400 block">TOP-3 RECALL</span>
                    <span className="text-base font-extrabold text-cyan-400">89.28%</span>
                  </div>
                </div>
                <p className="text-[10px] text-slate-300 mt-2 font-mono bg-black/40 p-1.5 rounded border border-white/5 text-center">
                  P_final = 0.42 · P_ConvNeXt + 0.58 · P_Swin
                </p>
              </div>
            </div>

            <div className="p-3 rounded-xl bg-white/5 border border-white/10 space-y-1.5 text-xs text-slate-300">
              <h4 className="font-bold text-white flex items-center space-x-1.5">
                <ShieldCheck className="w-4 h-4 text-emerald-400" />
                <span>Clinical Complementarity</span>
              </h4>
              <p className="text-[10px] leading-relaxed text-slate-400">
                CNNs possess strict translation equivariance suited for localized dermatoscopic patterns (e.g. dots, globules, streaks). Meanwhile, Swin Transformers capture contextual cross-lesion dependencies over long spatial expanses via shifted-window self-attention.
              </p>
            </div>

          </div>

          <div className="pt-3 border-t border-white/10 text-[9px] font-mono text-slate-500 text-center">
            Dermsemble Dual-Stream Clin-AI Ensemble • Dermatology Research Platform
          </div>
        </div>
      )}

      {/* Floating Interaction Tooltip */}
      <div className="absolute top-20 right-4 z-10 pointer-events-none px-3 py-1.5 rounded-full bg-gray-900/80 backdrop-blur-md border border-white/10 text-[10px] text-slate-300 flex items-center space-x-1.5 hidden md:flex">
        <Move3d className="w-3.5 h-3.5 text-emerald-400" />
        <span>Drag to orbit • Scroll to zoom • Click 3D tensor blocks to inspect</span>
      </div>

    </div>
  );
};
export default DualStream3DArchitecture;
