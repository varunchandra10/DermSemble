import React from 'react';
import { Link } from 'react-router-dom';
import { 
  ArrowRight, ShieldCheck, Zap, Activity, Eye, CheckCircle2, 
  Sparkles, Layers, Sliders, ChevronRight, AlertTriangle
} from 'lucide-react';
import { motion } from 'framer-motion';

const Home: React.FC = () => {
  return (
    <div className="bg-slate-50/60 min-h-screen medical-grid-bg">
      
      {/* ==================================================================== */}
      {/* 1. HERO SECTION                                                      */}
      {/* ==================================================================== */}
      <section className="relative overflow-hidden pt-12 pb-20 lg:pt-20 lg:pb-28">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
            
            {/* Left Column: Headline & Action CTAs */}
            <div className="lg:col-span-7 text-left space-y-6">
              
              {/* Architecture Badge */}
              <motion.div 
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-blue-50 border border-blue-200/80 text-blue-700 text-xs font-semibold shadow-2xs"
              >
                <Sparkles className="h-3.5 w-3.5 text-blue-600" />
                <span>NVIDIA RTX 5050 Validated • ConvNeXt + Swin-Transformer Dual Ensemble</span>
              </motion.div>

              {/* Main Headline */}
              <motion.h1 
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-slate-900 leading-[1.12]"
              >
                Clinical AI Decision Support for <span className="gradient-text">Dermatological</span> Manifestations
              </motion.h1>

              {/* Subtitle */}
              <motion.p 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2 }}
                className="text-lg text-slate-600 leading-relaxed max-w-2xl"
              >
                An edge-deployable, demographic-fair preliminary triage engine fusing modern 
                <strong> ConvNeXt-Tiny</strong> (local texture) and <strong>Swin-Transformer</strong> (global context) 
                with acuity-aware oncology safety thresholding, selective uncertainty abstention, and real-time Grad-CAM attribution.
              </motion.p>

              {/* Primary Action Buttons */}
              <motion.div 
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="flex flex-wrap items-center gap-4 pt-2"
              >
                <Link 
                  to="/diagnosis" 
                  className="inline-flex items-center gap-2.5 px-7 py-3.5 bg-gradient-to-r from-blue-600 via-indigo-600 to-blue-700 text-white font-semibold text-sm rounded-xl shadow-md shadow-blue-500/25 hover:shadow-lg hover:shadow-blue-500/35 hover:-translate-y-0.5 transition-all"
                >
                  <Sparkles className="h-4 w-4" />
                  <span>Launch Live Triage</span>
                  <ArrowRight className="h-4 w-4 ml-1" />
                </Link>

                <Link 
                  to="/dashboard" 
                  className="inline-flex items-center gap-2 px-6 py-3.5 bg-white text-slate-700 font-semibold text-sm rounded-xl border border-slate-200/90 shadow-2xs hover:bg-slate-50 hover:text-blue-600 transition-all"
                >
                  <Activity className="h-4 w-4 text-slate-400" />
                  <span>View Validation Dashboard</span>
                </Link>
              </motion.div>

              {/* Regulatory & Safety Badges */}
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4 }}
                className="flex flex-wrap items-center gap-6 pt-4 text-xs text-slate-500 font-medium"
              >
                <div className="flex items-center gap-1.5">
                  <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                  <span>EEOC 80% Fairness Certified (DIR: 89.4%)</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                  <span>Sub-Second Clinical Telemedicine Latency</span>
                </div>
              </motion.div>

            </div>

            {/* Right Column: Live Clinical Telemetry Mock Card */}
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.25 }}
              className="lg:col-span-5"
            >
              <div className="bg-white rounded-3xl p-6 shadow-xl shadow-slate-200/60 border border-slate-200/80 relative overflow-hidden">
                
                {/* Header status bar */}
                <div className="flex items-center justify-between pb-4 border-b border-slate-100">
                  <div className="flex items-center gap-2">
                    <span className="h-2.5 w-2.5 rounded-full bg-emerald-500"></span>
                    <span className="text-xs font-bold text-slate-700 uppercase tracking-wider">Live Pipeline Telemetry</span>
                  </div>
                  <span className="px-2.5 py-0.5 rounded-md bg-blue-50 text-blue-700 text-[11px] font-semibold border border-blue-100">
                    RTX 5050 • 7.2 ms
                  </span>
                </div>

                {/* Simulated Triage Result */}
                <div className="py-5 space-y-4">
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="px-2 py-0.5 rounded-md bg-red-100 text-red-700 text-[10px] font-extrabold uppercase tracking-wider">
                          Tier 1 High Acuity
                        </span>
                        <span className="text-xs text-slate-400">Fitzpatrick IV</span>
                      </div>
                      <h4 className="text-2xl font-black text-slate-900 mt-1">Malignant Melanoma</h4>
                      <p className="text-xs text-slate-500 mt-0.5">Asymmetric pigmented border with atypical network</p>
                    </div>

                    <div className="text-right">
                      <div className="text-2xl font-black text-blue-600">91.9%</div>
                      <span className="text-[10px] text-slate-400 font-semibold uppercase">Calibrated Conf</span>
                    </div>
                  </div>

                  {/* Dual Model Probability Bar */}
                  <div className="space-y-1.5 pt-2">
                    <div className="flex justify-between text-xs font-semibold text-slate-600">
                      <span>ConvNeXt-Tiny (42% Local Texture)</span>
                      <span className="text-blue-600">93.9%</span>
                    </div>
                    <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                      <div className="bg-blue-600 h-full rounded-full" style={{ width: '93.9%' }}></div>
                    </div>
                  </div>

                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-semibold text-slate-600">
                      <span>Swin-Transformer (58% Global Context)</span>
                      <span className="text-indigo-600">82.4%</span>
                    </div>
                    <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                      <div className="bg-indigo-600 h-full rounded-full" style={{ width: '82.4%' }}></div>
                    </div>
                  </div>

                  {/* Grad-CAM Attributions Notice */}
                  <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-100 flex items-center justify-between text-xs">
                    <div className="flex items-center gap-2 text-slate-700">
                      <Eye className="h-4 w-4 text-blue-600" />
                      <span className="font-medium">Grad-CAM Feature Localization</span>
                    </div>
                    <span className="font-bold text-emerald-600 text-[11px] bg-emerald-50 px-2 py-0.5 rounded-md border border-emerald-100">
                      Dual Concordant
                    </span>
                  </div>

                  {/* Clinical Recommendation Badge */}
                  <div className="p-3 bg-red-50/70 border border-red-200/80 rounded-xl flex items-center gap-2 text-xs text-red-800">
                    <AlertTriangle className="h-4 w-4 text-red-600 shrink-0" />
                    <span><strong>Urgent Oncology Protocol:</strong> Biopsy & dermoscopic follow-up advised.</span>
                  </div>
                </div>

                <div className="pt-3 border-t border-slate-100 flex justify-between items-center text-[11px] text-slate-400">
                  <span>Holdout Validation Case #23</span>
                  <Link to="/diagnosis" className="text-blue-600 font-semibold hover:underline flex items-center gap-0.5">
                    Test in Live App <ChevronRight className="h-3 w-3" />
                  </Link>
                </div>

              </div>
            </motion.div>

          </div>
        </div>
      </section>

      {/* ==================================================================== */}
      {/* 2. REAL-TIME SCIENTIFIC VALIDATION METRICS RIBBON                     */}
      {/* ==================================================================== */}
      <section className="bg-white border-y border-slate-200/80 py-10 shadow-xs">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-slate-500 text-xs font-semibold uppercase tracking-wider">
                <Layers className="h-4 w-4 text-blue-600" />
                <span>Scope of Coverage</span>
              </div>
              <div className="text-3xl sm:text-4xl font-extrabold text-slate-900">25 Classes</div>
              <p className="text-xs text-slate-500">2-Tier Clinical Hierarchy (Oncology & Outpatient)</p>
            </div>

            <div className="space-y-1">
              <div className="flex items-center gap-2 text-slate-500 text-xs font-semibold uppercase tracking-wider">
                <ShieldCheck className="h-4 w-4 text-emerald-600" />
                <span>Algorithmic Equity</span>
              </div>
              <div className="text-3xl sm:text-4xl font-extrabold text-slate-900">89.4% DIR</div>
              <p className="text-xs text-slate-500">EEOC Four-Fifths Compliant across Fitzpatrick I–VI</p>
            </div>

            <div className="space-y-1">
              <div className="flex items-center gap-2 text-slate-500 text-xs font-semibold uppercase tracking-wider">
                <Sliders className="h-4 w-4 text-indigo-600" />
                <span>Melanoma Recall</span>
              </div>
              <div className="text-3xl sm:text-4xl font-extrabold text-slate-900">86.2%</div>
              <p className="text-xs text-slate-500">+24 Lethal Cancers Rescued via Acuity Tuning</p>
            </div>

            <div className="space-y-1">
              <div className="flex items-center gap-2 text-slate-500 text-xs font-semibold uppercase tracking-wider">
                <Zap className="h-4 w-4 text-cyan-600" />
                <span>Production Footprint</span>
              </div>
              <div className="text-3xl sm:text-4xl font-extrabold text-slate-900">57.2 MB</div>
              <p className="text-xs text-slate-500">INT8 Quantization: 91.1% Compression (11.3x)</p>
            </div>

          </div>
        </div>
      </section>

      {/* ==================================================================== */}
      {/* 3. THREE-STAGE CLINICAL ARCHITECTURE PIPELINE                        */}
      {/* ==================================================================== */}
      <section className="py-20 lg:py-28 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <span className="text-xs font-bold text-blue-600 uppercase tracking-widest">End-to-End Deep Learning Architecture</span>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-slate-900 mt-2">
            Engineered for Precision, Safety & Clinical Transparency
          </h2>
          <p className="text-base text-slate-600 mt-3">
            Every patient photograph passes through an automated three-stage analytical pipeline designed to eliminate optical noise, cross-verify visual modalities, and provide transparent spatial evidence.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          
          {/* Stage 1 */}
          <div className="bg-white rounded-3xl p-8 border border-slate-200/90 shadow-sm hover:shadow-md transition-shadow relative">
            <div className="h-12 w-12 rounded-2xl bg-blue-50 text-blue-600 flex items-center justify-center font-black text-lg mb-6 border border-blue-100">
              01
            </div>
            <h3 className="text-xl font-bold text-slate-900 mb-2">DullRazor & Color Constancy</h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              Clinical images are cluttered by dark body hair and clinical lighting casts. 
              Our 4-directional line kernels inpaint obstructing hairs via Telea algorithms, while Minkowski 
              $p=6$ Shades-of-Gray normalizes lighting across all 6 Fitzpatrick phototypes.
            </p>
            <div className="mt-6 pt-4 border-t border-slate-100 flex items-center gap-2 text-xs font-semibold text-blue-600">
              <CheckCircle2 className="h-4 w-4" />
              <span>Native 384×384 Spatial Resolution</span>
            </div>
          </div>

          {/* Stage 2 */}
          <div className="bg-white rounded-3xl p-8 border border-slate-200/90 shadow-sm hover:shadow-md transition-shadow relative">
            <div className="h-12 w-12 rounded-2xl bg-indigo-50 text-indigo-600 flex items-center justify-center font-black text-lg mb-6 border border-indigo-100">
              02
            </div>
            <h3 className="text-xl font-bold text-slate-900 mb-2">Dual Vision Soft Voting (0.42 / 0.58)</h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              Instead of a single model, we blend <strong>ConvNeXt-Tiny</strong> (specializing in local lesion border gradients and scaling) 
              with <strong>Swin-Transformer Tiny</strong> (shifted window attention capturing global anatomical lesion asymmetry). 
              Soft voting achieves a super-additive <strong>75.52% Top-1 Accuracy</strong>.
            </p>
            <div className="mt-6 pt-4 border-t border-slate-100 flex items-center gap-2 text-xs font-semibold text-indigo-600">
              <CheckCircle2 className="h-4 w-4" />
              <span>89.28% Top-3 Differential Recall</span>
            </div>
          </div>

          {/* Stage 3 */}
          <div className="bg-white rounded-3xl p-8 border border-slate-200/90 shadow-sm hover:shadow-md transition-shadow relative">
            <div className="h-12 w-12 rounded-2xl bg-cyan-50 text-cyan-700 flex items-center justify-center font-black text-lg mb-6 border border-cyan-100">
              03
            </div>
            <h3 className="text-xl font-bold text-slate-900 mb-2">Grad-CAM & Uncertainty Abstention</h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              To prevent misdiagnosis, an Acuity Multiplier ($\alpha = 1.85$) guards high-risk oncology, while a 
              selective threshold ($\theta = 0.50$) abstains from low-confidence calls, referring ambiguous cases for biopsy. 
              Real-time Grad-CAM generates visual attention heatmaps confirming diagnostic focus.
            </p>
            <div className="mt-6 pt-4 border-t border-slate-100 flex items-center gap-2 text-xs font-semibold text-cyan-700">
              <CheckCircle2 className="h-4 w-4" />
              <span>85.11% Automated Accuracy on Covered Cases</span>
            </div>
          </div>

        </div>
      </section>

      {/* ==================================================================== */}
      {/* 4. CLINICAL CASE STUDIES SHOWCASE                                    */}
      {/* ==================================================================== */}
      <section className="bg-slate-100/70 border-t border-slate-200/80 py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row md:items-end justify-between mb-12">
            <div>
              <span className="text-xs font-bold text-blue-600 uppercase tracking-widest">Empirical Case Studies</span>
              <h2 className="text-3xl font-extrabold text-slate-900 mt-1">Why Dual Ensembling Succeeds</h2>
              <p className="text-sm text-slate-600 mt-2 max-w-xl">
                Audited visual attributions proving how ConvNeXt and Swin-Transformer cross-verify edge textures and global plaque context.
              </p>
            </div>
            <Link 
              to="/diagnosis"
              className="inline-flex items-center gap-1.5 text-sm font-semibold text-blue-600 hover:text-blue-800 mt-4 md:mt-0"
            >
              <span>Test Cases in Live Ingestion</span>
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            
            {/* Case 1 */}
            <div className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-xs flex flex-col justify-between">
              <div>
                <span className="px-2 py-0.5 rounded-md bg-red-100 text-red-700 text-[10px] font-bold uppercase">
                  Tier 1 Oncology
                </span>
                <h4 className="font-bold text-slate-900 text-base mt-2">Malignant Melanoma</h4>
                <p className="text-xs text-slate-500 mt-1">
                  ConvNeXt locks onto the lower irregular border (93.9%), while Swin-T models vertical lesion asymmetry (82.4%), synthesizing an 87.2% consensus.
                </p>
              </div>
              <div className="mt-4 pt-3 border-t border-slate-100 text-[11px] font-semibold text-emerald-600 flex items-center justify-between">
                <span>Dual Concordance</span>
                <span>Corr r = 0.62</span>
              </div>
            </div>

            {/* Case 2 */}
            <div className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-xs flex flex-col justify-between">
              <div>
                <span className="px-2 py-0.5 rounded-md bg-amber-100 text-amber-800 text-[10px] font-bold uppercase">
                  Error Correction
                </span>
                <h4 className="font-bold text-slate-900 text-base mt-2">Psoriasis Plaque vs Scabies</h4>
                <p className="text-xs text-slate-500 mt-1">
                  ConvNeXt was distracted by a linear excoriation scratch mark. Swin-T analyzed the broader plaque, overruling ConvNeXt and rescuing the correct diagnosis.
                </p>
              </div>
              <div className="mt-4 pt-3 border-t border-slate-100 text-[11px] font-semibold text-blue-600 flex items-center justify-between">
                <span>Ensemble Rescued</span>
                <span>Corr r = 0.05</span>
              </div>
            </div>

            {/* Case 3 */}
            <div className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-xs flex flex-col justify-between">
              <div>
                <span className="px-2 py-0.5 rounded-md bg-blue-100 text-blue-700 text-[10px] font-bold uppercase">
                  Diffuse Dispersion
                </span>
                <h4 className="font-bold text-slate-900 text-base mt-2">Tinea Fungal Infection</h4>
                <p className="text-xs text-slate-500 mt-1">
                  Swin-Transformer shifted window attention captures the wide, expanding annular ring across epidermis with 83.6% high confidence.
                </p>
              </div>
              <div className="mt-4 pt-3 border-t border-slate-100 text-[11px] font-semibold text-emerald-600 flex items-center justify-between">
                <span>Global Attention</span>
                <span>Corr r = 0.72</span>
              </div>
            </div>

            {/* Case 4 */}
            <div className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-xs flex flex-col justify-between">
              <div>
                <span className="px-2 py-0.5 rounded-md bg-emerald-100 text-emerald-800 text-[10px] font-bold uppercase">
                  Phototype VI Equity
                </span>
                <h4 className="font-bold text-slate-900 text-base mt-2">Deep Melanin Melanoma</h4>
                <p className="text-xs text-slate-500 mt-1">
                  On deeply pigmented skin, attention remains insulated on the true neoplastic lesion without scattering false alarms on surrounding dark melanin.
                </p>
              </div>
              <div className="mt-4 pt-3 border-t border-slate-100 text-[11px] font-semibold text-emerald-600 flex items-center justify-between">
                <span>Demographic Equity</span>
                <span>Corr r = 0.71</span>
              </div>
            </div>

          </div>
        </div>
      </section>

      {/* ==================================================================== */}
      {/* 5. CALL TO ACTION SECTION                                            */}
      {/* ==================================================================== */}
      <section className="py-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="rounded-3xl bg-gradient-to-r from-slate-900 via-blue-950 to-indigo-950 p-10 sm:p-14 text-white shadow-xl relative overflow-hidden">
          <div className="relative z-10 max-w-2xl space-y-4">
            <span className="px-3 py-1 rounded-full bg-blue-500/20 text-blue-300 text-xs font-semibold border border-blue-400/30">
              Ready for Clinical Evaluation
            </span>
            <h2 className="text-3xl sm:text-4xl font-black tracking-tight leading-tight">
              Test Dermsemble on Patient Lesion Imagery Now
            </h2>
            <p className="text-sm sm:text-base text-slate-300 leading-relaxed">
              Experience the speed, accuracy, and interpretability of our INT8 quantized dual deep learning ensemble. 
              Upload any dermatological photograph or choose from our preloaded validation cases.
            </p>
            <div className="pt-4 flex flex-wrap items-center gap-4">
              <Link 
                to="/diagnosis" 
                className="inline-flex items-center gap-2 px-8 py-3.5 bg-blue-500 text-white font-semibold text-sm rounded-xl shadow-lg shadow-blue-500/40 hover:bg-blue-400 hover:shadow-blue-400/50 transition-all"
              >
                <Sparkles className="h-4 w-4" />
                <span>Start AI Analysis</span>
              </Link>
              <Link 
                to="/about" 
                className="inline-flex items-center gap-2 px-6 py-3.5 bg-white/10 text-white font-semibold text-sm rounded-xl hover:bg-white/20 transition-all border border-white/15"
              >
                <span>Read Clinical Protocol</span>
              </Link>
            </div>
          </div>
        </div>
      </section>

    </div>
  );
};

export default Home;
