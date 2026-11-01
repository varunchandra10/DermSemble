import React from 'react';
import { Link } from 'react-router-dom';
import { ShieldCheck, ExternalLink } from 'lucide-react';

const Footer: React.FC = () => {
  return (
    <footer className="bg-white border-t border-slate-200/80 pt-12 pb-8 text-slate-600">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 pb-10 border-b border-slate-100">
          
          {/* Col 1: Brand & Identity */}
          <div className="space-y-3 md:col-span-1">
            <div className="flex items-center gap-2">
              <div className="h-8 w-8 rounded-lg bg-blue-600 flex items-center justify-center text-white">
                <ShieldCheck className="h-5 w-5" />
              </div>
              <span className="font-bold text-lg text-slate-900 tracking-tight">
                Derm<span className="text-blue-600">semble</span><span className="text-cyan-600 text-xs ml-1">AI</span>
              </span>
            </div>
            <p className="text-xs text-slate-500 leading-relaxed">
              Clinical decision support system for dermatological triage powered by an INT8 dual-vision ensemble.
            </p>
            <div className="flex items-center gap-2 text-[11px] text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded-md border border-emerald-200/80 w-fit">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-500"></span>
              <span>FastAPI Backend v2.0 Ready</span>
            </div>
          </div>

          {/* Col 2: Navigation Links */}
          <div className="space-y-2">
            <h5 className="text-xs font-bold text-slate-900 uppercase tracking-wider">Clinical Suite</h5>
            <ul className="space-y-1.5 text-xs">
              <li><Link to="/diagnosis" className="hover:text-blue-600 transition-colors">Live Lesion Triage</Link></li>
              <li><Link to="/dashboard" className="hover:text-blue-600 transition-colors">Demographic Fairness Audit</Link></li>
              <li><Link to="/model-insights" className="hover:text-blue-600 transition-colors">Dual Vision Architecture</Link></li>
              <li><Link to="/about" className="hover:text-blue-600 transition-colors">Clinical Urgency Protocol</Link></li>
            </ul>
          </div>

          {/* Col 3: Research & Verification Artifacts */}
          <div className="space-y-2">
            <h5 className="text-xs font-bold text-slate-900 uppercase tracking-wider">Research Artifacts</h5>
            <ul className="space-y-1.5 text-xs">
              <li><a href="http://localhost:8000/docs" target="_blank" rel="noreferrer" className="flex items-center gap-1 hover:text-blue-600 transition-colors">FastAPI Swagger UI <ExternalLink className="h-3 w-3" /></a></li>
              <li><span className="text-slate-400">4,599 Clinical Holdout Cohort</span></li>
              <li><span className="text-slate-400">EEOC 80% Rule Certified</span></li>
              <li><span className="text-slate-400">ONNX INT8 Quantization (57 MB)</span></li>
            </ul>
          </div>

          {/* Col 4: Medical Disclaimer */}
          <div className="space-y-2 md:col-span-1">
            <h5 className="text-xs font-bold text-slate-900 uppercase tracking-wider">Regulatory Disclaimer</h5>
            <p className="text-[11px] text-slate-500 leading-relaxed bg-slate-50 p-3 rounded-xl border border-slate-200/60">
              <strong>Academic & Clinical Triage Tool:</strong> This application is intended to assist medical professionals with preliminary classification. It does not replace tissue biopsy or in-person dermoscopy by a board-certified dermatologist.
            </p>
          </div>

        </div>

        {/* Bottom Credits */}
        <div className="pt-6 flex flex-col sm:flex-row justify-between items-center text-xs text-slate-400 gap-4">
          <p>© {new Date().getFullYear()} Dermsemble Research Group. B.Tech Capstone Project.</p>
          <p className="flex items-center gap-1">
            Engineered with PyTorch, ONNX Runtime & React
          </p>
        </div>

      </div>
    </footer>
  );
};

export default Footer;
