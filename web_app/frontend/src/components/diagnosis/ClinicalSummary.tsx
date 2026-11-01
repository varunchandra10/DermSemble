import React from 'react';
import { 
  FileText, 
  AlertTriangle, 
  Microscope, 
  Stethoscope, 
  Pill, 
  ShieldAlert, 
  Palette, 
  Printer, 
  Layers,
  CheckCircle2
} from 'lucide-react';

interface Props {
  report: string;
}

const ClinicalSummary: React.FC<Props> = ({ report }) => {
  // Helper to extract a named section from the report text
  const extractSection = (heading: string, nextHeadings: string[]) => {
    const startIdx = report.indexOf(heading);
    if (startIdx === -1) return null;
    
    const contentStart = startIdx + heading.length;
    let endIdx = report.length;

    for (const next of nextHeadings) {
      const idx = report.indexOf(next, contentStart);
      if (idx !== -1 && idx < endIdx) {
        endIdx = idx;
      }
    }

    return report.slice(contentStart, endIdx).trim();
  };

  const allHeadings = [
    'CLINICAL DIAGNOSTIC IMPRESSION:',
    '⚠️ CLINICAL UNCERTAINTY ALERT',
    'DIFFERENTIAL DIAGNOSTIC CONSIDERATIONS:',
    'PATHOPHYSIOLOGY & ETIOLOGY:',
    'DERMATOLOGICAL MORPHOLOGY:',
    'RECOMMENDED DIAGNOSTIC WORKUP:',
    'FRONTLINE MANAGEMENT GUIDELINES:',
    'RISK NOTE:'
  ];

  const impression = extractSection('CLINICAL DIAGNOSTIC IMPRESSION:', allHeadings.slice(1));
  const isAbstained = report.includes('CLINICAL UNCERTAINTY ALERT');
  const abstentionAlert = isAbstained ? extractSection('⚠️ CLINICAL UNCERTAINTY ALERT (SYSTEM ABSTENTION TRIGGERED):', allHeadings.slice(2)) : null;
  const differentials = extractSection('DIFFERENTIAL DIAGNOSTIC CONSIDERATIONS:', allHeadings.slice(3));
  const pathophysiology = extractSection('PATHOPHYSIOLOGY & ETIOLOGY:', allHeadings.slice(4));
  const morphology = extractSection('DERMATOLOGICAL MORPHOLOGY:', allHeadings.slice(5));
  const workup = extractSection('RECOMMENDED DIAGNOSTIC WORKUP:', allHeadings.slice(6));
  const management = extractSection('FRONTLINE MANAGEMENT GUIDELINES:', allHeadings.slice(7));
  const riskNote = extractSection('RISK NOTE:', []);

  const isTier1 = report.includes('Tier 1');

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="bg-white rounded-3xl shadow-sm border border-slate-200/90 overflow-hidden space-y-6 p-6 sm:p-8">
      
      {/* Header bar */}
      <div className="flex flex-wrap items-center justify-between pb-5 border-b border-slate-100 gap-4">
        <div className="flex items-center gap-3">
          <div className={`h-10 w-10 rounded-xl flex items-center justify-center font-bold ${
            isTier1 
              ? 'bg-red-50 text-red-600 border border-red-200/80' 
              : 'bg-blue-50 text-blue-600 border border-blue-200/80'
          }`}>
            <FileText className="h-5 w-5" />
          </div>
          <div>
            <h3 className="text-xl font-extrabold text-slate-900 tracking-tight">Structured Clinical Report</h3>
            <p className="text-xs text-slate-500 font-medium">Curated Evidence-Based Dermatological Knowledge Base</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <span className={`px-3 py-1 rounded-full text-xs font-bold border ${
            isTier1
              ? 'bg-red-50 text-red-700 border-red-200'
              : 'bg-blue-50 text-blue-700 border-blue-200'
          }`}>
            {isTier1 ? 'TIER 1 • HIGH ACUITY' : 'TIER 2 • ROUTINE OUTPATIENT'}
          </span>
          <button
            onClick={handlePrint}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 text-slate-600 hover:text-slate-900 hover:bg-slate-50 text-xs font-semibold transition-all shadow-2xs"
            title="Print or export clinical record"
          >
            <Printer className="h-3.5 w-3.5" />
            <span>Export / Print</span>
          </button>
        </div>
      </div>

      {/* Uncertainty Alert Banner (if Abstention Triggered) */}
      {isAbstained && (
        <div className="p-4 bg-amber-50/90 border border-amber-300/80 rounded-2xl flex items-start gap-3.5 shadow-xs">
          <AlertTriangle className="h-5 w-5 text-amber-600 flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h4 className="text-xs font-extrabold text-amber-900 uppercase tracking-wider">
              Clinical Uncertainty Policy Triggered (θ = 0.50)
            </h4>
            <p className="text-xs text-amber-800 leading-relaxed font-medium">
              {abstentionAlert || "Model confidence is below the safety threshold. Automated diagnosis withheld. Immediate dermatologist review recommended."}
            </p>
          </div>
        </div>
      )}

      {/* Primary Diagnostic Impression */}
      {impression && (
        <div className={`p-5 rounded-2xl border ${
          isTier1 ? 'bg-red-50/40 border-red-200/80' : 'bg-slate-50 border-slate-200/80'
        }`}>
          <div className="flex items-center gap-2 mb-2">
            <Stethoscope className={`h-4 w-4 ${isTier1 ? 'text-red-600' : 'text-blue-600'}`} />
            <span className="text-xs font-bold uppercase tracking-wider text-slate-700">Diagnostic Impression</span>
          </div>
          <p className="text-sm text-slate-800 whitespace-pre-line leading-relaxed font-medium">
            {impression}
          </p>
        </div>
      )}

      {/* Differential Candidates */}
      {differentials && (
        <div className="p-4 bg-slate-50/70 rounded-2xl border border-slate-200/70">
          <div className="flex items-center gap-2 mb-2">
            <Layers className="h-4 w-4 text-indigo-600" />
            <span className="text-xs font-bold uppercase tracking-wider text-slate-700">Differential Diagnostic Candidates</span>
          </div>
          <div className="text-xs text-slate-700 whitespace-pre-line leading-relaxed font-mono">
            {differentials}
          </div>
        </div>
      )}

      {/* Two-Column Grid: Pathophysiology & Morphology */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        
        {/* Pathophysiology */}
        {pathophysiology && (
          <div className="p-5 bg-white rounded-2xl border border-slate-200/80 shadow-2xs space-y-2">
            <div className="flex items-center gap-2 text-indigo-600">
              <Microscope className="h-4 w-4" />
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-800">Etiology & Pathophysiology</h4>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              {pathophysiology}
            </p>
          </div>
        )}

        {/* Morphology Across Phototypes */}
        {morphology && (
          <div className="p-5 bg-white rounded-2xl border border-slate-200/80 shadow-2xs space-y-2">
            <div className="flex items-center gap-2 text-purple-600">
              <Palette className="h-4 w-4" />
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-800">Morphology (Fitzpatrick I–VI)</h4>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed whitespace-pre-line">
              {morphology}
            </p>
          </div>
        )}

      </div>

      {/* Workup & Management */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        
        {/* Diagnostic Workup */}
        {workup && (
          <div className="p-5 bg-white rounded-2xl border border-slate-200/80 shadow-2xs space-y-2">
            <div className="flex items-center gap-2 text-emerald-600">
              <CheckCircle2 className="h-4 w-4" />
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-800">Recommended Workup</h4>
            </div>
            <div className="text-xs text-slate-600 leading-relaxed whitespace-pre-line">
              {workup}
            </div>
          </div>
        )}

        {/* Frontline Management */}
        {management && (
          <div className="p-5 bg-white rounded-2xl border border-slate-200/80 shadow-2xs space-y-2">
            <div className="flex items-center gap-2 text-blue-600">
              <Pill className="h-4 w-4" />
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-800">Frontline Management</h4>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              {management}
            </p>
          </div>
        )}

      </div>

      {/* Risk Note & Contraindications */}
      {riskNote && (
        <div className={`p-4 rounded-2xl border flex items-start gap-3 ${
          isTier1 
            ? 'bg-red-50/70 border-red-200 text-red-900' 
            : 'bg-amber-50/70 border-amber-200 text-amber-900'
        }`}>
          <ShieldAlert className={`h-5 w-5 flex-shrink-0 mt-0.5 ${
            isTier1 ? 'text-red-600' : 'text-amber-600'
          }`} />
          <div className="space-y-1">
            <h5 className="text-xs font-bold uppercase tracking-wider">
              Critical Red Flag / Contraindication Note
            </h5>
            <p className="text-xs leading-relaxed font-medium">
              {riskNote}
            </p>
          </div>
        </div>
      )}

      {/* Fallback for unparsed or legacy report text */}
      {!impression && (
        <p className="text-sm text-slate-700 whitespace-pre-line leading-relaxed">
          {report}
        </p>
      )}

    </div>
  );
};

export default ClinicalSummary;
