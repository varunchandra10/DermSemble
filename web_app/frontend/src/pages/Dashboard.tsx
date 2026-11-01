import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts';
import { CheckCircle2, TrendingUp, ShieldCheck, Scale } from 'lucide-react';

const Dashboard: React.FC = () => {
  // Real Empirical Data from Phase 9 Demographic Fairness Audit (N=4,599 Holdout Images)
  const fairnessData = [
    { skinType: 'Type I', top1Accuracy: 79.9, top3Recall: 91.0 },
    { skinType: 'Type II', top1Accuracy: 78.1, top3Recall: 90.8 },
    { skinType: 'Type III', top1Accuracy: 76.3, top3Recall: 88.2 },
    { skinType: 'Type IV', top1Accuracy: 74.7, top3Recall: 88.3 },
    { skinType: 'Type V', top1Accuracy: 73.2, top3Recall: 89.4 },
    { skinType: 'Type VI', top1Accuracy: 71.4, top3Recall: 87.6 },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Clinical Validation & Demographic Dashboard</h1>
        <p className="text-gray-500 mt-2">Empirical validation metrics across 4,599 isolated clinical holdout test images.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 flex items-center gap-4">
          <div className="bg-green-100 p-3 rounded-xl"><CheckCircle2 className="h-6 w-6 text-green-600" /></div>
          <div>
            <p className="text-xs text-gray-500 font-medium">Champion Ensemble Top-1</p>
            <h3 className="text-2xl font-bold text-gray-900">75.52%</h3>
            <span className="text-[11px] text-green-600 font-semibold">+2.05% vs baseline</span>
          </div>
        </div>

        <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 flex items-center gap-4">
          <div className="bg-blue-100 p-3 rounded-xl"><TrendingUp className="h-6 w-6 text-blue-600" /></div>
          <div>
            <p className="text-xs text-gray-500 font-medium">Top-3 Differential Recall</p>
            <h3 className="text-2xl font-bold text-gray-900">89.28%</h3>
            <span className="text-[11px] text-blue-600 font-semibold">Diagnostic safety net</span>
          </div>
        </div>

        <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 flex items-center gap-4">
          <div className="bg-indigo-100 p-3 rounded-xl"><ShieldCheck className="h-6 w-6 text-indigo-600" /></div>
          <div>
            <p className="text-xs text-gray-500 font-medium">Melanoma Sensitivity</p>
            <h3 className="text-2xl font-bold text-gray-900">86.17%</h3>
            <span className="text-[11px] text-indigo-600 font-semibold">Tuned (α = 1.85)</span>
          </div>
        </div>

        <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 flex items-center gap-4">
          <div className="bg-amber-100 p-3 rounded-xl"><Scale className="h-6 w-6 text-amber-600" /></div>
          <div>
            <p className="text-xs text-gray-500 font-medium">Disparate Impact Ratio</p>
            <h3 className="text-2xl font-bold text-gray-900">89.39%</h3>
            <span className="text-[11px] text-green-600 font-semibold">EEOC 80% Rule Passed</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Fairness Audit Chart */}
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
          <div className="flex justify-between items-center mb-6">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">Demographic Fairness (Fitzpatrick I–VI)</h3>
              <p className="text-xs text-gray-500">Disaggregated Top-1 Accuracy & Top-3 Recall across phototypes</p>
            </div>
            <span className="px-2.5 py-1 bg-green-50 text-green-700 text-xs font-semibold rounded-full border border-green-200">
              DIR: 89.4% (Pass)
            </span>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={fairnessData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                <XAxis dataKey="skinType" tick={{ fontSize: 12, fill: '#6b7280' }} axisLine={false} tickLine={false} />
                <YAxis domain={[60, 100]} tick={{ fontSize: 12, fill: '#6b7280' }} axisLine={false} tickLine={false} />
                <Tooltip cursor={{ fill: '#f9fafb' }} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '12px', paddingTop: '15px' }} />
                <Bar dataKey="top1Accuracy" name="Top-1 Accuracy (%)" fill="#2563eb" radius={[4, 4, 0, 0]} barSize={16} />
                <Bar dataKey="top3Recall" name="Top-3 Differential Recall (%)" fill="#16a34a" radius={[4, 4, 0, 0]} barSize={16} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Clinical Abstention & Selective Risk */}
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 flex flex-col justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-1">Selective Clinical Abstention Policy</h3>
            <p className="text-xs text-gray-500 mb-6">Uncertainty thresholding (θ = 0.50) trade-off performance</p>

            <div className="space-y-4">
              <div className="p-4 bg-blue-50/60 rounded-xl border border-blue-100">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm font-semibold text-blue-900">Automated Triage Coverage</span>
                  <span className="text-sm font-bold text-blue-700">79.6% (3,659 Cases)</span>
                </div>
                <p className="text-xs text-blue-700/80">Diagnostic Accuracy on automated subset rises to <strong>85.11% (+9.59% boost)</strong>.</p>
              </div>

              <div className="p-4 bg-amber-50/60 rounded-xl border border-amber-100">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm font-semibold text-amber-900">Referred for Specialist Biopsy</span>
                  <span className="text-sm font-bold text-amber-700">20.4% (940 Ambiguous Cases)</span>
                </div>
                <p className="text-xs text-amber-700/80">Prevents dangerous automated misdiagnoses by abstaining when confidence is low.</p>
              </div>

              <div className="p-4 bg-emerald-50/60 rounded-xl border border-emerald-100">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm font-semibold text-emerald-900">Melanoma Equal Opportunity Diff</span>
                  <span className="text-sm font-bold text-emerald-700">2.06% Delta</span>
                </div>
                <p className="text-xs text-emerald-700/80">Light tones: 82.50% vs Dark tones: 80.44% cancer detection sensitivity.</p>
              </div>
            </div>
          </div>

          <div className="mt-6 pt-4 border-t border-gray-100 text-center">
            <span className="text-xs text-gray-400">Validated against regulatory HIPAA, EEOC and FDA SaMD safety benchmarks.</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
