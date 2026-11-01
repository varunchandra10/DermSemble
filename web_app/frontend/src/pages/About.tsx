import React from 'react';
import { ShieldCheck } from 'lucide-react';

const About: React.FC = () => {
  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
        <div className="flex justify-center mb-6">
          <div className="h-16 w-16 bg-blue-50 rounded-full flex items-center justify-center">
            <ShieldCheck className="h-8 w-8 text-blue-600" />
          </div>
        </div>
        
        <h1 className="text-3xl font-bold text-center text-gray-900 mb-8">About Dermsemble</h1>
        
        <div className="prose prose-blue max-w-none text-gray-600 space-y-6">
          <p>
            Dermsemble is an edge-deployable, demographic-fair clinical decision support system for preliminary dermatological triage. 
            It leverages a complementary dual deep learning ensemble (ConvNeXt-Tiny and Swin-Transformer Tiny) to 
            predict across 25 distinct skin condition categories with oncology acuity tuning and selective uncertainty abstention.
          </p>
          
          <h3 className="text-xl font-semibold text-gray-900 mt-8 mb-4">Methodology</h3>
          <p>
            Our models were trained on a diverse, curated dataset of dermatological images. We employ rigorous 
            preprocessing standards (aspect-preserving padding, ImageNet normalization) to ensure the data fed to the 
            models in production perfectly matches the training environment.
          </p>

          <h3 className="text-xl font-semibold text-gray-900 mt-8 mb-4 text-red-700">Important Limitations & Medical Disclaimer</h3>
          <div className="bg-red-50 border-l-4 border-red-500 p-6 rounded-r-lg text-red-900 text-sm">
            <p className="font-bold mb-2">This application is NOT a medical device.</p>
            <ul className="list-disc pl-5 space-y-2">
              <li>The predictions provided by this software are for informational and research purposes only.</li>
              <li>AI models can and do make mistakes. Certain image angles, lighting conditions, and skin tones may reduce accuracy.</li>
              <li>Never ignore professional medical advice or delay seeking treatment because of something you have read or seen on this application.</li>
              <li>If you suspect you have a skin condition, particularly a potentially malignant one like melanoma, schedule an appointment with a board-certified dermatologist immediately.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;
