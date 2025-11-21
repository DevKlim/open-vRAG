import React, { useState, useRef, useEffect } from 'react';
import { 
  AlertCircle, Play, Upload, Layers, Terminal, Settings, 
  Cpu, Activity, FileText, Zap, MessageSquare, Sliders, 
  LayoutDashboard, FileJson, ChevronDown, ChevronRight, Bot
} from 'lucide-react';

function App() {
  // --- State ---
  const [activeTab, setActiveTab] = useState('qa');
  const [logs, setLogs] = useState<string>('System Ready.\n');
  const [isProcessing, setIsProcessing] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  
  // Form State
  const [videoUrl, setVideoUrl] = useState('https://www.youtube.com/watch?v=XHTskNem4co');
  const [model, setModel] = useState('vertex'); // Default to Vertex for modern feel
  
  // Model Parameters
  const [temperature, setTemperature] = useState(0.4);
  const [topP, setTopP] = useState(0.8);
  const [maxTokens, setMaxTokens] = useState(2048);
  const [samplingFps, setSamplingFps] = useState(0.2);
  const [perceptions, setPerceptions] = useState(3);
  const [repPenalty, setRepPenalty] = useState(1.1);

  // Refs
  const logEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const appendLog = (text: string) => {
    setLogs(prev => prev + text);
  };

  // --- Handlers ---
  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsProcessing(true);
    setLogs(`[SYSTEM] Initializing ${activeTab.toUpperCase()} pipeline...\n[SYSTEM] Selected Model: ${model}\n`);

    const formData = new FormData(e.currentTarget);
    
    // Explicitly append slider values to ensure they are sent for ALL models
    formData.set('temperature', temperature.toString());
    formData.set('top_p', topP.toString());
    formData.set('max_new_tokens', maxTokens.toString());
    formData.set('sampling_fps', samplingFps.toString());
    formData.set('num_perceptions', perceptions.toString());
    formData.set('repetition_penalty', repPenalty.toString());

    let endpoint = '/process';
    if (activeTab === 'labeling') endpoint = '/label_video';
    if (activeTab === 'batch') endpoint = '/batch_label';

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      if (!response.body) throw new Error("No response body");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        
        const lines = chunk.split('\n\n');
        for (const line of lines) {
          if (line.startsWith('data:')) {
             const msg = line.replace('data:', '').trim();
             if(msg) appendLog(msg + '\n');
          }
          if (line.startsWith('event: close')) {
             setIsProcessing(false);
          }
        }
      }
    } catch (err: any) {
      appendLog(`\n[ERROR]: ${err.message}\n`);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleFileBoxClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
        appendLog(`[USER] Selected file: ${e.target.files[0].name}\n`);
    }
  };

  // Helper component for Sliders
  const SliderControl = ({ label, name, value, setValue, min, max, step }: any) => (
    <div className="space-y-2">
        <div className="flex justify-between items-center">
            <span className="text-xs font-medium text-slate-400">{label}</span>
            <span className="text-[10px] font-mono bg-slate-800 px-2 py-0.5 rounded text-vchat-accent">{value}</span>
        </div>
        <input 
            type="range" 
            name={name}
            min={min} max={max} step={step} 
            value={value} 
            onChange={(e) => setValue(parseFloat(e.target.value))}
            className="w-full h-1.5 bg-slate-700 rounded-full appearance-none cursor-pointer accent-vchat-accent hover:accent-indigo-400 transition-all"
        />
    </div>
  );

  return (
    <div className="flex h-screen w-full bg-[#09090b] text-slate-200 font-sans overflow-hidden selection:bg-vchat-accent selection:text-white">
      
      {/* LEFT PANEL: CONFIGURATION */}
      <div className="w-[400px] flex flex-col border-r border-slate-800/60 bg-[#0c0c0e]">
        {/* Header */}
        <div className="h-16 flex items-center px-6 border-b border-slate-800/60 bg-[#0c0c0e]/50 backdrop-blur-sm">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-600 to-violet-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-sm font-bold text-white tracking-tight">AI Studio <span className="font-normal text-slate-500">| vChat</span></h1>
            </div>
          </div>
        </div>

        {/* Scrollable Form */}
        <div className="flex-1 overflow-y-auto custom-scrollbar p-6 space-y-8">
          <form id="control-form" onSubmit={handleSubmit} className="space-y-8">
            
            {/* INPUT SOURCE */}
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-xs font-bold text-slate-500 uppercase tracking-wider">
                <LayoutDashboard className="w-3 h-3" /> Input Source
              </div>
              <div className="group relative">
                <input 
                  type="text" 
                  name="video_url" 
                  value={videoUrl}
                  onChange={(e) => setVideoUrl(e.target.value)}
                  placeholder="https://youtube.com/watch?v=..."
                  className="w-full bg-slate-900/50 border border-slate-800 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-vchat-accent/50 focus:border-vchat-accent/50 outline-none transition-all placeholder:text-slate-700 text-slate-300 group-hover:border-slate-700"
                />
                <div className="absolute inset-0 rounded-lg ring-1 ring-inset ring-white/5 pointer-events-none"></div>
              </div>
            </div>

            {/* MODEL SELECTOR */}
            <div className="space-y-4">
               <div className="flex items-center gap-2 text-xs font-bold text-slate-500 uppercase tracking-wider">
                <Cpu className="w-3 h-3" /> Model Engine
              </div>
              
              <div className="relative">
                <select 
                  name="model_selection" 
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  className="w-full appearance-none bg-slate-900/50 border border-slate-800 rounded-lg px-4 py-3 text-sm outline-none focus:border-vchat-accent/50 transition-colors text-slate-300 cursor-pointer hover:border-slate-700"
                >
                  <option value="vertex">Google Vertex AI (Recommended)</option>
                  <option value="gemini">Google AI Studio (Gemini)</option>
                  <option value="default">Local Model (VideoChat-R1.5)</option>
                  <option value="custom">Custom Fine-Tuned LORA</option>
                </select>
                <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 pointer-events-none" />
              </div>

              {/* Dynamic Auth Fields */}
              {model === 'gemini' && (
                <div className="p-4 bg-indigo-500/5 border border-indigo-500/10 rounded-lg space-y-3 animate-in fade-in slide-in-from-top-1">
                  <div className="flex items-center gap-2 text-indigo-400 text-xs font-medium mb-1">
                    <Zap className="w-3 h-3" /> API Configuration
                  </div>
                  <input type="password" name="gemini_api_key" placeholder="Gemini API Key" className="w-full bg-black/20 border border-indigo-500/20 rounded px-3 py-2 text-xs focus:border-indigo-500/50 outline-none" />
                  <input type="text" name="gemini_model_name" defaultValue="models/gemini-1.5-pro-latest" placeholder="Model Name" className="w-full bg-black/20 border border-indigo-500/20 rounded px-3 py-2 text-xs focus:border-indigo-500/50 outline-none" />
                </div>
              )}

              {model === 'vertex' && (
                 <div className="p-4 bg-emerald-500/5 border border-emerald-500/10 rounded-lg space-y-3 animate-in fade-in slide-in-from-top-1">
                  <div className="flex items-center gap-2 text-emerald-400 text-xs font-medium mb-1">
                     <Activity className="w-3 h-3" /> Cloud Configuration
                  </div>
                  <input type="text" name="vertex_project_id" placeholder="GCP Project ID" className="w-full bg-black/20 border border-emerald-500/20 rounded px-3 py-2 text-xs focus:border-emerald-500/50 outline-none" />
                  <div className="grid grid-cols-2 gap-2">
                     <input type="text" name="vertex_location" defaultValue="us-central1" placeholder="Region" className="w-full bg-black/20 border border-emerald-500/20 rounded px-3 py-2 text-xs focus:border-emerald-500/50 outline-none" />
                     <input type="text" name="vertex_model_name" defaultValue="gemini-1.5-pro-preview-0409" placeholder="Model" className="w-full bg-black/20 border border-emerald-500/20 rounded px-3 py-2 text-xs focus:border-emerald-500/50 outline-none" />
                  </div>
                  <input type="password" name="vertex_api_key" placeholder="API Key (Optional)" className="w-full bg-black/20 border border-emerald-500/20 rounded px-3 py-2 text-xs focus:border-emerald-500/50 outline-none" />
                </div>
              )}
            </div>

            {/* MODEL TUNING (Available for ALL models now) */}
            <div className="space-y-4">
               <button 
                 type="button" 
                 onClick={() => setShowAdvanced(!showAdvanced)}
                 className="flex items-center gap-2 text-xs font-bold text-slate-500 uppercase tracking-wider hover:text-slate-300 transition-colors"
               >
                  <Sliders className="w-3 h-3" /> Model Parameters
                  <ChevronRight className={`w-3 h-3 transition-transform ${showAdvanced ? 'rotate-90' : ''}`} />
               </button>
               
               {showAdvanced && (
                   <div className="p-4 bg-slate-900/30 border border-slate-800 rounded-lg space-y-5 animate-in fade-in slide-in-from-top-2">
                      <SliderControl label="Temperature" name="temperature" value={temperature} setValue={setTemperature} min={0} max={2} step={0.1} />
                      <SliderControl label="Top P" name="top_p" value={topP} setValue={setTopP} min={0} max={1} step={0.05} />
                      <SliderControl label="Max Output Tokens" name="max_new_tokens" value={maxTokens} setValue={setMaxTokens} min={256} max={8192} step={64} />
                      
                      {(model === 'default' || model === 'custom') && (
                        <>
                          <div className="h-px bg-slate-800 my-2" />
                          <SliderControl label="Sampling FPS" name="sampling_fps" value={samplingFps} setValue={setSamplingFps} min={0.1} max={5} step={0.1} />
                          <SliderControl label="Reasoning Steps" name="num_perceptions" value={perceptions} setValue={setPerceptions} min={1} max={5} step={1} />
                        </>
                      )}
                   </div>
               )}
            </div>

            {/* TASK MODE TABS */}
            <div className="space-y-4">
               <div className="flex items-center gap-2 text-xs font-bold text-slate-500 uppercase tracking-wider">
                <FileText className="w-3 h-3" /> Task Type
              </div>
              
              <div className="grid grid-cols-3 gap-1 p-1 bg-slate-900 rounded-lg border border-slate-800">
                 {[
                   { id: 'qa', label: 'Q&A Check', icon: MessageSquare },
                   { id: 'labeling', label: 'Auto-Label', icon: FileJson },
                   { id: 'batch', label: 'Batch Job', icon: Layers }
                 ].map(tab => (
                    <button
                      key={tab.id}
                      type="button"
                      onClick={() => setActiveTab(tab.id)}
                      className={`flex items-center justify-center gap-2 py-2.5 text-xs font-medium rounded-md transition-all duration-200 ${
                        activeTab === tab.id 
                        ? 'bg-slate-800 text-white shadow-sm ring-1 ring-white/5' 
                        : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/50'
                      }`}
                    >
                      <tab.icon className="w-3.5 h-3.5" />
                      {tab.label}
                    </button>
                 ))}
              </div>

              {/* TASK CONTENT */}
              <div className="pt-2">
                {activeTab === 'qa' && (
                  <div className="space-y-4 animate-in fade-in slide-in-from-left-1">
                    <div className="space-y-2">
                        <label className="text-xs font-medium text-slate-400">Prompt / Question</label>
                        <textarea 
                            name="question" 
                            rows={3} 
                            defaultValue="Analyze the video for any factual inconsistencies or manipulated content." 
                            className="w-full bg-slate-900/50 border border-slate-800 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-vchat-accent/50 outline-none text-slate-300 placeholder:text-slate-600 resize-none" 
                        />
                    </div>
                    
                    <div className="p-4 bg-slate-900/30 border border-slate-800 rounded-lg">
                        <span className="text-xs font-bold text-slate-500 uppercase tracking-wider block mb-3">Factuality Modules</span>
                        <div className="space-y-2">
                            <label className="flex items-center gap-3 p-2 rounded hover:bg-white/5 cursor-pointer transition-colors group">
                                <input type="checkbox" name="check_visuals" value="true" className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-vchat-accent focus:ring-offset-0 focus:ring-1 focus:ring-vchat-accent/50" /> 
                                <span className="text-sm text-slate-300 group-hover:text-white">Visual Artifact Detection</span>
                            </label>
                            <label className="flex items-center gap-3 p-2 rounded hover:bg-white/5 cursor-pointer transition-colors group">
                                <input type="checkbox" name="check_content" value="true" className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-vchat-accent focus:ring-offset-0 focus:ring-1 focus:ring-vchat-accent/50" /> 
                                <span className="text-sm text-slate-300 group-hover:text-white">Claim Verification & Logic</span>
                            </label>
                            <label className="flex items-center gap-3 p-2 rounded hover:bg-white/5 cursor-pointer transition-colors group">
                                <input type="checkbox" name="check_audio" value="true" className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-vchat-accent focus:ring-offset-0 focus:ring-1 focus:ring-vchat-accent/50" /> 
                                <span className="text-sm text-slate-300 group-hover:text-white">Audio Forensics</span>
                            </label>
                        </div>
                    </div>
                  </div>
                )}

                {activeTab === 'labeling' && (
                   <div className="p-4 bg-amber-500/5 border border-amber-500/10 rounded-lg space-y-4 animate-in fade-in slide-in-from-left-1">
                      <div className="flex gap-3">
                         <div className="mt-0.5"><AlertCircle className="w-4 h-4 text-amber-500" /></div>
                         <div className="text-xs text-amber-200/80 leading-relaxed">
                            Generates "Ali Arsanjani Factuality Factors" and "Veracity Vectors". 
                            Results are saved to JSON and appended to the dataset CSV.
                         </div>
                      </div>
                      <label className="flex items-center gap-3 p-3 bg-black/20 rounded border border-amber-500/20 cursor-pointer hover:border-amber-500/40 transition-all">
                          <input type="checkbox" name="include_comments" defaultChecked className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-amber-500 focus:ring-offset-0" /> 
                          <span className="text-sm font-medium text-amber-100">Include Chain-of-Thought Reasoning</span>
                      </label>
                   </div>
                )}

                {activeTab === 'batch' && (
                   <div className="space-y-3 animate-in fade-in slide-in-from-left-1">
                      {/* Compact Upload Box */}
                      <div 
                        onClick={handleFileBoxClick}
                        className="relative border-2 border-dashed border-slate-700 rounded-xl p-4 text-center hover:border-vchat-accent hover:bg-vchat-accent/5 transition-all cursor-pointer group"
                      >
                         <div className="flex flex-col items-center justify-center gap-2">
                            <div className="w-10 h-10 rounded-full bg-slate-800 flex items-center justify-center group-hover:scale-105 transition-transform">
                                <Upload className="w-4 h-4 text-slate-400 group-hover:text-vchat-accent transition-colors" />
                            </div>
                            <div>
                                <p className="text-sm font-medium text-slate-300 group-hover:text-white">Click to Upload CSV</p>
                                <p className="text-[10px] text-slate-500">Must contain "link" column</p>
                            </div>
                        </div>
                        <input 
                            type="file" 
                            name="csv_file" 
                            accept=".csv" 
                            ref={fileInputRef} 
                            onChange={handleFileChange}
                            hidden 
                        />
                      </div>
                   </div>
                )}
              </div>
            </div>

            {/* HIDDEN INPUTS FOR LOCAL MODEL PROMPTS */}
            <input type="hidden" name="prompt_glue" value={`Answer the question: "[QUESTION]" according to the content of the video.\nOutput your think process within the <think> </think> tags.\nThen, provide your answer within the <answer> </answer> tags. At the same time, in the <glue> </glue> tags, present the precise time period in seconds of the video clips on which you base your answer in the format of [(s1, e1), (s2, e2), ...].`} />
            <input type="hidden" name="prompt_final" value={`Answer the question: "[QUESTION]" according to the content of the video.\nOutput your think process within the <think> </think> tags.\nThen, provide your answer within the <answer> </answer> tags.`} />

          </form>
        </div>

        {/* FOOTER */}
        <div className="p-6 border-t border-slate-800/60 bg-[#0c0c0e]">
          <button 
            type="submit" 
            form="control-form"
            disabled={isProcessing}
            className={`w-full py-3.5 px-4 rounded-lg font-bold text-sm shadow-lg shadow-indigo-500/10 transition-all flex items-center justify-center gap-2
              ${isProcessing 
                ? 'bg-slate-800 text-slate-400 cursor-wait border border-slate-700' 
                : 'bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white hover:shadow-indigo-500/25 active:scale-[0.98]'}`}
          >
            {isProcessing ? (
              <><div className="w-4 h-4 border-2 border-slate-400 border-t-transparent rounded-full animate-spin"/> Processing Task...</>
            ) : (
              <><Play className="w-4 h-4 fill-current" /> Run {activeTab === 'batch' ? 'Batch Job' : 'Analysis'}</>
            )}
          </button>
        </div>
      </div>

      {/* RIGHT PANEL: OUTPUT */}
      <div className="flex-1 flex flex-col relative bg-[#09090b]">
        {/* Background Gradients */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
             <div className="absolute top-[-10%] right-[-5%] w-[800px] h-[600px] rounded-full bg-indigo-900/10 blur-[120px]" />
             <div className="absolute bottom-[-10%] left-[-5%] w-[600px] h-[600px] rounded-full bg-blue-900/5 blur-[100px]" />
        </div>

        {/* Header */}
        <div className="h-16 border-b border-slate-800/60 bg-[#09090b]/80 backdrop-blur flex justify-between items-center px-8 z-10">
            <div className="flex items-center gap-3 text-slate-400">
                <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.4)] animate-pulse"></div>
                <span className="text-xs font-mono font-medium tracking-wide">TERMINAL OUTPUT</span>
            </div>
            <div className="flex gap-2">
                 <button onClick={() => setLogs('')} className="text-xs font-medium text-slate-500 hover:text-slate-200 px-3 py-1.5 rounded hover:bg-white/5 transition-colors">Clear Log</button>
                 <button className="text-xs font-medium text-indigo-400 hover:text-indigo-300 px-3 py-1.5 rounded border border-indigo-500/20 bg-indigo-500/10 hover:bg-indigo-500/20 transition-colors">Export</button>
            </div>
        </div>

        {/* Terminal Area */}
        <div className="flex-1 p-8 overflow-hidden relative z-10 flex flex-col">
            <div className="flex-1 bg-black/40 rounded-xl border border-slate-800/60 shadow-inner overflow-hidden flex flex-col backdrop-blur-sm relative group">
                
                {/* Log Content */}
                <div className="flex-1 p-6 overflow-y-auto font-mono text-[13px] leading-relaxed scroll-smooth custom-scrollbar" id="terminal-scroll">
                     {logs.split('\n').map((line, i) => {
                        if(!line) return <div key={i} className="min-h-[0.5rem]" />;
                        if(line.includes('[ERROR]')) return <div key={i} className="text-red-400 font-bold bg-red-500/10 p-2 rounded my-1 border-l-2 border-red-500">{line}</div>;
                        if(line.includes('[SYSTEM]')) return <div key={i} className="text-indigo-300 font-medium mt-2 mb-1">{line}</div>;
                        if(line.includes('Step')) return <div key={i} className="text-emerald-400 font-bold mt-4 pb-1 border-b border-emerald-500/20 inline-block">{line}</div>;
                        if(line.includes('TOON')) return <div key={i} className="text-amber-200/80 pl-4 border-l border-amber-500/20 my-1 italic">{line}</div>;
                        if(line.includes('<thinking>')) return <div key={i} className="text-slate-500 pl-4 my-1">{line}</div>;
                        
                        return <div key={i} className="text-slate-300/90 hover:bg-white/5 px-1 rounded transition-colors">{line}</div>;
                     })}
                     <div ref={logEndRef} />
                </div>
                
                {/* Footer Status */}
                <div className="h-8 bg-[#0c0c0e] border-t border-slate-800/60 px-4 flex justify-between items-center text-[10px] text-slate-500 font-mono uppercase tracking-wider">
                    <div className="flex items-center gap-2">
                        <Terminal className="w-3 h-3" />
                        <span>Ready</span>
                    </div>
                    <span>v2.4.0-RC1</span>
                </div>
            </div>
        </div>

      </div>
    </div>
  )
}

export default App