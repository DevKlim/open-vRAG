import React, { useState, useRef, useEffect } from 'react';
import { 
  AlertCircle, Play, Upload, Layers, Terminal, Cpu, Activity, 
  FileText, Zap, MessageSquare, Sliders, LayoutDashboard, FileJson, 
  ChevronDown, ChevronRight, Bot, Database, Trash2, Eye, StopCircle, List,
  CheckCircle, XCircle, BrainCircuit, Edit3, ClipboardList, CheckSquare,
  BarChart2, TrendingUp, TrendingDown, Scale, ExternalLink
} from 'lucide-react';

function App() {
  const [activeTab, setActiveTab] = useState('queue');
  const [logs, setLogs] = useState<string>('System Ready.\n');
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Data States
  const [dataList, setDataList] = useState<any[]>([]);
  const [queueList, setQueueList] = useState<any[]>([]);
  const [workflowList, setWorkflowList] = useState<any[]>([]);
  const [comparisonList, setComparisonList] = useState<any[]>([]);
  
  const [expandedRow, setExpandedRow] = useState<string | null>(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  // Manual Labeling Modal
  const [labelingItem, setLabelingItem] = useState<any>(null);
  const [manualForm, setManualForm] = useState({
      visual: 5, audio: 5, source: 5, logic: 5, emotion: 5,
      va: 5, vc: 5, ac: 5,
      final: 50,
      reasoning: '',
      tags: ''
  });

  const [videoUrl, setVideoUrl] = useState('');
  const [model, setModel] = useState('vertex'); 
  const [reasoningMethod, setReasoningMethod] = useState('cot');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const logEndRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  // Refresh Data logic
  useEffect(() => {
    if (activeTab === 'moderation') {
      fetch('/manage/list').then(res => res.json()).then(setDataList).catch(err => console.error("Data Load Error:", err));
    }
    if (activeTab === 'queue') {
      fetch('/queue/list').then(res => res.json()).then(setQueueList).catch(err => console.error("Queue Load Error:", err));
    }
    if (activeTab === 'workflow') {
      fetch('/workflow/status').then(res => res.json()).then(setWorkflowList).catch(err => console.error("Workflow Load Error:", err));
    }
    if (activeTab === 'analytics') {
      fetch('/manage/comparison_data').then(res => res.json()).then(setComparisonList).catch(err => console.error("Analytics Load Error:", err));
    }
  }, [activeTab, refreshTrigger]);

  const appendLog = (text: string) => setLogs(prev => prev + text);

  // --- Handlers ---
  
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.length) return;
    const file = e.target.files[0];
    const fd = new FormData();
    fd.append("file", file);
    
    appendLog(`[SYSTEM] Uploading ${file.name} to queue...\n`);
    try {
      const res = await fetch('/queue/upload_csv', { method: 'POST', body: fd });
      const data = await res.json();
      if(data.error) throw new Error(data.error);
      appendLog(`[SYSTEM] Upload complete. Added ${data.added} links.\n`);
      setRefreshTrigger(prev => prev + 1);
    } catch (err: any) {
      appendLog(`[ERROR] Upload failed: ${err.message}\n`);
    }
  };

  const handleStartQueue = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsProcessing(true);
    setLogs("[SYSTEM] Starting Queue Processing...\n");

    const form = document.getElementById('control-form') as HTMLFormElement;
    const formData = new FormData(form);

    try {
      const response = await fetch('/queue/run', { method: 'POST', body: formData });
      if (!response.body) throw new Error("No response");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        chunk.split('\n\n').forEach(line => {
          if (line.startsWith('data:')) appendLog(line.replace('data:', '').trim() + '\n');
          if (line.startsWith('event: close')) setIsProcessing(false);
        });
      }
    } catch (err: any) {
      appendLog(`\n[ERROR]: ${err.message}\n`);
    } finally {
      setIsProcessing(false);
      setRefreshTrigger(prev => prev + 1);
    }
  };

  const handleStopQueue = async () => {
      await fetch('/queue/stop', { method: 'POST' });
      appendLog("[USER] Stop signal sent. Finishing current item...\n");
  };

  const handleDelete = async (id: string, link: string) => {
    if (!confirm("Delete this entry? This allows re-labeling of the link in the queue.")) return;
    try {
      const res = await fetch(`/manage/delete?id=${id}&link=${encodeURIComponent(link)}`, { method: 'DELETE' });
      const json = await res.json();
      if (json.status === 'deleted') {
        setRefreshTrigger(prev => prev + 1);
      } else {
        alert("Error deleting: " + JSON.stringify(json));
      }
    } catch (e) { alert("Fail: " + e); }
  };

  const handleQueueDelete = async (link: string) => {
    if(!confirm("Remove this link from queue?")) return;
    try {
        const res = await fetch(`/queue/delete?link=${encodeURIComponent(link)}`, { method: 'DELETE' });
        const json = await res.json();
        if(json.status === 'success') {
            setRefreshTrigger(prev => prev + 1);
        } else {
            alert("Error: " + json.message);
        }
    } catch(err) {
        console.error(err);
    }
  }

  // --- Manual Labeling Handlers ---

  const parseScore = (val: any) => {
      if (!val) return 5;
      const str = String(val).replace(/[^\d]/g, '');
      const num = parseInt(str);
      return isNaN(num) ? 5 : num;
  };

  const openLabelingModal = (item: any) => {
      // Logic to handle source: workflow (item.ai_data) vs moderation (item direct)
      let ai = item.ai_data || {};
      
      // Fallback: If coming from Moderation tab, item itself is the AI data
      if (!item.ai_data && item.source_type === 'auto') {
          ai = {
              visual: item.visual_integrity_score,
              final: item.final_veracity_score,
              reasoning: item.final_reasoning,
              tags: item.tags
          };
      }
      
      setLabelingItem(item);
      setManualForm({
          visual: parseScore(ai.visual),
          audio: 5, source: 5, logic: 5, emotion: 5,
          va: 5, vc: 5, ac: 5,
          final: parseScore(ai.final || 50),
          reasoning: ai.reasoning || '',
          tags: ai.tags || ''
      });
  };

  const submitManualLabel = async () => {
      if(!labelingItem) return;
      
      const payload = {
          link: labelingItem.link,
          caption: "Manual Label via WebUI",
          labels: {
            visual_integrity_score: manualForm.visual,
            audio_integrity_score: manualForm.audio,
            source_credibility_score: manualForm.source,
            logical_consistency_score: manualForm.logic,
            emotional_manipulation_score: manualForm.emotion,
            video_audio_score: manualForm.va,
            video_caption_score: manualForm.vc,
            audio_caption_score: manualForm.ac,
            final_veracity_score: manualForm.final,
            reasoning: manualForm.reasoning
          },
          tags: manualForm.tags,
          stats: { platform: "webui" }
      };

      try {
          const res = await fetch('/extension/save_manual', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify(payload)
          });
          const json = await res.json();
          if(json.status === 'saved') {
              alert("Manual Label Saved!");
              setLabelingItem(null);
              setRefreshTrigger(prev => prev + 1);
          } else {
              alert("Error: " + JSON.stringify(json));
          }
      } catch(e: any) {
          alert("Error: " + e.message);
      }
  };

  // Helper to segment workflow
  const pendingVerification = workflowList.filter(row => row.ai_status === 'Labeled' && row.manual_status !== 'Completed');
  const pendingAI = workflowList.filter(row => row.ai_status !== 'Labeled' && row.manual_status !== 'Completed');
  const completed = workflowList.filter(row => row.manual_status === 'Completed');

  // Helper for Analytics Summary
  const calculateStats = () => {
      if(comparisonList.length === 0) return { avgDelta: 0, bias: 0 };
      let totalDelta = 0;
      let totalAbsDelta = 0;
      comparisonList.forEach(c => {
          totalDelta += c.deltas.final; // Raw delta (+ means AI > Human)
          totalAbsDelta += Math.abs(c.deltas.final);
      });
      return {
          avgMAE: (totalAbsDelta / comparisonList.length).toFixed(1),
          bias: (totalDelta / comparisonList.length).toFixed(1)
      };
  };
  const stats = calculateStats();

  return (
    <div className="flex h-screen w-full bg-[#09090b] text-slate-200 font-sans overflow-hidden">
      
      {/* LEFT PANEL */}
      <div className="w-[380px] flex flex-col border-r border-slate-800/60 bg-[#0c0c0e]">
        <div className="h-16 flex items-center px-6 border-b border-slate-800/60">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-sm font-bold text-white">vChat <span className="text-slate-500">Manager</span></h1>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-6 custom-scrollbar">
          <form id="control-form" className="space-y-6">
            <div className="grid grid-cols-2 gap-1 p-1 bg-slate-900 rounded-lg border border-slate-800">
               {[{id:'queue',l:'Ingest Queue',i:List}, {id:'workflow',l:'Labeling Workflow',i:ClipboardList}, {id:'moderation',l:'Dataset',i:Database}, {id:'analytics',l:'Showcase',i:BarChart2}].map(t => (
                  <button key={t.id} type="button" onClick={() => setActiveTab(t.id)}
                    className={`flex items-center justify-center gap-2 py-2 text-xs font-medium rounded ${activeTab===t.id ? 'bg-slate-800 text-white' : 'text-slate-500'}`}>
                    <t.i className="w-3 h-3" /> {t.l}
                  </button>
               ))}
            </div>

            <div className="space-y-3">
              <label className="text-xs font-bold text-slate-500 uppercase">Model Engine</label>
              <select name="model_selection" value={model} onChange={(e) => setModel(e.target.value)}
                className="w-full bg-slate-900 border border-slate-800 rounded px-3 py-2 text-xs outline-none focus:border-indigo-500">
                <option value="vertex">Google Vertex AI</option>
                <option value="gemini">Google Gemini API</option>
              </select>

              {model === 'gemini' && (
                <div className="space-y-2">
                  <input type="password" name="gemini_api_key" placeholder="Gemini API Key" className="w-full bg-black/20 border border-slate-800 rounded px-3 py-2 text-xs" />
                  <input type="text" name="gemini_model_name" defaultValue="models/gemini-2.0-flash-exp" className="w-full bg-black/20 border border-slate-800 rounded px-3 py-2 text-xs" />
                </div>
              )}
              {model === 'vertex' && (
                 <div className="space-y-2">
                   <input type="text" name="vertex_project_id" placeholder="GCP Project ID" className="w-full bg-black/20 border border-slate-800 rounded px-3 py-2 text-xs" />
                   <div className="grid grid-cols-2 gap-2">
                      <input type="text" name="vertex_location" defaultValue="us-central1" className="w-full bg-black/20 border border-slate-800 rounded px-3 py-2 text-xs" />
                      <input type="text" name="vertex_model_name" defaultValue="gemini-1.5-pro-preview-0409" className="w-full bg-black/20 border border-slate-800 rounded px-3 py-2 text-xs" />
                   </div>
                 </div>
              )}
            </div>

            <div className="space-y-3">
                <label className="text-xs font-bold text-slate-500 uppercase flex items-center gap-2">
                    <BrainCircuit className="w-3 h-3" /> Reasoning Architecture
                </label>
                <div className="grid grid-cols-2 gap-2">
                    <button type="button" 
                        onClick={() => setReasoningMethod('cot')}
                        className={`py-2 px-3 rounded text-xs border ${reasoningMethod === 'cot' ? 'bg-indigo-900/40 border-indigo-500 text-indigo-300' : 'bg-slate-900 border-slate-800 text-slate-500 hover:border-slate-700'}`}>
                        Standard CoT
                    </button>
                    <button type="button" 
                        onClick={() => setReasoningMethod('fcot')}
                        className={`py-2 px-3 rounded text-xs border ${reasoningMethod === 'fcot' ? 'bg-indigo-900/40 border-indigo-500 text-indigo-300' : 'bg-slate-900 border-slate-800 text-slate-500 hover:border-slate-700'}`}>
                        Fractal CoT
                    </button>
                </div>
                <input type="hidden" name="reasoning_method" value={reasoningMethod} />
                <p className="text-[10px] text-slate-500">
                    {reasoningMethod === 'cot' ? "Single-pass linear chain of thought." : "Recursive Multi-Scale (Macro → Meso → Consensus)."}
                </p>
            </div>

            {activeTab === 'queue' && (
               <div className="space-y-4">
                 <div onClick={() => fileInputRef.current?.click()} className="border-2 border-dashed border-slate-700 hover:border-indigo-500 hover:bg-indigo-500/5 rounded-xl p-4 text-center cursor-pointer transition-colors">
                     <Upload className="w-5 h-5 mx-auto text-slate-500 mb-1" />
                     <p className="text-xs text-slate-400">Upload CSV to Queue</p>
                     <input type="file" ref={fileInputRef} onChange={handleFileUpload} accept=".csv" hidden />
                 </div>
                 <div className="p-3 bg-indigo-900/10 border border-indigo-500/20 rounded-lg flex justify-between">
                    <p className="text-xs text-indigo-300">Pending: {queueList.filter(x=>x.status==='Pending').length}</p>
                    <p className="text-xs text-indigo-300">Total: {queueList.length}</p>
                 </div>
               </div>
            )}
            
            <div className="flex items-center gap-2 mt-4 bg-slate-900 p-2 rounded border border-slate-800">
               <input type="checkbox" name="include_comments" id="include_comments" value="true" className="rounded bg-slate-800 border-slate-700 text-indigo-500 focus:ring-offset-0 focus:ring-0" />
               <label htmlFor="include_comments" className="text-xs text-slate-400 select-none cursor-pointer">Include Reasoning (Detailed Schema)</label>
            </div>
          </form>
        </div>

        <div className="p-6 border-t border-slate-800/60 bg-[#0c0c0e]">
           {activeTab === 'queue' ? (
              <div className="flex gap-2">
                 <button onClick={handleStartQueue} disabled={isProcessing} className={`flex-1 py-3 rounded-lg font-bold text-xs flex items-center justify-center gap-2 ${isProcessing ? 'bg-slate-800 text-slate-400' : 'bg-emerald-600 hover:bg-emerald-500 text-white'}`}>
                    <Play className="w-4 h-4" /> Start Batch
                 </button>
                 {isProcessing && (
                     <button onClick={handleStopQueue} className="px-4 bg-red-900/50 text-red-400 border border-red-900 rounded-lg hover:bg-red-900/80">
                        <StopCircle className="w-4 h-4" />
                     </button>
                 )}
              </div>
           ) : activeTab === 'manual' ? (
             <button type="submit" form="control-form" className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-xs font-bold text-white flex justify-center gap-2"><Play className="w-4 h-4"/> Run Labeler</button>
           ) : (
             <button onClick={() => setRefreshTrigger(x=>x+1)} className="w-full py-3 bg-slate-800 hover:bg-slate-700 rounded-lg text-xs font-bold text-white">Refresh List</button>
           )}
        </div>
      </div>

      {/* RIGHT PANEL */}
      <div className="flex-1 flex flex-col bg-[#09090b] overflow-hidden relative">
         <div className="h-16 border-b border-slate-800/60 bg-[#09090b]/80 backdrop-blur flex justify-between items-center px-8 z-10">
            <span className="text-xs font-mono font-medium text-slate-400 tracking-wide">{activeTab.toUpperCase()} VIEW</span>
            <button onClick={() => setLogs('')} className="text-[10px] text-slate-600 hover:text-slate-400">Clear Logs</button>
         </div>

         <div className="flex-1 p-6 overflow-hidden flex flex-col z-10">
            {activeTab === 'queue' && (
                <div className="flex-1 flex flex-col gap-4">
                    <div className="h-1/2 bg-slate-900/30 border border-slate-800 rounded-xl overflow-auto custom-scrollbar">
                        <table className="w-full text-left text-xs text-slate-400">
                            <thead className="bg-slate-900 text-slate-300 sticky top-0"><tr><th className="p-3">Link</th><th className="p-3">Ingested</th><th className="p-3">Status</th><th className="p-3 text-right">Action</th></tr></thead>
                            <tbody className="divide-y divide-slate-800/50">
                                {queueList.map((q,i) => (
                                    <tr key={i} className="hover:bg-white/5">
                                        <td className="p-3 truncate max-w-[300px] text-sky-500">
                                            <a href={q.link} target="_blank" rel="noopener noreferrer" className="hover:underline flex items-center gap-1">
                                                {q.link} <ExternalLink className="w-3 h-3"/>
                                            </a>
                                        </td>
                                        <td className="p-3 text-slate-500">{q.timestamp}</td>
                                        <td className="p-3"><span className={`px-2 py-0.5 rounded ${q.status==='Processed' ? 'bg-emerald-500/10 text-emerald-500' : 'bg-amber-500/10 text-amber-500'}`}>{q.status}</span></td>
                                        <td className="p-3 text-right">
                                            <button onClick={()=>handleQueueDelete(q.link)} className="text-slate-500 hover:text-red-500 p-1">
                                                <Trash2 className="w-4 h-4"/>
                                            </button>
                                        </td>
                                    </tr>
                                ))}
                                {queueList.length===0 && <tr><td colSpan={4} className="p-4 text-center">Queue empty. Upload CSV or use Extension.</td></tr>}
                            </tbody>
                        </table>
                    </div>
                    <div className="h-1/2 bg-black/40 border border-slate-800 rounded-xl p-4 font-mono text-[11px] text-slate-300 overflow-auto">
                        <pre>{logs}</pre>
                        <div ref={logEndRef} />
                    </div>
                </div>
            )}

            {activeTab === 'analytics' && (
                <div className="flex-1 overflow-auto custom-scrollbar flex flex-col gap-6">
                    {/* Header Stats */}
                    <div className="grid grid-cols-4 gap-4">
                        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
                            <h4 className="text-xs text-slate-500 uppercase font-bold">Total Verified</h4>
                            <div className="text-2xl font-bold text-white mt-1">{comparisonList.length}</div>
                        </div>
                        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
                            <h4 className="text-xs text-slate-500 uppercase font-bold">MAE (Mean Err)</h4>
                            <div className="text-2xl font-bold text-sky-400 mt-1">{stats.avgMAE}</div>
                            <div className="text-[10px] text-slate-500">Average absolute deviation</div>
                        </div>
                        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
                            <h4 className="text-xs text-slate-500 uppercase font-bold">AI Bias</h4>
                            <div className="text-2xl font-bold text-amber-400 mt-1">{Number(stats.bias) > 0 ? "+" : ""}{stats.bias}</div>
                            <div className="text-[10px] text-slate-500">Positive = AI scores higher than human</div>
                        </div>
                    </div>

                    {/* Comparison Chart List */}
                    <div className="bg-slate-900/30 border border-slate-800 rounded-xl overflow-hidden">
                       <div className="p-4 bg-slate-950 border-b border-slate-800 flex justify-between items-center">
                            <h3 className="text-sm font-bold text-white flex items-center gap-2">
                                <Scale className="w-4 h-4 text-indigo-500"/> Verification Showcase
                            </h3>
                       </div>
                       <table className="w-full text-left text-xs text-slate-400">
                           <thead className="bg-slate-900 text-slate-300">
                               <tr>
                                   <th className="p-3 w-1/4">Video / Link</th>
                                   <th className="p-3 w-1/2">Score Comparison (AI vs Manual)</th>
                                   <th className="p-3 text-right">Delta</th>
                               </tr>
                           </thead>
                           <tbody className="divide-y divide-slate-800/50">
                               {comparisonList.map((item, i) => (
                                   <tr key={i} className="hover:bg-white/5">
                                       <td className="p-3">
                                           <div className="text-sky-500 truncate max-w-[200px]" title={item.link}>
                                               <a href={item.link} target="_blank" rel="noopener noreferrer" className="hover:underline flex items-center gap-1">
                                                   {item.link} <ExternalLink className="w-3 h-3"/>
                                               </a>
                                           </div>
                                           <div className="text-[10px] text-slate-600 font-mono">{item.id}</div>
                                       </td>
                                       <td className="p-3">
                                           {/* Visual Integrity Bar */}
                                           <div className="flex items-center gap-2 mb-1">
                                               <span className="w-16 text-[10px] text-slate-500">Visual</span>
                                               <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden flex relative">
                                                   <div className="absolute top-0 bottom-0 bg-indigo-500/50" style={{left:0, width:`${item.scores.visual.ai * 10}%`}}></div>
                                                   <div className="absolute top-0 bottom-0 border-l-2 border-emerald-500 h-full" style={{left:`${item.scores.visual.manual * 10}%`}}></div>
                                               </div>
                                               <span className="text-[10px] font-mono"><span className="text-indigo-400">{item.scores.visual.ai}</span> / <span className="text-emerald-400">{item.scores.visual.manual}</span></span>
                                           </div>
                                            {/* Final Veracity Bar */}
                                           <div className="flex items-center gap-2">
                                               <span className="w-16 text-[10px] text-slate-500 font-bold">Final</span>
                                               <div className="flex-1 h-3 bg-slate-800 rounded-full overflow-hidden flex relative">
                                                   {/* AI Score */}
                                                   <div className="bg-indigo-600 h-full" style={{width:`${item.scores.final.ai}%`}}></div>
                                               </div>
                                                {/* Manual Marker */}
                                               <div className="w-1 h-3 bg-emerald-500 -ml-1 z-10"></div>
                                               <span className="text-[10px] font-mono"><span className="text-indigo-400">{item.scores.final.ai}</span> / <span className="text-emerald-400">{item.scores.final.manual}</span></span>
                                           </div>
                                       </td>
                                       <td className="p-3 text-right">
                                           <div className={`font-bold ${Math.abs(item.deltas.final) > 20 ? 'text-red-500' : 'text-slate-400'}`}>
                                               {item.deltas.final > 0 ? "+" : ""}{item.deltas.final}
                                           </div>
                                       </td>
                                   </tr>
                               ))}
                           </tbody>
                       </table>
                    </div>
                </div>
            )}

            {activeTab === 'workflow' && (
                <div className="flex-1 overflow-auto custom-scrollbar flex flex-col gap-6">
                   
                   {/* Summary Cards */}
                   <div className="grid grid-cols-3 gap-4">
                       <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
                           <h4 className="text-xs text-slate-500 uppercase font-bold">Pending Manual Review</h4>
                           <div className="text-2xl font-bold text-white mt-1">{pendingVerification.length}</div>
                           <div className="text-[10px] text-amber-500 mt-1">Ready for verification</div>
                       </div>
                       <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
                           <h4 className="text-xs text-slate-500 uppercase font-bold">Ingestion Queue</h4>
                           <div className="text-2xl font-bold text-white mt-1">{pendingAI.length}</div>
                           <div className="text-[10px] text-sky-500 mt-1">Waiting for AI labeling</div>
                       </div>
                       <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
                           <h4 className="text-xs text-slate-500 uppercase font-bold">Total Verified</h4>
                           <div className="text-2xl font-bold text-white mt-1">{completed.length}</div>
                           <div className="text-[10px] text-emerald-500 mt-1">Manually confirmed</div>
                       </div>
                   </div>

                   {/* Main Section: Ready for Manual Verification */}
                   <div className="bg-slate-900/30 border border-slate-800 rounded-xl overflow-hidden flex flex-col min-h-[300px]">
                       <div className="p-4 bg-slate-950 border-b border-slate-800 flex justify-between items-center">
                            <h3 className="text-sm font-bold text-white flex items-center gap-2">
                                <AlertCircle className="w-4 h-4 text-amber-500"/> Needs Verification (Priority)
                            </h3>
                            <span className="text-xs text-slate-500">AI labeled links missing manual review.</span>
                       </div>
                       <div className="flex-1 overflow-auto">
                        <table className="w-full text-left text-xs text-slate-400">
                                <thead className="bg-slate-900 text-slate-300 sticky top-0">
                                <tr>
                                    <th className="p-4">Link</th>
                                    <th className="p-4">AI Score</th>
                                    <th className="p-4 text-right">Action</th>
                                </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-800/50">
                                {pendingVerification.map((row, i) => (
                                    <tr key={i} className="hover:bg-white/5 cursor-pointer" onClick={() => openLabelingModal(row)}>
                                        <td className="p-4 truncate max-w-[400px] text-sky-400">
                                            <a href={row.link} target="_blank" rel="noopener noreferrer" onClick={(e) => e.stopPropagation()} className="hover:underline flex items-center gap-1">
                                                {row.link} <ExternalLink className="w-3 h-3"/>
                                            </a>
                                        </td>
                                        <td className="p-4">
                                            <span className="flex items-center gap-1 text-emerald-400">
                                                <BrainCircuit className="w-3 h-3"/> {row.ai_data?.final}
                                            </span>
                                        </td>
                                        <td className="p-4 text-right">
                                            <button className="px-4 py-1.5 bg-indigo-600 text-white rounded font-bold hover:bg-indigo-500 shadow-lg shadow-indigo-500/20">Verify</button>
                                        </td>
                                    </tr>
                                ))}
                                {pendingVerification.length === 0 && (
                                    <tr><td colSpan={3} className="p-8 text-center text-emerald-500">No pending verifications. Good job!</td></tr>
                                )}
                                </tbody>
                        </table>
                       </div>
                   </div>

                   {/* Secondary: Ingestion Queue */}
                   <div className="bg-slate-900/30 border border-slate-800 rounded-xl overflow-hidden">
                       <div className="p-3 bg-slate-950/50 border-b border-slate-800">
                            <h3 className="text-xs font-bold text-slate-400 uppercase">Ingestion Queue (Waiting for AI)</h3>
                       </div>
                       <table className="w-full text-left text-xs text-slate-500">
                            <tbody className="divide-y divide-slate-800/50">
                            {pendingAI.slice(0, 10).map((row, i) => (
                                <tr key={i}>
                                    <td className="p-3 truncate max-w-[400px] opacity-60">
                                        <a href={row.link} target="_blank" rel="noopener noreferrer" className="hover:underline hover:text-sky-400 flex items-center gap-1">
                                            {row.link} <ExternalLink className="w-3 h-3"/>
                                        </a>
                                    </td>
                                    <td className="p-3 text-right">
                                        <span className="px-2 py-0.5 bg-slate-800 rounded text-slate-400">Pending AI</span>
                                    </td>
                                </tr>
                            ))}
                            {pendingAI.length > 10 && <tr><td colSpan={2} className="p-3 text-center opacity-50">...and {pendingAI.length - 10} more</td></tr>}
                            </tbody>
                       </table>
                   </div>

                   {/* Verified History */}
                   <div className="bg-slate-900/30 border border-slate-800 rounded-xl overflow-hidden mb-8">
                       <div className="p-3 bg-slate-950/50 border-b border-slate-800">
                            <h3 className="text-xs font-bold text-slate-400 uppercase">Verification History</h3>
                       </div>
                       <table className="w-full text-left text-xs text-slate-500">
                            <tbody className="divide-y divide-slate-800/50">
                            {completed.slice(0, 10).map((row, i) => (
                                <tr key={i}>
                                    <td className="p-3 truncate max-w-[400px] opacity-60 text-emerald-500/50">
                                        <a href={row.link} target="_blank" rel="noopener noreferrer" className="hover:underline hover:text-emerald-400 flex items-center gap-1">
                                            {row.link} <ExternalLink className="w-3 h-3"/>
                                        </a>
                                    </td>
                                    <td className="p-3 opacity-60">Tags: {row.manual_tags || "-"}</td>
                                    <td className="p-3 text-right">
                                        <CheckSquare className="w-4 h-4 text-emerald-600 inline"/>
                                    </td>
                                </tr>
                            ))}
                            </tbody>
                       </table>
                   </div>
                </div>
            )}

            {activeTab === 'moderation' && (
                <div className="flex-1 bg-slate-900/30 border border-slate-800 rounded-xl overflow-auto custom-scrollbar">
                   <table className="w-full text-left text-xs text-slate-400">
                        <thead className="bg-slate-900 text-slate-300 sticky top-0">
                           <tr><th className="p-4">ID / Source</th><th className="p-4">Link / Caption</th><th className="p-4">Scores (V/A/F)</th><th className="p-4">Status</th><th className="p-4 text-right">Action</th></tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/50">
                           {dataList.map((row, i) => (
                             <React.Fragment key={i}>
                               <tr onClick={() => setExpandedRow(expandedRow === row.id ? null : row.id)} className={`hover:bg-white/5 cursor-pointer ${expandedRow===row.id?'bg-white/5':''}`}>
                                 <td className="p-4">
                                     <div className="font-mono text-indigo-400 font-bold">{row.id || 'N/A'}</div>
                                     <span className="text-[9px] uppercase px-1.5 py-0.5 rounded bg-slate-800 text-slate-500">{row.source_type}</span>
                                 </td>
                                 <td className="p-4">
                                     <div className="truncate max-w-[250px] text-white mb-1" title={row.caption}>{row.caption || 'No Caption'}</div>
                                     <div className="truncate max-w-[250px] text-[10px] text-slate-600">
                                        <a href={row.link} target="_blank" rel="noopener noreferrer" onClick={(e) => e.stopPropagation()} className="hover:text-indigo-400 hover:underline flex items-center gap-1">
                                             {row.link} <ExternalLink className="w-3 h-3"/>
                                        </a>
                                     </div>
                                     {row.tags && <div className="mt-1 text-[9px] text-emerald-400 font-mono">{row.tags}</div>}
                                 </td>
                                 <td className="p-4 font-mono">
                                     <span title="Visual" className="text-emerald-400">{row.visual_integrity_score}</span> / 
                                     <span title="Audio" className="text-sky-400">{row.audio_integrity_score}</span> / 
                                     <span title="Final" className="text-white font-bold">{row.final_veracity_score}</span>
                                 </td>
                                 <td className="p-4">
                                     {row.source_type === 'auto' && (
                                         row.manual_verification_status === 'Verified' ?
                                         <span className="text-emerald-500 flex items-center gap-1"><CheckCircle className="w-3 h-3"/> Verified</span> :
                                         <div className="flex items-center gap-2">
                                            <span className="text-amber-500">Need Manual</span>
                                            <button onClick={(e) => { e.stopPropagation(); openLabelingModal(row); }} 
                                                className="px-2 py-1 bg-indigo-600 hover:bg-indigo-500 text-white rounded text-[10px]">
                                                Verify
                                            </button>
                                         </div>
                                     )}
                                 </td>
                                 <td className="p-4 text-right"><button onClick={(e)=>{e.stopPropagation(); handleDelete(row.id, row.link)}} className="hover:text-red-400 p-2"><Trash2 className="w-4 h-4"/></button></td>
                               </tr>
                               {expandedRow === row.id && (
                                   <tr><td colSpan={5} className="bg-slate-950 p-6 border-b border-slate-800">
                                      <div className="grid grid-cols-2 gap-6">
                                         <div className="space-y-4">
                                             <div>
                                                 <h4 className="text-indigo-400 text-[10px] font-bold uppercase mb-2">Prompt Used</h4>
                                                 <div className="bg-black/30 border border-slate-800 rounded p-2 h-32 overflow-auto text-[9px] font-mono text-slate-500">
                                                     {row.json_data?.meta_info?.prompt_used || "Prompt not saved in legacy data."}
                                                 </div>
                                             </div>
                                             <div>
                                                 <h4 className="text-indigo-400 text-[10px] font-bold uppercase mb-2">Reasoning</h4>
                                                 <p className="text-sm text-slate-300 italic">{row.final_reasoning}</p>
                                             </div>
                                         </div>
                                         <div>
                                             <h4 className="text-indigo-400 text-[10px] font-bold uppercase mb-2">Raw JSON Data</h4>
                                             <div className="h-[300px] overflow-auto border border-slate-800 rounded p-3 bg-black/50 custom-scrollbar">
                                                 <pre className="text-[10px] font-mono text-emerald-500">{JSON.stringify(row.json_data || row, null, 2)}</pre>
                                             </div>
                                         </div>
                                      </div>
                                   </td></tr>
                               )}
                             </React.Fragment>
                           ))}
                        </tbody>
                   </table>
                </div>
            )}

            {activeTab === 'manual' && (
                <div className="flex-1 bg-black/40 border border-slate-800 rounded-xl p-4 font-mono text-[11px] text-slate-300 overflow-auto">
                    <pre>{logs}</pre>
                    <div ref={logEndRef} />
                </div>
            )}
         </div>

         {/* MANUAL LABELING MODAL */}
         {labelingItem && (
             <div className="absolute inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-8">
                 <div className="bg-[#0f172a] border border-slate-700 rounded-xl w-full max-w-4xl h-full max-h-full overflow-y-auto shadow-2xl flex flex-col">
                     <div className="p-4 border-b border-slate-800 flex justify-between items-center bg-[#1e293b]">
                         <h2 className="text-lg font-bold text-white">Manual Verification</h2>
                         <button onClick={() => setLabelingItem(null)} className="text-slate-400 hover:text-white"><XCircle className="w-6 h-6"/></button>
                     </div>
                     <div className="p-6 flex-1 overflow-y-auto">
                         <div className="mb-6 bg-slate-900 p-4 rounded border border-slate-800">
                             <div className="flex items-center gap-2 mb-2">
                                <a href={labelingItem.link} target="_blank" rel="noopener noreferrer" className="text-xs text-indigo-400 font-mono hover:underline flex items-center gap-2">
                                     {labelingItem.link} <ExternalLink className="w-3 h-3"/>
                                </a>
                             </div>
                             {labelingItem.ai_data?.reasoning && (
                                 <div className="text-xs text-slate-500 italic border-l-2 border-indigo-500 pl-3">
                                     "AI Reasoning: {labelingItem.ai_data.reasoning}"
                                 </div>
                             )}
                             {!labelingItem.ai_data && labelingItem.final_reasoning && (
                                 <div className="text-xs text-slate-500 italic border-l-2 border-indigo-500 pl-3">
                                     "AI Reasoning: {labelingItem.final_reasoning}"
                                 </div>
                             )}
                         </div>

                         <div className="grid grid-cols-2 gap-8">
                             <div className="space-y-4">
                                 <h3 className="text-xs font-bold text-slate-400 uppercase">Veracity Vectors (1-10)</h3>
                                 {['visual', 'audio', 'source', 'logic'].map(k => (
                                     <div key={k} className="flex items-center gap-4">
                                         <label className="w-24 text-xs capitalize text-slate-300">{k}</label>
                                         <input type="range" min="1" max="10" value={(manualForm as any)[k]} 
                                            onChange={e => setManualForm({...manualForm, [k]: parseInt(e.target.value)})} 
                                            className="flex-1 accent-indigo-500" />
                                         <span className="w-6 text-center text-sm font-bold text-indigo-400">{(manualForm as any)[k]}</span>
                                     </div>
                                 ))}
                                 
                                 <h3 className="text-xs font-bold text-slate-400 uppercase mt-6">Modalities (1-10)</h3>
                                 {['va', 'vc', 'ac'].map(k => (
                                     <div key={k} className="flex items-center gap-4">
                                         <label className="w-24 text-xs uppercase text-slate-300">{k}</label>
                                         <input type="range" min="1" max="10" value={(manualForm as any)[k]} 
                                            onChange={e => setManualForm({...manualForm, [k]: parseInt(e.target.value)})} 
                                            className="flex-1 accent-emerald-500" />
                                         <span className="w-6 text-center text-sm font-bold text-emerald-400">{(manualForm as any)[k]}</span>
                                     </div>
                                 ))}
                             </div>

                             <div className="space-y-4">
                                 <div>
                                     <label className="text-xs font-bold text-slate-400 uppercase block mb-2">Final Veracity Score (1-100)</label>
                                     <div className="flex items-center gap-4">
                                         <input type="range" min="1" max="100" value={manualForm.final} 
                                            onChange={e => setManualForm({...manualForm, final: parseInt(e.target.value)})} 
                                            className="flex-1 accent-amber-500 h-2" />
                                         <span className="text-xl font-bold text-amber-500">{manualForm.final}</span>
                                     </div>
                                 </div>

                                 <div>
                                     <label className="text-xs font-bold text-slate-400 uppercase block mb-2">Reasoning</label>
                                     <textarea className="w-full bg-slate-900 border border-slate-700 rounded p-3 text-xs text-white h-24"
                                         value={manualForm.reasoning} onChange={e => setManualForm({...manualForm, reasoning: e.target.value})}
                                         placeholder="Why did you assign these scores?"
                                     />
                                 </div>

                                 <div>
                                     <label className="text-xs font-bold text-slate-400 uppercase block mb-2">Tags (comma separated)</label>
                                     <input type="text" className="w-full bg-slate-900 border border-slate-700 rounded p-3 text-xs text-white"
                                         value={manualForm.tags} onChange={e => setManualForm({...manualForm, tags: e.target.value})}
                                         placeholder="political, viral, deepfake..."
                                     />
                                 </div>
                             </div>
                         </div>
                     </div>
                     <div className="p-4 border-t border-slate-800 bg-[#1e293b] flex justify-end gap-3">
                         <button onClick={() => setLabelingItem(null)} className="px-4 py-2 text-slate-400 hover:text-white">Cancel</button>
                         <button onClick={submitManualLabel} className="px-6 py-2 bg-indigo-600 hover:bg-indigo-500 text-white font-bold rounded">Save Manual Label</button>
                     </div>
                 </div>
             </div>
         )}
      </div>
    </div>
  )
}

export default App