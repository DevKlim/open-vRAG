import React, { useState, useRef, useEffect } from 'react';
import { 
  AlertCircle, Play, Upload, Layers, Terminal, Cpu, Activity, 
  FileText, Zap, MessageSquare, Sliders, LayoutDashboard, FileJson, 
  ChevronDown, ChevronRight, Bot, Database, Trash2, Eye, StopCircle, List,
  CheckCircle, XCircle
} from 'lucide-react';

function App() {
  const [activeTab, setActiveTab] = useState('queue');
  const [logs, setLogs] = useState<string>('System Ready.\n');
  const [isProcessing, setIsProcessing] = useState(false);
  const [dataList, setDataList] = useState<any[]>([]);
  const [queueList, setQueueList] = useState<any[]>([]);
  const [expandedRow, setExpandedRow] = useState<string | null>(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const [videoUrl, setVideoUrl] = useState('');
  const [model, setModel] = useState('vertex'); 
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
            <div className="grid grid-cols-3 gap-1 p-1 bg-slate-900 rounded-lg border border-slate-800">
               {[{id:'queue',l:'Queue',i:List}, {id:'moderation',l:'Moderation',i:Database}, {id:'manual',l:'Labeler',i:Play}].map(t => (
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

            {activeTab === 'manual' && (
               <div className="space-y-2">
                   <label className="text-xs font-bold text-slate-500 uppercase">Single Video URL</label>
                   <input type="text" name="video_url" value={videoUrl} onChange={e=>setVideoUrl(e.target.value)} className="w-full bg-slate-900 border border-slate-800 rounded px-3 py-2 text-xs" />
               </div>
            )}
            
            <input type="hidden" name="include_comments" value="true" />
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
           ) : activeTab === 'moderation' ? (
              <button onClick={() => setRefreshTrigger(x=>x+1)} className="w-full py-3 bg-slate-800 hover:bg-slate-700 rounded-lg text-xs font-bold text-white">Refresh List</button>
           ) : (
             <button type="submit" form="control-form" className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-xs font-bold text-white flex justify-center gap-2"><Play className="w-4 h-4"/> Run Labeler</button>
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
                            <thead className="bg-slate-900 text-slate-300 sticky top-0"><tr><th className="p-3">Link</th><th className="p-3">Ingested</th><th className="p-3">Status</th></tr></thead>
                            <tbody className="divide-y divide-slate-800/50">
                                {queueList.map((q,i) => (
                                    <tr key={i} className="hover:bg-white/5">
                                        <td className="p-3 truncate max-w-[300px] text-sky-500">{q.link}</td>
                                        <td className="p-3 text-slate-500">{q.timestamp}</td>
                                        <td className="p-3"><span className={`px-2 py-0.5 rounded ${q.status==='Processed' ? 'bg-emerald-500/10 text-emerald-500' : 'bg-amber-500/10 text-amber-500'}`}>{q.status}</span></td>
                                    </tr>
                                ))}
                                {queueList.length===0 && <tr><td colSpan={3} className="p-4 text-center">Queue empty. Upload CSV or use Extension.</td></tr>}
                            </tbody>
                        </table>
                    </div>
                    <div className="h-1/2 bg-black/40 border border-slate-800 rounded-xl p-4 font-mono text-[11px] text-slate-300 overflow-auto">
                        <pre>{logs}</pre>
                        <div ref={logEndRef} />
                    </div>
                </div>
            )}

            {activeTab === 'moderation' && (
                <div className="flex-1 bg-slate-900/30 border border-slate-800 rounded-xl overflow-auto custom-scrollbar">
                   <table className="w-full text-left text-xs text-slate-400">
                        <thead className="bg-slate-900 text-slate-300 sticky top-0">
                           <tr><th className="p-4">ID / Source</th><th className="p-4">Link / Caption</th><th className="p-4">Scores (V/A/F)</th><th className="p-4 text-right">Action</th></tr>
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
                                     <div className="truncate max-w-[250px] text-[10px] text-slate-600">{row.link}</div>
                                 </td>
                                 <td className="p-4 font-mono">
                                     <span title="Visual" className="text-emerald-400">{row.visual_integrity_score}</span> / 
                                     <span title="Audio" className="text-sky-400">{row.audio_integrity_score}</span> / 
                                     <span title="Final" className="text-white font-bold">{row.final_veracity_score}</span>
                                 </td>
                                 <td className="p-4 text-right"><button onClick={(e)=>{e.stopPropagation(); handleDelete(row.id, row.link)}} className="hover:text-red-400 p-2"><Trash2 className="w-4 h-4"/></button></td>
                               </tr>
                               {expandedRow === row.id && (
                                   <tr><td colSpan={4} className="bg-slate-950 p-6 border-b border-slate-800">
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
      </div>
    </div>
  )
}

export default App
