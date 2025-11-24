// content.js - Injects buttons, Side Panel, and Handles Scraping
console.log("[vChat] Extension loaded.");

let debounceTimer = null;
let sidePanel = null;
let panelContent = null;
let toggleBtn = null;
let currentLink = "";
let currentCaption = "";
let currentPlatform = "";
let currentStats = {};

// --- UI INITIALIZATION ---

function initSidePanel() {
    if (document.getElementById('vchat-panel')) return;

    sidePanel = document.createElement('div');
    sidePanel.id = 'vchat-panel';
    sidePanel.className = 'hidden'; 
    
    // Header + Tabs
    sidePanel.innerHTML = `
        <div class="vchat-panel-header">
            <span class="vchat-title">vChat Assistant</span>
            <button class="vchat-close-btn" id="vchat-close" title="Close">×</button>
        </div>
        <div class="vchat-tabs">
            <button class="vchat-tab active" data-tab="comments">💬 Comments</button>
            <button class="vchat-tab" data-tab="labeling">📝 Labeling</button>
        </div>
        <div class="vchat-panel-content" id="vchat-content">
            <div class="vchat-status-msg">
                Select a post to begin.
            </div>
        </div>
    `;
    
    document.body.appendChild(sidePanel);

    // Floating Toggle
    toggleBtn = document.createElement('div');
    toggleBtn.id = 'vchat-toggle';
    toggleBtn.innerHTML = '⚡';
    toggleBtn.title = "Open vChat Panel";
    toggleBtn.onclick = togglePanel;
    document.body.appendChild(toggleBtn);

    // Events
    document.getElementById('vchat-close').onclick = togglePanel;
    panelContent = document.getElementById('vchat-content');

    // Tab Switching
    sidePanel.querySelectorAll('.vchat-tab').forEach(btn => {
        btn.addEventListener('click', () => {
            sidePanel.querySelectorAll('.vchat-tab').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const tabName = btn.getAttribute('data-tab');
            if (currentLink) {
                renderPanelForLink(currentLink, currentCaption, currentPlatform, tabName);
            }
        });
    });
}

function togglePanel() {
    if (sidePanel.classList.contains('hidden')) {
        sidePanel.classList.remove('hidden');
        toggleBtn.style.display = 'none';
    } else {
        sidePanel.classList.add('hidden');
        toggleBtn.style.display = 'flex';
    }
}

// --- BUTTON INJECTION & SCRAPING ---

function injectTwitterButtons() {
    const tweets = document.querySelectorAll('article[data-testid="tweet"]');
    tweets.forEach(tweet => {
        const actionBar = tweet.querySelector('[role="group"]');
        if (actionBar && !actionBar.querySelector('.vchat-btn-group')) {
            const timeElement = tweet.querySelector('a[href*="/status/"]');
            const tweetLink = timeElement ? timeElement.href : window.location.href;
            const textElement = tweet.querySelector('[data-testid="tweetText"]');
            const caption = textElement ? textElement.innerText : "";
            
            // Scrape Stats (approximate based on aria-labels or text)
            const stats = {
                likes: parseMetric(tweet.querySelector('[data-testid="like"]')),
                shares: parseMetric(tweet.querySelector('[data-testid="retweet"]')),
                comments: parseMetric(tweet.querySelector('[data-testid="reply"]')),
                platform: 'twitter'
            };

            createButtonUI(actionBar, tweetLink, caption, 'twitter', stats);
        }
    });
}

function parseMetric(element) {
    if (!element) return 0;
    const aria = element.getAttribute('aria-label'); 
    if (aria) {
        // Example: "100 likes" -> extract 100
        const match = aria.match(/(\d+(?:,\d+)*(?:\.\d+)?[KMB]?)/);
        if (match) return parseKMB(match[1]);
    }
    // Fallback to text content
    const txt = element.innerText;
    if (txt) return parseKMB(txt);
    return 0;
}

function parseKMB(str) {
    if (!str) return 0;
    str = str.toUpperCase().replace(/,/g, '');
    let mul = 1;
    if (str.includes('K')) mul = 1000;
    else if (str.includes('M')) mul = 1000000;
    else if (str.includes('B')) mul = 1000000000;
    const val = parseFloat(str.replace(/[KMB]/g, ''));
    return Math.floor(val * mul) || 0;
}

function injectRedditButtons() {
    const posts = document.querySelectorAll('shreddit-post');
    posts.forEach(post => {
        const actionRow = post.shadowRoot ? post.shadowRoot.querySelector('[slot="actions"]') : post.querySelector('[slot="actions"]');
        const permalink = post.getAttribute('permalink');
        
        if (actionRow && permalink && !actionRow.querySelector('.vchat-btn-group')) {
            const fullLink = "https://www.reddit.com" + permalink;
            const title = post.getAttribute('post-title') || "Reddit Post";
            
            const stats = {
                likes: parseInt(post.getAttribute('score') || 0), // Upvotes
                shares: 0, // Reddit doesn't easily show share counts
                comments: parseInt(post.getAttribute('comment-count') || 0),
                platform: 'reddit'
            };

            createButtonUI(actionRow, fullLink, title, 'reddit', stats);
        }
    });
}

function createButtonUI(container, link, caption, platform, stats) {
    const wrapper = document.createElement('div');
    wrapper.className = 'vchat-btn-group';
    wrapper.onclick = (e) => e.stopPropagation();
    
    const btnIngest = document.createElement('button');
    btnIngest.className = 'vchat-btn';
    btnIngest.innerHTML = '⚡ Batch';
    btnIngest.onclick = (e) => { e.stopPropagation(); handleIngest(link, btnIngest); };
    
    const btnOpen = document.createElement('button');
    btnOpen.className = 'vchat-btn comments';
    btnOpen.innerHTML = '🔍 Analyze';
    btnOpen.onclick = (e) => { 
        e.stopPropagation(); 
        openPanel(link, caption, platform, stats); 
    };

    wrapper.appendChild(btnIngest);
    wrapper.appendChild(btnOpen);
    container.appendChild(wrapper);
}

// --- ACTIONS ---

function handleIngest(link, btn) {
    btn.innerText = '...';
    chrome.runtime.sendMessage({type: 'INGEST_LINK', link: link}, (res) => {
        if (res && res.success) {
            btn.innerText = '✔';
            btn.style.backgroundColor = '#10b981';
        } else {
            btn.innerText = '❌';
            btn.style.backgroundColor = '#ef4444';
        }
        setTimeout(() => { btn.innerText = '⚡ Batch'; btn.style.backgroundColor = ''; }, 2000);
    });
}

function openPanel(link, caption, platform, stats) {
    currentLink = link;
    currentCaption = caption;
    currentPlatform = platform;
    currentStats = stats;
    
    if (sidePanel.classList.contains('hidden')) togglePanel();
    
    // Default to comments tab initially
    const activeTab = sidePanel.querySelector('.vchat-tab.active').getAttribute('data-tab');
    renderPanelForLink(link, caption, platform, activeTab);
}

function renderPanelForLink(link, caption, platform, tab) {
    if (tab === 'comments') {
        renderCommentsTab(link, platform);
    } else {
        renderLabelingTab(link, caption);
    }
}

// --- COMMENTS TAB ---

function renderCommentsTab(link, platform) {
    panelContent.innerHTML = `
        <div class="vchat-section-title">Target Post</div>
        <div class="vchat-link-preview">${link}</div>
        <div class="vchat-stats-row">
             <span>❤️ ${currentStats.likes || 0}</span>
             <span>💬 ${currentStats.comments || 0}</span>
             <span>🔄 ${currentStats.shares || 0}</span>
        </div>
        <div class="vchat-status-msg" id="scan-msg">
            <span style="font-size:20px;">🔍</span> Scanning...
        </div>
    `;

    setTimeout(() => {
        const comments = scrapeCommentsFromDOM(platform, 20);
        renderCommentList(link, comments);
    }, 500);
}

function scrapeCommentsFromDOM(platform, limit) {
    let comments = [];
    if (platform === 'twitter') {
        // Look for comment cells
        const nodes = document.querySelectorAll('[data-testid="cellInnerDiv"]');
        nodes.forEach((cell) => {
            const textNode = cell.querySelector('[data-testid="tweetText"]');
            const userNode = cell.querySelector('[data-testid="User-Name"]');
            
            if (textNode && userNode && comments.length < limit) {
                const text = textNode.innerText;
                // Extract handle (e.g. @username)
                const handle = userNode.innerText.split('\n').find(s => s.startsWith('@')) || "Unknown";
                
                if (text.length > 2) {
                    comments.push({ author: handle, text: text });
                }
            }
        });
    } else if (platform === 'reddit') {
        const nodes = document.querySelectorAll('shreddit-comment');
        nodes.forEach(n => {
            const content = n.querySelector('[slot="comment"]');
            const author = n.getAttribute('author') || "Unknown";
            if (content && comments.length < limit) {
                comments.push({ author: author, text: content.innerText });
            }
        });
    }
    return comments;
}

function renderCommentList(link, comments) {
    const listHtml = comments.length ? 
        comments.map(c => `<div class="vchat-comment-item"><b>${c.author}</b>: ${c.text}</div>`).join('') :
        `<div class="vchat-status-msg">No comments found visible.<br>Scroll down and Rescan.</div>`;

    const container = document.getElementById('scan-msg').parentElement; // panel content
    
    // Re-render inside content
    container.innerHTML = `
        <div class="vchat-section-title">Target Post</div>
        <div class="vchat-link-preview">${link}</div>
        
        <div class="vchat-section-title" style="display:flex; justify-content:space-between;">
            <span>Detected (${comments.length})</span>
            <button id="btn-refresh-comments" style="background:none; border:none; color:#6366f1; cursor:pointer; font-weight:bold; font-size:11px;">↻ RESCAN</button>
        </div>
        <div id="comment-list" style="max-height: 350px; overflow-y: auto; margin-bottom: 20px;">
            ${listHtml}
        </div>

        <button id="btn-save-all" class="vchat-action-btn">Save Comments</button>
    `;

    document.getElementById('btn-refresh-comments').onclick = () => {
        const newComments = scrapeCommentsFromDOM(currentPlatform, 30);
        renderCommentList(link, newComments);
    };

    document.getElementById('btn-save-all').onclick = () => {
        const btn = document.getElementById('btn-save-all');
        btn.innerHTML = 'Saving...';
        chrome.runtime.sendMessage({
            type: 'SAVE_COMMENTS', 
            payload: {link: link, comments: comments}
        }, (res) => {
            if(res && res.success) {
                btn.innerText = '✔ Saved';
                btn.style.background = '#10b981';
            } else {
                btn.innerText = '❌ Error';
            }
        });
    };
}

// --- LABELING TAB ---

function renderLabelingTab(link, caption) {
    const formHtml = `
        <div class="vchat-section-title">Manual Labeling</div>
        
        <div class="form-group">
            <label class="form-label">Visual Integrity</label>
            <div class="range-container">
                <input type="range" id="score-visual" min="1" max="10" value="5" class="form-input">
                <span id="val-visual" class="range-val">5</span>
            </div>
        </div>
        
        <div class="form-group">
            <label class="form-label">Audio Integrity</label>
            <div class="range-container">
                <input type="range" id="score-audio" min="1" max="10" value="5" class="form-input">
                <span id="val-audio" class="range-val">5</span>
            </div>
        </div>

        <div class="form-group">
            <label class="form-label">Source Credibility</label>
            <div class="range-container">
                <input type="range" id="score-source" min="1" max="10" value="5" class="form-input">
                <span id="val-source" class="range-val">5</span>
            </div>
        </div>

        <div class="form-group">
            <label class="form-label">Logical Consistency</label>
            <div class="range-container">
                <input type="range" id="score-logic" min="1" max="10" value="5" class="form-input">
                <span id="val-logic" class="range-val">5</span>
            </div>
        </div>

        <div class="vchat-section-title" style="margin-top:15px;">Modality Alignment</div>
        
        <div class="form-group">
            <label class="form-label">Video ↔ Audio</label>
            <div class="range-container">
                <input type="range" id="score-va" min="1" max="10" value="5" class="form-input">
                <span id="val-va" class="range-val">5</span>
            </div>
        </div>

        <div class="form-group">
            <label class="form-label">Video ↔ Caption</label>
            <div class="range-container">
                <input type="range" id="score-vc" min="1" max="10" value="5" class="form-input">
                <span id="val-vc" class="range-val">5</span>
            </div>
        </div>

        <div class="vchat-section-title" style="margin-top:15px;">Final Assessment</div>
        <div class="form-group">
            <label class="form-label">Veracity Score (1-100)</label>
            <div class="range-container">
                <input type="range" id="score-final" min="1" max="100" value="50" class="form-input">
                <span id="val-final" class="range-val">50</span>
            </div>
        </div>

        <textarea id="input-reasoning" class="vchat-input" placeholder="Reasoning notes..." style="margin-top:10px;"></textarea>
        
        <button id="btn-submit-label" class="vchat-action-btn" style="background:#6366f1;">Save Label</button>
    `;
    
    panelContent.innerHTML = formHtml;

    // Bind Sliders
    ['visual', 'audio', 'source', 'logic', 'va', 'vc', 'final'].forEach(key => {
        const sl = document.getElementById(`score-${key}`);
        const val = document.getElementById(`val-${key}`);
        sl.oninput = () => val.innerText = sl.value;
    });

    document.getElementById('btn-submit-label').onclick = () => {
        const btn = document.getElementById('btn-submit-label');
        btn.innerText = "Saving...";
        
        const payload = {
            link: link,
            caption: caption,
            stats: currentStats, // Include stats in manual label
            labels: {
                visual_integrity_score: parseInt(document.getElementById('score-visual').value),
                audio_integrity_score: parseInt(document.getElementById('score-audio').value),
                source_credibility_score: parseInt(document.getElementById('score-source').value),
                logical_consistency_score: parseInt(document.getElementById('score-logic').value),
                
                video_audio_score: parseInt(document.getElementById('score-va').value),
                video_caption_score: parseInt(document.getElementById('score-vc').value),
                audio_caption_score: 5, // Default/Implied
                
                final_veracity_score: parseInt(document.getElementById('score-final').value),
                reasoning: document.getElementById('input-reasoning').value
            }
        };

        chrome.runtime.sendMessage({type: 'SAVE_MANUAL', payload: payload}, (res) => {
            if (res && res.success) {
                btn.innerText = "✔ Saved!";
                btn.style.background = "#10b981";
                setTimeout(() => btn.innerText = "Save Label", 2000);
            } else {
                alert("Error: " + (res ? res.error : "Unknown"));
                btn.innerText = "Save Label";
            }
        });
    };
}

// --- INIT ---
initSidePanel();

const observer = new MutationObserver(() => {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
        injectTwitterButtons();
        injectRedditButtons();
    }, 800);
});
observer.observe(document.body, { childList: true, subtree: true });