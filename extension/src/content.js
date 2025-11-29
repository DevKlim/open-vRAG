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
        const match = aria.match(/(\d+(?:,\d+)*(?:\.\d+)?[KMB]?)/);
        if (match) return parseKMB(match[1]);
    }
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

// --- REDDIT INJECTION LOGIC ---

// Helper: Recursively search Shadow DOM for a condition
function findElementInShadow(root, predicate) {
    if (!root) return null;
    if (predicate(root)) return root;
    
    // Check direct children
    const children = root.children ? Array.from(root.children) : [];
    for (let child of children) {
        if (predicate(child)) return child;
        
        // Recurse into child's shadow root if it exists
        if (child.shadowRoot) {
            const res = findElementInShadow(child.shadowRoot, predicate);
            if (res) return res;
        }
        
        // Recurse into child's children
        const res = findElementInShadow(child, predicate);
        if (res) return res;
    }
    return null;
}

function injectRedditButtons() {
    // Select all post elements
    const posts = document.querySelectorAll('shreddit-post');
    
    posts.forEach(post => {
        // Avoid double injection
        if (post.dataset.vchatInjected === "true") {
             // Optional: Check if it's still in DOM correctly? 
             return;
        }

        let actionRow = null;

        // 1. Look for shreddit-action-row explicitly (Common in post view and new feed)
        // This is often a direct child or in shadow
        actionRow = post.querySelector('shreddit-action-row');
        if (!actionRow && post.shadowRoot) {
            actionRow = post.shadowRoot.querySelector('shreddit-action-row');
        }

        // 2. Direct Slot Search (Light DOM)
        if (!actionRow) actionRow = post.querySelector('[slot="actions"]');
        
        // 3. Direct Slot Search (Shadow DOM)
        if (!actionRow && post.shadowRoot) {
            actionRow = post.shadowRoot.querySelector('[slot="actions"]');
        }

        // 4. Deep Search for "Share" button container if specific components not found
        if (!actionRow && post.shadowRoot) {
            const shareBtn = findElementInShadow(post.shadowRoot, (el) => {
                if (!el) return false;
                const tag = el.tagName;
                // Check standard buttons or links with "share" intent
                if (tag === 'BUTTON' || tag === 'A') {
                    const label = (el.getAttribute('aria-label') || "").toLowerCase();
                    const name = (el.getAttribute('name') || "").toLowerCase();
                    const text = (el.innerText || "").toLowerCase();
                    if (label.includes('share') || name.includes('share') || text.includes('share')) return true;
                }
                // Check shreddit specific components
                if (tag === 'SHREDDIT-COMMENT-BUTTON') return true;
                return false;
            });
            
            if (shareBtn) {
                // The parent is typically the flex container for actions
                actionRow = shareBtn.parentElement;
            }
        }

        if (actionRow) {
            if (actionRow.querySelector('.vchat-btn-group')) {
                post.dataset.vchatInjected = "true";
                return;
            }

            // --- LINK EXTRACTION ---
            let permalink = post.getAttribute('permalink');
            // Fallback: If permalink is missing (common on single post view for the main post), 
            // use window location if it looks like a post.
            if (!permalink && window.location.pathname.includes('/comments/')) {
                 permalink = window.location.pathname;
            }

            if (!permalink) {
                // console.log("[vChat] No permalink found for post.");
                return; 
            }

            let fullLink = permalink;
            if (!fullLink.startsWith('http')) {
                fullLink = "https://www.reddit.com" + permalink;
            }

            const title = post.getAttribute('post-title') || document.title || "Reddit Post";
            
            const stats = {
                likes: parseInt(post.getAttribute('score') || 0),
                shares: 0,
                comments: parseInt(post.getAttribute('comment-count') || 0),
                platform: 'reddit'
            };

            // console.log(`[vChat] Injecting into Post: ${title.substring(0, 20)}...`);
            createButtonUI(actionRow, fullLink, title, 'reddit', stats);
            post.dataset.vchatInjected = "true";
        }
    });
}

function injectRedditCommentButtons() {
    const comments = document.querySelectorAll('shreddit-comment');
    
    comments.forEach(comment => {
        if (comment.dataset.vchatInjected === "true") return;

        let actionRow = null;
        
        // 1. Look for explicit action row
        actionRow = comment.querySelector('shreddit-comment-action-row');
        if (!actionRow && comment.shadowRoot) {
            actionRow = comment.shadowRoot.querySelector('shreddit-comment-action-row');
        }

        // 2. Light DOM Slot
        if (!actionRow) actionRow = comment.querySelector('[slot="actions"]');
        
        // 3. Shadow DOM Slot
        if (!actionRow && comment.shadowRoot) {
            actionRow = comment.shadowRoot.querySelector('[slot="actions"]');
        }

        // 4. Deep Search
        if (!actionRow && comment.shadowRoot) {
            const btn = findElementInShadow(comment.shadowRoot, (el) => {
                if (el.tagName === 'BUTTON') {
                    const txt = (el.innerText || "").toLowerCase();
                    if (txt.includes('reply')) return true;
                }
                return false;
            });
            if (btn) actionRow = btn.parentElement;
        }

        if (actionRow) {
            if (actionRow.querySelector('.vchat-btn-group')) {
                comment.dataset.vchatInjected = "true";
                return;
            }

            let permalink = comment.getAttribute('permalink');
            if (!permalink) return;
            
            let fullLink = permalink;
            if (!fullLink.startsWith('http')) {
                fullLink = "https://www.reddit.com" + permalink;
            }

            const author = comment.getAttribute('author') || "Unknown";
            
            let text = "";
            const contentSlot = comment.querySelector('[slot="comment"]');
            if (contentSlot) text = contentSlot.innerText;
            else {
                 const shadowContent = findElementInShadow(comment.shadowRoot, el => el.id === 'comment-content');
                 if(shadowContent) text = shadowContent.innerText;
            }
            if(!text) text = "Comment content hidden";

            const caption = `Comment by u/${author}: ${text.substring(0, 100).replace(/\n/g, ' ')}...`;

            const stats = {
                likes: parseInt(comment.getAttribute('score') || 0),
                shares: 0,
                comments: 0,
                platform: 'reddit'
            };

            createButtonUI(actionRow, fullLink, caption, 'reddit', stats);
            comment.dataset.vchatInjected = "true";
        }
    });
}

function createButtonUI(container, link, caption, platform, stats) {
    // Avoid duplicates in container
    if (container.querySelector('.vchat-btn-group')) return;

    const wrapper = document.createElement('div');
    wrapper.className = 'vchat-btn-group';
    
    // Default Styles
    wrapper.style.display = 'flex';
    wrapper.style.gap = '6px';
    wrapper.style.marginLeft = '10px';
    wrapper.style.zIndex = '999';
    wrapper.style.position = 'relative';

    // Platform Specific Overrides
    if (platform === 'reddit') {
        wrapper.style.display = 'inline-flex';
        wrapper.style.alignItems = 'center';
        wrapper.style.marginLeft = '8px';
        wrapper.style.height = '100%'; 
        // Ensure visibility
        wrapper.style.opacity = '1'; 
        wrapper.style.visibility = 'visible';
    }

    wrapper.onclick = (e) => e.stopPropagation();
    
    const btnIngest = document.createElement('button');
    btnIngest.className = 'vchat-btn';
    btnIngest.innerHTML = '⚡';
    btnIngest.title = "Add to Queue";
    btnIngest.style.backgroundColor = '#6366f1'; 
    btnIngest.style.color = '#ffffff';
    btnIngest.style.border = 'none';
    btnIngest.style.padding = '4px 8px';
    btnIngest.style.borderRadius = '999px';
    btnIngest.style.fontSize = '12px';
    btnIngest.style.fontWeight = 'bold';
    btnIngest.style.cursor = 'pointer';

    btnIngest.onclick = (e) => { e.stopPropagation(); handleIngest(link, btnIngest); };
    
    const btnOpen = document.createElement('button');
    btnOpen.className = 'vchat-btn comments';
    btnOpen.innerHTML = '🔍';
    btnOpen.title = "Analyze Veracity";
    btnOpen.style.backgroundColor = '#f59e0b';
    btnOpen.style.color = '#ffffff';
    btnOpen.style.border = 'none';
    btnOpen.style.padding = '4px 8px';
    btnOpen.style.borderRadius = '999px';
    btnOpen.style.fontSize = '12px';
    btnOpen.style.fontWeight = 'bold';
    btnOpen.style.cursor = 'pointer';

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
    btn.innerHTML = '...';
    chrome.runtime.sendMessage({type: 'INGEST_LINK', link: link}, (res) => {
        if (res && res.success) {
            btn.innerHTML = '✔';
            btn.style.backgroundColor = '#10b981';
        } else {
            btn.innerHTML = '❌';
            btn.style.backgroundColor = '#ef4444';
        }
        setTimeout(() => { btn.innerHTML = '⚡'; btn.style.backgroundColor = '#6366f1'; }, 2000);
    });
}

function openPanel(link, caption, platform, stats) {
    currentLink = link;
    currentCaption = caption;
    currentPlatform = platform;
    currentStats = stats;
    
    if (sidePanel.classList.contains('hidden')) togglePanel();
    
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
        <div class="vchat-section-title">Target Content</div>
        <div class="vchat-link-preview">${link}</div>
        <div class="vchat-stats-row">
             <span>❤️ ${currentStats.likes || 0}</span>
             <span>💬 ${currentStats.comments || 0}</span>
        </div>
        <div class="vchat-status-msg" id="scan-msg">
            <span style="font-size:20px;">🔍</span> Scanning context...
        </div>
    `;

    setTimeout(() => {
        const comments = scrapeCommentsFromDOM(platform, 30);
        renderCommentList(link, comments);
    }, 500);
}

function scrapeCommentsFromDOM(platform, limit) {
    let comments = [];
    if (platform === 'twitter') {
        const nodes = document.querySelectorAll('[data-testid="cellInnerDiv"]');
        nodes.forEach((cell) => {
            const textNode = cell.querySelector('[data-testid="tweetText"]');
            const userNode = cell.querySelector('[data-testid="User-Name"]');
            if (textNode && userNode && comments.length < limit) {
                const text = textNode.innerText;
                const handle = userNode.innerText.split('\n').find(s => s.startsWith('@')) || "Unknown";
                if (text.length > 2) comments.push({ author: handle, text: text });
            }
        });
    } else if (platform === 'reddit') {
        // Scrape visible shreddit comments
        const nodes = document.querySelectorAll('shreddit-comment');
        nodes.forEach(n => {
            const content = n.querySelector('[slot="comment"]');
            const author = n.getAttribute('author') || "Unknown";
            // Helper to get text if slot is empty (sometimes in shadow)
            let txt = "";
            if (content) txt = content.innerText;
            else {
                const shadowC = findElementInShadow(n.shadowRoot, el => el.id === 'comment-content');
                if (shadowC) txt = shadowC.innerText;
            }
            
            if (txt && txt.length > 2 && comments.length < limit) {
                comments.push({ author: author, text: txt });
            }
        });
    }
    return comments;
}

function renderCommentList(link, comments) {
    const listHtml = comments.length ? 
        comments.map(c => `<div class="vchat-comment-item"><b>${c.author}</b>: ${c.text}</div>`).join('') :
        `<div class="vchat-status-msg">No context found.<br>Scroll to load comments.</div>`;

    const container = document.getElementById('scan-msg').parentElement;
    
    container.innerHTML = `
        <div class="vchat-section-title">Target Content</div>
        <div class="vchat-link-preview">${link}</div>
        
        <div class="vchat-section-title" style="display:flex; justify-content:space-between;">
            <span>Context (${comments.length})</span>
            <button id="btn-refresh-comments" style="background:none; border:none; color:#6366f1; cursor:pointer; font-weight:bold; font-size:11px;">↻ RESCAN</button>
        </div>
        <div id="comment-list" style="max-height: 350px; overflow-y: auto; margin-bottom: 20px;">
            ${listHtml}
        </div>

        <button id="btn-save-all" class="vchat-action-btn">Save Context</button>
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

    ['visual', 'audio', 'source', 'logic', 'va', 'vc', 'final'].forEach(key => {
        const sl = document.getElementById(`score-${key}`);
        const val = document.getElementById(`val-${key}`);
        if(sl) sl.oninput = () => val.innerText = sl.value;
    });

    document.getElementById('btn-submit-label').onclick = () => {
        const btn = document.getElementById('btn-submit-label');
        btn.innerText = "Saving...";
        
        const payload = {
            link: link,
            caption: caption,
            stats: currentStats,
            labels: {
                visual_integrity_score: parseInt(document.getElementById('score-visual').value),
                audio_integrity_score: parseInt(document.getElementById('score-audio').value),
                source_credibility_score: parseInt(document.getElementById('score-source').value),
                logical_consistency_score: parseInt(document.getElementById('score-logic').value),
                
                video_audio_score: parseInt(document.getElementById('score-va').value),
                video_caption_score: parseInt(document.getElementById('score-vc').value),
                audio_caption_score: 5,
                
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
        injectRedditCommentButtons();
    }, 500); 
});

// Observe Body for changes (Navigation in SPA)
observer.observe(document.body, { childList: true, subtree: true });

// Also interval check for safety in complex SPAs
setInterval(() => {
    injectTwitterButtons();
    injectRedditButtons();
    injectRedditCommentButtons();
}, 2000);