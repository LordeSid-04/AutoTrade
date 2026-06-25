// --- SUPABASE AUTH & SAAS LOGIC ---
const supabaseUrl = 'https://lwpxwnogsjpfsqtybbot.supabase.co';
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx3cHh3bm9nc2pwZnNxdHliYm90Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODE4OTAyODQsImV4cCI6MjA5NzQ2NjI4NH0.ZrXtmBbY27uHsJdsEgEDolK9fuja0OEiV3zBKuIKgI4';
const supabaseClient = supabase.createClient(supabaseUrl, supabaseKey);

let authToken = null;

async function checkSession() {
    const { data, error } = await supabaseClient.auth.getSession();
    if (data && data.session) {
        authToken = data.session.access_token;
        document.getElementById('auth-modal').classList.add('hidden');
        document.getElementById('btn-logout').classList.remove('hidden');
        initializeApp();
    } else {
        document.getElementById('auth-modal').classList.remove('hidden');
        document.getElementById('btn-logout').classList.add('hidden');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    checkSession();
    
    document.getElementById('btn-signup').addEventListener('click', async () => {
        const email = document.getElementById('auth-email').value;
        const password = document.getElementById('auth-password').value;
        if (!email || !password) return;
        
        const { data, error } = await supabaseClient.auth.signUp({ email, password });
        if (error) {
            const errEl = document.getElementById('auth-error');
            errEl.textContent = error.message;
            errEl.classList.remove('hidden');
        } else {
            alert('Signup successful! You can now login.');
        }
    });
    
    document.getElementById('auth-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = document.getElementById('auth-email').value;
        const password = document.getElementById('auth-password').value;
        
        const { data, error } = await supabaseClient.auth.signInWithPassword({ email, password });
        if (error) {
            const errEl = document.getElementById('auth-error');
            errEl.textContent = error.message;
            errEl.classList.remove('hidden');
        } else {
            authToken = data.session.access_token;
            document.getElementById('auth-modal').classList.add('hidden');
            document.getElementById('btn-logout').classList.remove('hidden');
            initializeApp();
        }
    });

    document.getElementById('btn-logout').addEventListener('click', async () => {
        await supabaseClient.auth.signOut();
        authToken = null;
        document.getElementById('auth-modal').classList.remove('hidden');
        document.getElementById('btn-logout').classList.add('hidden');
    });
});

// Wrapper for fetch to include Authorization header automatically
const originalFetch = window.fetch;
window.fetch = async function() {
    let [resource, config] = arguments;
    if(resource.startsWith('/api/') && authToken) {
        if(config === undefined) config = {};
        if(config.headers === undefined) config.headers = {};
        config.headers['Authorization'] = `Bearer ${authToken}`;
    }
    return await originalFetch(resource, config);
};


// --- APP LOGIC ---

function initializeApp() {
    initNavigation();
    initWatchlist();
    fetchAccountData();
    initCharting();
    fetchPositionsForPie();
    fetchConfig();
    fetchStrategy();
    fetchOrders();
    fetchActivities();
    initBacktestingView();
}

// 0. NAVIGATION
function initNavigation() {
    const navBtns = document.querySelectorAll('.nav-btn');
    const sections = document.querySelectorAll('.view-section');

    navBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const target = btn.getAttribute('data-target');
            
            navBtns.forEach(b => {
                b.classList.remove('btn-active');
                b.classList.add('text-tv-textMuted');
            });
            btn.classList.add('btn-active');
            btn.classList.remove('text-tv-textMuted');
            
            sections.forEach(s => s.classList.remove('active'));
            document.getElementById(`view-${target}`).classList.add('active');
            
            // Trigger Plotly resize on a short delay to allow elements to transition and calculate sizes correctly
            setTimeout(() => {
                const charts = ['trading-chart', 'main-chart', 'interactive-backtest-chart', 'allocation-chart'];
                charts.forEach(id => {
                    const el = document.getElementById(id);
                    if (el && el.data) {
                        Plotly.Plots.resize(el);
                    }
                });
            }, 100);
        });
    });
}

// 0.5 BACKTEST REPORT LOADER (Interactive)
let allBacktestData = null;
let currentBacktestRegime = null;
let currentBacktestTimeframe = 'ALL';
let activeBacktestModels = new Set([
    'Buy & Hold',
    'Markowitz MVO (Rolling)',
    'Hierarchical LLM+CQL',
    'Hierarchical LLM+TD3BC'
]);

const BT_COLOR_MAP = {
    'Buy & Hold': '#a1a1aa',
    'Markowitz MVO (Rolling)': '#fb923c',
    'Offline CQL Only': '#60a5fa',
    'Offline TD3BC Only': '#c084fc',
    'Hierarchical LLM+CQL': '#4ade80',
    'Hierarchical LLM+TD3BC': '#2dd4bf'
};

async function initBacktestingView() {
    try {
        const res = await fetch('/api/backtest');
        if (!res.ok) throw new Error("Failed to fetch backtest data");
        allBacktestData = await res.json();
        
        const selector = document.getElementById('regime-selector');
        selector.innerHTML = '';
        allBacktestData.regimes.forEach((r, idx) => {
            const opt = document.createElement('option');
            opt.value = r;
            opt.innerText = r;
            selector.appendChild(opt);
        });
        
        currentBacktestRegime = allBacktestData.regimes[0];
        
        selector.addEventListener('change', (e) => {
            currentBacktestRegime = e.target.value;
            renderBacktestRegime();
        });
        
        // Timeframe bindings
        const tfBtns = document.querySelectorAll('.bt-tf-btn');
        tfBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                tfBtns.forEach(b => {
                    b.classList.remove('bg-tv-blue', 'text-white', 'hover:bg-tv-blueHover');
                    b.classList.add('hover:bg-tv-border', 'text-tv-textMuted');
                });
                e.target.classList.remove('hover:bg-tv-border', 'text-tv-textMuted');
                e.target.classList.add('bg-tv-blue', 'text-white', 'hover:bg-tv-blueHover');
                
                const tf = e.target.getAttribute('data-tf');
                currentBacktestTimeframe = tf;
                applyBacktestTimeframe(tf);
            });
        });

        // Model switcher bindings for Backtesting
        const modelBtns = document.querySelectorAll('.bt-model-btn');
        modelBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const modelName = btn.getAttribute('data-model');
                if (activeBacktestModels.has(modelName)) {
                    if (activeBacktestModels.size > 1) { // Keep at least one trace
                        activeBacktestModels.delete(modelName);
                        btn.classList.remove('bg-tv-blue', 'text-white', 'hover:bg-tv-blueHover');
                        btn.classList.add('bg-tv-bg', 'text-tv-textMuted', 'hover:bg-tv-border');
                    }
                } else {
                    activeBacktestModels.add(modelName);
                    btn.classList.remove('bg-tv-bg', 'text-tv-textMuted', 'hover:bg-tv-border');
                    btn.classList.add('bg-tv-blue', 'text-white', 'hover:bg-tv-blueHover');
                }
                renderBacktestRegime();
            });
        });
        
        renderBacktestRegime();
    } catch(e) {
        console.error("Failed to init interactive backtesting", e);
    }
}

function renderBacktestRegime() {
    if (!allBacktestData || !currentBacktestRegime) return;
    
    // 1. Render Chart
    const regimeCurves = allBacktestData.curves[currentBacktestRegime];
    let traces = [];
    
    // Define the specific models to show
    const modelsToShow = [
        'Buy & Hold',
        'Markowitz MVO (Rolling)',
        'Offline CQL Only',
        'Offline TD3BC Only',
        'Hierarchical LLM+CQL',
        'Hierarchical LLM+TD3BC'
    ];
    
    regimeCurves.forEach(curve => {
        if (activeBacktestModels.has(curve.strategy)) {
            traces.push({
                x: curve.dates,
                y: curve.values,
                type: 'scatter',
                mode: 'lines',
                name: curve.strategy,
                line: { 
                    color: BT_COLOR_MAP[curve.strategy] || '#ffffff',
                    width: curve.strategy.startsWith('Hierarchical') ? 3 : 2
                }
            });
        }
    });

    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#D1D4DC', family: 'Inter' },
        xaxis: { gridcolor: '#2A2E39', type: 'date', showline: true, linecolor: '#2A2E39' },
        yaxis: { gridcolor: '#2A2E39', title: 'Normalized Equity (Base 100)', showline: true, linecolor: '#2A2E39', hoverformat: '.2f' },
        margin: { t: 20, r: 20, l: 60, b: 40 },
        showlegend: true,
        legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(19, 23, 34, 0.8)', bordercolor: '#2A2E39', borderwidth: 1 },
        autosize: true,
        hovermode: 'x unified'
    };

    Plotly.newPlot('interactive-backtest-chart', traces, layout, {
        displayModeBar: true, 
        responsive: true,
        scrollZoom: true,
        doubleClick: false
    });
    
    // Attach double click event listener to chart to reset back to current timeframe (instead of full range)
    const chartDiv = document.getElementById('interactive-backtest-chart');
    chartDiv.on('plotly_doubleclick', () => {
        applyBacktestTimeframe(currentBacktestTimeframe);
        return false; // prevent default zoom reset to ALL
    });

    // Reset timeframe selection to ALL visually and logically upon regime change
    currentBacktestTimeframe = 'ALL';
    const tfBtns = document.querySelectorAll('.bt-tf-btn');
    tfBtns.forEach(btn => {
        const tf = btn.getAttribute('data-tf');
        if (tf === 'ALL') {
            btn.classList.remove('hover:bg-tv-border', 'text-tv-textMuted');
            btn.classList.add('bg-tv-blue', 'text-white', 'hover:bg-tv-blueHover');
        } else {
            btn.classList.add('hover:bg-tv-border', 'text-tv-textMuted');
            btn.classList.remove('bg-tv-blue', 'text-white', 'hover:bg-tv-blueHover');
        }
    });
    applyBacktestTimeframe('ALL');
    
    // 2. Render Table
    const tbody = document.getElementById('interactive-backtest-table');
    tbody.innerHTML = '';
    
    // Filter metrics for current regime
    const regimeMetrics = allBacktestData.metrics.filter(m => m.Regime.includes(currentBacktestRegime));
    
    modelsToShow.forEach(modelName => {
        const rowData = regimeMetrics.find(m => m.Strategy === modelName);
        if (rowData) {
            const isActive = activeBacktestModels.has(modelName);
            const isMain = modelName.startsWith('Hierarchical');
            const retClass = parseFloat(rowData['Total Return']) >= 0 ? 'text-tv-green' : 'text-tv-red';
            
            tbody.innerHTML += `
                <tr class="hover:bg-tv-panel transition-colors ${isActive ? '' : 'opacity-40'}">
                    <td class="px-4 py-3 border-l-4 font-sans ${isMain ? 'font-bold border-tv-blue text-white' : 'border-transparent text-tv-textMuted'}">${modelName}</td>
                    <td class="px-4 py-3 text-right ${retClass}">${(parseFloat(rowData['Total Return'])*100).toFixed(2)}%</td>
                    <td class="px-4 py-3 text-right">${parseFloat(rowData['Sharpe']).toFixed(2)}</td>
                    <td class="px-4 py-3 text-right text-tv-red">${(parseFloat(rowData['Max Drawdown'])*100).toFixed(2)}%</td>
                    <td class="px-4 py-3 text-right">${(parseFloat(rowData['Ann. Vol'])*100).toFixed(2)}%</td>
                </tr>
            `;
        }
    });
}

function applyBacktestTimeframe(tf) {
    const chartDiv = document.getElementById('interactive-backtest-chart');
    if (!chartDiv.data) return;
    
    const dates = chartDiv.data[0].x;
    if (!dates || dates.length === 0) return;
    
    const end = new Date(dates[dates.length - 1]);
    let start = new Date(dates[0]);
    
    switch(tf) {
        case '1D': start = new Date(end); start.setDate(end.getDate() - 1); break;
        case '5D': start = new Date(end); start.setDate(end.getDate() - 5); break;
        case '1W': start = new Date(end); start.setDate(end.getDate() - 7); break;
        case '1M': start = new Date(end); start.setMonth(end.getMonth() - 1); break;
        case '3M': start = new Date(end); start.setMonth(end.getMonth() - 3); break;
        case '6M': start = new Date(end); start.setMonth(end.getMonth() - 6); break;
        case '1Y': start = new Date(end); start.setFullYear(end.getFullYear() - 1); break;
        case 'ALL': start = new Date(dates[0]); break;
    }
    
    Plotly.relayout(chartDiv, {
        'xaxis.range': [start.toISOString().split('T')[0], end.toISOString().split('T')[0]]
    });
}

// 1. WATCHLIST (Live Simulation)
let currentTickers = [];

function initWatchlist() {
    const container = document.getElementById('watchlist-container');
    
    async function updateWatchlist() {
        try {
            const res = await fetch('/api/market-data-live');
            const data = await res.json();
            if (data.error || !data.data) return;
            
            let newTickers = data.data;
            if (currentTickers.length === 0) {
                // Initial render
                container.innerHTML = newTickers.map((t, i) => `
                    <div class="flex px-3 py-2 border-b border-tv-border hover:bg-tv-panel cursor-pointer text-xs items-center" id="ticker-row-${t.sym}">
                        <div class="flex-1 font-bold text-white">${t.sym}</div>
                        <div class="w-16 text-right font-mono" id="ticker-price-${t.sym}">${t.price.toFixed(2)}</div>
                        <div class="w-16 text-right font-mono ${t.chg >= 0 ? 'text-tv-green' : 'text-tv-red'}" id="ticker-chg-${t.sym}">
                            ${t.chg >= 0 ? '+' : ''}${t.chg.toFixed(2)}%
                        </div>
                    </div>
                `).join('');
            } else {
                // Update specific rows and trigger flash
                newTickers.forEach(t => {
                    const oldT = currentTickers.find(x => x.sym === t.sym);
                    if (oldT && oldT.price !== t.price) {
                        const row = document.getElementById(`ticker-row-${t.sym}`);
                        const pEl = document.getElementById(`ticker-price-${t.sym}`);
                        const cEl = document.getElementById(`ticker-chg-${t.sym}`);
                        if (row && pEl && cEl) {
                            pEl.innerText = t.price.toFixed(2);
                            cEl.innerText = `${t.chg >= 0 ? '+' : ''}${t.chg.toFixed(2)}%`;
                            cEl.className = `w-16 text-right font-mono ${t.chg >= 0 ? 'text-tv-green' : 'text-tv-red'}`;
                            
                            row.classList.remove('flash-up', 'flash-down');
                            void row.offsetWidth; // trigger reflow
                            row.classList.add(t.price > oldT.price ? 'flash-up' : 'flash-down');
                        }
                    }
                });
            }
            currentTickers = newTickers;
        } catch (e) {
            console.error("Failed to fetch live watchlist data:", e);
        }
    }
    
    updateWatchlist();
    setInterval(updateWatchlist, 5000);
}

// 2. ACCOUNT DATA
const formatCurrency = (val) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(val);

async function fetchAccountData() {
    try {
        const res = await fetch('/api/alpaca/account');
        const data = await res.json();
        
        const dot = document.getElementById('status-dot');
        const status = document.getElementById('alpaca-status');

        if (data.error) {
            dot.className = "w-2 h-2 rounded-full bg-tv-textMuted";
            status.innerHTML = `<div class="w-2 h-2 rounded-full bg-tv-textMuted" id="status-dot"></div> <span class="text-tv-textMuted">Alpaca: Unlinked</span>`;
            return;
        }
        
        dot.className = "w-2 h-2 rounded-full bg-tv-green";
        status.innerHTML = `<div class="w-2 h-2 rounded-full bg-tv-green" id="status-dot"></div> Connected`;
        
        document.getElementById('metric-cash').innerText = formatCurrency(data.cash);
        document.getElementById('metric-portfolio').innerText = formatCurrency(data.portfolio_value);
        document.getElementById('metric-buying-power').innerText = formatCurrency(data.buying_power);
    } catch (e) {
        console.error("Account fetch failed:", e);
        document.getElementById('status-dot').className = "w-2 h-2 rounded-full bg-tv-textMuted";
        document.getElementById('alpaca-status').innerHTML = `<div class="w-2 h-2 rounded-full bg-tv-textMuted" id="status-dot"></div> <span class="text-tv-textMuted">Alpaca: Unlinked</span>`;
        document.getElementById('metric-cash').innerText = "---";
        document.getElementById('metric-portfolio').innerText = "---";
        document.getElementById('metric-buying-power').innerText = "---";
    }
}

// 3. PIE CHART ALLOCATION
async function fetchPositionsForPie() {
    let labels = [];
    let values = [];
    try {
        const res = await fetch('/api/alpaca/positions');
        const data = await res.json();
        if (!data.error && Object.keys(data).length > 0) {
            for (const [sym, pos] of Object.entries(data)) {
                labels.push(sym);
                values.push(pos.market_value);
            }
        } else {
            throw new Error("No data");
        }
    } catch (e) {
        console.error("Pie chart fetch failed:", e);
        document.getElementById('allocation-chart').innerHTML = "<div class='text-center mt-10 text-tv-textMuted'>No positions loaded</div>";
        return;
    }

    const trace = {
        labels: labels,
        values: values,
        type: 'pie',
        hole: 0.5,
        marker: {
            colors: ['#2962FF', '#089981', '#F23645', '#E2B93B', '#9C27B0', '#787B86']
        },
        textinfo: 'label+percent',
        textposition: 'outside',
        hoverinfo: 'label+value+percent'
    };

    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 10, b: 10, l: 10, r: 10 },
        showlegend: false,
        font: { color: '#D1D4DC', family: 'Inter' }
    };

    Plotly.newPlot('allocation-chart', [trace], layout, {displayModeBar: false});
}

// 4. CHARTING & MODEL SWITCHER
let currentBacktestData = null;
let currentMetricsData = null;

const FORMAT_MAP = {
    'BuyAndHold': 'Buy & Hold',
    'MVO': 'Markowitz MVO (Rolling)',
    'RL_Model': 'Offline CQL Only',
    'RL_LLM': 'Hierarchical LLM+CQL'
};

async function initCharting() {
    try {
        const res = await fetch('/api/backtest/data');
        const data = await res.json();
        if (!data.error && data.dates) {
            currentBacktestData = data;
        }
        
        const mRes = await fetch('/api/backtest');
        const mData = await mRes.json();
        if (!mData.error && mData.metrics) {
            currentMetricsData = mData.metrics;
        }
    } catch(e) { console.error("Backtest fetch failed."); }

    let activeModels = new Set(['RL_LLM']);

    function drawChart() {
        let traces = [];

        // Draw Equity Curves
        if (currentBacktestData) {
            activeModels.forEach(modelKey => {
                if (currentBacktestData[modelKey]) {
                    traces.push({
                        x: currentBacktestData.dates,
                        y: currentBacktestData[modelKey],
                        type: 'scatter',
                        mode: 'lines',
                        name: FORMAT_MAP[modelKey],
                        line: { width: modelKey === 'RL_LLM' ? 3 : 2 }
                    });
                }
            });
        }
        
        if (traces.length === 0) {
            return;
        }

        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#787B86', family: 'Inter' },
            xaxis: { gridcolor: '#2A2E39', rangeslider: {visible: false} },
            yaxis: { gridcolor: '#2A2E39', title: 'Equity ($)' },
            margin: { t: 20, r: 50, l: 50, b: 30 },
            showlegend: true,
            legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(30, 34, 45, 0.8)' },
            autosize: true
        };

        Plotly.newPlot('main-chart', traces, layout, {displayModeBar: false, responsive: true});

        // Update Bottom Panel Metrics
        if (currentMetricsData && activeModels.size > 0) {
            // Find the most recently added model or just the last one
            const modelKey = Array.from(activeModels).pop();
            const stratName = FORMAT_MAP[modelKey];
            const metricsObj = currentMetricsData.find(x => x.Strategy === stratName);
            if (metricsObj) {
                document.getElementById('active-model-label').innerText = stratName;
                document.getElementById('metric-return').innerText = `${(metricsObj['Total Return']*100).toFixed(2)}%`;
                document.getElementById('metric-sharpe').innerText = metricsObj.Sharpe.toFixed(2);
                document.getElementById('metric-drawdown').innerText = `${(metricsObj['Max Drawdown']*100).toFixed(2)}%`;
                document.getElementById('metric-vol').innerText = `${(metricsObj['Ann. Vol']*100).toFixed(2)}%`;
                
                document.getElementById('metric-return').className = `py-2 font-mono text-right ${metricsObj['Total Return'] < 0 ? 'text-tv-red' : 'text-tv-green'}`;
                document.getElementById('metric-drawdown').className = `py-2 font-mono text-right text-tv-red`;
            }
        }
    }

    // Event listeners for model switcher
    const buttons = document.querySelectorAll('#model-switcher button');
    buttons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const modelKey = e.target.getAttribute('data-model');
            
            if (activeModels.has(modelKey)) {
                if (activeModels.size > 1) { // Always keep at least 1 graph
                    activeModels.delete(modelKey);
                    e.target.classList.remove('bg-tv-blue', 'hover:bg-tv-blueHover');
                    e.target.classList.add('bg-tv-bg', 'hover:bg-tv-border');
                }
            } else {
                activeModels.add(modelKey);
                e.target.classList.remove('bg-tv-bg', 'hover:bg-tv-border');
                e.target.classList.add('bg-tv-blue', 'hover:bg-tv-blueHover');
            }
            
            drawChart();
        });
    });

    // Draw initial
    drawChart();
}

// 5. TRADING VIEW (Strategy, Orders, Activities)

let currentSymbol = 'SPY';
let currentInterval = '1d';
let isBuySide = true;

async function fetchStrategy() {
    try {
        const res = await fetch('/api/strategy');
        const data = await res.json();
        
        document.getElementById('ai-regime-display').innerText = `[ ${data.regime.toUpperCase()} ]`;
        document.getElementById('ai-regime-display').className = `font-bold cursor-help ${data.regime === 'risk-off' ? 'text-tv-red' : 'text-tv-green'}`;
        
        if (data.dialogue) {
            let logHtml = '';
            data.dialogue.forEach(d => {
                logHtml += `<div class="mb-2"><span class="font-bold text-white">${d.agent}:</span> ${d.text}</div>`;
            });
            document.getElementById('ai-reasoning-content').innerHTML = logHtml;
        }
        
        if (data.evidence) {
            document.getElementById('ai-evidence-content').innerText = data.evidence;
        } else {
            document.getElementById('ai-evidence-content').innerText = "Evidence context derived from RAG macro corpus.";
        }
        
        // Populate AI recommendation based on current symbol and caps
        let recText = "HOLD";
        let recColor = "text-tv-textMuted";
        if (data.sector_caps) {
            const capObj = data.sector_caps.find(x => x.ticker === currentSymbol);
            if (capObj && capObj.cap > 0.1) {
                recText = `ACCUMULATE (Cap: ${(capObj.cap*100).toFixed(0)}%)`;
                recColor = "text-tv-green";
            } else if (capObj && capObj.cap <= 0.1) {
                recText = "REDUCE / AVOID";
                recColor = "text-tv-red";
            }
        }
        document.getElementById('ai-rec-action').innerText = recText;
        document.getElementById('ai-rec-action').className = `font-bold ${recColor}`;
        
    } catch (e) {
        console.error("Strategy fetch failed:", e);
    }
}

document.getElementById('btn-plan-strategy').addEventListener('click', async (e) => {
    const btn = e.target;
    btn.disabled = true;
    btn.innerHTML = "EXECUTING...";
    document.getElementById('rebalance-status').innerText = "";
    
    try {
        const res = await fetch('/api/rebalance', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({live: false}) // Demo mode
        });
        const data = await res.json();
        if (data.status === 'started') {
            document.getElementById('rebalance-status').innerText = "> TASK_STARTED";
            document.getElementById('rebalance-status').className = "mt-1 text-[10px] text-center text-tv-green";
        } else {
            throw new Error(data.detail || "Failed");
        }
    } catch(err) {
        document.getElementById('rebalance-status').innerText = "> ERROR: " + err.message;
        document.getElementById('rebalance-status').className = "mt-1 text-[10px] text-center text-tv-red";
    } finally {
        setTimeout(() => {
            btn.innerHTML = "> RE-EVALUATE PORTFOLIO";
            btn.disabled = false;
        }, 3000);
    }
});

// Interactive Charting Data
async function fetchChartData(symbol, interval) {
    document.getElementById('chart-loader').classList.remove('hidden');
    try {
        const res = await fetch(`/api/market-data-history?ticker=${symbol}&interval=${interval}`);
        const data = await res.json();
        if (data.error || !data.data || data.data.length === 0) {
            console.error("No data");
            return;
        }
        
        const x = data.data.map(d => {
            if (typeof d.time === 'number') {
                return new Date(d.time * 1000); // Plotly handles JS Date objects well
            }
            return d.time;
        });
        const open = data.data.map(d => d.open);
        const high = data.data.map(d => d.high);
        const low = data.data.map(d => d.low);
        const close = data.data.map(d => d.close);
        
        const latestPrice = close[close.length - 1];
        document.getElementById('live-price-display').innerText = `${symbol} $${latestPrice.toFixed(2)}`;
        
        const trace = {
            x: x,
            open: open,
            high: high,
            low: low,
            close: close,
            type: 'candlestick',
            name: symbol,
            increasing: {line: {color: '#089981'}},
            decreasing: {line: {color: '#F23645'}}
        };
        
        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#787B86', family: 'Inter' },
            xaxis: { 
                gridcolor: '#2A2E39', 
                rangeslider: {visible: false},
                type: 'date',
                rangebreaks: [
                    { bounds: ["sat", "mon"] }
                ]
            },
            yaxis: { gridcolor: '#2A2E39', title: 'Price ($)' },
            margin: { t: 10, r: 50, l: 50, b: 30 },
            dragmode: 'pan',
            newshape: { line: { color: '#2962FF', width: 2 } },
            autosize: true
        };

        const config = {
            displayModeBar: true,
            modeBarButtonsToAdd: ['drawline', 'drawrect', 'eraseshape'],
            displaylogo: false,
            responsive: true,
            scrollZoom: true
        };

        Plotly.newPlot('trading-chart', [trace], layout, config);
        
    } catch(e) {
        console.error(e);
    } finally {
        document.getElementById('chart-loader').classList.add('hidden');
    }
}

document.getElementById('symbol-search').addEventListener('change', (e) => {
    currentSymbol = e.target.value.toUpperCase();
    e.target.value = currentSymbol;
    fetchChartData(currentSymbol, currentInterval);
    fetchStrategy(); // Refresh AI recommendation for new symbol
});

const tfButtons = document.querySelectorAll('#timeframe-switcher button');
tfButtons.forEach(btn => {
    btn.addEventListener('click', (e) => {
        tfButtons.forEach(b => {
            b.classList.remove('bg-tv-blue', 'hover:bg-tv-blueHover');
            b.classList.add('bg-tv-bg', 'hover:bg-tv-border');
        });
        e.target.classList.remove('bg-tv-bg', 'hover:bg-tv-border');
        e.target.classList.add('bg-tv-blue', 'hover:bg-tv-blueHover');
        
        currentInterval = e.target.getAttribute('data-tf');
        fetchChartData(currentSymbol, currentInterval);
    });
});

// Initial Chart load triggered inside navigation or here
setTimeout(() => fetchChartData('SPY', '1d'), 500);

// Drawing Tools Bindings
document.getElementById('btn-draw-line').addEventListener('click', () => {
    Plotly.relayout('trading-chart', {'dragmode': 'drawline'});
});
document.getElementById('btn-draw-rect').addEventListener('click', () => {
    Plotly.relayout('trading-chart', {'dragmode': 'drawrect'});
});
document.getElementById('btn-erase').addEventListener('click', () => {
    Plotly.relayout('trading-chart', {'dragmode': 'pan', 'shapes': []});
});

// Trade Panel Toggle
document.getElementById('btn-toggle-trade').addEventListener('click', () => {
    const panel = document.getElementById('trade-panel');
    if (panel.classList.contains('hidden')) {
        panel.classList.remove('hidden');
    } else {
        panel.classList.add('hidden');
    }
});

// Order Entry UI
document.getElementById('btn-side-buy').addEventListener('click', () => {
    isBuySide = true;
    document.getElementById('btn-side-buy').classList.replace('bg-tv-bg', 'bg-tv-green');
    document.getElementById('btn-side-buy').classList.replace('text-tv-textMuted', 'text-white');
    document.getElementById('btn-side-sell').classList.replace('bg-tv-red', 'bg-tv-bg');
    document.getElementById('btn-side-sell').classList.replace('text-white', 'text-tv-textMuted');
    
    const submitBtn = document.getElementById('btn-submit-order');
    submitBtn.classList.replace('bg-tv-red', 'bg-tv-green');
    submitBtn.classList.replace('hover:bg-[#9c1825]', 'hover:bg-[#067a67]');
    submitBtn.classList.replace('shadow-tv-red/20', 'shadow-tv-green/20');
});

document.getElementById('btn-side-sell').addEventListener('click', () => {
    isBuySide = false;
    document.getElementById('btn-side-sell').classList.replace('bg-tv-bg', 'bg-tv-red');
    document.getElementById('btn-side-sell').classList.replace('text-tv-textMuted', 'text-white');
    document.getElementById('btn-side-buy').classList.replace('bg-tv-green', 'bg-tv-bg');
    document.getElementById('btn-side-buy').classList.replace('text-white', 'text-tv-textMuted');
    
    const submitBtn = document.getElementById('btn-submit-order');
    submitBtn.classList.replace('bg-tv-green', 'bg-tv-red');
    submitBtn.classList.replace('hover:bg-[#067a67]', 'hover:bg-[#9c1825]');
    submitBtn.classList.replace('shadow-tv-green/20', 'shadow-tv-red/20');
});

document.getElementById('order-type').addEventListener('change', (e) => {
    if (e.target.value === 'limit') {
        document.getElementById('limit-price-row').classList.remove('hidden');
        document.getElementById('limit-price-row').classList.add('flex');
    } else {
        document.getElementById('limit-price-row').classList.add('hidden');
        document.getElementById('limit-price-row').classList.remove('flex');
    }
});

document.getElementById('btn-submit-order').addEventListener('click', async () => {
    const qty = parseFloat(document.getElementById('order-qty').value);
    const type = document.getElementById('order-type').value;
    let limitPrice = null;
    if (type === 'limit') {
        limitPrice = parseFloat(document.getElementById('order-limit').value);
    }
    
    const payload = {
        symbol: currentSymbol,
        qty: qty,
        side: isBuySide ? 'buy' : 'sell',
        type: type,
        limit_price: limitPrice,
        time_in_force: 'gtc'
    };
    
    const statusMsg = document.getElementById('order-status-msg');
    statusMsg.innerText = "Submitting...";
    statusMsg.className = "text-xs text-center mt-1 text-tv-textMuted block";
    
    try {
        const res = await fetch('/api/alpaca/trade', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.status === 'success') {
            statusMsg.innerText = `Success! ID: ${data.order_id.substring(0,8)}...`;
            statusMsg.className = "text-xs text-center mt-1 text-tv-green block";
            setTimeout(() => { fetchOrders(); fetchActivities(); }, 1000);
        } else {
            throw new Error(data.error || "Failed to submit");
        }
    } catch(e) {
        statusMsg.innerText = `Error: ${e.message}`;
        statusMsg.className = "text-xs text-center mt-1 text-tv-red block";
    }
});

async function fetchOrders() {
    try {
        const res = await fetch('/api/alpaca/orders');
        const data = await res.json();
        // Since we unified activities and orders in the new UI into a small list,
        // we will combine them in the fetchActivities block, or just display activities.
        // For simplicity, we just use fetchActivities to populate the Recent Activity tab.
    } catch (e) { console.error(e); }
}

async function fetchActivities() {
    try {
        const res = await fetch('/api/alpaca/activities');
        const data = await res.json();
        const tbody = document.getElementById('activities-table-body');
        tbody.innerHTML = '';
        if (data.error || !data.length) {
            tbody.innerHTML = `<div class="text-tv-textMuted text-center p-2">No recent activity.</div>`;
            return;
        }
        
        data.slice(0, 15).forEach(a => {
            const isBuy = a.side === 'buy';
            tbody.innerHTML += `
                <div class="flex justify-between items-center bg-tv-bg p-2 rounded">
                    <div>
                        <div class="font-bold text-white">${a.symbol} <span class="${isBuy ? 'text-tv-green' : 'text-tv-red'}">${isBuy ? 'BUY' : 'SELL'}</span></div>
                        <div class="text-[10px] text-tv-textMuted">${a.date ? a.date.substring(0,16).replace('T', ' ') : ''}</div>
                    </div>
                    <div class="text-right">
                        <div class="font-mono text-white">${a.qty || ''} @ ${a.price ? formatCurrency(a.price) : 'MKT'}</div>
                        <div class="text-[10px] text-tv-textMuted">${a.activity_type}</div>
                    </div>
                </div>
            `;
        });
    } catch (e) { console.error(e); }
}

// 6. SETTINGS VIEW

async function fetchConfig() {
    try {
        const res = await fetch('/api/config');
        const data = await res.json();
        // Just verify it works, we don't display keys for security
    } catch (e) { console.error(e); }
}

document.getElementById('btn-save-config').addEventListener('click', async () => {
    const openai = document.getElementById('cfg-openai').value;
    const alpacaKey = document.getElementById('cfg-alpaca-key').value;
    const alpacaSec = document.getElementById('cfg-alpaca-secret').value;
    
    const payload = {};
    if (openai) payload.openai_key = openai;
    if (alpacaKey) payload.alpaca_key = alpacaKey;
    if (alpacaSec) payload.alpaca_secret = alpacaSec;
    
    try {
        const res = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        const status = document.getElementById('config-status');
        if (data.status === 'success') {
            status.innerText = "> CREDENTIALS_SAVED_SUCCESSFULLY";
            status.className = "mt-2 text-xs text-center text-tv-green";
            setTimeout(() => fetchAccountData(), 1000); // refresh account status
        } else {
            status.innerText = "> ERROR_SAVING_CREDENTIALS";
            status.className = "mt-2 text-xs text-center text-tv-red";
        }
    } catch (e) {
        console.error(e);
    }
});
