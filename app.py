import os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")

from flask import Flask, request, jsonify, render_template_string
import financial_advisor_combined_ml as fa

app = Flask(__name__)
loaded = fa.load_models()

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Financial Advisor AI</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}
  .header{background:linear-gradient(135deg,#1e3a5f,#0ea5e9);padding:24px 32px}
  .header h1{font-size:1.8rem;font-weight:700}
  .header p{color:#bae6fd;margin-top:4px;font-size:0.85rem}
  .container{max-width:1100px;margin:28px auto;padding:0 16px}
  .card{background:#1e293b;border-radius:12px;padding:22px;margin-bottom:20px;border:1px solid #334155}
  .card h2{font-size:0.85rem;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-bottom:14px}
  .grid-2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  .grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
  .grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
  label{display:block;font-size:0.78rem;color:#94a3b8;margin-bottom:3px}
  input,select,textarea{width:100%;padding:8px 11px;background:#0f172a;border:1px solid #334155;border-radius:8px;color:#e2e8f0;font-size:0.88rem}
  input:focus,select:focus,textarea:focus{outline:none;border-color:#0ea5e9}
  textarea{resize:vertical;min-height:60px}
  .btn{width:100%;padding:13px;background:linear-gradient(135deg,#0ea5e9,#6366f1);border:none;border-radius:10px;color:#fff;font-size:1rem;font-weight:600;cursor:pointer;margin-top:8px}
  .btn:hover{opacity:.9} .btn:disabled{opacity:.5;cursor:not-allowed}
  #results{display:none}
  .metric{background:#0f172a;border-radius:10px;padding:14px;text-align:center;border:1px solid #334155}
  .metric .val{font-size:1.35rem;font-weight:700;color:#0ea5e9}
  .metric .lbl{font-size:0.72rem;color:#64748b;margin-top:3px}
  .score-bar{height:10px;background:#1e293b;border-radius:5px;overflow:hidden;margin-top:6px}
  .score-fill{height:100%;border-radius:5px;transition:width 1s ease}
  .section{background:#0f172a;border-radius:10px;padding:16px;border-left:3px solid #0ea5e9;margin-bottom:10px}
  .section h3{font-size:0.88rem;color:#0ea5e9;margin-bottom:8px;font-weight:600}
  .section p{font-size:0.8rem;color:#94a3b8;line-height:1.6;margin-bottom:8px}
  .section ul{list-style:none}
  .section ul li{font-size:0.8rem;color:#cbd5e1;padding:2px 0}
  .section ul li::before{content:"✓ ";color:#34d399}
  .week-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}
  .week{background:#1e293b;border-radius:8px;padding:10px}
  .week h4{font-size:0.78rem;color:#f59e0b;margin-bottom:6px}
  .week ul li::before{content:"→ ";color:#f59e0b}
  .pot-bar{display:flex;align-items:center;gap:8px;margin:3px 0}
  .pot-bar .lbl{font-size:0.78rem;color:#94a3b8;width:100px}
  .pot-bar .bar{flex:1;height:7px;background:#1e293b;border-radius:4px;overflow:hidden}
  .pot-bar .bar-fill{height:100%;background:linear-gradient(90deg,#0ea5e9,#6366f1);border-radius:4px}
  .pot-bar .amt{font-size:0.78rem;color:#e2e8f0;width:75px;text-align:right}
  .risk-selector{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:4px}
  .risk-opt{background:#0f172a;border:2px solid #334155;border-radius:10px;padding:10px 8px;text-align:center;cursor:pointer;transition:all .2s}
  .risk-opt:hover{border-color:#0ea5e9}
  .risk-opt .risk-icon{display:block;font-size:1.4rem;margin-bottom:3px}
  .risk-opt .risk-name{display:block;font-size:0.8rem;font-weight:700;color:#e2e8f0}
  .risk-opt .risk-desc{display:block;font-size:0.68rem;color:#64748b;margin-top:2px}
  .risk-opt.risk-active.low{border-color:#34d399;background:#022c22}
  .risk-opt.risk-active.medium{border-color:#f59e0b;background:#2d1a00}
  .risk-opt.risk-active.high{border-color:#f87171;background:#2d0a0a}
  .risk-advice{border-radius:12px;padding:18px;margin-bottom:10px}
  .risk-advice.low{background:#022c22;border:1px solid #34d399}
  .risk-advice.medium{background:#2d1a00;border:1px solid #f59e0b}
  .risk-advice.high{background:#2d0a0a;border:1px solid #f87171}
  .risk-advice h3{font-size:1rem;font-weight:700;margin-bottom:10px}
  .risk-advice.low h3{color:#34d399}
  .risk-advice.medium h3{color:#f59e0b}
  .risk-advice.high h3{color:#f87171}
  .risk-advice .alloc-row{display:flex;align-items:center;gap:10px;margin:5px 0}
  .risk-advice .alloc-label{font-size:0.78rem;color:#94a3b8;width:70px}
  .risk-advice .alloc-bar{flex:1;height:8px;background:#1e293b;border-radius:4px;overflow:hidden}
  .risk-advice .alloc-fill{height:100%;border-radius:4px}
  .risk-advice .alloc-pct{font-size:0.78rem;color:#e2e8f0;width:35px;text-align:right}
  .risk-advice ul{list-style:none;margin-top:10px}
  .risk-advice ul li{font-size:0.8rem;color:#cbd5e1;padding:3px 0}
  .risk-advice ul li::before{content:"✓ ";color:#34d399}
  .badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:0.72rem;font-weight:600}
  .badge-good{background:#064e3b;color:#34d399}
  .badge-fair{background:#451a03;color:#fb923c}
  .badge-bad{background:#450a0a;color:#f87171}
  .ml-tag{background:#1e3a5f;color:#7dd3fc;padding:3px 8px;border-radius:4px;font-size:0.72rem;margin-right:4px}
  .error{background:#450a0a;border:1px solid #f87171;color:#f87171;padding:12px;border-radius:8px;margin-top:10px}
  @media(max-width:640px){.grid-2,.grid-3,.grid-4,.week-grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="header">
  <h1>💰 Financial Advisor AI</h1>
  <p>6-Model ML Pipeline · M1 GradientBoosting R²=0.83 · M2 R²=0.94 · M3 R²=0.94 · M4 RandomForest Acc=100% · M5 R²=0.95 · M6 KMeans</p>
</div>
<div class="container">
  <form id="form">

    <div class="card">
      <h2>👤 Personal Profile</h2>
      <div class="grid-3">
        <div><label>Full Name</label><input name="name" placeholder="e.g. Rahul Sharma" required></div>
        <div><label>Monthly Income (₹)</label><input type="number" name="income" placeholder="e.g. 80000" required min="1"></div>
        <div><label>Age</label><input type="number" name="age" placeholder="e.g. 28" required min="18" max="80"></div>
      </div>
      <div class="grid-3" style="margin-top:10px">
        <div><label>Dependents</label><input type="number" name="dependents" placeholder="e.g. 2" required min="0"></div>
        <div><label>Occupation</label>
          <select name="occupation">
            <option value="" disabled selected>Select occupation</option>
            <option>Professional</option>
            <option>Self_Employed</option>
            <option>Student</option>
            <option>Retired</option>
          </select>
        </div>
        <div><label>City Tier</label>
          <select name="city_tier">
            <option value="" disabled selected>Select city tier</option>
            <option>Tier_1</option><option>Tier_2</option><option>Tier_3</option>
          </select>
        </div>
      </div>
      <div class="grid-3" style="margin-top:10px">
        <div><label>Location / City</label><input name="location" placeholder="e.g. Pune" required></div>
        <div><label>Business / Profession Type</label><input name="business_type" placeholder="e.g. Dairy Farming"></div>
        <div><label>Current Monthly Savings (₹)</label><input type="number" name="monthly_savings" placeholder="e.g. 10000" required min="0"></div>
      </div>
      <div class="grid-2" style="margin-top:10px">
        <div><label>Financial Goal</label><input name="goal" placeholder="e.g. buy a house worth 50 lakhs in 5 years" required></div>
        <div>
          <label>Risk Appetite</label>
          <input type="hidden" name="risk_level" id="risk_level_input" value="Medium Risk">
          <div class="risk-selector">
            <div class="risk-opt" data-val="Low Risk" onclick="selectRisk(this)">
              <span class="risk-icon">🛡️</span>
              <span class="risk-name">Low Risk</span>
              <span class="risk-desc">Safe &amp; Stable</span>
            </div>
            <div class="risk-opt risk-active" data-val="Medium Risk" onclick="selectRisk(this)">
              <span class="risk-icon">⚖️</span>
              <span class="risk-name">Medium Risk</span>
              <span class="risk-desc">Balanced Growth</span>
            </div>
            <div class="risk-opt" data-val="High Risk" onclick="selectRisk(this)">
              <span class="risk-icon">🚀</span>
              <span class="risk-name">High Risk</span>
              <span class="risk-desc">Aggressive Returns</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <button type="submit" class="btn" id="submitBtn">🔍 Generate Financial Roadmap</button>
    <div id="errorBox"></div>
  </form>

  <div id="results">

    <!-- ML Outputs -->
    <div class="card">
      <h2>🤖 6-Model ML Outputs</h2>
      <div class="grid-4" id="ml-metrics"></div>
      <div style="margin-top:10px" id="ml-tags"></div>
    </div>

    <!-- Sections 1-9 -->
    <div id="sections-container"></div>

    <!-- Section 10: 30-Day Plan -->
    <div class="card" id="action-plan" style="display:none">
      <h2>✅ Your 30-Day Action Plan</h2>
      <div class="week-grid" id="week-grid"></div>
    </div>

  </div>
</div>

<script>
const fmt = n => '₹' + Math.round(n).toLocaleString('en-IN');

document.getElementById('form').addEventListener('submit', async e => {
  e.preventDefault();
  const btn = document.getElementById('submitBtn');
  btn.disabled = true; btn.textContent = '⏳ Running 6 ML Models...';
  document.getElementById('errorBox').innerHTML = '';

  const fd = new FormData(e.target);
  const body = {};
  fd.forEach((v, k) => {
    const n = Number(v);
    body[k] = (!isNaN(n) && v !== '' && k !== 'name' && k !== 'location' && k !== 'business_type' && k !== 'goal' && k !== 'occupation' && k !== 'city_tier' && k !== 'risk_level') ? n : v;
  });

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    const d = await res.json();
    if (d.error) throw new Error(d.error);

    const ml = d.ml_outputs;

    // ML metrics grid
    const sc = ml.M5_health_score;
    const scoreColor = sc >= 65 ? '#34d399' : sc >= 40 ? '#fb923c' : '#f87171';
    document.getElementById('ml-metrics').innerHTML = `
      <div class="metric"><div class="val">${ml.M1_savings_pct}%</div><div class="lbl">M1 · Recommended Savings %</div></div>
      <div class="metric"><div class="val">${fmt(ml.M2_savings_amt)}/mo</div><div class="lbl">M2 · Monthly Savings Target</div></div>
      <div class="metric"><div class="val">${fmt(ml.M3_disposable)}/mo</div><div class="lbl">M3 · Disposable Income</div></div>
      <div class="metric">
        <div class="val" style="color:${scoreColor}">${sc}/100</div>
        <div class="lbl">M5 · Financial Health Score</div>
        <div class="score-bar"><div class="score-fill" style="width:${sc}%;background:${scoreColor}"></div></div>
      </div>`;

    document.getElementById('ml-tags').innerHTML =
      `<span class="ml-tag">M4 Budget: ${ml.M4_budget_profile}</span>` +
      `<span class="ml-tag">M6 ${ml.M6_invest_cluster}</span>`;

    // Sections 1–9
    const rm = d.roadmap;
    const sectionKeys = [
      '1_FINANCIAL_HEALTH_SNAPSHOT','2_PERSONALIZED_GOAL_STRATEGY',
      '3_SMART_BUDGETING_PLAN','4_INVESTMENT_ROADMAP',
      '5_EMERGENCY_AND_SAVINGS','6_RISK_PROTECTION_PLAN',
      '7_TAX_OPTIMIZATION','8_BUSINESS_FINANCE_GUIDANCE',
      '9_LOCATION_SPECIFIC_BENEFITS'
    ];

    document.getElementById('sections-container').innerHTML = sectionKeys.map(k => {
      const s = rm[k]; if (!s) return '';
      const bullets = (s.bullets || []).map(b => `<li>${b}</li>`).join('');
      const tips = (s.tips || []).map(t => `<li>${t}</li>`).join('');
      return `<div class="card"><div class="section">
        <h3>${s.title || k}</h3>
        <p>${s.narrative || ''}</p>
        ${bullets ? `<ul>${bullets}</ul>` : ''}
        ${tips ? `<div style="margin-top:8px;color:#f59e0b;font-size:0.78rem;font-weight:600">Tips:</div><ul>${tips}</ul>` : ''}
      </div></div>`;
    }).join('');

    // Risk-Based Advice Section
    const riskVal = body.risk_level;
    const riskClass = riskVal === 'Low Risk' ? 'low' : riskVal === 'High Risk' ? 'high' : 'medium';
    const riskIcon  = riskVal === 'Low Risk' ? '🛡️' : riskVal === 'High Risk' ? '🚀' : '⚖️';
    const riskAlloc = riskVal === 'Low Risk'
      ? {equity:20, debt:55, gold:15, liquid:10,
         instruments:['Post Office FD','PPF (7.1%)','NSC (7.7%)','SCSS (8.2%)','Debt Mutual Fund'],
         returns:'6–8% CAGR', horizon:'Short to Medium term (1–5 yrs)',
         tips:['Prioritise capital protection over growth','Avoid equity exposure above 20%','Use sweep-in FD for idle cash','Ladder FDs across 1, 2, 3-year tenures']}
      : riskVal === 'High Risk'
      ? {equity:75, debt:10, gold:5, liquid:10,
         instruments:['Mid/Small-cap MF','Direct Equity (NIFTY stocks)','ELSS','NPS Tier-I','Sovereign Gold Bond'],
         returns:'12–16% CAGR', horizon:'Long term (7+ yrs)',
         tips:['Stay invested through market volatility','Use SIP to average out market fluctuations','Review portfolio every 6 months','Keep 10% liquid for opportunities']}
      : {equity:50, debt:30, gold:10, liquid:10,
         instruments:['Large-cap Index Fund','PPF','Hybrid MF','ELSS','Liquid MF'],
         returns:'9–12% CAGR', horizon:'Medium to Long term (3–7 yrs)',
         tips:['Rebalance portfolio annually','Increase equity allocation as income grows','Use ELSS for tax saving + growth','Maintain 6-month emergency fund']};

    const allocBars = [
      {label:'Equity', pct: riskAlloc.equity, color: riskClass==='low'?'#34d399':riskClass==='high'?'#f87171':'#f59e0b'},
      {label:'Debt',   pct: riskAlloc.debt,   color:'#7dd3fc'},
      {label:'Gold',   pct: riskAlloc.gold,   color:'#fbbf24'},
      {label:'Liquid', pct: riskAlloc.liquid, color:'#a78bfa'},
    ].map(a => `
      <div class="alloc-row">
        <span class="alloc-label">${a.label}</span>
        <div class="alloc-bar"><div class="alloc-fill" style="width:${a.pct}%;background:${a.color}"></div></div>
        <span class="alloc-pct">${a.pct}%</span>
      </div>`).join('');

    const riskSection = `
      <div class="card">
        <div class="risk-advice ${riskClass}">
          <h3>${riskIcon} ${riskVal} — Personalised Investment Advice</h3>
          <p style="font-size:0.8rem;color:#94a3b8;margin-bottom:12px">
            Based on your <strong style="color:#e2e8f0">${riskVal}</strong> profile,
            here is your tailored investment strategy targeting <strong style="color:#e2e8f0">${riskAlloc.returns}</strong>.
            Suitable horizon: ${riskAlloc.horizon}.
          </p>
          <div style="margin-bottom:10px">${allocBars}</div>
          <div style="font-size:0.78rem;color:#94a3b8;margin-bottom:6px;font-weight:600">Recommended Instruments:</div>
          <ul>${riskAlloc.instruments.map(i=>`<li>${i}</li>`).join('')}</ul>
          <div style="font-size:0.78rem;color:#94a3b8;margin-top:10px;margin-bottom:6px;font-weight:600">Key Tips:</div>
          <ul>${riskAlloc.tips.map(t=>`<li>${t}</li>`).join('')}</ul>
        </div>
      </div>`;

    document.getElementById('sections-container').innerHTML += riskSection;

    // Section 10: 30-Day Plan
    const plan = rm['10_30_DAY_ACTION_PLAN'];
    if (plan) {
      document.getElementById('action-plan').style.display = 'block';
      document.getElementById('week-grid').innerHTML = ['week_1','week_2','week_3','week_4'].map((w,i) =>
        `<div class="week"><h4>Week ${i+1}</h4><ul>${(plan[w]||[]).map(a=>`<li>${a}</li>`).join('')}</ul></div>`
      ).join('');
    }

    document.getElementById('results').style.display = 'block';
    document.getElementById('results').scrollIntoView({behavior:'smooth'});
  } catch(err) {
    document.getElementById('errorBox').innerHTML = `<div class="error">Error: ${err.message}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = '🔍 Generate Financial Roadmap';
  }
});
function selectRisk(el) {
  document.querySelectorAll('.risk-opt').forEach(o => o.classList.remove('risk-active','low','medium','high'));
  el.classList.add('risk-active');
  const val = el.dataset.val;
  el.classList.add(val === 'Low Risk' ? 'low' : val === 'High Risk' ? 'high' : 'medium');
  document.getElementById('risk_level_input').value = val;
}
// init active state style on load
document.querySelector('.risk-opt.risk-active').classList.add('medium');
</script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        d = request.get_json(force=True)
        result = fa.predict(
            name           = str(d.get("name", "User")),
            income         = float(d.get("income", 0)),
            age            = int(d.get("age", 30)),
            dependents     = int(d.get("dependents", 0)),
            occupation     = str(d.get("occupation", "Self_Employed")),
            city_tier      = str(d.get("city_tier", "Tier_2")),
            location       = str(d.get("location", "")),
            business_type  = str(d.get("business_type", "")),
            monthly_savings= float(d.get("monthly_savings", 0)),
            goal           = str(d.get("goal", "")),
            risk_level     = str(d.get("risk_level", "Medium Risk")),
            rent           = float(d.get("rent", 0)),
            loan_repayment = float(d.get("loan_repayment", 0)),
            insurance      = float(d.get("insurance", 0)),
            groceries      = float(d.get("groceries", 0)),
            transport      = float(d.get("transport", 0)),
            eating_out     = float(d.get("eating_out", 0)),
            entertainment  = float(d.get("entertainment", 0)),
            utilities      = float(d.get("utilities", 0)),
            healthcare     = float(d.get("healthcare", 0)),
            education      = float(d.get("education", 0)),
            miscellaneous  = float(d.get("miscellaneous", 0)),
            loaded         = loaded,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/health")
def health():
    return jsonify({"status": "ok", "models": 6})

if __name__ == "__main__":
    print("Financial Advisor AI running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
