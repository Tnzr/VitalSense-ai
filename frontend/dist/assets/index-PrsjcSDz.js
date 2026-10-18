(function(){const a=document.createElement("link").relList;if(a&&a.supports&&a.supports("modulepreload"))return;for(const t of document.querySelectorAll('link[rel="modulepreload"]'))s(t);new MutationObserver(t=>{for(const o of t)if(o.type==="childList")for(const l of o.addedNodes)l.tagName==="LINK"&&l.rel==="modulepreload"&&s(l)}).observe(document,{childList:!0,subtree:!0});function n(t){const o={};return t.integrity&&(o.integrity=t.integrity),t.referrerPolicy&&(o.referrerPolicy=t.referrerPolicy),t.crossOrigin==="use-credentials"?o.credentials="include":t.crossOrigin==="anonymous"?o.credentials="omit":o.credentials="same-origin",o}function s(t){if(t.ep)return;t.ep=!0;const o=n(t);fetch(t.href,o)}})();const m="/v1",c=localStorage.getItem("asculto_api_key")||"",f={"abnormal-flag":"pill-bad","reference-flag":"pill-warn"},p={"abnormal-flag":"abnormality flagged","reference-flag":"reference-only screen flagged"};function g(){return c?{authorization:`Bearer ${c}`}:{}}async function h(e,a){const n=new FormData;n.append("file",e),n.append("heads",String(a.heads||"")),n.append("explanation",String(a.explanation||"standard")),n.append("figures","all"),n.append("temporal",String(!!a.temporal)),n.append("temporal_mode","multiscale"),n.append("scales","1,3,15"),n.append("ai_narrative","true"),n.append("patient_context",String(a.patient_context||""));const s=await fetch(`${m}/analyze`,{method:"POST",headers:g(),body:n});if(!s.ok)throw new Error(`HTTP ${s.status}: ${await s.text()}`);return await s.json()}function i(e){return e.replace(/[&<>"]/g,a=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"})[a])}function v(e){var l,d;const a=document.getElementById("results");if(e.error){a.innerHTML=`<div class="card warn"><h2>Could not analyze</h2><pre>${i(e.error)}</pre></div>`;return}const n=e.heads.map(r=>`<tr><td>${i(r.label)}</td><td><strong>${i(r.pred)}</strong></td>
      <td>${(r.confidence*100).toFixed(1)}%</td><td>${r.n_models} folds</td></tr>`).join(""),s=e.narrative.map(r=>`<li>${i(r)}</li>`).join(""),t=Object.entries(e.figures||{}).map(([r,u])=>`<figure><img src="${u}" alt="${i(r)}" /></figure>`).join(""),o=e.ai_narrative?`<div class="card"><h2>Personalized AI narrative
         <span class="muted">${i(e.ai_narrative.provider)} · ${i(e.ai_narrative.model)}</span></h2>
         <p>${i(e.ai_narrative.summary)}</p>
         <h3>Medical</h3><ul>${e.ai_narrative.recommendations.medical.map(r=>`<li>${i(r)}</li>`).join("")}</ul>
         <h3>Diet</h3><ul>${e.ai_narrative.recommendations.diet.map(r=>`<li>${i(r)}</li>`).join("")}</ul>
         <h3>Environment</h3><ul>${e.ai_narrative.recommendations.environment.map(r=>`<li>${i(r)}</li>`).join("")}</ul>
         <h3>Next steps</h3><ul>${e.ai_narrative.next_steps.map(r=>`<li>${i(r)}</li>`).join("")}</ul>
         <p class="muted">${i(e.ai_narrative.disclaimer)}</p></div>`:"";a.innerHTML=`
    <section class="report-head">
      <h1>Report — ${i(e.filename)}</h1>
      <span class="pill ${f[e.verdict??""]??"pill-ok"}">
        ${p[e.verdict??""]??"no abnormality flagged"}</span>
    </section>
    ${(d=(l=e.quality)==null?void 0:l.flags)!=null&&d.length?`<div class="card warn"><strong>Quality:</strong> ${i(e.quality.flags.join("; "))}</div>`:""}
    <div class="card"><h2>Summary</h2><ul class="narrative">${s}</ul></div>
    ${o}
    <div class="card"><h2>Predictions</h2>
      <table><thead><tr><th>Head</th><th>Prediction</th><th>Confidence</th><th>Ensemble</th></tr></thead>
      <tbody>${n}</tbody></table></div>
    ${t?`<div class="card"><h2>Figures</h2><div class="figures">${t}</div></div>`:""}
  `}document.getElementById("analyze-form").addEventListener("submit",async e=>{var s;e.preventDefault();const a=document.getElementById("status"),n=(s=document.getElementById("file").files)==null?void 0:s[0];if(n){a.textContent="Analyzing…";try{const t=await h(n,{explanation:document.getElementById("explanation").value,temporal:document.getElementById("temporal").checked,patient_context:document.getElementById("patient_context").value});v(t),a.textContent=""}catch(t){a.textContent=`Error: ${t.message}`}}});"serviceWorker"in navigator&&location.protocol.startsWith("http")&&window.addEventListener("load",()=>{navigator.serviceWorker.register("/sw.js").catch(()=>{})});
//# sourceMappingURL=index-PrsjcSDz.js.map
