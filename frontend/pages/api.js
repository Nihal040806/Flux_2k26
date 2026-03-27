/**
 * FraudShield AI - Shared API Client
 * Provides helper functions for all frontend pages to communicate with the backend.
 */
const API_BASE = window.location.origin;

async function apiGet(endpoint) {
    const res = await fetch(`${API_BASE}${endpoint}`);
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    return res.json();
}

async function apiPost(endpoint, body) {
    const res = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    return res.json();
}

function formatCurrency(n) {
    if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`;
    if (n >= 1_000) return `$${(n / 1_000).toFixed(0)}K`;
    return `$${n.toFixed(2)}`;
}

function riskColor(score) {
    if (score >= 70) return 'red';
    if (score >= 35) return 'amber';
    return 'emerald';
}

function statusBadge(status) {
    const colors = { Fraudulent: 'red', Suspicious: 'amber', Safe: 'emerald' };
    const c = colors[status] || 'slate';
    return `<span class="px-2.5 py-1 rounded-full bg-${c}-50 text-${c}-600 text-[10px] font-black uppercase border border-${c}-100">${status}</span>`;
}
