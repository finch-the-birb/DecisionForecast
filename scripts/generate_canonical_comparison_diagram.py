"""Generate side-by-side publication comparison diagram for Canonical Dual vs Hierarchical TimeXer."""

from __future__ import annotations
from pathlib import Path


def generate_comparison_svg() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1920 1080" width="100%" height="100%">
  <defs>
    <style>
      .main-title { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 22px; font-weight: bold; fill: #1A365D; }
      .col-title { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 16px; font-weight: bold; }
      .box-title { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 12px; font-weight: bold; }
      .box-desc { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 10px; }
      .tensor-shape { font-family: 'SF Mono', Monaco, Inconsolata, Consolas, monospace; font-size: 9px; font-weight: 600; }
      .arrow { stroke: #4A5568; stroke-width: 1.5; fill: none; marker-end: url(#arr); }
      .arrow-ts { stroke: #27AE60; stroke-width: 1.5; fill: none; marker-end: url(#arr-g); }
      .arrow-text { stroke: #8E44AD; stroke-width: 1.5; fill: none; marker-end: url(#arr-p); }
    </style>
    <marker id="arr" markerWidth="6" markerHeight="5" refX="5" refY="2.5" orient="auto">
      <polygon points="0 0, 6 2.5, 0 5" fill="#4A5568" />
    </marker>
    <marker id="arr-g" markerWidth="6" markerHeight="5" refX="5" refY="2.5" orient="auto">
      <polygon points="0 0, 6 2.5, 0 5" fill="#27AE60" />
    </marker>
    <marker id="arr-p" markerWidth="6" markerHeight="5" refX="5" refY="2.5" orient="auto">
      <polygon points="0 0, 6 2.5, 0 5" fill="#8E44AD" />
    </marker>
  </defs>

  <rect width="1920" height="1080" fill="#FFFFFF" />

  <!-- Main Banner -->
  <rect x="40" y="20" width="1840" height="55" rx="8" fill="#E8F1FC" stroke="#2B6CB0" stroke-width="2" />
  <text x="960" y="46" text-anchor="middle" class="main-title">Architectural Comparison: Canonical Dual vs. Hierarchical TimeXer</text>
  <text x="960" y="64" text-anchor="middle" font-family="sans-serif" font-size="12px" fill="#4A5568">Publication Architecture Specification for Multivariate Time-Series LTSF with Multi-Modal News and Technical Signals</text>

  <!-- LEFT PANEL: DUAL -->
  <rect x="40" y="90" width="900" height="960" rx="10" fill="#FAFCFF" stroke="#2B6CB0" stroke-width="1.5" />
  <text x="490" y="120" text-anchor="middle" class="col-title" fill="#1A365D">Paradigm A: Canonical Dual TimeXer (L = 2)</text>
  <text x="490" y="138" text-anchor="middle" font-family="sans-serif" font-size="11px" fill="#4A5568">Parallel Disentangled Cross-Modal Distillation in Every Layer</text>

  <!-- Dual Forecast -->
  <rect x="390" y="155" width="200" height="35" rx="6" fill="#E8F8F5" stroke="#1ABC9C" stroke-width="1.5" />
  <text x="490" y="177" text-anchor="middle" class="box-title" fill="#0E6251">Forecast: y_hat [B, H=7]</text>

  <!-- Dual Head -->
  <rect x="310" y="205" width="360" height="40" rx="6" fill="#EDF2F7" stroke="#4A5568" stroke-width="1.2" />
  <text x="490" y="225" text-anchor="middle" class="box-title" fill="#1A202C">FlattenHead: Close Tokens [B, N_p+2, d] → H=7</text>

  <!-- Dual Layer 2 -->
  <rect x="70" y="260" width="840" height="220" rx="8" fill="#FFFFFF" stroke="#CBD5E0" stroke-width="1.5" stroke-dasharray="6 3" />
  <text x="90" y="280" class="box-title" fill="#2D3748">Layer 2 (Broadcast &amp; Refine)</text>
  <rect x="290" y="285" width="400" height="30" rx="4" fill="#EDF2F7" stroke="#A0AEC0" />
  <text x="490" y="304" text-anchor="middle" class="box-desc" fill="#2D3748">Conv1d FFN + Residual + Norm over [P; G_ts; G_text]</text>
  <rect x="110" y="330" width="260" height="50" rx="6" fill="#EAFBF1" stroke="#27AE60" />
  <text x="240" y="350" text-anchor="middle" class="box-title" fill="#145A32">Cross-Attn: G_ts × cross_ts</text>
  <text x="240" y="368" text-anchor="middle" class="tensor-shape" fill="#27AE60">Q=[B·C, 1, d], K,V=[B, 25, d]</text>
  <rect x="610" y="330" width="260" height="50" rx="6" fill="#F5EEF8" stroke="#8E44AD" />
  <text x="740" y="350" text-anchor="middle" class="box-title" fill="#4A235A">Cross-Attn: G_text × cross_text</text>
  <text x="740" y="368" text-anchor="middle" class="tensor-shape" fill="#8E44AD">Q=[B·C, 1, d], K,V=[B, 15, d]</text>
  <rect x="170" y="400" width="640" height="50" rx="6" fill="#FEF5E7" stroke="#D35400" />
  <text x="490" y="422" text-anchor="middle" class="box-title" fill="#7E5109">Self-Attention over [P; G_ts^(1); G_text^(1)]</text>
  <text x="490" y="438" text-anchor="middle" class="box-desc" fill="#B9770E">Patches absorb Layer 1 updated exogenous signals from both bridges</text>

  <!-- Dual Layer 1 -->
  <rect x="70" y="495" width="840" height="220" rx="8" fill="#FFFFFF" stroke="#CBD5E0" stroke-width="1.5" stroke-dasharray="6 3" />
  <text x="90" y="515" class="box-title" fill="#2D3748">Layer 1 (Initial Endogenous Gathering &amp; Exogenous Filtering)</text>
  <rect x="290" y="520" width="400" height="30" rx="4" fill="#EDF2F7" stroke="#A0AEC0" />
  <text x="490" y="539" text-anchor="middle" class="box-desc" fill="#2D3748">Conv1d FFN + Residual + Norm over [P; G_ts; G_text]</text>
  <rect x="110" y="565" width="260" height="50" rx="6" fill="#EAFBF1" stroke="#27AE60" />
  <text x="240" y="585" text-anchor="middle" class="box-title" fill="#145A32">Cross-Attn: G_ts × cross_ts</text>
  <text x="240" y="603" text-anchor="middle" class="tensor-shape" fill="#27AE60">Q=[B·C, 1, d], K,V=[B, 25, d]</text>
  <rect x="610" y="565" width="260" height="50" rx="6" fill="#F5EEF8" stroke="#8E44AD" />
  <text x="740" y="585" text-anchor="middle" class="box-title" fill="#4A235A">Cross-Attn: G_text × cross_text</text>
  <text x="740" y="603" text-anchor="middle" class="tensor-shape" fill="#8E44AD">Q=[B·C, 1, d], K,V=[B, 15, d]</text>
  <rect x="170" y="635" width="640" height="50" rx="6" fill="#FEF5E7" stroke="#D35400" />
  <text x="490" y="657" text-anchor="middle" class="box-title" fill="#7E5109">Self-Attention over [P; G_ts; G_text]</text>
  <text x="490" y="673" text-anchor="middle" class="box-desc" fill="#B9770E">Bridges G_ts and G_text gather endogenous price dynamic from patches</text>

  <!-- Dual Embeddings -->
  <rect x="250" y="730" width="480" height="65" rx="6" fill="#E8F1FC" stroke="#2B6CB0" />
  <text x="490" y="752" text-anchor="middle" class="box-title" fill="#1A365D">EnEmbeddingDual: Patching + Patch-FFT + Dual Concat</text>
  <text x="490" y="770" text-anchor="middle" class="box-desc" fill="#2874A6">ValEmb + PosEmb + FreqEmb; Concat [P; G_ts; G_text] → [B·C, N_p+2, d]</text>
  <rect x="70" y="730" width="160" height="65" rx="6" fill="#EAFBF1" stroke="#27AE60" />
  <text x="150" y="755" text-anchor="middle" class="box-title" fill="#145A32">Inverted TS Proj</text>
  <text x="150" y="775" text-anchor="middle" class="tensor-shape" fill="#27AE60">[B, 25, d]</text>
  <rect x="750" y="730" width="160" height="65" rx="6" fill="#F5EEF8" stroke="#8E44AD" />
  <text x="830" y="755" text-anchor="middle" class="box-title" fill="#4A235A">Text Proj</text>
  <text x="830" y="775" text-anchor="middle" class="tensor-shape" fill="#8E44AD">[B, 15, d]</text>

  <!-- Dual Inputs -->
  <rect x="70" y="815" width="160" height="45" rx="6" fill="#EAFBF1" stroke="#27AE60" />
  <text x="150" y="842" text-anchor="middle" class="box-title" fill="#145A32">25 Indicators [B,60,25]</text>
  <rect x="250" y="815" width="480" height="45" rx="6" fill="#E8F1FC" stroke="#2B6CB0" />
  <text x="490" y="842" text-anchor="middle" class="box-title" fill="#1A365D">OHLCV Channels [B, 60, 5]</text>
  <rect x="750" y="815" width="160" height="45" rx="6" fill="#F5EEF8" stroke="#8E44AD" />
  <text x="830" y="842" text-anchor="middle" class="box-title" fill="#4A235A">15D Text [B, 15]</text>


  <!-- RIGHT PANEL: HIERARCHICAL -->
  <rect x="980" y="90" width="900" height="960" rx="10" fill="#FFFCFA" stroke="#D35400" stroke-width="1.5" />
  <text x="1430" y="120" text-anchor="middle" class="col-title" fill="#7E5109">Paradigm B: Canonical Hierarchical TimeXer (L = 3)</text>
  <text x="1430" y="138" text-anchor="middle" font-family="sans-serif" font-size="11px" fill="#4A5568">Sequential Progression: Micro Indicators → Macro News → Consolidation</text>

  <!-- Hier Forecast -->
  <rect x="1330" y="155" width="200" height="35" rx="6" fill="#E8F8F5" stroke="#1ABC9C" stroke-width="1.5" />
  <text x="1430" y="177" text-anchor="middle" class="box-title" fill="#0E6251">Forecast: y_hat [B, H=7]</text>

  <!-- Hier Head -->
  <rect x="1250" y="205" width="360" height="40" rx="6" fill="#EDF2F7" stroke="#4A5568" stroke-width="1.2" />
  <text x="1430" y="225" text-anchor="middle" class="box-title" fill="#1A202C">FlattenHead: Close Tokens [B, N_p+2, d] → H=7</text>

  <!-- Hier Layer 3 -->
  <rect x="1010" y="260" width="840" height="125" rx="8" fill="#FEF9E7" stroke="#F39C12" stroke-width="1.5" stroke-dasharray="6 3" />
  <text x="1030" y="280" class="box-title" fill="#7D6608">Layer 3: Cross-Modal Consolidation</text>
  <rect x="1230" y="285" width="400" height="28" rx="4" fill="#EDF2F7" stroke="#CBD5E0" />
  <text x="1430" y="303" text-anchor="middle" class="box-desc" fill="#2D3748">Conv1d FFN + Residual + Norm</text>
  <rect x="1110" y="320" width="640" height="45" rx="6" fill="#FEF5E7" stroke="#D35400" />
  <text x="1430" y="340" text-anchor="middle" class="box-title" fill="#7E5109">Consolidation Self-Attn: [P; G_ts^(2); G_text^(2)]</text>
  <text x="1430" y="355" text-anchor="middle" class="box-desc" fill="#B9770E">All patches harmonize news sentiment and indicator state without cross-attn</text>

  <!-- Hier Layer 2 -->
  <rect x="1010" y="400" width="840" height="155" rx="8" fill="#FDFEFE" stroke="#8E44AD" stroke-width="1.5" stroke-dasharray="6 3" />
  <text x="1030" y="420" class="box-title" fill="#4A235A">Layer 2: Semantic Macro-Structure (Text Attention)</text>
  <rect x="1230" y="425" width="400" height="28" rx="4" fill="#EDF2F7" stroke="#CBD5E0" />
  <text x="1430" y="443" text-anchor="middle" class="box-desc" fill="#2D3748">Conv1d FFN: tokens [P^(2); G_ts^(2); G_text^(2)]</text>
  <rect x="1450" y="460" width="360" height="38" rx="6" fill="#F5EEF8" stroke="#8E44AD" />
  <text x="1630" y="478" text-anchor="middle" class="box-title" fill="#4A235A">Cross-Attn ONLY G_text × cross_text</text>
  <text x="1630" y="492" text-anchor="middle" class="tensor-shape" fill="#8E44AD">G_ts bypasses via identity/residual</text>
  <rect x="1110" y="505" width="640" height="40" rx="6" fill="#FEF5E7" stroke="#D35400" />
  <text x="1430" y="523" text-anchor="middle" class="box-title" fill="#7E5109">Self-Attn: Patches &amp; G_text absorb enriched G_ts^(1) from Layer 1</text>
  <text x="1430" y="537" text-anchor="middle" class="box-desc" fill="#B9770E">Conditions news interpretation on technical momentum</text>

  <!-- Hier Layer 1 -->
  <rect x="1010" y="570" width="840" height="155" rx="8" fill="#FDFEFE" stroke="#27AE60" stroke-width="1.5" stroke-dasharray="6 3" />
  <text x="1030" y="590" class="box-title" fill="#145A32">Layer 1: Technical Micro-Structure (Indicator Attention)</text>
  <rect x="1230" y="595" width="400" height="28" rx="4" fill="#EDF2F7" stroke="#CBD5E0" />
  <text x="1430" y="613" text-anchor="middle" class="box-desc" fill="#2D3748">Conv1d FFN: tokens [P^(1); G_ts^(1); G_text^(1)]</text>
  <rect x="1050" y="630" width="360" height="38" rx="6" fill="#EAFBF1" stroke="#27AE60" />
  <text x="1230" y="648" text-anchor="middle" class="box-title" fill="#145A32">Cross-Attn ONLY G_ts × cross_ts</text>
  <text x="1230" y="662" text-anchor="middle" class="tensor-shape" fill="#27AE60">G_text bypasses via identity/residual</text>
  <rect x="1110" y="675" width="640" height="40" rx="6" fill="#FEF5E7" stroke="#D35400" />
  <text x="1430" y="693" text-anchor="middle" class="box-title" fill="#7E5109">Initial Self-Attn: [P; G_ts; G_text]</text>
  <text x="1430" y="707" text-anchor="middle" class="box-desc" fill="#B9770E">G_ts gathers baseline price structure before indicator query</text>

  <!-- Hier Embeddings -->
  <rect x="1190" y="740" width="480" height="65" rx="6" fill="#E8F1FC" stroke="#2B6CB0" />
  <text x="1430" y="762" text-anchor="middle" class="box-title" fill="#1A365D">EnEmbeddingDual: Patching + Frequency Embedding</text>
  <text x="1430" y="780" text-anchor="middle" class="box-desc" fill="#2874A6">ValEmb + PosEmb + Patch-FFT; Concat [P; G_ts; G_text] → [B·C, N_p+2, d]</text>
  <rect x="1010" y="740" width="160" height="65" rx="6" fill="#EAFBF1" stroke="#27AE60" />
  <text x="1090" y="765" text-anchor="middle" class="box-title" fill="#145A32">Inverted TS Proj</text>
  <text x="1090" y="785" text-anchor="middle" class="tensor-shape" fill="#27AE60">[B, 25, d]</text>
  <rect x="1690" y="740" width="160" height="65" rx="6" fill="#F5EEF8" stroke="#8E44AD" />
  <text x="1770" y="765" text-anchor="middle" class="box-title" fill="#4A235A">Text Proj</text>
  <text x="1770" y="785" text-anchor="middle" class="tensor-shape" fill="#8E44AD">[B, 15, d]</text>

  <!-- Hier Inputs -->
  <rect x="1010" y="815" width="160" height="45" rx="6" fill="#EAFBF1" stroke="#27AE60" />
  <text x="1090" y="842" text-anchor="middle" class="box-title" fill="#145A32">25 Indicators [B,60,25]</text>
  <rect x="1190" y="815" width="480" height="45" rx="6" fill="#E8F1FC" stroke="#2B6CB0" />
  <text x="1430" y="842" text-anchor="middle" class="box-title" fill="#1A365D">OHLCV Channels [B, 60, 5]</text>
  <rect x="1690" y="815" width="160" height="45" rx="6" fill="#F5EEF8" stroke="#8E44AD" />
  <text x="1770" y="842" text-anchor="middle" class="box-title" fill="#4A235A">15D Text [B, 15]</text>

  <!-- Bottom Synthesis Box -->
  <rect x="70" y="885" width="1780" height="145" rx="8" fill="#F8FAFC" stroke="#94A3B8" stroke-width="1.5" />
  <text x="960" y="912" text-anchor="middle" font-family="sans-serif" font-size="14px" font-weight="bold" fill="#0F172A">Comparative Paradigm Summary &amp; Theoretical Properties</text>
  
  <text x="100" y="940" font-family="sans-serif" font-size="11px" font-weight="bold" fill="#1E293B">Property</text>
  <text x="400" y="940" font-family="sans-serif" font-size="11px" font-weight="bold" fill="#1E293B">Canonical Dual TimeXer (c1_dual)</text>
  <text x="1100" y="940" font-family="sans-serif" font-size="11px" font-weight="bold" fill="#1E293B">Canonical Hierarchical TimeXer (c1_hierarchical)</text>

  <line x1="90" y1="948" x2="1830" y2="948" stroke="#CBD5E1" stroke-width="1" />

  <text x="100" y="968" font-family="sans-serif" font-size="11px" fill="#475569">Cross-modal topology:</text>
  <text x="400" y="968" font-family="sans-serif" font-size="11px" fill="#475569">Parallel: G_ts and G_text query respective modalities in every layer</text>
  <text x="1100" y="968" font-family="sans-serif" font-size="11px" fill="#475569">Sequential: Layer 1 = Micro (TS) → Layer 2 = Macro (Text) → Layer 3 = Consolidation</text>

  <text x="100" y="990" font-family="sans-serif" font-size="11px" fill="#475569">Inter-modal conditioning:</text>
  <text x="400" y="990" font-family="sans-serif" font-size="11px" fill="#475569">Symmetric in Self-Attn: G_ts and G_text attend to each other simultaneously</text>
  <text x="1100" y="990" font-family="sans-serif" font-size="11px" fill="#475569">Directional: Text layer explicitly conditions sentiment on prior technical momentum state</text>

  <text x="100" y="1012" font-family="sans-serif" font-size="11px" fill="#475569">Computational cost:</text>
  <text x="400" y="1012" font-family="sans-serif" font-size="11px" fill="#475569">2 Self-Attn + 4 Cross-Attn total (O(2·N^2 + 2·(C_ts + C_text)))</text>
  <text x="1100" y="1012" font-family="sans-serif" font-size="11px" fill="#475569">3 Self-Attn + 2 Cross-Attn total (O(3·N^2 + C_ts + C_text))</text>
</svg>"""


def main() -> None:
    figures_dir = Path("Articles/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    svg_content = generate_comparison_svg()
    out_svg = figures_dir / "canonical_dual_vs_hierarchical_comparison.svg"
    out_svg.write_text(svg_content, encoding="utf-8")
    print(f"Saved Comparison SVG to {out_svg} ({len(svg_content)} bytes)")


if __name__ == "__main__":
    main()
