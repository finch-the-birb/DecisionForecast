"""Generate publication-grade architecture comparison diagrams for DecisionForecast.

Generates:
1. Articles/figures/model_comparison_architectures.drawio (editable Draw.io XML with math support)
2. Articles/figures/model_comparison_architectures.svg (standalone crisp vector graphic)

Architectures compared:
(a) c1_late_fusion: TimeXer with Exogenous Text Encoder + Late Residual MLP Fusion of Indicators
(b) c1_hierarchical: Sequential Modality Hierarchy (Layer 1: Indicators via G_ts -> Layer 2: Text via G_text)
(c) c1_inverted: iTransformer-Style Inverted Variate Tokens (Time patches cross-attend to 25 indicator tokens)
"""

from __future__ import annotations

import html
from pathlib import Path


def create_drawio_xml() -> str:
    """Builds valid Draw.io XML document with generous spacing and zero overlapping elements."""

    elements = []
    cell_id = 2

    def next_id() -> str:
        nonlocal cell_id
        cid = f"cell_{cell_id}"
        cell_id += 1
        return cid

    def add_box(
        val: str,
        x: int,
        y: int,
        w: int,
        h: int,
        fill: str,
        stroke: str,
        font_color: str = "#1A202C",
        font_size: int = 11,
        bold: bool = False,
        rounded: int = 1,
        dashed: int = 0,
        align: str = "center",
        valign: str = "middle",
        custom_style: str = "",
    ) -> str:
        cid = next_id()
        dash_str = "dashed=1;dashPattern=8 4;" if dashed else "dashed=0;"
        b_str = "fontStyle=1;" if bold else ""
        escaped_val = html.escape(val).replace("\n", "&#xa;")
        style = (
            f"rounded={rounded};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
            f"strokeWidth=1.5;fontFamily=Helvetica,Arial,sans-serif;fontSize={font_size};"
            f"fontColor={font_color};align={align};verticalAlign={valign};{dash_str}{b_str}{custom_style}"
        )
        xml = (
            f'<mxCell id="{cid}" value="{escaped_val}" style="{style}" vertex="1" parent="1">\n'
            f'  <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />\n'
            f"</mxCell>"
        )
        elements.append(xml)
        return cid

    def add_edge(
        src: str,
        tgt: str,
        label: str = "",
        stroke: str = "#4A5568",
        stroke_width: float = 1.5,
        exit_pt: tuple[float, float] = (0.5, 0.0),
        entry_pt: tuple[float, float] = (0.5, 1.0),
        dashed: int = 0,
        edge_style: str = "orthogonalEdgeStyle",
    ) -> str:
        cid = next_id()
        dash_str = "dashed=1;dashPattern=6 3;" if dashed else "dashed=0;"
        escaped_label = html.escape(label).replace("\n", "&#xa;")
        style = (
            f"edgeStyle={edge_style};rounded=0;orthogonalLoop=1;jettySize=auto;html=1;"
            f"strokeColor={stroke};strokeWidth={stroke_width};fontSize=10;fontColor=#2D3748;"
            f"fontFamily=Helvetica,Arial,sans-serif;labelBackgroundColor=#FFFFFF;{dash_str}"
            f"exitX={exit_pt[0]};exitY={exit_pt[1]};exitDx=0;exitDy=0;"
            f"entryX={entry_pt[0]};entryY={entry_pt[1]};entryDx=0;entryDy=0;"
        )
        xml = (
            f'<mxCell id="{cid}" value="{escaped_label}" style="{style}" edge="1" parent="1" source="{src}" target="{tgt}">\n'
            f'  <mxGeometry relative="1" as="geometry" />\n'
            f"</mxCell>"
        )
        elements.append(xml)
        return cid

    # Top Header & Banners
    add_box(
        "DecisionForecast Architecture Comparison: Exogenous Modality Interaction Paradigms",
        50, 30, 2220, 48,
        fill="#2B6CB0", stroke="#2B6CB0", font_color="#FFFFFF", font_size=18, bold=True, rounded=1
    )
    add_box(
        "Benchmarking Long-Term Series Forecasting (H=7) on FNSPID: Endogenous Price vs. Exogenous Technical Indicators vs. Text Sentiment",
        50, 85, 2220, 28,
        fill="#EDF2F7", stroke="#CBD5E0", font_color="#4A5568", font_size=12, bold=False, rounded=1
    )

    # Legend Elements
    leg_y = 125
    add_box("MODALITY COLOR CODE:", 50, leg_y, 180, 32, fill="none", stroke="none", font_color="#2D3748", font_size=11, bold=True, align="left")
    add_box("Endogenous Price (OHLCV / FFT)", 240, leg_y, 250, 32, fill="#E8F1FC", stroke="#2B6CB0", font_color="#1A365D", font_size=11, bold=True)
    add_box("Exogenous Indicators (TreeSHAP 25)", 510, leg_y, 270, 32, fill="#EAFBF1", stroke="#27AE60", font_color="#145A32", font_size=11, bold=True)
    add_box("Exogenous Text (FinBERT 15D)", 800, leg_y, 250, 32, fill="#F5EEF8", stroke="#8E44AD", font_color="#4A235A", font_size=11, bold=True)
    add_box("Modality Attention & Gated Bridge", 1070, leg_y, 260, 32, fill="#FEF5E7", stroke="#D35400", font_color="#7E5109", font_size=11, bold=True)
    add_box("Trainable Query Token (G)", 1350, leg_y, 220, 32, fill="#FDEBD0", stroke="#B9770E", font_color="#7D6608", font_size=11, bold=True)
    add_box("Forecast Head & Prediction", 1590, leg_y, 230, 32, fill="#EBEDEF", stroke="#2C3E50", font_color="#1B2631", font_size=11, bold=True)

    # =========================================================================
    # COLUMN A: c1_late_fusion (Left: X = 50 to 750, Center ~ 400)
    # =========================================================================
    ca_x = 50
    ca_w = 700

    add_box(
        "(a) c1_late_fusion: TimeXer with Late Residual Indicator Fusion\nChampion (Paper Test MSE: 0.7687, DA: 54.3% | Dev Test MSE: 0.688)",
        ca_x, 175, ca_w, 55,
        fill="#EBF8FF", stroke="#3182CE", font_color="#2B6CB0", font_size=13, bold=True, rounded=1
    )

    # A - Forecast Head & Pred
    a_pred = add_box(
        "Forecast Horizon Output\n$$\\hat{Y} \\in \\mathbb{R}^{B \\times 7}$$\n(Multi-step Price Forecast)",
        ca_x + 225, 255, 250, 55,
        fill="#EBEDEF", stroke="#2C3E50", font_color="#1B2631", font_size=11, bold=True
    )
    a_head = add_box(
        "Forecast Head: Linear Projection\n$$\\text{Linear}(d_{\\text{model}}=64 \\to H=7)$$",
        ca_x + 225, 335, 250, 50,
        fill="#EBEDEF", stroke="#2C3E50", font_color="#1B2631", font_size=11
    )
    a_fused = add_box(
        "Late Residual Fusion & Normalization\n$$Z_{\\text{final}} = \\text{LayerNorm}(Z_{\\text{pool}} + Z_{\\text{tech}}) \\in \\mathbb{R}^{B \\times 64}$$",
        ca_x + 150, 410, 400, 55,
        fill="#D5E8D4", stroke="#27AE60", font_color="#196F3D", font_size=11, bold=True
    )
    a_pool = add_box(
        "Patch Pooling (Last Patch)\n$$Z_{\\text{pool}} = Z_{\\text{patch}}[:, -1, :] \\in \\mathbb{R}^{B \\times 64}$$",
        ca_x + 70, 490, 240, 50,
        fill="#E8F1FC", stroke="#2B6CB0", font_color="#1A365D", font_size=11
    )
    a_mlp = add_box(
        "Exogenous Technical MLP\n$$\\text{Linear}(25 \\to 64) \\to \\text{GELU} \\to \\text{Linear}(64 \\to 64)$$\n$$Z_{\\text{tech}} \\in \\mathbb{R}^{B \\times 64}$$",
        ca_x + 380, 485, 270, 60,
        fill="#EAFBF1", stroke="#27AE60", font_color="#145A32", font_size=11
    )

    # A - Backbone Container
    add_box(
        "TimeXer Encoder Backbone (e_layers = 1 or 2)\nProcesses High-Frequency Price Dynamics & Exogenous Text Context",
        ca_x + 20, 570, 420, 470,
        fill="#FAFAFA", stroke="#CBD5E0", font_color="#718096", font_size=11, bold=True, dashed=1, align="left", valign="top"
    )

    a_bridge = add_box(
        "Global-to-Patch Gated Bridge\n$$\\Delta = \\text{CrossAttn}(Q=Z_{\\text{patch}}, K=G_{\\text{en}}, V=G_{\\text{en}})$$\n$$Z_{\\text{patch}} \\leftarrow \\text{LayerNorm}(Z_{\\text{patch}} + \\tanh(\\alpha) \\cdot \\Delta)$$\n$$[B, 9, 64]$$",
        ca_x + 40, 630, 380, 75,
        fill="#FEF5E7", stroke="#D35400", font_color="#7E5109", font_size=10
    )
    a_self = add_box(
        "Joint Time-Domain Self-Attention & FFN\n$$\\text{SelfAttn}([Z_{\\text{patch}}; G_{\\text{en}}]) \\in \\mathbb{R}^{B \\times 10 \\times 64}$$",
        ca_x + 40, 735, 380, 55,
        fill="#FEF5E7", stroke="#D35400", font_color="#7E5109", font_size=11
    )
    a_cross_text = add_box(
        "Text Cross-Attention\n$$\\text{CrossAttn}(Q=[Z_{\\text{patch}}; G_{\\text{en}}], K=X_{\\text{text}}, V=X_{\\text{text}})$$\nUpdates Price Patches & Global Token with Market Sentiment",
        ca_x + 40, 820, 380, 65,
        fill="#FEF5E7", stroke="#D35400", font_color="#7E5109", font_size=10
    )
    a_concat = add_box(
        "Sequence Concatenation\n$$[Z_{\\text{patch}}; G_{\\text{en}}] \\in \\mathbb{R}^{B \\times 10 \\times 64}$$",
        ca_x + 40, 915, 230, 48,
        fill="#EDF2F7", stroke="#CBD5E0", font_color="#2D3748", font_size=11
    )
    a_gen = add_box(
        "Trainable Token\n$$G_{\\text{en}} = \\text{nn.Parameter}$$\n$$[B, 1, 64]$$",
        ca_x + 290, 915, 130, 48,
        fill="#FDEBD0", stroke="#B9770E", font_color="#7D6608", font_size=9, bold=True
    )

    # A - Embeddings
    a_patch_proj = add_box(
        "Patch Embedding + PosEmb\n$$\\text{Linear}(66 \\to 64) + E_{\\text{pos}}$$\n$$Z_{\\text{patch}} \\in \\mathbb{R}^{B \\times 9 \\times 64}$$",
        ca_x + 40, 1070, 210, 60,
        fill="#E8F1FC", stroke="#2B6CB0", font_color="#1A365D", font_size=10
    )
    a_unfold = add_box(
        "Time Unfold (P=12, S=6) + rFFT\n$$5 \\times 12 = 60 \\text{ values} + 6 \\text{ FFT bins}$$\n$$[B, 9, 66]$$",
        ca_x + 40, 1160, 210, 60,
        fill="#E8F1FC", stroke="#2B6CB0", font_color="#1A365D", font_size=10
    )
    a_text_proj = add_box(
        "Text Linear Projection\n$$\\text{Linear}(15 \\to 64)$$\n$$X_{\\text{text}} \\in \\mathbb{R}^{B \\times 60 \\times 64}$$",
        ca_x + 270, 1100, 160, 60,
        fill="#F5EEF8", stroke="#8E44AD", font_color="#4A235A", font_size=10
    )
    a_ts_slice = add_box(
        "Last Bar Slice (Bypasses Backbone)\n$$X_{\\text{ts}}[:, -1, :] \\in \\mathbb{R}^{B \\times 25}$$\n(TreeSHAP indicators on last bar $T$)",
        ca_x + 450, 1100, 230, 60,
        fill="#EAFBF1", stroke="#27AE60", font_color="#145A32", font_size=10, bold=True
    )

    # A - Inputs (Bottom)
    a_in_price = add_box(
        "Endogenous Price Input\n$$X_{\\text{price}} \\in \\mathbb{R}^{B \\times 60 \\times 5}$$\n[Close, Volume, Open, High, Low]",
        ca_x + 40, 1260, 210, 65,
        fill="#E8F1FC", stroke="#2B6CB0", font_color="#1A365D", font_size=10, bold=True
    )
    a_in_text = add_box(
        "Exogenous Text Input\n$$X_{\\text{text}} \\in \\mathbb{R}^{B \\times 60 \\times 15}$$\n(FinBERT Sentiment Scores)",
        ca_x + 270, 1260, 160, 65,
        fill="#F5EEF8", stroke="#8E44AD", font_color="#4A235A", font_size=10, bold=True
    )
    a_in_ts = add_box(
        "Exogenous Technical Indicators\n$$X_{\\text{ts}} \\in \\mathbb{R}^{B \\times 60 \\times 25}$$\n(TreeSHAP fold-1 technical features)",
        ca_x + 450, 1260, 230, 65,
        fill="#EAFBF1", stroke="#27AE60", font_color="#145A32", font_size=10, bold=True
    )

    # A - Connections
    add_edge(a_in_price, a_unfold, "[B, 60, 5]")
    add_edge(a_unfold, a_patch_proj, "[B, 9, 66]")
    add_edge(a_patch_proj, a_concat, "[B, 9, 64]")
    add_edge(a_gen, a_concat, "[B, 1, 64]", exit_pt=(0.0, 0.5), entry_pt=(1.0, 0.5))
    add_edge(a_in_text, a_text_proj, "[B, 60, 15]")
    add_edge(a_text_proj, a_cross_text, "[B, 60, 64]")
    add_edge(a_concat, a_cross_text, "[B, 10, 64]")
    add_edge(a_cross_text, a_self, "[B, 10, 64]")
    add_edge(a_self, a_bridge, "[B, 10, 64]")
    add_edge(a_bridge, a_pool, "[B, 9, 64]")
    add_edge(a_in_ts, a_ts_slice, "[B, 60, 25]")
    add_edge(a_ts_slice, a_mlp, "[B, 25]")
    add_edge(a_pool, a_fused, "[B, 64]")
    add_edge(a_mlp, a_fused, "[B, 64]")
    add_edge(a_fused, a_head, "[B, 64]")
    add_edge(a_head, a_pred, "[B, 7]")


    # =========================================================================
    # COLUMN B: c1_hierarchical (Middle: X = 810 to 1510, Center ~ 1160)
    # =========================================================================
    cb_x = 810
    cb_w = 700

    add_box(
        "(b) c1_hierarchical: Sequential Modality Hierarchy\nLayer 1: Indicators (G_ts) -> Layer 2: Text (G_text) (Paper Test MSE: 0.7720, DA: 53.4%)",
        cb_x, 175, cb_w, 55,
        fill="#F0FFF4", stroke="#38A169", font_color="#22543D", font_size=13, bold=True, rounded=1
    )

    # B - Pred & Head
    b_pred = add_box(
        "Forecast Horizon Output\n$$\\hat{Y} \\in \\mathbb{R}^{B \\times 7}$$\n(Multi-step Price Forecast)",
        cb_x + 225, 255, 250, 55,
        fill="#EBEDEF", stroke="#2C3E50", font_color="#1B2631", font_size=11, bold=True
    )
    b_head = add_box(
        "Forecast Head: Linear Projection\n$$\\text{Linear}(d_{\\text{model}}=64 \\to H=7)$$",
        cb_x + 225, 335, 250, 50,
        fill="#EBEDEF", stroke="#2C3E50", font_color="#1B2631", font_size=11
    )
    b_pool = add_box(
        "Patch Pooling (Last Patch from Layer 2)\n$$Z_{\\text{pool}} = Z_{\\text{patch}}^{(2)}[:, -1, :] \\in \\mathbb{R}^{B \\times 64}$$",
        cb_x + 200, 410, 300, 50,
        fill="#E8F1FC", stroke="#2B6CB0", font_color="#1A365D", font_size=11
    )

    # B - Layer 2 Container (Text Modulation)
    # Total container height: 260px, header top padding: 45px
    add_box(
        "Layer 2: Text Sentiment Modulation Layer (layer_text)\nAbsorbs FinBERT sentiment into price patches via G_text and Gated Bridge",
        cb_x + 20, 480, 660, 250,
        fill="#FDFEFE", stroke="#8E44AD", font_color="#4A235A", font_size=11, bold=True, dashed=1, align="left", valign="top"
    )
    b_bridge2 = add_box(
        "Gated Token Bridge 2\n$$Z_{\\text{patch}}^{(2)} = \\text{LayerNorm}(Z_{\\text{patch}} + \\tanh(\\alpha_2) \\cdot \\text{CrossAttn}(Z_{\\text{patch}}, G_{\\text{text}}))$$\n$$[B, 9, 64]$$",
        cb_x + 40, 535, 420, 52,
        fill="#FEF5E7", stroke="#D35400", font_color="#7E5109", font_size=10
    )
    b_self2 = add_box(
        "Joint Self-Attention & FFN: $$\\text{SelfAttn}([Z_{\\text{patch}}^{(1)}; G_{\\text{text}}]) \\in \\mathbb{R}^{B \\times 10 \\times 64}$$",
        cb_x + 40, 605, 420, 45,
        fill="#FEF5E7", stroke="#D35400", font_color="#7E5109", font_size=10
    )
    b_cross2 = add_box(
        "Text Cross-Attention\n$$G_{\\text{text}} \\leftarrow \\text{LayerNorm}(G_{\\text{text}} + \\text{CrossAttn}(G_{\\text{text}}, \\text{Text}_{\\text{exo}}))$$",
        cb_x + 40, 665, 420, 48,
        fill="#FEF5E7", stroke="#D35400", font_color="#7E5109", font_size=10
    )
    b_gtext = add_box(
        "Trainable Query Token\n$$G_{\\text{text}} = \\text{nn.Parameter}$$\n$$[B, 1, 64]$$",
        cb_x + 480, 665, 180, 48,
        fill="#FDEBD0", stroke="#B9770E", font_color="#7D6608", font_size=10, bold=True
    )

    # B - Layer 1 Container (Market Regime Indicators)
    add_box(
        "Layer 1: Market Regime Indicator Layer (layer_ts)\nAbsorbs 25 technical indicator series into price patches via G_ts and Gated Bridge",
        cb_x + 20, 755, 660, 250,
        fill="#FDFEFE", stroke="#27AE60", font_color="#145A32", font_size=11, bold=True, dashed=1, align="left", valign="top"
    )
    b_bridge1 = add_box(
        "Gated Token Bridge 1\n$$Z_{\\text{patch}}^{(1)} = \\text{LayerNorm}(Z_{\\text{patch}} + \\tanh(\\alpha_1) \\cdot \\text{CrossAttn}(Z_{\\text{patch}}, G_{\\text{ts}}))$$\n$$[B, 9, 64]$$",
        cb_x + 40, 810, 420, 52,
        fill="#FEF5E7", stroke="#D35400", font_color="#7E5109", font_size=10
    )
    b_self1 = add_box(
        "Joint Self-Attention & FFN: $$\\text{SelfAttn}([Z_{\\text{patch}}^{(0)}; G_{\\text{ts}}]) \\in \\mathbb{R}^{B \\times 10 \\times 64}$$",
        cb_x + 40, 880, 420, 45,
        fill="#FEF5E7", stroke="#D35400", font_color="#7E5109", font_size=10
    )
    b_cross1 = add_box(
        "Indicators Cross-Attention\n$$G_{\\text{ts}} \\leftarrow \\text{LayerNorm}(G_{\\text{ts}} + \\text{CrossAttn}(G_{\\text{ts}}, TS_{\\text{exo}}))$$",
        cb_x + 40, 940, 420, 48,
        fill="#FEF5E7", stroke="#D35400", font_color="#7E5109", font_size=10
    )
    b_gts = add_box(
        "Trainable Query Token\n$$G_{\\text{ts}} = \\text{nn.Parameter}$$\n$$[B, 1, 64]$$",
        cb_x + 480, 940, 180, 48,
        fill="#FDEBD0", stroke="#B9770E", font_color="#7D6608", font_size=10, bold=True
    )

    # B - Architecture Note Box
    add_box(
        "Structural Note on c1_hierarchical:\n"
        "Non-canonical attention order: Cross-Attn runs before Self-Attn so that G_ts absorbs exogenous features before being mixed with patches in a single layer pass. Gated bridge provides residual modulation.",
        cb_x + 20, 1020, 660, 40,
        fill="#FFFBEB", stroke="#D97706", font_color="#92400E", font_size=9, bold=False, rounded=1
    )

    # B - Embeddings
    b_patch_proj = add_box(
        "Price Patch Projection\n$$\\text{Linear}(30 \\to 64) \\implies Z_{\\text{patch}}^{(0)} \\in \\mathbb{R}^{B \\times 9 \\times 64}$$",
        cb_x + 40, 1080, 230, 60,
        fill="#E8F1FC", stroke="#2B6CB0", font_color="#1A365D", font_size=10
    )
    b_unfold = add_box(
        "Unfold (P=12, S=6) + rFFT\n$$2 \\times 12 = 24 \\text{ values} + 6 \\text{ FFT bins} = 30$$\n$$[B, 9, 30]$$",
        cb_x + 40, 1160, 230, 60,
        fill="#E8F1FC", stroke="#2B6CB0", font_color="#1A365D", font_size=10
    )
    b_ts_proj = add_box(
        "Indicator Projection\n$$\\text{Linear}(25 \\to 64)$$\n$$TS_{\\text{exo}} \\in \\mathbb{R}^{B \\times 60 \\times 64}$$",
        cb_x + 290, 1120, 180, 60,
        fill="#EAFBF1", stroke="#27AE60", font_color="#145A32", font_size=10
    )
    b_text_proj = add_box(
        "Text Projection\n$$\\text{Linear}(15 \\to 64)$$\n$$\\text{Text}_{\\text{exo}} \\in \\mathbb{R}^{B \\times 60 \\times 64}$$",
        cb_x + 490, 1120, 180, 60,
        fill="#F5EEF8", stroke="#8E44AD", font_color="#4A235A", font_size=10
    )

    # B - Inputs (Bottom)
    b_in_price = add_box(
        "Endogenous Price Input\n$$X_{\\text{price}} \\in \\mathbb{R}^{B \\times 60 \\times 2}$$\n[Close, Volume channels]",
        cb_x + 40, 1260, 230, 65,
        fill="#E8F1FC", stroke="#2B6CB0", font_color="#1A365D", font_size=10, bold=True
    )
    b_in_ts = add_box(
        "Exogenous Technical Series\n$$X_{\\text{ts}} \\in \\mathbb{R}^{B \\times 60 \\times 25}$$\n(Full Temporal Indicators)",
        cb_x + 290, 1260, 180, 65,
        fill="#EAFBF1", stroke="#27AE60", font_color="#145A32", font_size=10, bold=True
    )
    b_in_text = add_box(
        "Exogenous Text Series\n$$X_{\\text{text}} \\in \\mathbb{R}^{B \\times 60 \\times 15}$$\n(FinBERT Sentiment Scores)",
        cb_x + 490, 1260, 180, 65,
        fill="#F5EEF8", stroke="#8E44AD", font_color="#4A235A", font_size=10, bold=True
    )

    # B - Connections (Clean orthogonal routing without collisions)
    add_edge(b_in_price, b_unfold, "[B, 60, 2]")
    add_edge(b_unfold, b_patch_proj, "[B, 9, 30]")
    add_edge(b_in_ts, b_ts_proj, "[B, 60, 25]")
    add_edge(b_in_text, b_text_proj, "[B, 60, 15]")
    add_edge(b_gts, b_cross1, "[B, 1, 64]", exit_pt=(0.0, 0.5), entry_pt=(1.0, 0.5))
    add_edge(b_ts_proj, b_cross1, "[B, 60, 64]", exit_pt=(0.5, 0.0), entry_pt=(0.7, 1.0))
    add_edge(b_patch_proj, b_self1, "[B, 9, 64]", exit_pt=(0.5, 0.0), entry_pt=(0.2, 1.0))
    add_edge(b_cross1, b_self1, "updated G_ts", exit_pt=(0.5, 0.0), entry_pt=(0.5, 1.0))
    add_edge(b_self1, b_bridge1, "[B, 10, 64]", exit_pt=(0.5, 0.0), entry_pt=(0.5, 1.0))
    add_edge(b_gtext, b_cross2, "[B, 1, 64]", exit_pt=(0.0, 0.5), entry_pt=(1.0, 0.5))
    add_edge(b_text_proj, b_cross2, "[B, 60, 64]", exit_pt=(0.5, 0.0), entry_pt=(0.7, 1.0))
    add_edge(b_bridge1, b_self2, "Z_patch (L1)", exit_pt=(0.5, 0.0), entry_pt=(0.2, 1.0))
    add_edge(b_cross2, b_self2, "updated G_text", exit_pt=(0.5, 0.0), entry_pt=(0.5, 1.0))
    add_edge(b_self2, b_bridge2, "[B, 10, 64]", exit_pt=(0.5, 0.0), entry_pt=(0.5, 1.0))
    add_edge(b_bridge2, b_pool, "Z_patch (L2) [B, 9, 64]")
    add_edge(b_pool, b_head, "[B, 64]")
    add_edge(b_head, b_pred, "[B, 7]")


    # =========================================================================
    # COLUMN C: c1_inverted (Right: X = 1570 to 2270, Center ~ 1920)
    # =========================================================================
    cc_x = 1570
    cc_w = 700

    add_box(
        "(c) c1_inverted: Inverted Variate Cross-Attention\niTransformer Inversion: 25 Variate Tokens (Paper Test MSE: 0.7725, DA: 53.7%)",
        cc_x, 175, cc_w, 55,
        fill="#FAF5FF", stroke="#805AD5", font_color="#44337A", font_size=13, bold=True, rounded=1
    )

    # C - Pred & Head
    c_pred = add_box(
        "Forecast Horizon Output\n$$\\hat{Y} \\in \\mathbb{R}^{B \\times 7}$$\n(Multi-step Price Forecast)",
        cc_x + 225, 255, 250, 55,
        fill="#EBEDEF", stroke="#2C3E50", font_color="#1B2631", font_size=11, bold=True
    )
    c_head = add_box(
        "Forecast Head: Linear Projection\n$$\\text{Linear}(d_{\\text{model}}=64 \\to H=7)$$",
        cc_x + 225, 335, 250, 50,
        fill="#EBEDEF", stroke="#2C3E50", font_color="#1B2631", font_size=11
    )
    c_pool = add_box(
        "Patch Pooling (Last Patch from Inverted Layer)\n$$Z_{\\text{pool}} = Z_{\\text{patch}}[:, -1, :] \\in \\mathbb{R}^{B \\times 64}$$",
        cc_x + 200, 410, 300, 50,
        fill="#E8F1FC", stroke="#2B6CB0", font_color="#1A365D", font_size=11
    )

    # C - Inverted Layer Container
    add_box(
        "Inverted Layer (_InvertedLayer, e_layers = 1)\nTemporal Price Patches cross-attend to 25 Inverted Variate Tokens",
        cc_x + 20, 480, 660, 460,
        fill="#FDFEFE", stroke="#805AD5", font_color="#44337A", font_size=11, bold=True, dashed=1, align="left", valign="top"
    )
    c_ffn = add_box(
        "Feed-Forward Network & LayerNorm\n$$\\text{FFN}([Z_{\\text{patch}}; G_{\\text{text}}]) \\in \\mathbb{R}^{B \\times 10 \\times 64}$$",
        cc_x + 40, 535, 420, 48,
        fill="#FEF5E7", stroke="#D35400", font_color="#7E5109", font_size=10
    )
    c_cross_var = add_box(
        "Inverted Variate Cross-Attention (Core Inversion Mechanism)\n$$Q = Z_{\\text{patch}} \\in \\mathbb{R}^{B \\times 9 \\times 64} \\quad (9 \\text{ temporal price patches})$$\n$$K, V = V_{\\text{ts}} \\in \\mathbb{R}^{B \\times 25 \\times 64} \\quad (25 \\text{ indicator variate tokens})$$\n$$\\Delta_{\\text{var}} = \\text{CrossAttn}(Q, K, V) \\implies Z_{\\text{patch}} \\leftarrow \\text{LN}(Z_{\\text{patch}} + \\Delta_{\\text{var}})$$\nAllows each time patch to selectively attend to technical indicators",
        cc_x + 40, 605, 420, 100,
        fill="#FEF5E7", stroke="#D35400", font_color="#7E5109", font_size=10, bold=True
    )
    c_self = add_box(
        "Joint Patch-Text Self-Attention\n$$\\text{SelfAttn}([Z_{\\text{patch}}; G_{\\text{text}}]) \\in \\mathbb{R}^{B \\times 10 \\times 64}$$\nConditions price patches on text summary",
        cc_x + 40, 725, 420, 50,
        fill="#FEF5E7", stroke="#D35400", font_color="#7E5109", font_size=10
    )
    c_cross_text = add_box(
        "Text Cross-Attention\n$$G_{\\text{text}} \\leftarrow \\text{LN}(G_{\\text{text}} + \\text{CrossAttn}(G_{\\text{text}}, \\text{Text}_{\\text{exo}}))$$",
        cc_x + 40, 795, 420, 48,
        fill="#FEF5E7", stroke="#D35400", font_color="#7E5109", font_size=10
    )
    c_gtext = add_box(
        "Trainable Text Token\n$$G_{\\text{text}} = \\text{nn.Parameter}$$\n$$[B, 1, 64]$$",
        cc_x + 480, 795, 180, 48,
        fill="#FDEBD0", stroke="#B9770E", font_color="#7D6608", font_size=10, bold=True
    )

    # C - Embeddings & Inversion
    c_patch_proj = add_box(
        "Price Patch Projection + PosEmb\n$$\\text{Linear}(66 \\to 64) \\implies Z_{\\text{patch}} \\in \\mathbb{R}^{B \\times 9 \\times 64}$$",
        cc_x + 40, 1070, 210, 60,
        fill="#E8F1FC", stroke="#2B6CB0", font_color="#1A365D", font_size=10
    )
    c_unfold = add_box(
        "Unfold (P=12, S=6) + rFFT\n$$5 \\times 12 = 60 \\text{ values} + 6 \\text{ FFT bins} = 66$$\n$$[B, 9, 66]$$",
        cc_x + 40, 1160, 210, 60,
        fill="#E8F1FC", stroke="#2B6CB0", font_color="#1A365D", font_size=10
    )
    c_invert_proj = add_box(
        "iTransformer Variate Inversion\n$$X_{\\text{ts}}^\\top \\in \\mathbb{R}^{B \\times 25 \\times 60} \\xrightarrow{\\text{Linear}(T=60 \\to 64)}$$\n$$V_{\\text{ts}} \\in \\mathbb{R}^{B \\times 25 \\times 64} \\quad (25 \\text{ Variate Tokens})$$",
        cc_x + 270, 1090, 240, 75,
        fill="#EAFBF1", stroke="#27AE60", font_color="#145A32", font_size=10, bold=True
    )
    c_text_proj = add_box(
        "Text Linear Projection\n$$\\text{Linear}(15 \\to 64)$$\n$$\\text{Text}_{\\text{exo}} \\in \\mathbb{R}^{B \\times 60 \\times 64}$$",
        cc_x + 525, 1100, 160, 60,
        fill="#F5EEF8", stroke="#8E44AD", font_color="#4A235A", font_size=10
    )

    # C - Inputs (Bottom)
    c_in_price = add_box(
        "Endogenous Price Input\n$$X_{\\text{price}} \\in \\mathbb{R}^{B \\times 60 \\times 5}$$\n[Close, Volume, Open, High, Low]",
        cc_x + 40, 1260, 210, 65,
        fill="#E8F1FC", stroke="#2B6CB0", font_color="#1A365D", font_size=10, bold=True
    )
    c_in_ts = add_box(
        "Exogenous Technical Series\n$$X_{\\text{ts}} \\in \\mathbb{R}^{B \\times 60 \\times 25}$$\n(25 Indicator Curves over lookback $T=60$)",
        cc_x + 270, 1260, 240, 65,
        fill="#EAFBF1", stroke="#27AE60", font_color="#145A32", font_size=10, bold=True
    )
    c_in_text = add_box(
        "Exogenous Text Input\n$$X_{\\text{text}} \\in \\mathbb{R}^{B \\times 60 \\times 15}$$\n(FinBERT Sentiment Scores)",
        cc_x + 525, 1260, 160, 65,
        fill="#F5EEF8", stroke="#8E44AD", font_color="#4A235A", font_size=10, bold=True
    )

    # C - Connections
    add_edge(c_in_price, c_unfold, "[B, 60, 5]")
    add_edge(c_unfold, c_patch_proj, "[B, 9, 66]")
    add_edge(c_in_ts, c_invert_proj, "Transpose [B, 25, 60]")
    add_edge(c_in_text, c_text_proj, "[B, 60, 15]")
    add_edge(c_gtext, c_cross_text, "[B, 1, 64]", exit_pt=(0.0, 0.5), entry_pt=(1.0, 0.5))
    add_edge(c_text_proj, c_cross_text, "[B, 60, 64]")
    add_edge(c_cross_text, c_self, "updated G_text")
    add_edge(c_patch_proj, c_self, "Z_patch [B, 9, 64]")
    add_edge(c_self, c_cross_var, "[B, 9, 64] Query")
    add_edge(c_invert_proj, c_cross_var, "V_ts [B, 25, 64] Key/Val", exit_pt=(0.5, 0.0), entry_pt=(0.8, 1.0))
    add_edge(c_cross_var, c_ffn, "[B, 10, 64]")
    add_edge(c_ffn, c_pool, "Z_patch [B, 9, 64]")
    add_edge(c_pool, c_head, "[B, 64]")
    add_edge(c_head, c_pred, "[B, 7]")

    # Build full XML document
    xml_header = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<mxfile host="app.diagrams.net" modified="2026-10-10T00:00:00.000Z" agent="Antigravity" version="24.0.0" type="device">\n'
        '  <diagram id="model-comparison" name="Architecture Comparison">\n'
        '    <mxGraphModel dx="2400" dy="1600" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="2320" pageHeight="1450" math="1" shadow="0">\n'
        "      <root>\n"
        '        <mxCell id="0" />\n'
        '        <mxCell id="1" parent="0" />\n'
    )
    xml_footer = "      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n"

    return xml_header + "\n".join(elements) + "\n" + xml_footer


def create_standalone_svg() -> str:
    """Builds a high-resolution, self-contained SVG with clean, uncrowded vector graphics."""

    svg_w = 2320
    svg_h = 1420

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}" width="{svg_w}" height="{svg_h}" style="background-color: #FFFFFF; font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, Helvetica, Arial, sans-serif;">',
        "<defs>",
        '  <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '    <path d="M 0 1 L 10 5 L 0 9 z" fill="#4A5568" />',
        "  </marker>",
        '  <marker id="arrow-green" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '    <path d="M 0 1 L 10 5 L 0 9 z" fill="#27AE60" />',
        "  </marker>",
        '  <marker id="arrow-purple" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '    <path d="M 0 1 L 10 5 L 0 9 z" fill="#8E44AD" />',
        "  </marker>",
        '  <filter id="shadow" x="-3%" y="-3%" width="106%" height="106%">',
        '    <feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.06"/>',
        "  </filter>",
        "</defs>",
    ]

    def draw_box(x, y, w, h, fill, stroke, title, lines=None, stroke_width=1.5, rx=8, dashed=False, title_color="#1A202C", title_size=13, body_size=11, bold_title=True):
        dash_attr = ' stroke-dasharray="6,4"' if dashed else ""
        svg_parts.append(f'  <g filter="url(#shadow)">')
        svg_parts.append(f'    <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"{dash_attr} />')
        
        start_y = y + 20 if lines else y + h // 2 + 5
        weight = "bold" if bold_title else "600"
        svg_parts.append(f'    <text x="{x + w//2}" y="{start_y}" text-anchor="middle" fill="{title_color}" font-size="{title_size}" font-weight="{weight}">{html.escape(title)}</text>')
        if lines:
            line_y = start_y + 17
            for line in lines:
                color = "#4A5568" if not line.startswith("[") else "#2B6CB0"
                weight_line = "bold" if line.startswith("[") else "normal"
                svg_parts.append(f'    <text x="{x + w//2}" y="{line_y}" text-anchor="middle" fill="{color}" font-size="{body_size}" font-weight="{weight_line}">{html.escape(line)}</text>')
                line_y += 15
        svg_parts.append("  </g>")

    def draw_arrow(x1, y1, x2, y2, label="", color="#4A5568", marker="arrow", dashed=False):
        dash_attr = ' stroke-dasharray="4,3"' if dashed else ""
        svg_parts.append(f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.8"{dash_attr} marker-end="url(#{marker})" />')
        if label:
            mx = (x1 + x2) // 2
            my = (y1 + y2) // 2
            pad_w = len(label) * 6 + 12
            svg_parts.append(f'  <rect x="{mx - pad_w//2}" y="{my - 8}" width="{pad_w}" height="16" rx="3" fill="#FFFFFF" fill-opacity="0.9" />')
            svg_parts.append(f'  <text x="{mx}" y="{my + 4}" text-anchor="middle" fill="#2D3748" font-size="10" font-weight="600">{html.escape(label)}</text>')

    # 1. Header Banner
    svg_parts.append('  <rect x="50" y="25" width="2220" height="48" rx="8" fill="#2B6CB0" />')
    svg_parts.append('  <text x="1160" y="55" text-anchor="middle" fill="#FFFFFF" font-size="19" font-weight="bold">DecisionForecast Architecture Comparison: Exogenous Modality Interaction Paradigms</text>')
    
    svg_parts.append('  <rect x="50" y="80" width="2220" height="30" rx="6" fill="#EDF2F7" stroke="#CBD5E0" stroke-width="1" />')
    svg_parts.append('  <text x="1160" y="100" text-anchor="middle" fill="#4A5568" font-size="12" font-weight="500">Benchmarking Long-Term Series Forecasting (H=7) on FNSPID: Endogenous Price vs. Exogenous Technical Indicators vs. Text Sentiment</text>')

    # 2. Modality Legend Bar
    leg_y = 125
    svg_parts.append(f'  <text x="60" y="{leg_y + 20}" fill="#2D3748" font-size="12" font-weight="bold">MODALITY COLOR CODE:</text>')
    draw_box(240, leg_y, 250, 32, "#E8F1FC", "#2B6CB0", "Endogenous Price (OHLCV / FFT)", title_color="#1A365D", title_size=11, bold_title=True)
    draw_box(510, leg_y, 270, 32, "#EAFBF1", "#27AE60", "Exogenous Indicators (TreeSHAP 25)", title_color="#145A32", title_size=11, bold_title=True)
    draw_box(800, leg_y, 250, 32, "#F5EEF8", "#8E44AD", "Exogenous Text (FinBERT 15D)", title_color="#4A235A", title_size=11, bold_title=True)
    draw_box(1070, leg_y, 260, 32, "#FEF5E7", "#D35400", "Modality Attention & Gated Bridge", title_color="#7E5109", title_size=11, bold_title=True)
    draw_box(1350, leg_y, 220, 32, "#FDEBD0", "#B9770E", "Trainable Query Token (G)", title_color="#7D6608", title_size=11, bold_title=True)
    draw_box(1590, leg_y, 230, 32, "#EBEDEF", "#2C3E50", "Forecast Head & Prediction", title_color="#1B2631", title_size=11, bold_title=True)

    # =========================================================================
    # COLUMN A: c1_late_fusion
    # =========================================================================
    ca_x = 50
    draw_box(ca_x, 175, 700, 54, "#EBF8FF", "#3182CE", "(a) c1_late_fusion: TimeXer with Late Residual Fusion", ["Champion — Paper Test MSE: 0.7687, DA: 54.3% (Best: 0.7608) | Dev: 0.688"], title_color="#2B6CB0", title_size=13, bold_title=True)

    draw_box(ca_x + 225, 255, 250, 50, "#EBEDEF", "#2C3E50", "Forecast Horizon Output", ["Y_hat ∈ R^[B × 7] (Normalized Prices)", "[B, 7]"], title_color="#1B2631")
    draw_box(ca_x + 225, 335, 250, 48, "#EBEDEF", "#2C3E50", "Forecast Head: Linear Projection", ["Linear(d_model=64 → H=7)"], title_color="#1B2631")
    draw_box(ca_x + 150, 410, 400, 52, "#D5E8D4", "#27AE60", "Late Residual Fusion & Normalization", ["Z_final = LayerNorm(Z_pool + Z_tech)", "[B, 64]"], title_color="#196F3D", title_size=12, bold_title=True)
    draw_box(ca_x + 70, 490, 240, 48, "#E8F1FC", "#2B6CB0", "Patch Pooling (Last Patch)", ["Z_pool = Z_patch[:, -1, :]", "[B, 64]"], title_color="#1A365D")
    draw_box(ca_x + 380, 485, 270, 58, "#EAFBF1", "#27AE60", "Exogenous Technical MLP", ["Linear(25 → 64) → GELU → Linear(64 → 64)", "Z_tech ∈ R^[B × 64]"], title_color="#145A32")

    draw_box(ca_x + 20, 570, 420, 460, "#FAFAFA", "#CBD5E0", "TimeXer Encoder Backbone (e_layers = 1 or 2)", ["Processes Price Dynamics & Cross-Attends to Text Context"], title_color="#718096", title_size=11, bold_title=True, dashed=True)
    draw_box(ca_x + 40, 630, 380, 75, "#FEF5E7", "#D35400", "Global-to-Patch Gated Bridge", ["Δ = CrossAttn(Q=Z_patch, K=G_en, V=G_en)", "Z_patch ← LayerNorm(Z_patch + tanh(α) · Δ)", "[B, 9, 64]"], title_color="#7E5109", title_size=11)
    draw_box(ca_x + 40, 735, 380, 55, "#FEF5E7", "#D35400", "Joint Self-Attention & FFN", ["SelfAttn([Z_patch; G_en]) ∈ R^[B × 10 × 64]"], title_color="#7E5109", title_size=11)
    draw_box(ca_x + 40, 820, 380, 65, "#FEF5E7", "#D35400", "Text Cross-Attention", ["CrossAttn(Q=[Z_patch; G_en], K=X_text, V=X_text)", "Injects Sentiment Modality into Encoder"], title_color="#7E5109", title_size=11)
    draw_box(ca_x + 40, 915, 230, 45, "#EDF2F7", "#CBD5E0", "Sequence Concatenation", ["[Z_patch; G_en] ∈ R^[B × 10 × 64]"], title_color="#2D3748", title_size=10)
    draw_box(ca_x + 290, 915, 130, 45, "#FDEBD0", "#B9770E", "Trainable G_en", ["G_en ∈ R^[B × 1 × 64]"], title_color="#7D6608", title_size=10, bold_title=True)

    draw_box(ca_x + 40, 1070, 210, 58, "#E8F1FC", "#2B6CB0", "Patch Projection + PosEmb", ["Linear(66 → 64) + PosEmb", "Z_patch ∈ R^[B × 9 × 64]"], title_color="#1A365D", title_size=10)
    draw_box(ca_x + 40, 1160, 210, 58, "#E8F1FC", "#2B6CB0", "Time Unfold (P=12, S=6) + rFFT", ["5 × 12 = 60 values + 6 FFT bins", "[B, 9, 66]"], title_color="#1A365D", title_size=10)
    draw_box(ca_x + 270, 1100, 160, 58, "#F5EEF8", "#8E44AD", "Text Linear Projection", ["Linear(15 → 64)", "[B, 60, 64]"], title_color="#4A235A", title_size=10)
    draw_box(ca_x + 450, 1100, 230, 58, "#EAFBF1", "#27AE60", "Last Bar Slice (Bypasses Backbone)", ["X_ts[:, -1, :] ∈ R^[B × 25]", "Latest Regime State at Bar T"], title_color="#145A32", title_size=10, bold_title=True)

    draw_box(ca_x + 40, 1260, 210, 65, "#E8F1FC", "#2B6CB0", "Endogenous Price Input", ["X_price ∈ R^[B × 60 × 5]", "[Close, Vol, Open, High, Low]"], title_color="#1A365D", title_size=11, bold_title=True)
    draw_box(ca_x + 270, 1260, 160, 65, "#F5EEF8", "#8E44AD", "Exogenous Text", ["X_text ∈ R^[B × 60 × 15]", "FinBERT Compact Scores"], title_color="#4A235A", title_size=11, bold_title=True)
    draw_box(ca_x + 450, 1260, 230, 65, "#EAFBF1", "#27AE60", "Exogenous Indicators", ["X_ts ∈ R^[B × 60 × 25]", "TreeSHAP Technical Indicators"], title_color="#145A32", title_size=11, bold_title=True)

    draw_arrow(ca_x + 145, 1260, ca_x + 145, 1218, "[B, 60, 5]")
    draw_arrow(ca_x + 145, 1160, ca_x + 145, 1128, "[B, 9, 66]")
    draw_arrow(ca_x + 145, 1070, ca_x + 145, 960, "[B, 9, 64]")
    draw_arrow(ca_x + 350, 1260, ca_x + 350, 1158, "[B, 60, 15]")
    draw_arrow(ca_x + 350, 1100, ca_x + 350, 885, "[B, 60, 64]", marker="arrow-purple")
    draw_arrow(ca_x + 565, 1260, ca_x + 565, 1158, "[B, 60, 25]")
    draw_arrow(ca_x + 565, 1100, ca_x + 565, 543, "[B, 25]", marker="arrow-green")
    draw_arrow(ca_x + 290, 937, ca_x + 270, 937)
    draw_arrow(ca_x + 155, 915, ca_x + 155, 885)
    draw_arrow(ca_x + 230, 820, ca_x + 230, 790)
    draw_arrow(ca_x + 230, 735, ca_x + 230, 705)
    draw_arrow(ca_x + 230, 630, ca_x + 190, 538, "[B, 9, 64]")
    draw_arrow(ca_x + 190, 490, ca_x + 270, 462, "[B, 64]")
    draw_arrow(ca_x + 515, 485, ca_x + 430, 462, "[B, 64]")
    draw_arrow(ca_x + 350, 410, ca_x + 350, 383, "[B, 64]")
    draw_arrow(ca_x + 350, 335, ca_x + 350, 305, "[B, 7]")


    # =========================================================================
    # COLUMN B: c1_hierarchical
    # =========================================================================
    cb_x = 810
    draw_box(cb_x, 175, 700, 54, "#F0FFF4", "#38A169", "(b) c1_hierarchical: Cascaded Modality Hierarchy", ["Paper Test MSE: 0.7720 (DA: 53.4%) | Dev Test MSE: 0.702 (std=0.0023)"], title_color="#22543D", title_size=13, bold_title=True)

    draw_box(cb_x + 225, 255, 250, 50, "#EBEDEF", "#2C3E50", "Forecast Horizon Output", ["Y_hat ∈ R^[B × 7] (Multi-step Forecast)", "[B, 7]"], title_color="#1B2631")
    draw_box(cb_x + 225, 335, 250, 48, "#EBEDEF", "#2C3E50", "Forecast Head: Linear Projection", ["Linear(d_model=64 → H=7)"], title_color="#1B2631")
    draw_box(cb_x + 200, 410, 300, 48, "#E8F1FC", "#2B6CB0", "Patch Pooling (Last Patch)", ["Z_pool = Z_patch^(2)[:, -1, :]", "[B, 64]"], title_color="#1A365D")

    # Layer 2 Container
    draw_box(cb_x + 20, 480, 660, 250, "#FDFEFE", "#8E44AD", "Layer 2: Text Sentiment Modulation Layer (layer_text)", ["Absorbs FinBERT features into price patches via G_text & Gated Bridge 2"], title_color="#4A235A", title_size=11, bold_title=True, dashed=True)
    draw_box(cb_x + 40, 535, 420, 52, "#FEF5E7", "#D35400", "Gated Token Bridge 2", ["Z_patch^(2) = LN(Z_patch + tanh(α_2) · CrossAttn(Z_patch, G_text))", "[B, 9, 64]"], title_color="#7E5109", title_size=10)
    draw_box(cb_x + 40, 605, 420, 45, "#FEF5E7", "#D35400", "Joint Self-Attention & FFN", ["SelfAttn([Z_patch^(1); G_text]) ∈ R^[B × 10 × 64]"], title_color="#7E5109", title_size=10)
    draw_box(cb_x + 40, 665, 420, 48, "#FEF5E7", "#D35400", "Text Cross-Attention", ["G_text ← LN(G_text + CrossAttn(G_text, Text_exo))"], title_color="#7E5109", title_size=10)
    draw_box(cb_x + 480, 665, 180, 48, "#FDEBD0", "#B9770E", "Trainable G_text", ["G_text ∈ R^[B × 1 × 64]"], title_color="#7D6608", title_size=10, bold_title=True)

    # Layer 1 Container
    draw_box(cb_x + 20, 755, 660, 250, "#FDFEFE", "#27AE60", "Layer 1: Market Regime Indicator Layer (layer_ts)", ["Absorbs 25 technical indicator series into price patches via G_ts & Gated Bridge 1"], title_color="#145A32", title_size=11, bold_title=True, dashed=True)
    draw_box(cb_x + 40, 810, 420, 52, "#FEF5E7", "#D35400", "Gated Token Bridge 1", ["Z_patch^(1) = LN(Z_patch + tanh(α_1) · CrossAttn(Z_patch, G_ts))", "[B, 9, 64]"], title_color="#7E5109", title_size=10)
    draw_box(cb_x + 40, 880, 420, 45, "#FEF5E7", "#D35400", "Joint Self-Attention & FFN", ["SelfAttn([Z_patch^(0); G_ts]) ∈ R^[B × 10 × 64]"], title_color="#7E5109", title_size=10)
    draw_box(cb_x + 40, 940, 420, 48, "#FEF5E7", "#D35400", "Indicators Cross-Attention", ["G_ts ← LN(G_ts + CrossAttn(G_ts, TS_exo))"], title_color="#7E5109", title_size=10)
    draw_box(cb_x + 480, 940, 180, 48, "#FDEBD0", "#B9770E", "Trainable G_ts", ["G_ts ∈ R^[B × 1 × 64]"], title_color="#7D6608", title_size=10, bold_title=True)

    draw_box(cb_x + 20, 1020, 660, 40, "#FFFBEB", "#D97706", "Structural Note: Non-canonical order (Cross-Attn -> Self-Attn -> Bridge)", ["Allows G_ts to absorb indicators before single-pass mixing with price patches"], title_color="#92400E", title_size=9, bold_title=True)

    draw_box(cb_x + 40, 1080, 230, 58, "#E8F1FC", "#2B6CB0", "Price Patch Projection", ["Linear(30 → 64) + PosEmb", "Z_patch^(0) ∈ R^[B × 9 × 64]"], title_color="#1A365D", title_size=10)
    draw_box(cb_x + 40, 1160, 230, 58, "#E8F1FC", "#2B6CB0", "Unfold (P=12, S=6) + rFFT", ["2 × 12 = 24 values + 6 FFT bins", "[B, 9, 30]"], title_color="#1A365D", title_size=10)
    draw_box(cb_x + 290, 1120, 180, 58, "#EAFBF1", "#27AE60", "Indicator Projection", ["Linear(25 → 64)", "TS_exo ∈ R^[B × 60 × 64]"], title_color="#145A32", title_size=10)
    draw_box(cb_x + 490, 1120, 180, 58, "#F5EEF8", "#8E44AD", "Text Projection", ["Linear(15 → 64)", "Text_exo ∈ R^[B × 60 × 64]"], title_color="#4A235A", title_size=10)

    draw_box(cb_x + 40, 1260, 230, 65, "#E8F1FC", "#2B6CB0", "Endogenous Price Input", ["X_price ∈ R^[B × 60 × 2]", "[Close, Volume channels]"], title_color="#1A365D", title_size=11, bold_title=True)
    draw_box(cb_x + 290, 1260, 180, 65, "#EAFBF1", "#27AE60", "Exogenous Indicators", ["X_ts ∈ R^[B × 60 × 25]", "Full Temporal Indicator Series"], title_color="#145A32", title_size=11, bold_title=True)
    draw_box(cb_x + 490, 1260, 180, 65, "#F5EEF8", "#8E44AD", "Exogenous Text", ["X_text ∈ R^[B × 60 × 15]", "FinBERT Compact Scores"], title_color="#4A235A", title_size=11, bold_title=True)

    draw_arrow(cb_x + 155, 1260, cb_x + 155, 1218, "[B, 60, 2]")
    draw_arrow(cb_x + 155, 1160, cb_x + 155, 1138, "[B, 9, 30]")
    draw_arrow(cb_x + 155, 1080, cb_x + 155, 925, "Z_patch^(0) [B, 9, 64]")
    draw_arrow(cb_x + 380, 1260, cb_x + 380, 1178, "[B, 60, 25]")
    draw_arrow(cb_x + 380, 1120, cb_x + 380, 988, "[B, 60, 64]", marker="arrow-green")
    draw_arrow(cb_x + 580, 1260, cb_x + 580, 1178, "[B, 60, 15]")
    draw_arrow(cb_x + 580, 1120, cb_x + 580, 713, "[B, 60, 64]", marker="arrow-purple")
    draw_arrow(cb_x + 480, 964, cb_x + 460, 964)
    draw_arrow(cb_x + 250, 940, cb_x + 250, 925)
    draw_arrow(cb_x + 250, 880, cb_x + 250, 862)
    draw_arrow(cb_x + 250, 810, cb_x + 250, 650, "Z_patch^(1) [B, 9, 64]")
    draw_arrow(cb_x + 480, 689, cb_x + 460, 689)
    draw_arrow(cb_x + 250, 665, cb_x + 250, 650)
    draw_arrow(cb_x + 250, 605, cb_x + 250, 587)
    draw_arrow(cb_x + 250, 535, cb_x + 320, 458, "Z_patch^(2) [B, 9, 64]")
    draw_arrow(cb_x + 350, 410, cb_x + 350, 383, "[B, 64]")
    draw_arrow(cb_x + 350, 335, cb_x + 350, 305, "[B, 7]")


    # =========================================================================
    # COLUMN C: c1_inverted
    # =========================================================================
    cc_x = 1570
    draw_box(cc_x, 175, 700, 54, "#FAF5FF", "#805AD5", "(c) c1_inverted: Inverted Variate Cross-Attention", ["Paper Test MSE: 0.7725 (Best: 0.7624, DA: 53.7%) — 25 Variate Tokens"], title_color="#44337A", title_size=13, bold_title=True)

    draw_box(cc_x + 225, 255, 250, 50, "#EBEDEF", "#2C3E50", "Forecast Horizon Output", ["Y_hat ∈ R^[B × 7] (Multi-step Forecast)", "[B, 7]"], title_color="#1B2631")
    draw_box(cc_x + 225, 335, 250, 48, "#EBEDEF", "#2C3E50", "Forecast Head: Linear Projection", ["Linear(d_model=64 → H=7)"], title_color="#1B2631")
    draw_box(cc_x + 200, 410, 300, 48, "#E8F1FC", "#2B6CB0", "Patch Pooling (Last Patch)", ["Z_pool = Z_patch[:, -1, :]", "[B, 64]"], title_color="#1A365D")

    draw_box(cc_x + 20, 480, 660, 460, "#FDFEFE", "#805AD5", "Inverted Layer (_InvertedLayer, e_layers = 1)", ["Temporal Price Patches cross-attend to 25 Inverted Variate Tokens"], title_color="#44337A", title_size=11, bold_title=True, dashed=True)
    draw_box(cc_x + 40, 535, 420, 48, "#FEF5E7", "#D35400", "Feed-Forward Network & LayerNorm", ["FFN([Z_patch; G_text]) ∈ R^[B × 10 × 64]"], title_color="#7E5109", title_size=10)
    draw_box(cc_x + 40, 605, 420, 100, "#FEF5E7", "#D35400", "Inverted Variate Cross-Attention (Core Inversion)", [
        "Q = Z_patch ∈ R^[B × 9 × 64] (9 temporal patches)",
        "K, V = V_ts ∈ R^[B × 25 × 64] (25 variate tokens)",
        "Δ_var = CrossAttn(Q, K, V)  →  Z_patch ← LN(Z_patch + Δ_var)",
        "Each time patch selectively queries technical indicators"
    ], title_color="#7E5109", title_size=11, bold_title=True)
    draw_box(cc_x + 40, 725, 420, 50, "#FEF5E7", "#D35400", "Joint Patch-Text Self-Attention", ["SelfAttn([Z_patch; G_text]) ∈ R^[B × 10 × 64]", "Conditions price patches on text sentiment"], title_color="#7E5109", title_size=10)
    draw_box(cc_x + 40, 795, 420, 48, "#FEF5E7", "#D35400", "Text Cross-Attention", ["G_text ← LN(G_text + CrossAttn(G_text, Text_exo))"], title_color="#7E5109", title_size=10)
    draw_box(cc_x + 480, 795, 180, 48, "#FDEBD0", "#B9770E", "Trainable G_text", ["G_text ∈ R^[B × 1 × 64]"], title_color="#7D6608", title_size=10, bold_title=True)

    draw_box(cc_x + 40, 1070, 210, 58, "#E8F1FC", "#2B6CB0", "Patch Projection + PosEmb", ["Linear(66 → 64) + PosEmb", "Z_patch ∈ R^[B × 9 × 64]"], title_color="#1A365D", title_size=10)
    draw_box(cc_x + 40, 1160, 210, 58, "#E8F1FC", "#2B6CB0", "Unfold (P=12, S=6) + rFFT", ["5 × 12 = 60 values + 6 FFT bins", "[B, 9, 66]"], title_color="#1A365D", title_size=10)
    draw_box(cc_x + 270, 1090, 240, 72, "#EAFBF1", "#27AE60", "iTransformer Variate Inversion", [
        "X_ts^T ∈ R^[B × 25 × 60] → Linear(T=60 → 64)",
        "V_ts ∈ R^[B × 25 × 64] (25 Variate Tokens)",
        "Lookback window projected to d_model"
    ], title_color="#145A32", title_size=10, bold_title=True)
    draw_box(cc_x + 525, 1100, 160, 58, "#F5EEF8", "#8E44AD", "Text Linear Projection", ["Linear(15 → 64)", "[B, 60, 64]"], title_color="#4A235A", title_size=10)

    draw_box(cc_x + 40, 1260, 210, 65, "#E8F1FC", "#2B6CB0", "Endogenous Price Input", ["X_price ∈ R^[B × 60 × 5]", "[Close, Vol, Open, High, Low]"], title_color="#1A365D", title_size=11, bold_title=True)
    draw_box(cc_x + 270, 1260, 240, 65, "#EAFBF1", "#27AE60", "Exogenous Indicators", ["X_ts ∈ R^[B × 60 × 25]", "25 Indicator Curves over lookback T=60"], title_color="#145A32", title_size=11, bold_title=True)
    draw_box(cc_x + 525, 1260, 160, 65, "#F5EEF8", "#8E44AD", "Exogenous Text", ["X_text ∈ R^[B × 60 × 15]", "FinBERT Compact Scores"], title_color="#4A235A", title_size=11, bold_title=True)

    draw_arrow(cc_x + 145, 1260, cc_x + 145, 1218, "[B, 60, 5]")
    draw_arrow(cc_x + 145, 1160, cc_x + 145, 1128, "[B, 9, 66]")
    draw_arrow(cc_x + 145, 1070, cc_x + 145, 775, "Z_patch [B, 9, 64]")
    draw_arrow(cc_x + 390, 1260, cc_x + 390, 1162, "Transpose [B, 25, 60]")
    draw_arrow(cc_x + 390, 1090, cc_x + 420, 705, "V_ts [B, 25, 64]", color="#27AE60", marker="arrow-green")
    draw_arrow(cc_x + 605, 1260, cc_x + 605, 1158, "[B, 60, 15]")
    draw_arrow(cc_x + 605, 1100, cc_x + 605, 843, "[B, 60, 64]", marker="arrow-purple")
    draw_arrow(cc_x + 480, 819, cc_x + 460, 819)
    draw_arrow(cc_x + 250, 795, cc_x + 250, 775)
    draw_arrow(cc_x + 250, 725, cc_x + 250, 705, "[B, 9, 64] Query")
    draw_arrow(cc_x + 250, 605, cc_x + 250, 583)
    draw_arrow(cc_x + 250, 535, cc_x + 320, 458, "Z_patch [B, 9, 64]")
    draw_arrow(cc_x + 350, 410, cc_x + 350, 383, "[B, 64]")
    draw_arrow(cc_x + 350, 335, cc_x + 350, 305, "[B, 7]")

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def main() -> None:
    out_dir = Path("Articles/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    drawio_path = out_dir / "model_comparison_architectures.drawio"
    svg_path = out_dir / "model_comparison_architectures.svg"

    print("Generating updated Draw.io XML with clean spacing...")
    drawio_xml = create_drawio_xml()
    drawio_path.write_text(drawio_xml, encoding="utf-8")
    print(f"Saved Draw.io file to: {drawio_path} ({len(drawio_xml)} bytes)")

    print("Generating updated standalone publication SVG...")
    svg_content = create_standalone_svg()
    svg_path.write_text(svg_content, encoding="utf-8")
    print(f"Saved standalone SVG to: {svg_path} ({len(svg_content)} bytes)")


if __name__ == "__main__":
    main()
