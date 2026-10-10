"""Generate publication-grade architecture diagrams for UPDATED Canonical TimeXer Dual and Hierarchical.

Complies with NeurIPS/ICLR/IEEE visual conventions:
- Soft Blue/Slate for Endogenous Price / OHLCV (#4A90E2 / #E8F1FC)
- Emerald Green for Exogenous Technical Indicators (#2ECC71 / #EAFBF1)
- Purple/Violet for Exogenous Text / Sentiment (#9B59B6 / #F5EEF8)
- Amber/Coral for Transformer Attention Layers (#E67E22 / #FEF5E7)
- Cyan/Teal for Frequency / FFT Embedding (#00A896 / #E0F7F6)
- Explicit tensor dimensions at each step
"""

from __future__ import annotations

import html
from pathlib import Path


def create_dual_diagram() -> tuple[str, str]:
    """Build Draw.io XML and standalone SVG for Canonical TimeXer Dual."""
    # We will build draw.io elements
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
        dashed: int = 0,
        font_color: str = "#2D3748",
        exit_pt: tuple[float, float] | None = None,
        entry_pt: tuple[float, float] | None = None,
    ) -> str:
        cid = next_id()
        dash_str = "dashed=1;dashPattern=6 3;" if dashed else ""
        exit_str = f"exitX={exit_pt[0]};exitY={exit_pt[1]};exitDx=0;exitDy=0;" if exit_pt else ""
        entry_str = f"entryX={entry_pt[0]};entryY={entry_pt[1]};entryDx=0;entryDy=0;" if entry_pt else ""
        escaped_label = html.escape(label).replace("\n", "&#xa;")
        style = (
            f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
            f"strokeColor={stroke};strokeWidth={stroke_width};fontColor={font_color};fontSize=10;"
            f"fontFamily=Helvetica,Arial,sans-serif;{dash_str}{exit_str}{entry_str}"
        )
        xml = (
            f'<mxCell id="{cid}" value="{escaped_label}" style="{style}" edge="1" parent="1" source="{src}" target="{tgt}">\n'
            f'  <mxGeometry relative="1" as="geometry" />\n'
            f"</mxCell>"
        )
        elements.append(xml)
        return cid

    # Top Title
    add_box(
        "Canonical TimeXer Dual Architecture (c1_dual)\n"
        "NeurIPS 2024 / TSLib Canonical Formulation with Parallel Cross-Modal Bridge Tokens",
        30, 20, 1140, 50,
        fill="#2B6CB0", stroke="#1A365D", font_color="#FFFFFF", font_size=16, bold=True
    )

    # 1. Output Forecast
    pred_box = add_box(
        "Forecast Horizon Output\n$$\\hat{y} \\in \\mathbb{R}^{B \\times H} \\quad (H = 7 \\text{ days Close price})$$",
        450, 90, 300, 50,
        fill="#E8F8F5", stroke="#1ABC9C", font_color="#0E6251", font_size=12, bold=True
    )

    # 2. Flatten Head
    head_box = add_box(
        "FlattenHead (Close Channel Target Projection)\n"
        "Extract Close tokens: $enc_{\\text{close}} = tokens[:, \\text{close\\_idx}, :, :] \\in \\mathbb{R}^{B \\times (N+2) \\times D}$\n"
        "Flatten: $(N+2) \\times D \\xrightarrow{\\text{nn.Linear}} H=7$\n"
        "Optional: Prototype Residual Consolidation",
        360, 160, 480, 65,
        fill="#EDF2F7", stroke="#4A5568", font_color="#1A202C", font_size=11, bold=True
    )

    # 3. Layer 2 Container
    add_box(
        "CANONICAL DUAL ENCODER LAYER 2 (of 2)\nSame structure: Cross-attentions update G_ts & G_text; Self-attention broadcasts knowledge into patches",
        40, 245, 1120, 280,
        fill="#F7FAFC", stroke="#CBD5E0", font_color="#2D3748", font_size=12, bold=True, dashed=1
    )

    l2_ffn = add_box(
        "Conv1d FFN + Residual + LayerNorm\n"
        "Conv1d(1x1) -> GELU -> Conv1d(1x1) over all $(N+2)$ tokens $\\in \\mathbb{R}^{(B \\times C) \\times (N+2) \\times D}$",
        380, 280, 440, 45,
        fill="#EDF2F7", stroke="#A0AEC0", font_color="#2D3748", font_size=10, bold=True
    )

    l2_cross_ts = add_box(
        "Cross-Attention TS\n"
        "$Q = G_{\\text{ts}}^{(2)} [B\\cdot C, 1, D]$\n"
        "$K, V = cross_{\\text{ts}} [B, 25, D]$\n"
        "$G_{\\text{ts}} \\leftarrow \\text{Norm}(G_{\\text{ts}} + \\Delta_{\\text{ts}})$",
        150, 345, 230, 75,
        fill="#EAFBF1", stroke="#27AE60", font_color="#145A32", font_size=10, bold=True
    )

    l2_cross_text = add_box(
        "Cross-Attention Text\n"
        "$Q = G_{\\text{text}}^{(2)} [B\\cdot C, 1, D]$\n"
        "$K, V = cross_{\\text{text}} [B, 15, D]$\n"
        "$G_{\\text{text}} \\leftarrow \\text{Norm}(G_{\\text{text}} + \\Delta_{\\text{text}})$",
        820, 345, 230, 75,
        fill="#F5EEF8", stroke="#8E44AD", font_color="#4A235A", font_size=10, bold=True
    )

    l2_self = add_box(
        "Self-Attention (Full Temporal & Inter-Modal Attention)\n"
        "Input: $[P_1^{(1)}, \\dots, P_N^{(1)}; G_{\\text{ts}}^{(1)}; G_{\\text{text}}^{(1)}] \\in \\mathbb{R}^{(B \\times C) \\times (N+2) \\times D}$\n"
        "PRICE PATCHES ABSORB EXOGENOUS KNOWLEDGE FROM G_ts AND G_text UPDATED IN LAYER 1!",
        240, 440, 720, 65,
        fill="#FEF5E7", stroke="#D35400", font_color="#7E5109", font_size=11, bold=True
    )

    # 4. Layer 1 Container
    add_box(
        "CANONICAL DUAL ENCODER LAYER 1 (of 2)\nInput: Initialized Endogenous Patches + Learnable Bridge Tokens [P; G_ts; G_text]",
        40, 545, 1120, 280,
        fill="#F7FAFC", stroke="#CBD5E0", font_color="#2D3748", font_size=12, bold=True, dashed=1
    )

    l1_ffn = add_box(
        "Conv1d FFN + Residual + LayerNorm\n"
        "Conv1d(1x1) -> GELU -> Conv1d(1x1) over all $(N+2)$ tokens $\\in \\mathbb{R}^{(B \\times C) \\times (N+2) \\times D}$",
        380, 580, 440, 45,
        fill="#EDF2F7", stroke="#A0AEC0", font_color="#2D3748", font_size=10, bold=True
    )

    l1_cross_ts = add_box(
        "Cross-Attention TS\n"
        "$Q = G_{\\text{ts}} [B\\cdot C, 1, D]$\n"
        "$K, V = cross_{\\text{ts}} [B, 25, D]$\n"
        "$G_{\\text{ts}} \\leftarrow \\text{Norm}(G_{\\text{ts}} + \\Delta_{\\text{ts}})$",
        150, 645, 230, 75,
        fill="#EAFBF1", stroke="#27AE60", font_color="#145A32", font_size=10, bold=True
    )

    l1_cross_text = add_box(
        "Cross-Attention Text\n"
        "$Q = G_{\\text{text}} [B\\cdot C, 1, D]$\n"
        "$K, V = cross_{\\text{text}} [B, 15, D]$\n"
        "$G_{\\text{text}} \\leftarrow \\text{Norm}(G_{\\text{text}} + \\Delta_{\\text{text}})$",
        820, 645, 230, 75,
        fill="#F5EEF8", stroke="#8E44AD", font_color="#4A235A", font_size=10, bold=True
    )

    l1_self = add_box(
        "Self-Attention (Full Temporal & Inter-Modal Attention)\n"
        "Input: $[P_1, \\dots, P_N; G_{\\text{ts}}; G_{\\text{text}}] \\in \\mathbb{R}^{(B \\times C) \\times (N+2) \\times D}$\n"
        "G_ts and G_text aggregate endogenous price dynamics; Patches exchange time structure.",
        240, 740, 720, 65,
        fill="#FEF5E7", stroke="#D35400", font_color="#7E5109", font_size=11, bold=True
    )

    # 5. Embeddings & Inputs Row
    en_embed = add_box(
        "EnEmbeddingDual (Endogenous Patching & Bridges)\n"
        "1. Unfold OHLCV: $[B, 5, T=60] \\xrightarrow{P=16, S=12} [B, 5, N_p, 16]$\n"
        "2. Value Projection: $\\text{Linear}(16 \\to D) + E_{\\text{pos}}$\n"
        "3. Local Patch-FFT: $+ \\text{Linear}(6 \\to D)(\\text{rFFT}(\\text{Close}))$\n"
        "4. Learnable Bridges: $[G_{\\text{ts}}, G_{\\text{text}}] \\in \\mathbb{R}^{1 \\times 5 \\times 2 \\times D}$\n"
        "Output: Tokens $[B \\times 5, N_p + 2, D]$",
        350, 850, 500, 95,
        fill="#E8F1FC", stroke="#2B6CB0", font_color="#1A365D", font_size=10, bold=True
    )

    ts_proj = add_box(
        "Inverted TS Projection\n"
        "$X_{\\text{ts}}^\\top \\in \\mathbb{R}^{B \\times 25 \\times 60}$\n"
        "$\\text{Linear}(60 \\to D)$\n"
        "$cross_{\\text{ts}} \\in \\mathbb{R}^{B \\times 25 \\times D}$",
        80, 855, 230, 85,
        fill="#EAFBF1", stroke="#27AE60", font_color="#145A32", font_size=10, bold=True
    )

    text_proj = add_box(
        "Text Projection\n"
        "$X_{\\text{text}} \\in \\mathbb{R}^{B \\times 15}$\n"
        "$\\text{Linear}(15 \\to D)$\n"
        "$cross_{\\text{text}} \\in \\mathbb{R}^{B \\times 1 \\times D}$",
        890, 855, 230, 85,
        fill="#F5EEF8", stroke="#8E44AD", font_color="#4A235A", font_size=10, bold=True
    )

    # Inputs at Bottom
    in_ts = add_box(
        "25 Exogenous TS Features\n$$X_{\\text{ts}} \\in \\mathbb{R}^{B \\times 60 \\times 25}$$\n(TreeSHAP Indicators)",
        80, 970, 230, 55,
        fill="#EAFBF1", stroke="#27AE60", font_color="#145A32", font_size=10, bold=True
    )

    in_endo = add_box(
        "Endogenous Price Channels (OHLCV)\n$$X_{\\text{endo}} \\in \\mathbb{R}^{B \\times 60 \\times 5}$$\n(Close, Volume, Open, High, Low)",
        350, 970, 500, 55,
        fill="#E8F1FC", stroke="#2B6CB0", font_color="#1A365D", font_size=11, bold=True
    )

    in_text = add_box(
        "15D Compact News Text\n$$X_{\\text{text}} \\in \\mathbb{R}^{B \\times 15}$$\n(FinBERT Sentiment)",
        890, 970, 230, 55,
        fill="#F5EEF8", stroke="#8E44AD", font_color="#4A235A", font_size=10, bold=True
    )

    # Connections
    add_edge(in_ts, ts_proj, "[B, 60, 25]")
    add_edge(in_endo, en_embed, "[B, 60, 5]")
    add_edge(in_text, text_proj, "[B, 15]")

    add_edge(en_embed, l1_self, "[B*5, N+2, D]")
    add_edge(ts_proj, l1_cross_ts, "cross_ts [B, 25, D]", exit_pt=(0.5, 0.0), entry_pt=(0.5, 1.0))
    add_edge(text_proj, l1_cross_text, "cross_text [B, 1, D]", exit_pt=(0.5, 0.0), entry_pt=(0.5, 1.0))

    # Cross connections L1
    add_edge(l1_self, l1_cross_ts, "Slice G_ts [B*5, 1, D]", exit_pt=(0.2, 0.0), entry_pt=(0.5, 1.0))
    add_edge(l1_self, l1_cross_text, "Slice G_text [B*5, 1, D]", exit_pt=(0.8, 0.0), entry_pt=(0.5, 1.0))
    add_edge(l1_cross_ts, l1_ffn, "G_ts updated", exit_pt=(0.5, 0.0), entry_pt=(0.2, 1.0))
    add_edge(l1_cross_text, l1_ffn, "G_text updated", exit_pt=(0.5, 0.0), entry_pt=(0.8, 1.0))

    # L1 to L2
    add_edge(l1_ffn, l2_self, "Tokens [B*5, N+2, D]", exit_pt=(0.5, 0.0), entry_pt=(0.5, 1.0))
    add_edge(ts_proj, l2_cross_ts, "cross_ts [B, 25, D]", exit_pt=(0.1, 0.0), entry_pt=(0.1, 1.0))
    add_edge(text_proj, l2_cross_text, "cross_text [B, 1, D]", exit_pt=(0.9, 0.0), entry_pt=(0.9, 1.0))

    # Cross connections L2
    add_edge(l2_self, l2_cross_ts, "Slice G_ts", exit_pt=(0.2, 0.0), entry_pt=(0.5, 1.0))
    add_edge(l2_self, l2_cross_text, "Slice G_text", exit_pt=(0.8, 0.0), entry_pt=(0.5, 1.0))
    add_edge(l2_cross_ts, l2_ffn, "G_ts updated", exit_pt=(0.5, 0.0), entry_pt=(0.2, 1.0))
    add_edge(l2_cross_text, l2_ffn, "G_text updated", exit_pt=(0.5, 0.0), entry_pt=(0.8, 1.0))

    # L2 to Head
    add_edge(l2_ffn, head_box, "[B*5, N+2, D]", exit_pt=(0.5, 0.0), entry_pt=(0.5, 1.0))
    add_edge(head_box, pred_box, "[B, 7]", exit_pt=(0.5, 0.0), entry_pt=(0.5, 1.0))

    drawio_content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<mxfile host="app.diagrams.net" modified="2026-10-10T14:45:00.000Z" agent="Antigravity" version="21.0.0" type="device">\n'
        '  <diagram id="canonical_c1_dual" name="Canonical TimeXer Dual">\n'
        '    <mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1200" pageHeight="1100" math="1" shadow="0">\n'
        '      <root>\n'
        '        <mxCell id="0" />\n'
        '        <mxCell id="1" parent="0" />\n'
        + "".join(elements) +
        '\n      </root>\n'
        '    </mxGraphModel>\n'
        '  </diagram>\n'
        '</mxfile>'
    )
    return drawio_content


def create_hierarchical_diagram() -> tuple[str, str]:
    """Build Draw.io XML for Canonical TimeXer Hierarchical (3 Layers)."""
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
        dashed: int = 0,
        font_color: str = "#2D3748",
        exit_pt: tuple[float, float] | None = None,
        entry_pt: tuple[float, float] | None = None,
    ) -> str:
        cid = next_id()
        dash_str = "dashed=1;dashPattern=6 3;" if dashed else ""
        exit_str = f"exitX={exit_pt[0]};exitY={exit_pt[1]};exitDx=0;exitDy=0;" if exit_pt else ""
        entry_str = f"entryX={entry_pt[0]};entryY={entry_pt[1]};entryDx=0;entryDy=0;" if entry_pt else ""
        escaped_label = html.escape(label).replace("\n", "&#xa;")
        style = (
            f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
            f"strokeColor={stroke};strokeWidth={stroke_width};fontColor={font_color};fontSize=10;"
            f"fontFamily=Helvetica,Arial,sans-serif;{dash_str}{exit_str}{entry_str}"
        )
        xml = (
            f'<mxCell id="{cid}" value="{escaped_label}" style="{style}" edge="1" parent="1" source="{src}" target="{tgt}">\n'
            f'  <mxGeometry relative="1" as="geometry" />\n'
            f"</mxCell>"
        )
        elements.append(xml)
        return cid

    # Top Title
    add_box(
        "Canonical TimeXer Hierarchical Architecture (c1_hierarchical)\n"
        "3-Layer Cross-Modal Progression: Technical Micro-structure -> Semantic Macro-structure -> Joint Consolidation",
        30, 20, 1140, 50,
        fill="#2B6CB0", stroke="#1A365D", font_color="#FFFFFF", font_size=16, bold=True
    )

    # 1. Output Forecast
    pred_box = add_box(
        "Forecast Horizon Output\n$$\\hat{y} \\in \\mathbb{R}^{B \\times H} \\quad (H = 7 \\text{ days Close price})$$",
        450, 90, 300, 50,
        fill="#E8F8F5", stroke="#1ABC9C", font_color="#0E6251", font_size=12, bold=True
    )

    # 2. Flatten Head
    head_box = add_box(
        "FlattenHead (Target Variable Projection)\n"
        "Extract Close: $enc_{\\text{close}} = tokens[:, \\text{close\\_idx}, :, :] \\in \\mathbb{R}^{B \\times (N+2) \\times D}$\n"
        "Flatten tokens -> Linear -> Forecast Horizon $H=7$",
        360, 160, 480, 55,
        fill="#EDF2F7", stroke="#4A5568", font_color="#1A202C", font_size=11, bold=True
    )

    # 3. Layer 3 Container (Consolidation)
    add_box(
        "LAYER 3: CROSS-MODAL CONSOLIDATION (Endogenous Integration)\n"
        "All price patches receive joint consolidated signals from enriched G_text^(2) and G_ts^(2)",
        50, 235, 1100, 160,
        fill="#FEF9E7", stroke="#F39C12", font_color="#7D6608", font_size=12, bold=True, dashed=1
    )

    l3_ffn = add_box(
        "Conv1d FFN + LayerNorm\nConv1d(1x1) -> GELU -> Conv1d(1x1) over all $(N+2)$ tokens",
        380, 270, 440, 40,
        fill="#EDF2F7", stroke="#CBD5E0", font_color="#2D3748", font_size=10, bold=True
    )

    l3_self = add_box(
        "Consolidation Self-Attention over [P; G_ts; G_text]\n"
        "$$\\text{SelfAttn}(tokens, tokens, tokens) \\quad \\text{in } \\mathbb{R}^{(B \\times C) \\times (N+2) \\times D}$$\n"
        "Cross-Attention is omitted here; patches naturally absorb and harmonize news sentiment and technical state!",
        240, 325, 720, 55,
        fill="#FEF5E7", stroke="#D35400", font_color="#7E5109", font_size=11, bold=True
    )

    # 4. Layer 2 Container (Macro / Text)
    add_box(
        "LAYER 2: SEMANTIC MACRO-STRUCTURE (News Sentiment Attention)\n"
        "Only G_text queries cross_text; Patches and G_text absorb updated technical token G_ts^(1)",
        50, 415, 1100, 220,
        fill="#FDFEFE", stroke="#8E44AD", font_color="#4A235A", font_size=12, bold=True, dashed=1
    )

    l2_ffn = add_box(
        "Conv1d FFN + LayerNorm\nTokens: $[P^{(2)}; G_{\\text{ts}}^{(2)}; G_{\\text{text}}^{(2)}] \\in \\mathbb{R}^{(B\\cdot C) \\times (N+2) \\times D}$",
        380, 450, 440, 40,
        fill="#EDF2F7", stroke="#CBD5E0", font_color="#2D3748", font_size=10, bold=True
    )

    l2_cross_text = add_box(
        "Cross-Attention ONLY G_text\n"
        "$Q = G_{\\text{text}}^{(1)} [B\\cdot C, 1, D]$\n"
        "$K, V = cross_{\\text{text}} [B, 15, D]$\n"
        "$G_{\\text{ts}}$ bypasses via Identity/Residual",
        680, 505, 300, 60,
        fill="#F5EEF8", stroke="#8E44AD", font_color="#4A235A", font_size=10, bold=True
    )

    l2_self = add_box(
        "Layer 2 Self-Attention over [P; G_ts; G_text]\n"
        "PRICE PATCHES AND G_text ABSORB THE ENRICHED TECHNICAL TOKEN G_ts^(1) FROM LAYER 1!\n"
        "Aligns news understanding with technical momentum/overbought state.",
        240, 575, 720, 50,
        fill="#FEF5E7", stroke="#D35400", font_color="#7E5109", font_size=11, bold=True
    )

    # 5. Layer 1 Container (Micro / TS)
    add_box(
        "LAYER 1: TECHNICAL MICRO-STRUCTURE (Indicator Attention)\n"
        "Only G_ts queries cross_ts; G_text bypasses via Identity/Residual",
        50, 655, 1100, 220,
        fill="#FDFEFE", stroke="#27AE60", font_color="#145A32", font_size=12, bold=True, dashed=1
    )

    l1_ffn = add_box(
        "Conv1d FFN + LayerNorm\nTokens: $[P^{(1)}; G_{\\text{ts}}^{(1)}; G_{\\text{text}}^{(1)}] \\in \\mathbb{R}^{(B\\cdot C) \\times (N+2) \\times D}$",
        380, 690, 440, 40,
        fill="#EDF2F7", stroke="#CBD5E0", font_color="#2D3748", font_size=10, bold=True
    )

    l1_cross_ts = add_box(
        "Cross-Attention ONLY G_ts\n"
        "$Q = G_{\\text{ts}} [B\\cdot C, 1, D]$\n"
        "$K, V = cross_{\\text{ts}} [B, 25, D]$\n"
        "$G_{\\text{text}}$ bypasses via Identity/Residual",
        220, 745, 300, 60,
        fill="#EAFBF1", stroke="#27AE60", font_color="#145A32", font_size=10, bold=True
    )

    l1_self = add_box(
        "Layer 1 Self-Attention over [P; G_ts; G_text]\n"
        "Initial endogenous temporal exchange between patches; G_ts gathers price context.",
        240, 815, 720, 50,
        fill="#FEF5E7", stroke="#D35400", font_color="#7E5109", font_size=11, bold=True
    )

    # 6. Embeddings Row
    en_embed = add_box(
        "EnEmbeddingDual (Endogenous Patching & Bridges)\n"
        "1. Unfold OHLCV: $[B, 5, T=60] \\xrightarrow{P=16, S=12} [B, 5, N_p, 16]$\n"
        "2. Value Projection: $\\text{Linear}(16 \\to D) + E_{\\text{pos}} + \\text{FrequencyEmbedding}$\n"
        "3. Learnable Bridges: $[G_{\\text{ts}}, G_{\\text{text}}] \\in \\mathbb{R}^{1 \\times 5 \\times 2 \\times D}$\n"
        "Output: Tokens $[B \\times 5, N_p + 2, D]$",
        350, 895, 500, 90,
        fill="#E8F1FC", stroke="#2B6CB0", font_color="#1A365D", font_size=10, bold=True
    )

    ts_proj = add_box(
        "Inverted TS Projection\n$X_{\\text{ts}}^\\top \\in \\mathbb{R}^{B \\times 25 \\times 60} \\xrightarrow{\\text{Linear}} [B, 25, D]$",
        80, 905, 230, 70,
        fill="#EAFBF1", stroke="#27AE60", font_color="#145A32", font_size=10, bold=True
    )

    text_proj = add_box(
        "Text Projection\n$X_{\\text{text}} \\in \\mathbb{R}^{B \\times 15} \\xrightarrow{\\text{Linear}} [B, 1, D]$",
        890, 905, 230, 70,
        fill="#F5EEF8", stroke="#8E44AD", font_color="#4A235A", font_size=10, bold=True
    )

    # Inputs at Bottom
    in_ts = add_box(
        "25 Exogenous TS Features\n$$X_{\\text{ts}} \\in \\mathbb{R}^{B \\times 60 \\times 25}$$",
        80, 995, 230, 50,
        fill="#EAFBF1", stroke="#27AE60", font_color="#145A32", font_size=10, bold=True
    )

    in_endo = add_box(
        "Endogenous Price Channels (OHLCV)\n$$X_{\\text{endo}} \\in \\mathbb{R}^{B \\times 60 \\times 5}$$",
        350, 995, 500, 50,
        fill="#E8F1FC", stroke="#2B6CB0", font_color="#1A365D", font_size=11, bold=True
    )

    in_text = add_box(
        "15D Compact News Text\n$$X_{\\text{text}} \\in \\mathbb{R}^{B \\times 15}$$",
        890, 995, 230, 50,
        fill="#F5EEF8", stroke="#8E44AD", font_color="#4A235A", font_size=10, bold=True
    )

    # Connections
    add_edge(in_ts, ts_proj, "[B, 60, 25]")
    add_edge(in_endo, en_embed, "[B, 60, 5]")
    add_edge(in_text, text_proj, "[B, 15]")

    add_edge(en_embed, l1_self, "[B*5, N+2, D]")
    add_edge(ts_proj, l1_cross_ts, "cross_ts [B, 25, D]", exit_pt=(0.5, 0.0), entry_pt=(0.5, 1.0))
    add_edge(l1_self, l1_cross_ts, "Slice G_ts", exit_pt=(0.3, 0.0), entry_pt=(0.5, 1.0))
    add_edge(l1_cross_ts, l1_ffn, "G_ts^(1) updated", exit_pt=(0.5, 0.0), entry_pt=(0.3, 1.0))

    # L1 to L2
    add_edge(l1_ffn, l2_self, "Tokens with G_ts^(1) [B*5, N+2, D]", exit_pt=(0.5, 0.0), entry_pt=(0.5, 1.0))
    add_edge(text_proj, l2_cross_text, "cross_text [B, 1, D]", exit_pt=(0.5, 0.0), entry_pt=(0.5, 1.0))
    add_edge(l2_self, l2_cross_text, "Slice G_text", exit_pt=(0.7, 0.0), entry_pt=(0.5, 1.0))
    add_edge(l2_cross_text, l2_ffn, "G_text^(2) updated", exit_pt=(0.5, 0.0), entry_pt=(0.7, 1.0))

    # L2 to L3
    add_edge(l2_ffn, l3_self, "Tokens with G_text^(2) & G_ts^(2)", exit_pt=(0.5, 0.0), entry_pt=(0.5, 1.0))
    add_edge(l3_self, l3_ffn, "[B*5, N+2, D]", exit_pt=(0.5, 0.0), entry_pt=(0.5, 1.0))

    # L3 to Head
    add_edge(l3_ffn, head_box, "[B*5, N+2, D]", exit_pt=(0.5, 0.0), entry_pt=(0.5, 1.0))
    add_edge(head_box, pred_box, "[B, 7]", exit_pt=(0.5, 0.0), entry_pt=(0.5, 1.0))

    drawio_content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<mxfile host="app.diagrams.net" modified="2026-10-10T14:45:00.000Z" agent="Antigravity" version="21.0.0" type="device">\n'
        '  <diagram id="canonical_c1_hierarchical" name="Canonical TimeXer Hierarchical">\n'
        '    <mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1200" pageHeight="1100" math="1" shadow="0">\n'
        '      <root>\n'
        '        <mxCell id="0" />\n'
        '        <mxCell id="1" parent="0" />\n'
        + "".join(elements) +
        '\n      </root>\n'
        '    </mxGraphModel>\n'
        '  </diagram>\n'
        '</mxfile>'
    )
    return drawio_content


def generate_svg_dual() -> str:
    """Generate standalone publication SVG for Canonical Dual TimeXer."""
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 1080" width="100%" height="100%">
  <defs>
    <style>
      .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 20px; font-weight: bold; fill: #1A365D; }
      .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 13px; fill: #4A5568; }
      .box-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 13px; font-weight: bold; }
      .box-desc { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11px; }
      .tensor-shape { font-family: "SF Mono", Monaco, Inconsolata, Consolas, monospace; font-size: 10px; font-weight: 600; fill: #2D3748; }
      .arrow { stroke: #4A5568; stroke-width: 1.8; fill: none; marker-end: url(#arrowhead); }
      .arrow-ts { stroke: #27AE60; stroke-width: 1.8; fill: none; marker-end: url(#arrowhead-green); }
      .arrow-text { stroke: #8E44AD; stroke-width: 1.8; fill: none; marker-end: url(#arrowhead-purple); }
    </style>
    <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#4A5568" />
    </marker>
    <marker id="arrowhead-green" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#27AE60" />
    </marker>
    <marker id="arrowhead-purple" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#8E44AD" />
    </marker>
  </defs>

  <!-- Background -->
  <rect width="1200" height="1080" fill="#FFFFFF" />

  <!-- Title Banner -->
  <rect x="40" y="25" width="1120" height="60" rx="8" fill="#E8F1FC" stroke="#2B6CB0" stroke-width="2" />
  <text x="600" y="50" text-anchor="middle" class="title">Canonical TimeXer Dual (c1_dual) Architecture</text>
  <text x="600" y="70" text-anchor="middle" class="subtitle">Parallel Disentangled Cross-Modal Bridge Tokens with Local Patch-FFT (NeurIPS 2024 / TSLib Formulation)</text>

  <!-- Forecast Output -->
  <rect x="450" y="105" width="300" height="50" rx="8" fill="#E8F8F5" stroke="#1ABC9C" stroke-width="2" />
  <text x="600" y="128" text-anchor="middle" class="box-title" fill="#0E6251">Forecast Horizon Output [B, H=7]</text>
  <text x="600" y="144" text-anchor="middle" class="box-desc" fill="#117864">ŷ = [Close_{t+1}, ..., Close_{t+7}] in z-space</text>

  <!-- FlattenHead -->
  <rect x="360" y="175" width="480" height="60" rx="8" fill="#EDF2F7" stroke="#4A5568" stroke-width="1.5" />
  <text x="600" y="198" text-anchor="middle" class="box-title" fill="#1A202C">FlattenHead: Extract Close Variable Tokens</text>
  <text x="600" y="218" text-anchor="middle" class="box-desc" fill="#4A5568">enc_close = tokens[:, close_idx, :, :] [B, N_p+2, d] → Flatten → Linear → [B, 7]</text>

  <!-- LAYER 2 CONTAINER -->
  <rect x="40" y="255" width="1120" height="260" rx="12" fill="#F7FAFC" stroke="#CBD5E0" stroke-width="2" stroke-dasharray="8 4" />
  <text x="70" y="280" class="box-title" fill="#2D3748">CANONICAL DUAL LAYER 2 (of 2)</text>
  <text x="70" y="296" class="box-desc" fill="#718096">Self-attention distributes Layer 1 exogenous learning into price patches; parallel bridges refine modalities</text>

  <!-- L2 FFN -->
  <rect x="390" y="285" width="420" height="40" rx="6" fill="#EDF2F7" stroke="#A0AEC0" stroke-width="1.5" />
  <text x="600" y="310" text-anchor="middle" class="box-desc" fill="#2D3748"><b>Conv1d FFN + Residual + Norm</b> over all (N_p + 2) tokens [B·C, N_p+2, d]</text>

  <!-- L2 Cross TS -->
  <rect x="140" y="345" width="250" height="70" rx="8" fill="#EAFBF1" stroke="#27AE60" stroke-width="1.8" />
  <text x="265" y="368" text-anchor="middle" class="box-title" fill="#145A32">Cross-Attention TS (Layer 2)</text>
  <text x="265" y="386" text-anchor="middle" class="box-desc" fill="#1E8449">Q = G_ts [B·C, 1, d], K,V = cross_ts [B, 25, d]</text>
  <text x="265" y="402" text-anchor="middle" class="tensor-shape" fill="#27AE60">G_ts ← Norm(G_ts + Δ_ts)</text>

  <!-- L2 Cross Text -->
  <rect x="810" y="345" width="250" height="70" rx="8" fill="#F5EEF8" stroke="#8E44AD" stroke-width="1.8" />
  <text x="935" y="368" text-anchor="middle" class="box-title" fill="#4A235A">Cross-Attention Text (Layer 2)</text>
  <text x="935" y="386" text-anchor="middle" class="box-desc" fill="#6C3483">Q = G_text [B·C, 1, d], K,V = cross_text [B, 15, d]</text>
  <text x="935" y="402" text-anchor="middle" class="tensor-shape" fill="#8E44AD">G_text ← Norm(G_text + Δ_text)</text>

  <!-- L2 Self Attention -->
  <rect x="240" y="435" width="720" height="65" rx="8" fill="#FEF5E7" stroke="#D35400" stroke-width="2" />
  <text x="600" y="458" text-anchor="middle" class="box-title" fill="#7E5109">Full Self-Attention over all tokens [P_1..P_N; G_ts; G_text]</text>
  <text x="600" y="476" text-anchor="middle" class="box-desc" fill="#B9770E">Patches absorb Layer 1 updated G_ts &amp; G_text; G_ts &amp; G_text align inter-modal synergy</text>
  <text x="600" y="491" text-anchor="middle" class="tensor-shape" fill="#D35400">Tokens shape: [B × 5, N_patches + 2, d_model]</text>

  <!-- LAYER 1 CONTAINER -->
  <rect x="40" y="535" width="1120" height="260" rx="12" fill="#F7FAFC" stroke="#CBD5E0" stroke-width="2" stroke-dasharray="8 4" />
  <text x="70" y="560" class="box-title" fill="#2D3748">CANONICAL DUAL LAYER 1 (of 2)</text>
  <text x="70" y="576" class="box-desc" fill="#718096">Initialized price patches &amp; learnable bridge tokens enter together; parallel cross-attentions filter modalities</text>

  <!-- L1 FFN -->
  <rect x="390" y="565" width="420" height="40" rx="6" fill="#EDF2F7" stroke="#A0AEC0" stroke-width="1.5" />
  <text x="600" y="590" text-anchor="middle" class="box-desc" fill="#2D3748"><b>Conv1d FFN + Residual + Norm</b> over all (N_p + 2) tokens [B·C, N_p+2, d]</text>

  <!-- L1 Cross TS -->
  <rect x="140" y="625" width="250" height="70" rx="8" fill="#EAFBF1" stroke="#27AE60" stroke-width="1.8" />
  <text x="265" y="648" text-anchor="middle" class="box-title" fill="#145A32">Cross-Attention TS (Layer 1)</text>
  <text x="265" y="666" text-anchor="middle" class="box-desc" fill="#1E8449">Q = G_ts [B·C, 1, d], K,V = cross_ts [B, 25, d]</text>
  <text x="265" y="682" text-anchor="middle" class="tensor-shape" fill="#27AE60">Price patches completely isolated from cross_ts</text>

  <!-- L1 Cross Text -->
  <rect x="810" y="625" width="250" height="70" rx="8" fill="#F5EEF8" stroke="#8E44AD" stroke-width="1.8" />
  <text x="935" y="648" text-anchor="middle" class="box-title" fill="#4A235A">Cross-Attention Text (Layer 1)</text>
  <text x="935" y="666" text-anchor="middle" class="box-desc" fill="#6C3483">Q = G_text [B·C, 1, d], K,V = cross_text [B, 15, d]</text>
  <text x="935" y="682" text-anchor="middle" class="tensor-shape" fill="#8E44AD">Price patches completely isolated from cross_text</text>

  <!-- L1 Self Attention -->
  <rect x="240" y="715" width="720" height="65" rx="8" fill="#FEF5E7" stroke="#D35400" stroke-width="2" />
  <text x="600" y="738" text-anchor="middle" class="box-title" fill="#7E5109">Initial Self-Attention over [P_1..P_N; G_ts; G_text]</text>
  <text x="600" y="756" text-anchor="middle" class="box-desc" fill="#B9770E">Patches model temporal dependencies; G_ts &amp; G_text absorb endogenous price dynamics</text>
  <text x="600" y="771" text-anchor="middle" class="tensor-shape" fill="#D35400">Tokens shape: [B × 5, N_patches + 2, d_model]</text>

  <!-- EMBEDDINGS ROW -->
  <!-- EnEmbeddingDual -->
  <rect x="340" y="815" width="520" height="95" rx="8" fill="#E8F1FC" stroke="#2B6CB0" stroke-width="2" />
  <text x="600" y="838" text-anchor="middle" class="box-title" fill="#1A365D">EnEmbeddingDual (Endogenous Patching &amp; Dual Bridge Concat)</text>
  <text x="600" y="856" text-anchor="middle" class="box-desc" fill="#2874A6">1. Unfold OHLCV [B, 5, 60] → [B, 5, N_p, 16] → Linear(16 → d) + PositionalEmbedding</text>
  <text x="600" y="873" text-anchor="middle" class="box-desc" fill="#00A896"><b>2. Local Patch-FFT:</b> + Linear(6 → d)(rFFT(Close)) additive Frequency Embedding</text>
  <text x="600" y="890" text-anchor="middle" class="tensor-shape" fill="#1B4F72">3. Concat learnable tokens: [Patches; G_ts; G_text] → [B × 5, N_p + 2, d_model]</text>

  <!-- Inverted TS Projection -->
  <rect x="80" y="820" width="230" height="85" rx="8" fill="#EAFBF1" stroke="#27AE60" stroke-width="1.8" />
  <text x="195" y="843" text-anchor="middle" class="box-title" fill="#145A32">Inverted TS Projection</text>
  <text x="195" y="861" text-anchor="middle" class="box-desc" fill="#1E8449">Transpose time &amp; indicators</text>
  <text x="195" y="878" text-anchor="middle" class="box-desc" fill="#1E8449">Linear(60 → d_model)</text>
  <text x="195" y="895" text-anchor="middle" class="tensor-shape" fill="#27AE60">cross_ts: [B, 25, d_model]</text>

  <!-- Text Projection -->
  <rect x="890" y="820" width="230" height="85" rx="8" fill="#F5EEF8" stroke="#8E44AD" stroke-width="1.8" />
  <text x="1005" y="843" text-anchor="middle" class="box-title" fill="#4A235A">Text Linear Projection</text>
  <text x="1005" y="861" text-anchor="middle" class="box-desc" fill="#6C3483">Compact FinBERT sentiment</text>
  <text x="1005" y="878" text-anchor="middle" class="box-desc" fill="#6C3483">Linear(15 → d_model)</text>
  <text x="1005" y="895" text-anchor="middle" class="tensor-shape" fill="#8E44AD">cross_text: [B, 15, d_model]</text>

  <!-- RAW INPUTS ROW -->
  <rect x="80" y="930" width="230" height="60" rx="8" fill="#EAFBF1" stroke="#27AE60" stroke-width="1.5" />
  <text x="195" y="955" text-anchor="middle" class="box-title" fill="#145A32">25 Technical Indicators</text>
  <text x="195" y="974" text-anchor="middle" class="tensor-shape" fill="#1E8449">X_ts: [B, 60, 25]</text>

  <rect x="340" y="930" width="520" height="60" rx="8" fill="#E8F1FC" stroke="#2B6CB0" stroke-width="1.5" />
  <text x="600" y="955" text-anchor="middle" class="box-title" fill="#1A365D">Endogenous Price Channels (OHLCV)</text>
  <text x="600" y="974" text-anchor="middle" class="tensor-shape" fill="#2874A6">X_endo: [B, 60, 5] (Close, Volume, Open, High, Low)</text>

  <rect x="890" y="930" width="230" height="60" rx="8" fill="#F5EEF8" stroke="#8E44AD" stroke-width="1.5" />
  <text x="1005" y="955" text-anchor="middle" class="box-title" fill="#4A235A">Compact News Sentiment</text>
  <text x="1005" y="974" text-anchor="middle" class="tensor-shape" fill="#6C3483">X_text: [B, 15] (Frozen Cache)</text>

  <!-- Arrows -->
  <path d="M 195 930 L 195 905" class="arrow-ts" />
  <path d="M 600 930 L 600 910" class="arrow" />
  <path d="M 1005 930 L 1005 905" class="arrow-text" />

  <path d="M 600 815 L 600 780" class="arrow" />
  <path d="M 195 820 L 195 695" class="arrow-ts" />
  <path d="M 1005 820 L 1005 695" class="arrow-text" />

  <path d="M 380 715 L 265 695" class="arrow" />
  <path d="M 820 715 L 935 695" class="arrow" />
  <path d="M 265 625 L 450 605" class="arrow" />
  <path d="M 935 625 L 750 605" class="arrow" />

  <path d="M 600 565 L 600 500" class="arrow" />
  <path d="M 195 625 L 195 415" class="arrow-ts" />
  <path d="M 1005 625 L 1005 415" class="arrow-text" />

  <path d="M 380 435 L 265 415" class="arrow" />
  <path d="M 820 435 L 935 415" class="arrow" />
  <path d="M 265 345 L 450 325" class="arrow" />
  <path d="M 935 345 L 750 325" class="arrow" />

  <path d="M 600 285 L 600 235" class="arrow" />
  <path d="M 600 175 L 600 155" class="arrow" />
</svg>"""


def generate_svg_hierarchical() -> str:
    """Generate standalone publication SVG for Canonical Hierarchical TimeXer."""
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 1080" width="100%" height="100%">
  <defs>
    <style>
      .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 20px; font-weight: bold; fill: #1A365D; }
      .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 13px; fill: #4A5568; }
      .box-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 13px; font-weight: bold; }
      .box-desc { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11px; }
      .tensor-shape { font-family: "SF Mono", Monaco, Inconsolata, Consolas, monospace; font-size: 10px; font-weight: 600; fill: #2D3748; }
      .arrow { stroke: #4A5568; stroke-width: 1.8; fill: none; marker-end: url(#arrowhead); }
      .arrow-ts { stroke: #27AE60; stroke-width: 1.8; fill: none; marker-end: url(#arrowhead-green); }
      .arrow-text { stroke: #8E44AD; stroke-width: 1.8; fill: none; marker-end: url(#arrowhead-purple); }
    </style>
    <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#4A5568" />
    </marker>
    <marker id="arrowhead-green" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#27AE60" />
    </marker>
    <marker id="arrowhead-purple" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#8E44AD" />
    </marker>
  </defs>

  <!-- Background -->
  <rect width="1200" height="1080" fill="#FFFFFF" />

  <!-- Title Banner -->
  <rect x="40" y="25" width="1120" height="60" rx="8" fill="#E8F1FC" stroke="#2B6CB0" stroke-width="2" />
  <text x="600" y="50" text-anchor="middle" class="title">Canonical TimeXer Hierarchical (c1_hierarchical) Architecture</text>
  <text x="600" y="70" text-anchor="middle" class="subtitle">3-Layer Progression: Micro Technical Indicators → Macro News Sentiment → Cross-Modal Consolidation</text>

  <!-- Forecast Output -->
  <rect x="450" y="105" width="300" height="50" rx="8" fill="#E8F8F5" stroke="#1ABC9C" stroke-width="2" />
  <text x="600" y="128" text-anchor="middle" class="box-title" fill="#0E6251">Forecast Horizon Output [B, H=7]</text>
  <text x="600" y="144" text-anchor="middle" class="box-desc" fill="#117864">ŷ = [Close_{t+1}, ..., Close_{t+7}] in z-space</text>

  <!-- FlattenHead -->
  <rect x="360" y="175" width="480" height="50" rx="8" fill="#EDF2F7" stroke="#4A5568" stroke-width="1.5" />
  <text x="600" y="198" text-anchor="middle" class="box-title" fill="#1A202C">FlattenHead: Extract Close Variable Tokens</text>
  <text x="600" y="215" text-anchor="middle" class="box-desc" fill="#4A5568">tokens[:, close_idx, :, :] [B, N_p+2, d] → Flatten → Linear → [B, 7]</text>

  <!-- LAYER 3 CONTAINER (CONSOLIDATION) -->
  <rect x="40" y="245" width="1120" height="160" rx="12" fill="#FEF9E7" stroke="#F39C12" stroke-width="2" stroke-dasharray="8 4" />
  <text x="70" y="270" class="box-title" fill="#7D6608">LAYER 3: CROSS-MODAL CONSOLIDATION</text>
  <text x="70" y="286" class="box-desc" fill="#B7950B">All price patches consolidate joint signals from G_text^(2) and G_ts^(2); Cross-Attention omitted</text>

  <!-- L3 FFN -->
  <rect x="390" y="295" width="420" height="35" rx="6" fill="#EDF2F7" stroke="#A0AEC0" stroke-width="1.5" />
  <text x="600" y="318" text-anchor="middle" class="box-desc" fill="#2D3748"><b>Conv1d FFN + Residual + Norm</b> over all (N_p + 2) tokens</text>

  <!-- L3 Self Attention -->
  <rect x="240" y="345" width="720" height="50" rx="8" fill="#FEF5E7" stroke="#D35400" stroke-width="2" />
  <text x="600" y="366" text-anchor="middle" class="box-title" fill="#7E5109">Consolidation Self-Attention over [P; G_ts^(2); G_text^(2)]</text>
  <text x="600" y="384" text-anchor="middle" class="box-desc" fill="#B9770E">Patches integrate both macro news sentiment and micro technical indicators seamlessly</text>

  <!-- LAYER 2 CONTAINER (MACRO / TEXT) -->
  <rect x="40" y="425" width="1120" height="210" rx="12" fill="#FDFEFE" stroke="#8E44AD" stroke-width="2" stroke-dasharray="8 4" />
  <text x="70" y="450" class="box-title" fill="#4A235A">LAYER 2: SEMANTIC MACRO-STRUCTURE (Text / Sentiment)</text>
  <text x="70" y="466" class="box-desc" fill="#7D3C98">G_text queries 15D compact text; Patches and G_text absorb technical token G_ts^(1) from Layer 1</text>

  <!-- L2 FFN -->
  <rect x="390" y="475" width="420" height="35" rx="6" fill="#EDF2F7" stroke="#A0AEC0" stroke-width="1.5" />
  <text x="600" y="498" text-anchor="middle" class="box-desc" fill="#2D3748"><b>Conv1d FFN + Residual + Norm</b>: tokens [P^(2); G_ts^(2); G_text^(2)]</text>

  <!-- L2 Cross Text -->
  <rect x="680" y="525" width="350" height="55" rx="8" fill="#F5EEF8" stroke="#8E44AD" stroke-width="1.8" />
  <text x="855" y="546" text-anchor="middle" class="box-title" fill="#4A235A">Cross-Attention ONLY G_text</text>
  <text x="855" y="563" text-anchor="middle" class="tensor-shape" fill="#8E44AD">Q = G_text [B·C, 1, d], K,V = cross_text [B, 15, d] (G_ts passes through)</text>

  <!-- L2 Self Attention -->
  <rect x="240" y="590" width="720" height="40" rx="8" fill="#FEF5E7" stroke="#D35400" stroke-width="2" />
  <text x="600" y="608" text-anchor="middle" class="box-title" fill="#7E5109">Self-Attention: Patches &amp; G_text absorb enriched technical token G_ts^(1)</text>
  <text x="600" y="622" text-anchor="middle" class="box-desc" fill="#B9770E">Aligns news understanding with technical momentum/overbought state</text>

  <!-- LAYER 1 CONTAINER (MICRO / TS) -->
  <rect x="40" y="655" width="1120" height="210" rx="12" fill="#FDFEFE" stroke="#27AE60" stroke-width="2" stroke-dasharray="8 4" />
  <text x="70" y="680" class="box-title" fill="#145A32">LAYER 1: TECHNICAL MICRO-STRUCTURE (25 TreeSHAP Indicators)</text>
  <text x="70" y="696" class="box-desc" fill="#1E8449">G_ts queries 25 indicators; G_text bypasses via identity/residual</text>

  <!-- L1 FFN -->
  <rect x="390" y="705" width="420" height="35" rx="6" fill="#EDF2F7" stroke="#A0AEC0" stroke-width="1.5" />
  <text x="600" y="728" text-anchor="middle" class="box-desc" fill="#2D3748"><b>Conv1d FFN + Residual + Norm</b>: tokens [P^(1); G_ts^(1); G_text^(1)]</text>

  <!-- L1 Cross TS -->
  <rect x="170" y="755" width="350" height="55" rx="8" fill="#EAFBF1" stroke="#27AE60" stroke-width="1.8" />
  <text x="345" y="776" text-anchor="middle" class="box-title" fill="#145A32">Cross-Attention ONLY G_ts</text>
  <text x="345" y="793" text-anchor="middle" class="tensor-shape" fill="#27AE60">Q = G_ts [B·C, 1, d], K,V = cross_ts [B, 25, d] (G_text passes through)</text>

  <!-- L1 Self Attention -->
  <rect x="240" y="820" width="720" height="40" rx="8" fill="#FEF5E7" stroke="#D35400" stroke-width="2" />
  <text x="600" y="838" text-anchor="middle" class="box-title" fill="#7E5109">Initial Self-Attention over [P_1..P_N; G_ts; G_text]</text>
  <text x="600" y="852" text-anchor="middle" class="box-desc" fill="#B9770E">Patches exchange temporal dynamic; G_ts gathers initial price state</text>

  <!-- EMBEDDINGS ROW -->
  <rect x="340" y="885" width="520" height="85" rx="8" fill="#E8F1FC" stroke="#2B6CB0" stroke-width="2" />
  <text x="600" y="908" text-anchor="middle" class="box-title" fill="#1A365D">EnEmbeddingDual (Patching + Frequency Embedding)</text>
  <text x="600" y="926" text-anchor="middle" class="box-desc" fill="#2874A6">Unfold OHLCV [B, 5, 60] → [B, 5, N_p, 16] → Linear(16 → d) + PosEmbedding + Patch-FFT</text>
  <text x="600" y="943" text-anchor="middle" class="tensor-shape" fill="#1B4F72">Concat [Patches; G_ts; G_text] → Tokens: [B × 5, N_p + 2, d_model]</text>

  <rect x="80" y="885" width="230" height="85" rx="8" fill="#EAFBF1" stroke="#27AE60" stroke-width="1.8" />
  <text x="195" y="908" text-anchor="middle" class="box-title" fill="#145A32">Inverted TS Projection</text>
  <text x="195" y="926" text-anchor="middle" class="box-desc" fill="#1E8449">Linear(60 → d_model)</text>
  <text x="195" y="943" text-anchor="middle" class="tensor-shape" fill="#27AE60">cross_ts: [B, 25, d_model]</text>

  <rect x="890" y="885" width="230" height="85" rx="8" fill="#F5EEF8" stroke="#8E44AD" stroke-width="1.8" />
  <text x="1005" y="908" text-anchor="middle" class="box-title" fill="#4A235A">Text Linear Projection</text>
  <text x="1005" y="926" text-anchor="middle" class="box-desc" fill="#6C3483">Linear(15 → d_model)</text>
  <text x="1005" y="943" text-anchor="middle" class="tensor-shape" fill="#8E44AD">cross_text: [B, 15, d_model]</text>

  <!-- RAW INPUTS -->
  <rect x="80" y="990" width="230" height="55" rx="8" fill="#EAFBF1" stroke="#27AE60" stroke-width="1.5" />
  <text x="195" y="1014" text-anchor="middle" class="box-title" fill="#145A32">25 Technical Indicators</text>
  <text x="195" y="1031" text-anchor="middle" class="tensor-shape" fill="#1E8449">X_ts: [B, 60, 25]</text>

  <rect x="340" y="990" width="520" height="55" rx="8" fill="#E8F1FC" stroke="#2B6CB0" stroke-width="1.5" />
  <text x="600" y="1014" text-anchor="middle" class="box-title" fill="#1A365D">Endogenous Price Channels (OHLCV)</text>
  <text x="600" y="1031" text-anchor="middle" class="tensor-shape" fill="#2874A6">X_endo: [B, 60, 5]</text>

  <rect x="890" y="990" width="230" height="55" rx="8" fill="#F5EEF8" stroke="#8E44AD" stroke-width="1.5" />
  <text x="1005" y="1014" text-anchor="middle" class="box-title" fill="#4A235A">Compact News Sentiment</text>
  <text x="1005" y="1031" text-anchor="middle" class="tensor-shape" fill="#6C3483">X_text: [B, 15]</text>

  <!-- Arrows -->
  <path d="M 195 990 L 195 970" class="arrow-ts" />
  <path d="M 600 990 L 600 970" class="arrow" />
  <path d="M 1005 990 L 1005 970" class="arrow-text" />

  <path d="M 600 885 L 600 860" class="arrow" />
  <path d="M 195 885 L 195 780 L 170 780" class="arrow-ts" />
  <path d="M 345 820 L 345 810" class="arrow" />
  <path d="M 345 755 L 450 740" class="arrow" />

  <path d="M 600 705 L 600 630" class="arrow" />
  <path d="M 1005 885 L 1005 550 L 1030 550" class="arrow-text" />
  <path d="M 855 590 L 855 580" class="arrow" />
  <path d="M 855 525 L 750 510" class="arrow" />

  <path d="M 600 475 L 600 395" class="arrow" />
  <path d="M 600 345 L 600 330" class="arrow" />
  <path d="M 600 295 L 600 225" class="arrow" />
  <path d="M 600 175 L 600 155" class="arrow" />
</svg>"""


def main() -> None:
    figures_dir = Path("Articles/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Updated Canonical TimeXer Dual Diagrams...")
    dual_drawio = create_dual_diagram()
    dual_svg = generate_svg_dual()

    dual_drawio_path = figures_dir / "canonical_c1_dual.drawio"
    dual_svg_path = figures_dir / "canonical_c1_dual.svg"

    dual_drawio_path.write_text(dual_drawio, encoding="utf-8")
    dual_svg_path.write_text(dual_svg, encoding="utf-8")
    print(f"Saved: {dual_drawio_path} ({len(dual_drawio)} bytes)")
    print(f"Saved: {dual_svg_path} ({len(dual_svg)} bytes)")

    print("\nGenerating Updated Canonical TimeXer Hierarchical Diagrams...")
    hier_drawio = create_hierarchical_diagram()
    hier_svg = generate_svg_hierarchical()

    hier_drawio_path = figures_dir / "canonical_c1_hierarchical.drawio"
    hier_svg_path = figures_dir / "canonical_c1_hierarchical.svg"

    hier_drawio_path.write_text(hier_drawio, encoding="utf-8")
    hier_svg_path.write_text(hier_svg, encoding="utf-8")
    print(f"Saved: {hier_drawio_path} ({len(hier_drawio)} bytes)")
    print(f"Saved: {hier_svg_path} ({len(hier_svg)} bytes)")


if __name__ == "__main__":
    main()
