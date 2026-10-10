"""Generate publication-grade architecture diagram of CANONICAL TimeXer (thuml/Time-Series-Library).

Official Tsinghua University / NeurIPS 2024 implementation:
- EnEmbedding: Patches + learnable glb_token concatenated immediately at model input: [P; G] in R^[B, N+1, d_model]
- DataEmbedding_inverted: C exogenous series projected over lookback T to C variate tokens in R^[B, C, d_model]
- EncoderLayer:
    1. Endogenous Self-Attention over [P; G]
    2. Extract G = tokens[:, -1:, :]
    3. Variate Cross-Attention: Query = G [B, 1, d], Key/Val = Variate Tokens [B, C, d]
    4. Re-concatenate: tokens = [P; G]
    5. FFN (Conv1d -> GELU -> Conv1d)
- Next layer: Self-Attention broadcasts G's exogenous knowledge into price patches!
"""

from __future__ import annotations

import html
from pathlib import Path


def create_drawio_xml() -> str:
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

    # Banner Header
    add_box(
        "Canonical TimeXer Architecture (Official TSLib / NeurIPS 2024)",
        50, 30, 1500, 48,
        fill="#2B6CB0", stroke="#2B6CB0", font_color="#FFFFFF", font_size=18, bold=True, rounded=1
    )
    add_box(
        "Wang et al., 'TimeXer: Empowering Canonical Transformers for Long-Term Time Series Forecasting with Exogenous Variables' (thuml/Time-Series-Library)",
        50, 85, 1500, 28,
        fill="#EDF2F7", stroke="#CBD5E0", font_color="#4A5568", font_size=12, bold=False, rounded=1
    )

    # Modality Legend Bar
    leg_y = 125
    add_box("COLOR CODE:", 50, leg_y, 130, 32, fill="none", stroke="none", font_color="#2D3748", font_size=11, bold=True, align="left")
    add_box("Endogenous Target Series (Price)", 190, leg_y, 250, 32, fill="#E8F1FC", stroke="#2B6CB0", font_color="#1A365D", font_size=11, bold=True)
    add_box("Exogenous Variables (Variate Tokens)", 460, leg_y, 280, 32, fill="#EAFBF1", stroke="#27AE60", font_color="#145A32", font_size=11, bold=True)
    add_box("Global Endogenous Token (G)", 760, leg_y, 230, 32, fill="#FDEBD0", stroke="#B9770E", font_color="#7D6608", font_size=11, bold=True)
    add_box("Transformer Self & Cross Attention", 1010, leg_y, 270, 32, fill="#FEF5E7", stroke="#D35400", font_color="#7E5109", font_size=11, bold=True)
    add_box("Projection & Forecast Head", 1300, leg_y, 250, 32, fill="#EBEDEF", stroke="#2C3E50", font_color="#1B2631", font_size=11, bold=True)

    # =========================================================================
    # LEFT / MAIN COLUMN: The Canonical Flow (x = 50 to 950)
    # =========================================================================
    mx_x = 50

    # Output & Head
    pred_box = add_box(
        "Forecast Horizon Output\n$$\\hat{Y} \\in \\mathbb{R}^{B \\times H}$$\n(Target Time Series Prediction)",
        mx_x + 320, 180, 260, 52,
        fill="#EBEDEF", stroke="#2C3E50", font_color="#1B2631", font_size=11, bold=True
    )
    head_box = add_box(
        "FlattenHead / Linear Projection\n$$\\text{Flatten}(\\text{enc\\_out}) \\xrightarrow{\\text{Linear}} \\hat{Y} \\in \\mathbb{R}^{B \\times H}$$",
        mx_x + 320, 255, 260, 48,
        fill="#EBEDEF", stroke="#2C3E50", font_color="#1B2631", font_size=11
    )

    # Canonical Transformer Layer Container (L layers)
    add_box(
        "Canonical TimeXer EncoderLayer (Repeated L times)\n"
        "Step 1: Endogenous Self-Attention  ->  Step 2: Global Token Exogenous Cross-Attention  ->  Step 3: FFN",
        mx_x + 20, 325, 860, 570,
        fill="#FDFEFE", stroke="#2B6CB0", font_color="#1A365D", font_size=12, bold=True, dashed=1, align="left", valign="top"
    )

    # Inside Layer:
    ln3_box = add_box(
        "LayerNorm 3 & Residual: $$\\text{LayerNorm}(x + y) \\in \\mathbb{R}^{B \\times (N+1) \\times D}$$",
        mx_x + 50, 375, 800, 42,
        fill="#EDF2F7", stroke="#CBD5E0", font_color="#2D3748", font_size=10
    )

    ffn_box = add_box(
        "Feed-Forward Network (FFN)\n"
        "$$y = \\text{Conv1d}(D \\to d_{\\text{ff}}) \\to \\text{GELU} \\to \\text{Conv1d}(d_{\\text{ff}} \\to D)$$\n"
        "Applied to all $(N+1)$ concatenated tokens: $[P_1, \\dots, P_N; G]$",
        mx_x + 50, 435, 800, 60,
        fill="#FEF5E7", stroke="#D35400", font_color="#7E5109", font_size=10
    )

    reconcat_box = add_box(
        "Token Re-concatenation: $x = [x[:, :-1, :]; x_{\\text{glb}}] \\in \\mathbb{R}^{B \\times (N+1) \\times D}$\n"
        "Patches and updated Global Token rejoin as a single sequence",
        mx_x + 50, 515, 800, 45,
        fill="#EDF2F7", stroke="#CBD5E0", font_color="#2D3748", font_size=10, bold=True
    )

    cross_box = add_box(
        "Step 2: Exogenous-to-Endogenous Cross-Attention (Variate-wise Attention)\n"
        "$$\\text{Query} = x_{\\text{glb}} \\in \\mathbb{R}^{B \\times 1 \\times D} \\quad (\\text{Only Global Token Queries!})$$\n"
        "$$\\text{Key}, \\text{Value} = \\text{cross} \\in \\mathbb{R}^{B \\times C \\times D} \\quad (C \\text{ Exogenous Variate Tokens})$$\n"
        "$$x_{\\text{glb}} \\leftarrow \\text{LayerNorm}(x_{\\text{glb}} + \\text{Dropout}(\\text{CrossAttn}(x_{\\text{glb}}, \\text{cross}, \\text{cross})))$$\n"
        "Complexity: $O(1 \\times C)$ — Patches NEVER attend to exogenous directly, eliminating noise & bloat!",
        mx_x + 50, 580, 800, 105,
        fill="#FEF5E7", stroke="#D35400", font_color="#7E5109", font_size=10, bold=True
    )

    extract_box = add_box(
        "Token Splitting: $x_{\\text{glb}} = x[:, -1, :] \\in \\mathbb{R}^{B \\times 1 \\times D}, \\quad \\text{patches} = x[:, :-1, :] \\in \\mathbb{R}^{B \\times N \\times D}$",
        mx_x + 50, 705, 800, 40,
        fill="#EDF2F7", stroke="#CBD5E0", font_color="#2D3748", font_size=10
    )

    self_box = add_box(
        "Step 1: Endogenous Self-Attention (Temporal-wise Attention)\n"
        "$$\\text{SelfAttn}(Q=x, K=x, V=x) \\quad \\text{over all } (N+1) \\text{ tokens } [P_1, \\dots, P_N; G]$$\n"
        "$$x \\leftarrow \\text{LayerNorm}(x + \\text{Dropout}(\\text{SelfAttn}(x, x, x)))$$\n"
        "Models temporal relationships between patches; Global token G aggregates endogenous price context!",
        mx_x + 50, 765, 800, 80,
        fill="#FEF5E7", stroke="#D35400", font_color="#7E5109", font_size=10, bold=True
    )

    add_box(
        "Input to Layer: $x \\in \\mathbb{R}^{B \\times (N+1) \\times D}$ (From EnEmbedding or Previous Layer)",
        mx_x + 50, 860, 800, 25,
        fill="#F0FFF4", stroke="#38A169", font_color="#22543D", font_size=10, bold=True
    )

    # Embeddings at Bottom
    en_embed_box = add_box(
        "EnEmbedding (Endogenous Series Embedding)\n"
        "1. Patches: $x_{\\text{unfold}} = \\text{unfold}(X_{\\text{endo}}, P, P) \\xrightarrow{\\text{Linear}} [B, N, D] + E_{\\text{pos}}$\n"
        "2. Global Token: $G = \\text{nn.Parameter}(\\text{randn}(1, 1, 1, D)) \\to [B, 1, D]$\n"
        "3. IMMEDIATE CONCATENATION AT INPUT: $x = [\\text{Patches}; G] \\in \\mathbb{R}^{B \\times (N+1) \\times D}$\n"
        "NO CROSS-ATTENTION OCCURS BEFORE ENCODER!",
        mx_x + 50, 915, 520, 95,
        fill="#E8F1FC", stroke="#2B6CB0", font_color="#1A365D", font_size=10, bold=True
    )

    ex_embed_box = add_box(
        "DataEmbedding_inverted (Exogenous Series Embedding)\n"
        "Transposes time and variates: $X_{\\text{exo}}^\\top \\in \\mathbb{R}^{B \\times C \\times T}$\n"
        "Temporal Projection: $\\text{Linear}(T \\to D) + E_{\\text{var}}$\n"
        "Yields $C$ Variate Tokens: $\\text{cross} \\in \\mathbb{R}^{B \\times C \\times D}$\n"
        "Each token represents an entire exogenous variable curve!",
        mx_x + 600, 915, 280, 95,
        fill="#EAFBF1", stroke="#27AE60", font_color="#145A32", font_size=10, bold=True
    )

    # Raw Inputs
    in_endo = add_box(
        "Endogenous Input Series\n$$X_{\\text{endo}} \\in \\mathbb{R}^{B \\times T \\times 1}$$\n(Target Series, e.g. Stock Price)",
        mx_x + 50, 1040, 520, 60,
        fill="#E8F1FC", stroke="#2B6CB0", font_color="#1A365D", font_size=11, bold=True
    )
    in_exo = add_box(
        "Exogenous Input Series\n$$X_{\\text{exo}} \\in \\mathbb{R}^{B \\times T \\times C}$$\n($C$ External Variables / Indicators)",
        mx_x + 600, 1040, 280, 60,
        fill="#EAFBF1", stroke="#27AE60", font_color="#145A32", font_size=11, bold=True
    )

    # Connections Main Column
    add_edge(in_endo, en_embed_box, "[B, T, 1]")
    add_edge(in_exo, ex_embed_box, "[B, T, C]")
    add_edge(en_embed_box, self_box, "[B, N+1, D] (Patches + G)")
    add_edge(ex_embed_box, cross_box, "cross [B, C, D] (Key, Value)", exit_pt=(0.5, 0.0), entry_pt=(0.85, 1.0))

    add_edge(self_box, extract_box, "[B, N+1, D]")
    add_edge(extract_box, cross_box, "x_glb [B, 1, D] (Query)", exit_pt=(0.3, 0.0), entry_pt=(0.3, 1.0))
    add_edge(cross_box, reconcat_box, "updated x_glb + patches", exit_pt=(0.5, 0.0), entry_pt=(0.5, 1.0))
    add_edge(reconcat_box, ffn_box, "[B, N+1, D]")
    add_edge(ffn_box, ln3_box, "[B, N+1, D]")
    add_edge(ln3_box, head_box, "enc_out [B, N+1, D]")
    add_edge(head_box, pred_box, "[B, H]")


    # =========================================================================
    # RIGHT COLUMN: The Crucial Insights & TSLib Code (x = 920 to 1540)
    # =========================================================================
    rx_x = 910
    rx_w = 640

    # Inset 1: Official TSLib Source Code
    add_box(
        "Official TSLib Code: models/TimeXer.py (Exact Excerpt)",
        rx_x, 180, rx_w, 40,
        fill="#EDF2F7", stroke="#4A5568", font_color="#1A202C", font_size=12, bold=True, rounded=1
    )

    code_str = (
        "# 1. IMMEDIATE CONCATENATION AT INPUT (EnEmbedding):\n"
        "glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))\n"
        "x = self.value_embedding(x) + self.position_embedding(x)\n"
        "x = torch.cat([x, glb], dim=2)  # [Patches; GLB] immediately!\n\n"
        "# 2. INSIDE ENCODER LAYER (EncoderLayer.forward):\n"
        "# Step A: Self-Attention on ALL tokens (Patches + GLB)\n"
        "x = x + self.dropout(self.self_attention(x, x, x)[0])\n"
        "x = self.norm1(x)\n\n"
        "# Step B: Extract ONLY GLB token for Cross-Attention\n"
        "x_glb = x[:, -1, :].unsqueeze(1)  # [B, 1, D]\n\n"
        "# Step C: Cross-Attention (Query=GLB, Key/Val=Variate Tokens)\n"
        "x_glb_attn = self.cross_attention(x_glb, cross, cross)[0]\n"
        "x_glb = self.norm2(x_glb + x_glb_attn)\n\n"
        "# Step D: Re-concatenate Patches and updated GLB\n"
        "y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)\n"
        "return self.norm3(x + self.ffn(y))"
    )
    add_box(
        code_str,
        rx_x, 230, rx_w, 310,
        fill="#2D3748", stroke="#1A202C", font_color="#68D391", font_size=10, bold=False, align="left", valign="top",
        custom_style="fontFamily=Courier New,Courier,monospace;"
    )

    # Inset 2: How information actually flows to Patches
    add_box(
        "How Do Patches Receive Exogenous Information in TimeXer?",
        rx_x, 560, rx_w, 40,
        fill="#EDF2F7", stroke="#4A5568", font_color="#1A202C", font_size=12, bold=True, rounded=1
    )
    add_box(
        "NO Separate Gated Bridge or Ad-Hoc Cross-Attentions are Needed!\n\n"
        "1. Layer L (Step 2): The Global Token $G$ absorbs exogenous variate features\n"
        "   through Cross-Attention: $G \\leftarrow \\text{CrossAttn}(G, \\text{cross}, \\text{cross})$.\n\n"
        "2. Layer L+1 (Step 1): In the NEXT layer, the updated $G$ enters Self-Attention\n"
        "   alongside the price patches $[P_1, \\dots, P_N; G]$.\n"
        "   Through the standard Self-Attention matrix $(N+1) \\times (N+1)$,\n"
        "   all patches $P_1 \\dots P_N$ freely attend to $G$ and receive exogenous context!\n\n"
        "Why c1_dual had a 'Frankenstein' layout:\n"
        "The project Dev agent artificially placed Cross-Attention BEFORE Self-Attention\n"
        "and added an external Gated Token Bridge with tanh(alpha) because they restricted\n"
        "the model to 1 layer and feared information wouldn't flow in a single pass.\n"
        "In canonical TimeXer, multi-layer stacking solves this naturally with ZERO ad-hoc gates!",
        rx_x, 610, rx_w, 235,
        fill="#FFFBEB", stroke="#D97706", font_color="#78350F", font_size=10, bold=False, align="left", valign="top"
    )

    # Inset 3: Architectural Contrast Table
    add_box(
        "TimeXer vs. DecisionForecast c1_dual Mutation",
        rx_x, 865, rx_w, 40,
        fill="#EDF2F7", stroke="#4A5568", font_color="#1A202C", font_size=12, bold=True, rounded=1
    )
    add_box(
        "Feature                 | Canonical TimeXer (TSLib) | DecisionForecast c1_dual\n"
        "------------------------|---------------------------|-------------------------\n"
        "GLB Token Input         | Concat immediately        | Concat AFTER Cross-Attn\n"
        "Order in Layer          | Self-Attn -> Cross-Attn   | Cross-Attn -> Self-Attn\n"
        "Exogenous Query         | ONLY GLB queries (1 x C)  | Dual GLB + Patches Bridge\n"
        "Information to Patches  | Via next layer Self-Attn  | Via tanh(alpha) Gated Bridge\n"
        "Cross-Attention Count   | Exactly 1 per layer       | 4 (Cross-TS, Cross-Text, 2 Bridge)",
        rx_x, 915, rx_w, 185,
        fill="#FFF5F5", stroke="#E53E3E", font_color="#742A2A", font_size=10, bold=False, align="left", valign="top",
        custom_style="fontFamily=Courier New,Courier,monospace;"
    )

    xml_header = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<mxfile host="app.diagrams.net" modified="2026-10-10T01:00:00.000Z" agent="Antigravity" version="24.0.0" type="device">\n'
        '  <diagram id="canonical-timexer" name="Canonical TimeXer TSLib">\n'
        '    <mxGraphModel dx="2000" dy="1400" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1600" pageHeight="1150" math="1" shadow="0">\n'
        "      <root>\n"
        '        <mxCell id="0" />\n'
        '        <mxCell id="1" parent="0" />\n'
    )
    xml_footer = "      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n"

    return xml_header + "\n".join(elements) + "\n" + xml_footer


def create_standalone_svg() -> str:
    svg_w = 1600
    svg_h = 1150

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}" width="{svg_w}" height="{svg_h}" style="background-color: #FFFFFF; font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, Helvetica, Arial, sans-serif;">',
        "<defs>",
        '  <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '    <path d="M 0 1 L 10 5 L 0 9 z" fill="#4A5568" />',
        "  </marker>",
        '  <marker id="arrow-green" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '    <path d="M 0 1 L 10 5 L 0 9 z" fill="#27AE60" />',
        "  </marker>",
        '  <filter id="shadow" x="-3%" y="-3%" width="106%" height="106%">',
        '    <feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.06"/>',
        "  </filter>",
        "</defs>",
    ]

    def draw_box(x, y, w, h, fill, stroke, title, lines=None, stroke_width=1.5, rx=8, dashed=False, title_color="#1A202C", title_size=13, body_size=11, bold_title=True, align="center"):
        dash_attr = ' stroke-dasharray="6,4"' if dashed else ""
        svg_parts.append(f'  <g filter="url(#shadow)">')
        svg_parts.append(f'    <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"{dash_attr} />')
        
        anchor = "middle" if align == "center" else "start"
        text_x = x + w // 2 if align == "center" else x + 16
        start_y = y + 20 if lines else y + h // 2 + 5
        weight = "bold" if bold_title else "600"
        svg_parts.append(f'    <text x="{text_x}" y="{start_y}" text-anchor="{anchor}" fill="{title_color}" font-size="{title_size}" font-weight="{weight}">{html.escape(title)}</text>')
        if lines:
            line_y = start_y + 17
            for line in lines:
                color = "#4A5568" if not line.startswith("[") else "#2B6CB0"
                weight_line = "bold" if line.startswith("[") or "Step" in line or "Canonical" in line else "normal"
                svg_parts.append(f'    <text x="{text_x}" y="{line_y}" text-anchor="{anchor}" fill="{color}" font-size="{body_size}" font-weight="{weight_line}">{html.escape(line)}</text>')
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

    # 1. Header
    svg_parts.append('  <rect x="50" y="25" width="1500" height="48" rx="8" fill="#2B6CB0" />')
    svg_parts.append('  <text x="800" y="55" text-anchor="middle" fill="#FFFFFF" font-size="19" font-weight="bold">Canonical TimeXer Architecture (Official TSLib / NeurIPS 2024)</text>')
    
    svg_parts.append('  <rect x="50" y="80" width="1500" height="30" rx="6" fill="#EDF2F7" stroke="#CBD5E0" stroke-width="1" />')
    svg_parts.append('  <text x="800" y="100" text-anchor="middle" fill="#4A5568" font-size="12" font-weight="500">Wang et al., \'TimeXer: Empowering Canonical Transformers for Long-Term Time Series Forecasting with Exogenous Variables\' (thuml/Time-Series-Library)</text>')

    # 2. Legend
    leg_y = 125
    svg_parts.append(f'  <text x="60" y="{leg_y + 20}" fill="#2D3748" font-size="12" font-weight="bold">COLOR CODE:</text>')
    draw_box(180, leg_y, 250, 32, "#E8F1FC", "#2B6CB0", "Endogenous Target Series (Price)", title_color="#1A365D", title_size=11, bold_title=True)
    draw_box(450, leg_y, 280, 32, "#EAFBF1", "#27AE60", "Exogenous Variables (Variate Tokens)", title_color="#145A32", title_size=11, bold_title=True)
    draw_box(750, leg_y, 230, 32, "#FDEBD0", "#B9770E", "Global Endogenous Token (G)", title_color="#7D6608", title_size=11, bold_title=True)
    draw_box(1000, leg_y, 270, 32, "#FEF5E7", "#D35400", "Transformer Self & Cross Attention", title_color="#7E5109", title_size=11, bold_title=True)
    draw_box(1290, leg_y, 250, 32, "#EBEDEF", "#2C3E50", "Projection & Forecast Head", title_color="#1B2631", title_size=11, bold_title=True)

    # 3. Main Column
    mx_x = 50
    draw_box(mx_x + 320, 180, 260, 50, "#EBEDEF", "#2C3E50", "Forecast Horizon Output", ["Y_hat ∈ R^[B × H] (Target Prediction)", "[B, H]"], title_color="#1B2631")
    draw_box(mx_x + 320, 255, 260, 48, "#EBEDEF", "#2C3E50", "FlattenHead / Linear Projection", ["Flatten(enc_out) → Linear → Y_hat"], title_color="#1B2631")

    # Layer Container
    draw_box(mx_x + 20, 325, 860, 560, "#FDFEFE", "#2B6CB0", "Canonical TimeXer EncoderLayer (L × Stack)", ["Step 1: Endogenous Self-Attention → Step 2: Global Token Exogenous Cross-Attention → Step 3: FFN"], title_color="#1A365D", title_size=12, bold_title=True, dashed=True)

    draw_box(mx_x + 50, 375, 800, 42, "#EDF2F7", "#CBD5E0", "LayerNorm 3 & Residual", ["LayerNorm(x + y) ∈ R^[B × (N+1) × D]"], title_color="#2D3748", title_size=10)
    draw_box(mx_x + 50, 435, 800, 60, "#FEF5E7", "#D35400", "Feed-Forward Network (FFN)", [
        "y = Conv1d(D → d_ff) → GELU → Conv1d(d_ff → D)",
        "Applied to all (N+1) tokens: [P_1, ..., P_N; G]"
    ], title_color="#7E5109", title_size=10)
    draw_box(mx_x + 50, 515, 800, 45, "#EDF2F7", "#CBD5E0", "Token Re-concatenation", ["x = [x[:, :-1, :]; x_glb] ∈ R^[B × (N+1) × D] (Sequence restored)"], title_color="#2D3748", title_size=10, bold_title=True)

    draw_box(mx_x + 50, 580, 800, 105, "#FEF5E7", "#D35400", "Step 2: Exogenous-to-Endogenous Cross-Attention (Variate-wise Attention)", [
        "Query = x_glb ∈ R^[B × 1 × D]  (ONLY Global Token Queries!)",
        "Key, Value = cross ∈ R^[B × C × D]  (C Exogenous Variate Tokens)",
        "x_glb ← LayerNorm(x_glb + Dropout(CrossAttn(x_glb, cross, cross)))",
        "Complexity: O(1 × C) — Price patches never attend to exogenous directly, eliminating noise!"
    ], title_color="#7E5109", title_size=10, bold_title=True)

    draw_box(mx_x + 50, 705, 800, 40, "#EDF2F7", "#CBD5E0", "Token Splitting", ["x_glb = x[:, -1, :] ∈ R^[B × 1 × D],   patches = x[:, :-1, :] ∈ R^[B × N × D]"], title_color="#2D3748", title_size=10)

    draw_box(mx_x + 50, 765, 800, 80, "#FEF5E7", "#D35400", "Step 1: Endogenous Self-Attention (Temporal-wise Attention)", [
        "SelfAttn(Q=x, K=x, V=x) over all (N+1) tokens [P_1, ..., P_N; G]",
        "x ← LayerNorm(x + Dropout(SelfAttn(x, x, x)))",
        "Models temporal dynamics between patches; G aggregates endogenous price context!"
    ], title_color="#7E5109", title_size=10, bold_title=True)

    # Embeddings
    draw_box(mx_x + 50, 915, 520, 95, "#E8F1FC", "#2B6CB0", "EnEmbedding (Endogenous Series Embedding)", [
        "1. Patches: x_unfold = unfold(X_endo, P, P) → Linear → [B, N, D] + PosEmb",
        "2. Global Token: G = nn.Parameter(randn(1, 1, 1, D)) → [B, 1, D]",
        "3. IMMEDIATE CONCATENATION: x = [Patches; G] ∈ R^[B × (N+1) × D]",
        "NO CROSS-ATTENTION OCCURS BEFORE ENCODER!"
    ], title_color="#1A365D", title_size=10, bold_title=True)

    draw_box(mx_x + 600, 915, 280, 95, "#EAFBF1", "#27AE60", "DataEmbedding_inverted (Exogenous)", [
        "Transposes time & variates: X_exo^T ∈ R^[B × C × T]",
        "Linear(T → D) yields C Variate Tokens",
        "cross ∈ R^[B × C × D]",
        "Each token = entire curve of 1 indicator!"
    ], title_color="#145A32", title_size=10, bold_title=True)

    # Raw Inputs
    draw_box(mx_x + 50, 1040, 520, 60, "#E8F1FC", "#2B6CB0", "Endogenous Input Series", ["X_endo ∈ R^[B × T × 1] (Target Price Series)"], title_color="#1A365D", title_size=11, bold_title=True)
    draw_box(mx_x + 600, 1040, 280, 60, "#EAFBF1", "#27AE60", "Exogenous Input Series", ["X_exo ∈ R^[B × T × C] (C External Variables)"], title_color="#145A32", title_size=11, bold_title=True)

    # Connections
    draw_arrow(mx_x + 310, 1040, mx_x + 310, 1010, "[B, T, 1]")
    draw_arrow(mx_x + 740, 1040, mx_x + 740, 1010, "[B, T, C]")
    draw_arrow(mx_x + 310, 915, mx_x + 310, 845, "[B, N+1, D] (Patches + G)")
    draw_arrow(mx_x + 740, 915, mx_x + 740, 685, "cross [B, C, D] (Key, Val)", color="#27AE60", marker="arrow-green")

    draw_arrow(mx_x + 450, 765, mx_x + 450, 745)
    draw_arrow(mx_x + 450, 705, mx_x + 450, 685, "x_glb [B, 1, D] (Query)")
    draw_arrow(mx_x + 450, 580, mx_x + 450, 560)
    draw_arrow(mx_x + 450, 515, mx_x + 450, 495)
    draw_arrow(mx_x + 450, 435, mx_x + 450, 417)
    draw_arrow(mx_x + 450, 375, mx_x + 450, 303, "enc_out [B, N+1, D]")
    draw_arrow(mx_x + 450, 255, mx_x + 450, 230, "[B, H]")

    # 4. Right Column Insets
    rx_x = 910
    rx_w = 640

    draw_box(rx_x, 180, rx_w, 40, "#EDF2F7", "#4A5568", "Official TSLib Code: models/TimeXer.py (Exact Flow)", title_size=12, bold_title=True)
    draw_box(rx_x, 230, rx_w, 310, "#2D3748", "#1A202C", "Official PyTorch Implementation", [
        "# 1. IMMEDIATE CONCATENATION AT INPUT (EnEmbedding):",
        "glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))",
        "x = self.value_embedding(x) + self.position_embedding(x)",
        "x = torch.cat([x, glb], dim=2)  # [Patches; GLB] immediately!",
        "",
        "# 2. INSIDE ENCODER LAYER (EncoderLayer.forward):",
        "# Step A: Self-Attention on ALL tokens (Patches + GLB)",
        "x = x + self.dropout(self.self_attention(x, x, x)[0])",
        "x = self.norm1(x)",
        "",
        "# Step B: Extract ONLY GLB token for Cross-Attention",
        "x_glb = x[:, -1, :].unsqueeze(1)  # [B, 1, D]",
        "",
        "# Step C: Cross-Attention (Query=GLB, Key/Val=Variate Tokens)",
        "x_glb_attn = self.cross_attention(x_glb, cross, cross)[0]",
        "x_glb = self.norm2(x_glb + x_glb_attn)",
        "",
        "# Step D: Re-concatenate Patches and updated GLB",
        "y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)",
        "return self.norm3(x + self.ffn(y))"
    ], title_color="#68D391", body_size=10, bold_title=True, align="left")

    draw_box(rx_x, 560, rx_w, 40, "#EDF2F7", "#4A5568", "How Do Patches Receive Exogenous Information in TimeXer?", title_size=12, bold_title=True)
    draw_box(rx_x, 610, rx_w, 235, "#FFFBEB", "#D97706", "Natural Propagation Across Layer Stacks", [
        "NO Separate Gated Bridge or Ad-Hoc Cross-Attentions are Needed!",
        "",
        "1. Layer L (Step 2): The Global Token G absorbs exogenous variate features",
        "   through Cross-Attention: G ← CrossAttn(G, cross, cross).",
        "",
        "2. Layer L+1 (Step 1): In the NEXT layer, the updated G enters Self-Attention",
        "   alongside the price patches [P_1, ..., P_N; G].",
        "   Through the standard Self-Attention matrix (N+1) × (N+1),",
        "   all patches P_1 ... P_N freely attend to G and receive exogenous context!",
        "",
        "Why c1_dual had a 'Frankenstein' layout:",
        "The project Dev agent artificially placed Cross-Attention BEFORE Self-Attention",
        "and added an external Gated Token Bridge with tanh(alpha) because they restricted",
        "the model to 1 layer and feared information wouldn't flow in a single pass.",
        "In canonical TimeXer, multi-layer stacking solves this naturally with ZERO ad-hoc gates!"
    ], body_size=10, bold_title=True, align="left")

    draw_box(rx_x, 865, rx_w, 40, "#EDF2F7", "#4A5568", "TimeXer vs. DecisionForecast c1_dual Mutation", title_size=12, bold_title=True)
    draw_box(rx_x, 915, rx_w, 185, "#FFF5F5", "#E53E3E", "Architectural Divergence Summary", [
        "Feature                 | Canonical TimeXer (TSLib) | DecisionForecast c1_dual",
        "------------------------|---------------------------|-------------------------",
        "GLB Token Input         | Concat immediately        | Concat AFTER Cross-Attn",
        "Order in Layer          | Self-Attn -> Cross-Attn   | Cross-Attn -> Self-Attn",
        "Exogenous Query         | ONLY GLB queries (1 x C)  | Dual GLB + Patches Bridge",
        "Information to Patches  | Via next layer Self-Attn  | Via tanh(alpha) Gated Bridge",
        "Cross-Attention Count   | Exactly 1 per layer       | 4 (Cross-TS, Cross-Text, 2 Bridge)"
    ], body_size=10, bold_title=True, align="left")

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def main() -> None:
    out_dir = Path("Articles/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    drawio_path = out_dir / "canonical_timexer_tslib.drawio"
    svg_path = out_dir / "canonical_timexer_tslib.svg"

    print("Generating Canonical TimeXer Draw.io XML...")
    drawio_xml = create_drawio_xml()
    drawio_path.write_text(drawio_xml, encoding="utf-8")
    print(f"Saved Draw.io file to: {drawio_path} ({len(drawio_xml)} bytes)")

    print("Generating Canonical TimeXer standalone publication SVG...")
    svg_content = create_standalone_svg()
    svg_path.write_text(svg_content, encoding="utf-8")
    print(f"Saved standalone SVG to: {svg_path} ({len(svg_content)} bytes)")


if __name__ == "__main__":
    main()
