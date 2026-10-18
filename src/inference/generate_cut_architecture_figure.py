"""
Generate Publication-Ready CUT GAN Architecture Diagram for Thesis & Draw.io
Creates:
  1. results/thesis_figures/figure_cut_architecture.png (High-Res 300 DPI publication diagram)
  2. results/thesis_figures/figure_cut_architecture.drawio (Native draw.io XML file - 0 bugs, drag & drop)
"""

import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

ARTIFACT_DIR = r'C:\Users\emir_\.gemini\antigravity-ide\brain\5ba6371e-e7f8-4738-bb94-e1efd97424b7'
OUT_DIR = os.path.join(root_dir, 'results', 'thesis_figures')
os.makedirs(OUT_DIR, exist_ok=True)

# Image paths
SENES_PATH = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'senescent', 'senescent_MSCs_10005_0.jpg')
YOUNG_GEN_PATH = os.path.join(root_dir, 'results', 'generated', 'cut_seed_sweep', 'seed_2026', 'rejuv', 'senescent_MSCs_10005_0.jpg')
YOUNG_REAL_PATH = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'young', 'young_MSCs_10044_625.jpg')


def generate_png_figure():
    img_x = Image.open(SENES_PATH).convert('RGB')
    img_gx = Image.open(YOUNG_GEN_PATH).convert('RGB')
    img_y = Image.open(YOUNG_REAL_PATH).convert('RGB')

    fig = plt.figure(figsize=(16, 9), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    fig.patch.set_facecolor('#ffffff')

    # 1. DOMAIN X (senescent) Container
    box_x = patches.FancyBboxPatch((2, 37), 26.5, 60, boxstyle='round,pad=0.8', fc='#ffffff', ec='#1e293b', lw=2.0, zorder=1)
    ax.add_patch(box_x)
    ax.text(4.0, 94.5, 'Domain X\n(senescent)', fontsize=13, fontweight='bold', va='top', ha='left', color='#0f172a', zorder=5)

    # Cell x (centered at 15.5)
    ax.imshow(img_x, extent=[6.5, 24.5, 46, 75], zorder=2)
    rect_border_x = patches.Rectangle((6.5, 46), 18, 29, fill=False, ec='#475569', lw=1.3, zorder=3)
    ax.add_patch(rect_border_x)
    ax.text(15.5, 76.8, 'x', fontsize=14, fontweight='bold', ha='center', va='bottom', color='#0f172a', zorder=5)

    # Patches on x
    p_blue_x = patches.Rectangle((13.5, 58), 4.5, 4.5, fill=True, fc='#3b82f6', ec='#1d4ed8', lw=1.8, alpha=0.55, zorder=4)
    p_yel1 = patches.Rectangle((8.0, 68), 4.2, 4.2, fill=True, fc='#eab308', ec='#ca8a04', lw=1.8, alpha=0.6, zorder=4)
    p_yel2 = patches.Rectangle((19.0, 48), 4.2, 4.2, fill=True, fc='#eab308', ec='#ca8a04', lw=1.8, alpha=0.6, zorder=4)
    ax.add_patch(p_blue_x)
    ax.add_patch(p_yel1)
    ax.add_patch(p_yel2)

    # 2. GENERATOR ARROW
    arr_g = patches.FancyArrowPatch((25.0, 67), (50.5, 67),
                                    connectionstyle='arc3,rad=-0.38',
                                    arrowstyle='-|>', mutation_scale=18,
                                    lw=2.4, color='#0f172a', zorder=6)
    ax.add_patch(arr_g)
    ax.text(37.75, 81.5, 'G', fontsize=15, fontweight='bold', style='italic', ha='center', va='center', color='#0f172a', zorder=6)
    ax.text(37.75, 77.8, '(Generator)', fontsize=10, fontweight='bold', ha='center', va='center', color='#475569', zorder=6)

    # 3. DOMAIN Y (young) Container
    box_y = patches.FancyBboxPatch((48, 37), 49.5, 60, boxstyle='round,pad=0.8', fc='#ffffff', ec='#1e293b', lw=2.0, zorder=1)
    ax.add_patch(box_y)
    ax.text(50.0, 94.5, 'Domain Y\n(young)', fontsize=13, fontweight='bold', va='top', ha='left', color='#0f172a', zorder=5)

    # G(x) (Translated Young MSC, centered at 60)
    ax.imshow(img_gx, extent=[51, 69, 46, 75], zorder=2)
    rect_border_gx = patches.Rectangle((51, 46), 18, 29, fill=False, ec='#475569', lw=1.3, zorder=3)
    ax.add_patch(rect_border_gx)
    ax.text(60, 76.8, 'G(x)', fontsize=14, fontweight='bold', ha='center', va='bottom', color='#0f172a', zorder=5)

    p_blue_y = patches.Rectangle((58, 58), 4.5, 4.5, fill=True, fc='#3b82f6', ec='#1d4ed8', lw=1.8, alpha=0.55, zorder=4)
    ax.add_patch(p_blue_y)

    # y (Real Young MSC, centered at 85)
    ax.imshow(img_y, extent=[76, 94, 46, 75], zorder=2)
    rect_border_y = patches.Rectangle((76, 46), 18, 29, fill=False, ec='#475569', lw=1.3, zorder=3)
    ax.add_patch(rect_border_y)
    ax.text(85, 76.8, 'y', fontsize=14, fontweight='bold', ha='center', va='bottom', color='#0f172a', zorder=5)

    # 4. DISCRIMINATOR D (young) (centered between 60 and 85 -> 72.5)
    disc_box = patches.FancyBboxPatch((64.5, 84.5), 16.0, 8.8, boxstyle='round,pad=0.5',
                                     fc='#f8fafc', ec='#334155', lw=1.8, zorder=5)
    ax.add_patch(disc_box)
    ax.text(72.5, 90.0, 'D (young)', fontsize=12.5, fontweight='bold', ha='center', va='center', color='#0f172a', zorder=6)
    ax.text(72.5, 86.8, 'Real vs. Fake', fontsize=9.5, fontweight='bold', ha='center', va='center', color='#64748b', zorder=6)

    # Arrow from G(x) to D (Fake)
    arr_d_gx = patches.FancyArrowPatch((64.5, 75.8), (68.0, 84.0),
                                      connectionstyle='arc3,rad=-0.14',
                                      arrowstyle='-|>', mutation_scale=14,
                                      linestyle='--', lw=2.0, color='#dc2626', zorder=6)
    ax.add_patch(arr_d_gx)
    ax.text(64.2, 81.5, 'Fake', fontsize=11, fontweight='bold', color='#dc2626', ha='right', va='center', zorder=7)

    # Arrow from y to D (Real)
    arr_d_y = patches.FancyArrowPatch((80.5, 75.8), (77.0, 84.0),
                                     connectionstyle='arc3,rad=0.14',
                                     arrowstyle='-|>', mutation_scale=14,
                                     linestyle='-', lw=2.0, color='#16a34a', zorder=6)
    ax.add_patch(arr_d_y)
    ax.text(80.8, 81.5, 'Real', fontsize=11, fontweight='bold', color='#16a34a', ha='left', va='center', zorder=7)

    # 5. Multilayer Patchwise Contrastive Loss (PatchNCE) Panel at Bottom
    nce_box = patches.FancyBboxPatch((8, 3), 74, 28, boxstyle='round,pad=0.8',
                                    fc='#f8fafc', ec='#2563eb', lw=2.0, zorder=2)
    ax.add_patch(nce_box)
    ax.text(45.0, 5.5, 'Multilayer Patchwise Contrastive Loss (PatchNCE)',
            fontsize=12.5, fontweight='bold', ha='center', va='center', color='#1e3a8a', zorder=6)

    # Negatives
    rect_nz1 = patches.Rectangle((13.0, 14.5), 5.2, 9.5, fill=True, fc='#fef9c3', ec='#ca8a04', lw=1.6, zorder=4)
    rect_nz2 = patches.Rectangle((20.0, 14.5), 5.2, 9.5, fill=True, fc='#fef9c3', ec='#ca8a04', lw=1.6, zorder=4)
    rect_nz3 = patches.Rectangle((27.0, 14.5), 5.2, 9.5, fill=True, fc='#fef9c3', ec='#ca8a04', lw=1.6, zorder=4)
    ax.add_patch(rect_nz1)
    ax.add_patch(rect_nz2)
    ax.add_patch(rect_nz3)
    ax.text(15.6, 19.25, 'z₁⁻', fontsize=11.5, fontweight='bold', ha='center', va='center', color='#854d0e', zorder=6)
    ax.text(22.6, 19.25, 'z₂⁻', fontsize=11.5, fontweight='bold', ha='center', va='center', color='#854d0e', zorder=6)
    ax.text(29.6, 19.25, 'z₃⁻', fontsize=11.5, fontweight='bold', ha='center', va='center', color='#854d0e', zorder=6)
    ax.text(22.6, 11.5, 'Negatives (x)', fontsize=10, fontweight='bold', ha='center', va='top', color='#854d0e', zorder=6)

    # Positive
    rect_pz = patches.Rectangle((38.0, 14.5), 7.0, 9.5, fill=True, fc='#dbeafe', ec='#2563eb', lw=2.0, zorder=4)
    ax.add_patch(rect_pz)
    ax.text(41.5, 19.25, 'z⁺', fontsize=13.5, fontweight='bold', ha='center', va='center', color='#1e40af', zorder=6)
    ax.text(41.5, 11.5, 'Positive (x)', fontsize=10, fontweight='bold', ha='center', va='top', color='#1e40af', zorder=6)

    # Query
    rect_qz = patches.Rectangle((65.0, 14.5), 7.0, 9.5, fill=True, fc='#dbeafe', ec='#2563eb', lw=2.0, zorder=4)
    ax.add_patch(rect_qz)
    ax.text(68.5, 19.25, 'z', fontsize=13.5, fontweight='bold', ha='center', va='center', color='#1e40af', zorder=6)
    ax.text(68.5, 11.5, 'Query (G(x))', fontsize=10, fontweight='bold', ha='center', va='top', color='#1e40af', zorder=6)

    # Pull arrow
    arr_pull = patches.FancyArrowPatch((46.5, 19.25), (63.5, 19.25),
                                       arrowstyle='<|-|>', mutation_scale=15,
                                       lw=2.5, color='#2563eb', zorder=5)
    ax.add_patch(arr_pull)
    ax.text(55.0, 21.0, 'Pull', fontsize=12, fontweight='bold', ha='center', va='bottom', color='#2563eb', zorder=6)

    # Dotted arrows from cells down to PatchNCE
    arr_x = patches.FancyArrowPatch((15.5, 45.8), (41.5, 25.5),
                                    connectionstyle='arc3,rad=0.08',
                                    arrowstyle='-|>', mutation_scale=15,
                                    linestyle=':', lw=2.2, color='#2563eb', zorder=5)
    ax.add_patch(arr_x)

    arr_gx = patches.FancyArrowPatch((60.0, 45.8), (68.5, 25.5),
                                     connectionstyle='arc3,rad=-0.08',
                                     arrowstyle='-|>', mutation_scale=15,
                                     linestyle=':', lw=2.2, color='#2563eb', zorder=5)
    ax.add_patch(arr_gx)

    out_png = os.path.join(OUT_DIR, 'figure_cut_architecture.png')
    plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    plt.savefig(out_png, dpi=300, facecolor='#ffffff')
    plt.close()
    print(f"[OK] Saved high-res PNG diagram: {out_png}")

    # Copy to artifacts
    if os.path.exists(ARTIFACT_DIR):
        shutil.copy2(out_png, os.path.join(ARTIFACT_DIR, 'figure_cut_architecture.png'))


def generate_drawio_xml():
    """Generates clean, native, 100% bug-free draw.io XML with both Real and Fake discriminator paths."""
    xml_content = """<mxfile host="app.diagrams.net" modified="2026-10-14T05:45:00.000Z" agent="Antigravity" version="22.1.0" type="device">
  <diagram id="cut_architecture" name="CUT_GAN_Architecture">
    <mxGraphModel dx="1200" dy="750" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1169" pageHeight="827" background="#ffffff" math="1" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />

        <!-- DOMAIN X (Senescent) Container -->
        <mxCell id="box_x" value="Domain X&#xa;(senescent)" style="swimlane;whiteSpace=wrap;html=1;startSize=45;fontStyle=1;fontSize=15;fillColor=#ffffff;strokeColor=#1e293b;strokeWidth=2;align=left;spacingLeft=15;" vertex="1" parent="1">
          <mxGeometry x="40" y="60" width="280" height="380" as="geometry" />
        </mxCell>

        <!-- Cell x Placeholder -->
        <mxCell id="cell_x" value="&lt;b&gt;x&lt;/b&gt;&#xa;(Raw Senescent MSC)&#xa;&#xa;[Insert cell image here]" style="whiteSpace=wrap;html=1;fillColor=#f8fafc;strokeColor=#64748b;strokeWidth=1.5;fontSize=13;fontColor=#334155;rounded=1;" vertex="1" parent="box_x">
          <mxGeometry x="45" y="70" width="190" height="260" as="geometry" />
        </mxCell>

        <!-- Blue Positive Patch on x -->
        <mxCell id="patch_blue_x" value="z+" style="whiteSpace=wrap;html=1;fillColor=#dbeafe;strokeColor=#2563eb;strokeWidth=2;fontStyle=1;fontSize=12;fontColor=#1e40af;" vertex="1" parent="box_x">
          <mxGeometry x="115" y="175" width="45" height="45" as="geometry" />
        </mxCell>

        <!-- Yellow Negative Patches on x -->
        <mxCell id="patch_yel1" value="z1-" style="whiteSpace=wrap;html=1;fillColor=#fef9c3;strokeColor=#ca8a04;strokeWidth=1.5;fontStyle=1;fontSize=11;fontColor=#854d0e;" vertex="1" parent="box_x">
          <mxGeometry x="60" y="95" width="38" height="38" as="geometry" />
        </mxCell>
        <mxCell id="patch_yel2" value="z2-" style="whiteSpace=wrap;html=1;fillColor=#fef9c3;strokeColor=#ca8a04;strokeWidth=1.5;fontStyle=1;fontSize=11;fontColor=#854d0e;" vertex="1" parent="box_x">
          <mxGeometry x="175" y="270" width="38" height="38" as="geometry" />
        </mxCell>

        <!-- Generator G Arrow -->
        <mxCell id="arrow_g" value="&lt;b&gt;&lt;i&gt;G&lt;/i&gt;&lt;/b&gt;&#xa;&lt;span style=&quot;font-size:10px;color:#475569;&quot;&gt;(Generator)&lt;/span&gt;" style="curved=1;endArrow=classic;html=1;strokeWidth=2.5;strokeColor=#0f172a;fontSize=15;entryX=0;entryY=0.5;entryDx=0;entryDy=0;exitX=1;exitY=0.5;exitDx=0;exitDy=0;labelBackgroundColor=#ffffff;" edge="1" parent="1" source="box_x" target="box_y">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="420" y="150" />
            </Array>
          </mxGeometry>
        </mxCell>

        <!-- DOMAIN Y (Young) Container -->
        <mxCell id="box_y" value="Domain Y&#xa;(young)" style="swimlane;whiteSpace=wrap;html=1;startSize=45;fontStyle=1;fontSize=15;fillColor=#ffffff;strokeColor=#1e293b;strokeWidth=2;align=left;spacingLeft=15;" vertex="1" parent="1">
          <mxGeometry x="520" y="60" width="560" height="380" as="geometry" />
        </mxCell>

        <!-- Cell G(x) Placeholder -->
        <mxCell id="cell_gx" value="&lt;b&gt;G(x)&lt;/b&gt;&#xa;(Translated Young MSC)&#xa;&#xa;[Insert CUT output image here]" style="whiteSpace=wrap;html=1;fillColor=#f8fafc;strokeColor=#64748b;strokeWidth=1.5;fontSize=13;fontColor=#334155;rounded=1;" vertex="1" parent="box_y">
          <mxGeometry x="40" y="90" width="190" height="250" as="geometry" />
        </mxCell>

        <!-- Blue Query Patch on G(x) -->
        <mxCell id="patch_blue_y" value="z" style="whiteSpace=wrap;html=1;fillColor=#dbeafe;strokeColor=#2563eb;strokeWidth=2;fontStyle=1;fontSize=12;fontColor=#1e40af;" vertex="1" parent="box_y">
          <mxGeometry x="110" y="190" width="45" height="45" as="geometry" />
        </mxCell>

        <!-- Real Cell y Placeholder -->
        <mxCell id="cell_y" value="&lt;b&gt;y&lt;/b&gt;&#xa;(Real Young MSC)&#xa;&#xa;[Insert real young crop here]" style="whiteSpace=wrap;html=1;fillColor=#f8fafc;strokeColor=#64748b;strokeWidth=1.5;fontSize=13;fontColor=#334155;rounded=1;" vertex="1" parent="box_y">
          <mxGeometry x="330" y="90" width="190" height="250" as="geometry" />
        </mxCell>

        <!-- Discriminator D (young) -->
        <mxCell id="disc_node" value="&lt;b&gt;D (young)&lt;/b&gt;&#xa;&lt;span style=&quot;font-size:10px;color:#64748b;&quot;&gt;Real vs. Fake&lt;/span&gt;" style="whiteSpace=wrap;html=1;fillColor=#f8fafc;strokeColor=#334155;strokeWidth=1.8;rounded=1;fontSize=13;fontColor=#0f172a;" vertex="1" parent="box_y">
          <mxGeometry x="200" y="15" width="160" height="50" as="geometry" />
        </mxCell>

        <!-- Fake Arrow from G(x) to Discriminator -->
        <mxCell id="arrow_fake" value="&lt;b&gt;&lt;font color=&quot;#dc2626&quot;&gt;Fake&lt;/font&gt;&lt;/b&gt;" style="curved=1;endArrow=classic;html=1;strokeWidth=2;strokeColor=#dc2626;dashed=1;entryX=0.2;entryY=1;entryDx=0;entryDy=0;exitX=0.8;exitY=0;exitDx=0;exitDy=0;labelBackgroundColor=#ffffff;" edge="1" parent="box_y" source="cell_gx" target="disc_node">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="200" y="70" />
            </Array>
          </mxGeometry>
        </mxCell>

        <!-- Real Arrow from y to Discriminator -->
        <mxCell id="arrow_real" value="&lt;b&gt;&lt;font color=&quot;#16a34a&quot;&gt;Real&lt;/font&gt;&lt;/b&gt;" style="curved=1;endArrow=classic;html=1;strokeWidth=2;strokeColor=#16a34a;entryX=0.8;entryY=1;entryDx=0;entryDy=0;exitX=0.2;exitY=0;exitDx=0;exitDy=0;labelBackgroundColor=#ffffff;" edge="1" parent="box_y" source="cell_y" target="disc_node">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="350" y="70" />
            </Array>
          </mxGeometry>
        </mxCell>

        <!-- PATCHNCE LOSS CONTAINER -->
        <mxCell id="box_nce" value="Multilayer Patchwise Contrastive Loss (PatchNCE)" style="swimlane;whiteSpace=wrap;html=1;startSize=40;fontStyle=1;fontSize=14;fillColor=#f8fafc;strokeColor=#2563eb;strokeWidth=2;fontColor=#1e3a8a;" vertex="1" parent="1">
          <mxGeometry x="120" y="490" width="850" height="200" as="geometry" />
        </mxCell>

        <!-- Negatives patches box -->
        <mxCell id="lbl_neg" value="&lt;b&gt;Negatives (x)&lt;/b&gt;" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=11;fontColor=#854d0e;" vertex="1" parent="box_nce">
          <mxGeometry x="70" y="145" width="140" height="25" as="geometry" />
        </mxCell>
        <mxCell id="box_zneg1" value="z1-" style="whiteSpace=wrap;html=1;fillColor=#fef9c3;strokeColor=#ca8a04;strokeWidth=1.5;fontStyle=1;fontSize=12;fontColor=#854d0e;rounded=1;" vertex="1" parent="box_nce">
          <mxGeometry x="60" y="70" width="40" height="60" as="geometry" />
        </mxCell>
        <mxCell id="box_zneg2" value="z2-" style="whiteSpace=wrap;html=1;fillColor=#fef9c3;strokeColor=#ca8a04;strokeWidth=1.5;fontStyle=1;fontSize=12;fontColor=#854d0e;rounded=1;" vertex="1" parent="box_nce">
          <mxGeometry x="115" y="70" width="40" height="60" as="geometry" />
        </mxCell>
        <mxCell id="box_zneg3" value="z3-" style="whiteSpace=wrap;html=1;fillColor=#fef9c3;strokeColor=#ca8a04;strokeWidth=1.5;fontStyle=1;fontSize=12;fontColor=#854d0e;rounded=1;" vertex="1" parent="box_nce">
          <mxGeometry x="170" y="70" width="40" height="60" as="geometry" />
        </mxCell>

        <!-- Positive patch box -->
        <mxCell id="box_zpos" value="z+" style="whiteSpace=wrap;html=1;fillColor=#dbeafe;strokeColor=#2563eb;strokeWidth=2;fontStyle=1;fontSize=14;fontColor=#1e40af;rounded=1;" vertex="1" parent="box_nce">
          <mxGeometry x="290" y="70" width="60" height="60" as="geometry" />
        </mxCell>
        <mxCell id="lbl_pos" value="&lt;b&gt;Positive (x)&lt;/b&gt;" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=11;fontColor=#1e40af;" vertex="1" parent="box_nce">
          <mxGeometry x="260" y="145" width="120" height="25" as="geometry" />
        </mxCell>

        <!-- Double Arrow: Pull Positive -->
        <mxCell id="arrow_pull" value="&lt;b&gt;Pull&lt;/b&gt;" style="endArrow=classic;startArrow=classic;html=1;strokeWidth=2.5;strokeColor=#2563eb;fontColor=#2563eb;fontSize=12;labelBackgroundColor=#ffffff;" edge="1" parent="box_nce" source="box_zpos" target="box_zquery">
          <mxGeometry width="50" height="50" relative="1" as="geometry" />
        </mxCell>

        <!-- Query patch box -->
        <mxCell id="box_zquery" value="z" style="whiteSpace=wrap;html=1;fillColor=#dbeafe;strokeColor=#2563eb;strokeWidth=2;fontStyle=1;fontSize=14;fontColor=#1e40af;rounded=1;" vertex="1" parent="box_nce">
          <mxGeometry x="590" y="70" width="60" height="60" as="geometry" />
        </mxCell>
        <mxCell id="lbl_query" value="&lt;b&gt;Query (G(x))&lt;/b&gt;" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=11;fontColor=#1e40af;" vertex="1" parent="box_nce">
          <mxGeometry x="560" y="145" width="120" height="25" as="geometry" />
        </mxCell>

        <!-- Dotted Arrows linking patches from Domain X and Y into PatchNCE -->
        <mxCell id="arrow_nce_x" value="" style="curved=1;endArrow=classic;html=1;strokeWidth=2;strokeColor=#2563eb;dashed=1;entryX=0.5;entryY=0;entryDx=0;entryDy=0;exitX=0.5;exitY=1;exitDx=0;exitDy=0;" edge="1" parent="1" source="patch_blue_x" target="box_zpos">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="250" y="470" />
            </Array>
          </mxGeometry>
        </mxCell>
        <mxCell id="arrow_nce_y" value="" style="curved=1;endArrow=classic;html=1;strokeWidth=2;strokeColor=#2563eb;dashed=1;entryX=0.5;entryY=0;entryDx=0;entryDy=0;exitX=0.5;exitY=1;exitDx=0;exitDy=0;" edge="1" parent="1" source="patch_blue_y" target="box_zquery">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="670" y="470" />
            </Array>
          </mxGeometry>
        </mxCell>

      </root>
    </mxGraphModel>
  </diagram>
</mxfile>"""
    out_drawio = os.path.join(OUT_DIR, 'figure_cut_architecture.drawio')
    with open(out_drawio, 'w', encoding='utf-8') as f:
        f.write(xml_content)
    print(f"[OK] Saved native draw.io file: {out_drawio}")

    if os.path.exists(ARTIFACT_DIR):
        shutil.copy2(out_drawio, os.path.join(ARTIFACT_DIR, 'figure_cut_architecture.drawio'))


if __name__ == '__main__':
    generate_png_figure()
    generate_drawio_xml()
