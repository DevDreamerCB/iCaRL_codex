from __future__ import annotations

import datetime as _dt
import os
import zipfile
from html import escape


OUT_PATH = "/data1/bochen/cbcontinual/iCaRL_codex/research/full_results_progression_20260317_zh.pptx"

SLIDES = [
    {
        "title": "iCaRL_codex 关键实验递进",
        "subtitle": "仅整理 3 seeds + 30 epochs 的 full 结果\n场景：跨被试类增量学习（AB -> BC -> CD）\n日期：2026-03-17",
        "body": [
            "目标：按“加了什么方法 -> 指标变成多少”整理成论文可用版本",
            "指标顺序统一为：task1 / task2 / task3",
            "当前 best full：84.18 / 60.75 / 49.36",
        ],
    },
    {
        "title": "任务设定",
        "body": [
            "stage1：训练被试 1,2,3 的 A,B；测试 AB",
            "stage2：训练被试 4,5,6 的 B,C；测试 ABC",
            "stage3：训练被试 7,8,9 的 C,D；测试 ABCD",
            "类别映射：A=左手，B=右手，C=双脚，D=舌头",
            "主目标：提高 task2、task3，尤其 task3 旧类保持",
        ],
    },
    {
        "title": "递进主线总览",
        "body": [
            "baseline -> 去掉 contrastive -> adapter -> LwF/replaymix2/oldweight",
            "-> normalized NME -> 更合理的 stage epochs",
            "-> stage3 feature distill + task-affine",
            "-> hybrid NME+logits -> new_only prototype blend -> affine2 微调",
            "结论：真正的大拐点是 normalized NME、hybrid、new_only prototype blend",
        ],
    },
    {
        "title": "0. 原始基线",
        "body": [
            "运行名：baseline_confirm",
            "方法：原始 iCaRL 基线",
            "结果：85.80 / 59.85 / 37.23",
            "结论：作为全部改进的起点，task3 明显偏低",
        ],
    },
    {
        "title": "1. 去掉对比损失",
        "body": [
            "运行名：no_contrastive_confirm",
            "添加方法：去掉 supervised contrastive loss",
            "结果：85.80 / 60.78 / 40.24",
            "相对基线：task2 +0.93，task3 +3.01",
            "结论：当前实现里的 contrastive loss 是负收益",
        ],
    },
    {
        "title": "2. 加入轻量 Adapter",
        "body": [
            "运行名：no_contrastive_adapter16_s2_confirm",
            "添加方法：embedding 后 task adapter(dim=16)，从 stage2 开始启用",
            "结果：85.34 / 61.52 / 41.34",
            "结论：轻量 PEFT 对后续阶段有帮助",
        ],
    },
    {
        "title": "3. 引入 LwF",
        "body": [
            "运行名：lwf015_replaymix2_adapter16_confirm",
            "添加方法：LwF + replaymix2",
            "结果：85.73 / 58.90 / 41.56",
            "结论：task3 继续涨，但 task2 先掉，说明旧类保持和新类学习开始冲突",
        ],
    },
    {
        "title": "4. 强化旧类权重",
        "body": [
            "运行名：lwf015_replaymix2_adapter16_mem36_oldweight25_confirm",
            "添加方法：memory=36 + oldweight=2.5",
            "结果：85.73 / 58.49 / 42.72",
            "结论：更偏向保护老类后，task3 继续提升",
        ],
    },
    {
        "title": "5. normalized NME",
        "body": [
            "运行名：lwf015_normnme_adapter16_mem36_oldweight25_confirm",
            "添加方法：normalized NME",
            "结果：86.11 / 59.46 / 44.85",
            "结论：一个大拐点，说明原型几何修正非常关键",
        ],
    },
    {
        "title": "6. 分阶段训练轮数",
        "body": [
            "运行名：lwf015_normnme_adapter16_mem36_oldweight25_stage101214_confirm",
            "添加方法：stage epochs 调整为 10,12,14",
            "结果：84.18 / 60.62 / 46.99",
            "结论：后两阶段更强正则/更长训练明显有效",
        ],
    },
    {
        "title": "7. oldweight 精修",
        "body": [
            "运行名：lwf015_normnme_adapter16_mem36_oldweight30_stage101212_confirm",
            "添加方法：oldweight 调到 3.0，schedule 调到 10,12,12",
            "结果：84.18 / 60.54 / 47.79",
            "结论：非 hybrid 主线基本成型",
        ],
    },
    {
        "title": "8. stage3 distill + task-affine",
        "body": [
            "运行名：lwf0175T15_normnme_adapter16_mem36_oldweight325_stage101212_stagefd0025_taskaffine_s3_confirm",
            "添加方法：LwF T=1.5 + stage3 feature distill + task-affine_s3",
            "结果：84.18 / 60.86 / 48.29",
            "结论：非 hybrid 成熟强线，task3 已到 48+",
        ],
    },
    {
        "title": "9. hybrid NME + logits",
        "body": [
            "运行名：lwf0175T15_normnmehybrid_s3old04_adapter16_mem36_oldweight3375_stage101212_stagefd0025_taskaffine_s3_confirm",
            "添加方法：stage3-only hybrid NME+logits",
            "结果：85.72 / 59.62 / 48.52",
            "结论：训练分类器与最终 NME 决策器的错配被部分修复",
        ],
    },
    {
        "title": "10. new_only prototype blend",
        "body": [
            "运行名：lwf0175T15_normnmehybrid_s3old04_ptblend07s3new_adapter16_mem36_oldweight30_stage101212_stagefd0025_taskaffine_s3_confirm",
            "添加方法：stage3-only + new_only prototype blend",
            "结果：84.18 / 60.80 / 48.91",
            "结论：只修正新增类原型，比全局原型修正更稳",
        ],
    },
    {
        "title": "11. affine2",
        "body": [
            "运行名：lwf0175T15_normnmehybrid_s3old04_ptblend07s3new_affine2_adapter16_mem36_oldweight30_stage101212_stagefd0025_taskaffine_s3_confirm",
            "添加方法：task-affine 从 stage2 开始启用",
            "结果：84.18 / 60.80 / 49.04",
            "结论：阶段条件化的轻量分布校正继续有效",
        ],
    },
    {
        "title": "12. stagefd 与 oldweight 精修",
        "body": [
            "运行名：lwf0175T15_normnmehybrid_s3old04_ptblend07s3new_affine2_adapter16_mem36_oldweight3125_stage101212_stagefd003_taskaffine_s3_confirm",
            "添加方法：stage3 distill=0.03 + oldweight=3.125",
            "结果：84.18 / 60.75 / 49.22",
            "结论：稳定进入 49+ 区间",
        ],
    },
    {
        "title": "13. ptblend alpha 精修",
        "body": [
            "运行名：lwf0175T15_normnmehybrid_s3old04_ptblend02s3new_affine2_adapter16_mem36_oldweight3125_stage101212_stagefd003_taskaffine_s3_confirm",
            "添加方法：将 new_only stage3 ptblend alpha 压到 0.2",
            "结果：84.18 / 60.75 / 49.36",
            "结论：当前最好 full",
        ],
    },
    {
        "title": "典型无效或负收益方法",
        "body": [
            "embedding_only_confirm：83.33 / 51.08 / 37.65",
            "proto_align_confirm：85.80 / 59.59 / 35.65",
            "task BN full：84.18 / 60.85 / 47.22",
            "group-bias 主线 full：84.18 / 61.01 / 48.63",
            "结论：并不是所有合理方法都能提升 task3",
        ],
    },
    {
        "title": "Replay 相关 full 结果",
        "body": [
            "lwf015_replaymix2_adapter16_mem42_oldweight2_confirm：85.73 / 58.36 / 42.95",
            "hard replay / MIR-lite full：84.18 / 60.75 / 49.36",
            "alignment-memory replay、DER-lite、kmeans herding 主要停留在 short，没形成正式突破",
            "结论：replay 仍重要，但当前最强提升来自 replay 与决策/原型校正的联合",
        ],
    },
    {
        "title": "当前结论与后续方向",
        "body": [
            "当前 best full：84.18 / 60.75 / 49.36",
            "最关键方法：no contrastive、normalized NME、hybrid、new_only ptblend、affine2",
            "当前平台：task3 已逼近 49~50，单纯 replay 小改动难再明显突破",
            "后续方向：更严格分析 task-conditioned 参数、prototype/refinement 更强联动、subject-invariant 约束",
        ],
    },
]


EMU_PER_INCH = 914400
SLIDE_W = 13.333 * EMU_PER_INCH
SLIDE_H = 7.5 * EMU_PER_INCH


def xml_header():
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'


def rels_xml(entries):
    body = "".join(
        f'<Relationship Id="{rid}" Type="{rtype}" Target="{target}"/>'
        for rid, rtype, target in entries
    )
    return (
        xml_header()
        + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + body
        + "</Relationships>"
    )


def content_types_xml(slide_count: int):
    overrides = [
        ("/ppt/presentation.xml", "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"),
        ("/ppt/presProps.xml", "application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"),
        ("/ppt/viewProps.xml", "application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"),
        ("/ppt/theme/theme1.xml", "application/vnd.openxmlformats-officedocument.theme+xml"),
        ("/ppt/tableStyles.xml", "application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml"),
        ("/ppt/slideMasters/slideMaster1.xml", "application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"),
        ("/ppt/slideLayouts/slideLayout1.xml", "application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"),
        ("/docProps/core.xml", "application/vnd.openxmlformats-package.core-properties+xml"),
        ("/docProps/app.xml", "application/vnd.openxmlformats-officedocument.extended-properties+xml"),
    ]
    overrides += [
        (f"/ppt/slides/slide{i}.xml", "application/vnd.openxmlformats-officedocument.presentationml.slide+xml")
        for i in range(1, slide_count + 1)
    ]
    body = [
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
    ]
    body += [f'<Override PartName="{part}" ContentType="{ctype}"/>' for part, ctype in overrides]
    return (
        xml_header()
        + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        + "".join(body)
        + "</Types>"
    )


def app_xml(slide_count: int):
    return xml_header() + f"""
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
  <PresentationFormat>On-screen Show (16:9)</PresentationFormat>
  <Slides>{slide_count}</Slides>
  <Notes>0</Notes>
  <HiddenSlides>0</HiddenSlides>
  <MMClips>0</MMClips>
  <ScaleCrop>false</ScaleCrop>
  <HeadingPairs>
    <vt:vector size="2" baseType="variant">
      <vt:variant><vt:lpstr>Slides</vt:lpstr></vt:variant>
      <vt:variant><vt:i4>{slide_count}</vt:i4></vt:variant>
    </vt:vector>
  </HeadingPairs>
  <TitlesOfParts>
    <vt:vector size="{slide_count}" baseType="lpstr">
      {''.join('<vt:lpstr>Slide</vt:lpstr>' for _ in range(slide_count))}
    </vt:vector>
  </TitlesOfParts>
  <Company>OpenAI Codex</Company>
  <LinksUpToDate>false</LinksUpToDate>
  <SharedDoc>false</SharedDoc>
  <HyperlinksChanged>false</HyperlinksChanged>
  <AppVersion>16.0000</AppVersion>
</Properties>
"""


def core_xml():
    now = _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return xml_header() + f"""
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>iCaRL_codex 实验递进整理</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
"""


def presentation_xml(slide_count: int):
    sld_ids = "".join(
        f'<p:sldId id="{256+i}" r:id="rId{6+i}"/>' for i in range(slide_count)
    )
    return xml_header() + f"""
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst>
    <p:sldMasterId id="2147483648" r:id="rId1"/>
  </p:sldMasterIdLst>
  <p:sldIdLst>
    {sld_ids}
  </p:sldIdLst>
  <p:sldSz cx="{int(SLIDE_W)}" cy="{int(SLIDE_H)}"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>
"""


def presentation_rels_xml(slide_count: int):
    entries = [
        ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster", "slideMasters/slideMaster1.xml"),
        ("rId2", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/presProps", "presProps.xml"),
        ("rId3", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/viewProps", "viewProps.xml"),
        ("rId4", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme", "theme/theme1.xml"),
        ("rId5", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/tableStyles", "tableStyles.xml"),
    ]
    entries += [
        (f"rId{6+i}", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide", f"slides/slide{i+1}.xml")
        for i in range(slide_count)
    ]
    return rels_xml(entries)


def pres_props_xml():
    return xml_header() + """
<p:presentationPr xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>
"""


def view_props_xml():
    return xml_header() + """
<p:viewPr xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:normalViewPr/>
  <p:slideViewPr/>
  <p:notesTextViewPr/>
  <p:gridSpacing cx="780288" cy="780288"/>
</p:viewPr>
"""


def table_styles_xml():
    return xml_header() + """
<a:tblStyleLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" def="{00000000-0000-0000-0000-000000000000}"/>
"""


def theme_xml():
    return xml_header() + """
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Simple Theme">
  <a:themeElements>
    <a:clrScheme name="Simple">
      <a:dk1><a:srgbClr val="1E2A38"/></a:dk1>
      <a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="243447"/></a:dk2>
      <a:lt2><a:srgbClr val="F7F9FC"/></a:lt2>
      <a:accent1><a:srgbClr val="2F6BFF"/></a:accent1>
      <a:accent2><a:srgbClr val="1FA187"/></a:accent2>
      <a:accent3><a:srgbClr val="F18F01"/></a:accent3>
      <a:accent4><a:srgbClr val="D7263D"/></a:accent4>
      <a:accent5><a:srgbClr val="6C5CE7"/></a:accent5>
      <a:accent6><a:srgbClr val="00A8E8"/></a:accent6>
      <a:hlink><a:srgbClr val="0563C1"/></a:hlink>
      <a:folHlink><a:srgbClr val="954F72"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="Simple">
      <a:majorFont><a:latin typeface="Arial"/></a:majorFont>
      <a:minorFont><a:latin typeface="Arial"/></a:minorFont>
    </a:fontScheme>
    <a:fmtScheme name="Simple">
      <a:fillStyleLst>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
      </a:fillStyleLst>
      <a:lnStyleLst>
        <a:ln w="9525" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>
      </a:lnStyleLst>
      <a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>
      <a:bgFillStyleLst><a:solidFill><a:schemeClr val="lt1"/></a:solidFill></a:bgFillStyleLst>
    </a:fmtScheme>
  </a:themeElements>
</a:theme>
"""


def slide_master_xml():
    return xml_header() + """
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld name="Master">
    <p:bg>
      <p:bgPr>
        <a:solidFill><a:srgbClr val="F7F9FC"/></a:solidFill>
      </p:bgPr>
    </p:bg>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" hlink="hlink" tx1="dk1" tx2="dk2"/>
  <p:sldLayoutIdLst>
    <p:sldLayoutId id="1" r:id="rId1"/>
  </p:sldLayoutIdLst>
  <p:txStyles/>
</p:sldMaster>
"""


def slide_master_rels_xml():
    return rels_xml([
        ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout", "../slideLayouts/slideLayout1.xml"),
        ("rId2", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme", "../theme/theme1.xml"),
    ])


def slide_layout_xml():
    return xml_header() + """
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank">
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>
"""


def slide_layout_rels_xml():
    return rels_xml([
        ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster", "../slideMasters/slideMaster1.xml"),
    ])


def text_paragraph(text: str, level: int = 0, font_size: int = 2200, bold: bool = False):
    text = escape(text)
    b = ' b="1"' if bold else ""
    return (
        f'<a:p><a:pPr lvl="{level}"/>'
        f'<a:r><a:rPr lang="zh-CN" sz="{font_size}"{b}/><a:t>{text}</a:t></a:r>'
        f'<a:endParaRPr lang="zh-CN" sz="{font_size}"/></a:p>'
    )


def textbox(shape_id: int, name: str, x: int, y: int, cx: int, cy: int, paragraphs: list[str], fill: str | None = None):
    sppr_fill = f"<a:solidFill><a:srgbClr val=\"{fill}\"/></a:solidFill>" if fill else "<a:noFill/>"
    tx = "".join(paragraphs)
    return f"""
<p:sp>
  <p:nvSpPr>
    <p:cNvPr id="{shape_id}" name="{escape(name)}"/>
    <p:cNvSpPr txBox="1"/>
    <p:nvPr/>
  </p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
    {sppr_fill}
  </p:spPr>
  <p:txBody>
    <a:bodyPr wrap="square" rtlCol="0"/>
    <a:lstStyle/>
    {tx}
  </p:txBody>
</p:sp>
"""


def slide_xml(slide: dict, idx: int):
    title = slide["title"]
    subtitle = slide.get("subtitle")
    bullets = slide.get("body", [])

    shapes = []
    # title
    shapes.append(
        textbox(
            2, f"Title {idx}",
            int(0.6 * EMU_PER_INCH), int(0.4 * EMU_PER_INCH),
            int(12.0 * EMU_PER_INCH), int(0.8 * EMU_PER_INCH),
            [text_paragraph(title, font_size=2800, bold=True)],
        )
    )
    # divider
    shapes.append(f"""
<p:sp>
  <p:nvSpPr><p:cNvPr id="3" name="Divider {idx}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="{int(0.6 * EMU_PER_INCH)}" y="{int(1.25 * EMU_PER_INCH)}"/><a:ext cx="{int(12.0 * EMU_PER_INCH)}" cy="{int(0.04 * EMU_PER_INCH)}"/></a:xfrm>
    <a:solidFill><a:srgbClr val="2F6BFF"/></a:solidFill>
  </p:spPr>
  <p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody>
</p:sp>
""")
    next_id = 4
    if subtitle:
        shapes.append(
            textbox(
                next_id, f"Subtitle {idx}",
                int(0.7 * EMU_PER_INCH), int(1.45 * EMU_PER_INCH),
                int(11.6 * EMU_PER_INCH), int(1.0 * EMU_PER_INCH),
                [text_paragraph(line, font_size=1800) for line in subtitle.split("\n")],
            )
        )
        y_start = 2.35
        next_id += 1
    else:
        y_start = 1.55
    para = [text_paragraph("• " + b, font_size=2000) for b in bullets]
    shapes.append(
        textbox(
            next_id, f"Body {idx}",
            int(0.8 * EMU_PER_INCH), int(y_start * EMU_PER_INCH),
            int(11.7 * EMU_PER_INCH), int(4.9 * EMU_PER_INCH),
            para,
        )
    )
    shape_xml = "".join(shapes)
    return xml_header() + f"""
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg>
      <p:bgPr><a:solidFill><a:srgbClr val="F7F9FC"/></a:solidFill></p:bgPr>
    </p:bg>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
      {shape_xml}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>
"""


def slide_rels_xml():
    return rels_xml([
        ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout", "../slideLayouts/slideLayout1.xml"),
    ])


def build_pptx(out_path: str):
    slide_count = len(SLIDES)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml(slide_count))
        zf.writestr("_rels/.rels", rels_xml([
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument", "ppt/presentation.xml"),
            ("rId2", "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties", "docProps/core.xml"),
            ("rId3", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties", "docProps/app.xml"),
        ]))
        zf.writestr("docProps/app.xml", app_xml(slide_count))
        zf.writestr("docProps/core.xml", core_xml())
        zf.writestr("ppt/presentation.xml", presentation_xml(slide_count))
        zf.writestr("ppt/_rels/presentation.xml.rels", presentation_rels_xml(slide_count))
        zf.writestr("ppt/presProps.xml", pres_props_xml())
        zf.writestr("ppt/viewProps.xml", view_props_xml())
        zf.writestr("ppt/tableStyles.xml", table_styles_xml())
        zf.writestr("ppt/theme/theme1.xml", theme_xml())
        zf.writestr("ppt/slideMasters/slideMaster1.xml", slide_master_xml())
        zf.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", slide_master_rels_xml())
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout_xml())
        zf.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", slide_layout_rels_xml())
        for i, slide in enumerate(SLIDES, start=1):
            zf.writestr(f"ppt/slides/slide{i}.xml", slide_xml(slide, i))
            zf.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", slide_rels_xml())


if __name__ == "__main__":
    build_pptx(OUT_PATH)
    print(OUT_PATH)
