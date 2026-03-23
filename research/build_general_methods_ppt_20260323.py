from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt


OUT = Path("/data1/bochen/cbcontinual/iCaRL_codex/research/general_methods_chain_20260323_zh.pptx")


def add_slide(prs, title, bullets, subtitle=None):
    layout = prs.slide_layouts[1] if subtitle else prs.slide_layouts[5]
    slide = prs.slides.add_slide(layout)
    slide.shapes.title.text = title

    if subtitle:
        tf = slide.placeholders[1].text_frame
        tf.clear()
        p = tf.paragraphs[0]
        p.text = subtitle
        p.font.size = Pt(18)
        for item in bullets:
            para = tf.add_paragraph()
            para.text = item
            para.level = 0
            para.font.size = Pt(22)
        return

    box = slide.shapes.add_textbox(Inches(0.8), Inches(1.35), Inches(11.6), Inches(5.5))
    tf = box.text_frame
    tf.word_wrap = True
    first = True
    for item in bullets:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.text = item
        p.level = 0
        p.font.size = Pt(22)


prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

slides = [
    (
        "通用连续学习方法链复验",
        [
            "目标：只保留适用于所有 task 的通用 continual 方法",
            "统一标准：stage_epochs = 10,12,12；screen = 1 seed",
            "从最原始 iCaRL（去掉 contrastive）开始逐步加方法",
            "不使用 stage-specific prototype 修补或 asymmetric BCE",
        ],
        "固定预算口径下的通用 pipeline",
    ),
    (
        "为什么要单独做这条链",
        [
            "之前更强的主线包含 stage2/stage3 条件性设计，虽然有效，但不够通用。",
            "这次的目标不是追最高分，而是确认哪些组件能被写成“对所有 task 都适用”的方法。",
            "因此本轮更关注方法的普适性和可解释性，而不是最终最优分数。",
        ],
        None,
    ),
    (
        "方法链步骤",
        [
            "gen00：原始 iCaRL（去掉 contrastive）",
            "gen01：+ task adapter dim=16，从 stage1 开始",
            "gen02：+ stage_epochs = 10,12,12",
            "gen03：+ oldweight = 3.125",
            "gen04：+ normalized NME",
            "gen05：+ LwF",
            "gen06：+ task affine，从 stage1 开始",
            "gen07：+ hybrid NME + logits",
        ],
        None,
    ),
    (
        "结果总览",
        [
            "gen00: 84.72 / 56.48 / 43.29",
            "gen01: 85.42 / 57.64 / 39.89",
            "gen02: 85.42 / 57.79 / 39.20",
            "gen03: 85.42 / 53.70 / 39.93",
            "gen04: 85.42 / 54.86 / 45.22",
            "gen05: 85.42 / 54.63 / 46.76",
            "gen06: 84.49 / 55.86 / 43.09",
            "gen07: 84.49 / 56.40 / 43.33",
        ],
        None,
    ),
    (
        "前半段结论",
        [
            "adapter16 from stage1 不是通用增益项，task3 明显变差。",
            "单独把训练预算改成 10,12,12 不能修复这个问题。",
            "全局 oldweight=3.125 也不适合作为通用组件，task2/task3 一起受损。",
        ],
        None,
    ),
    (
        "为什么 normNME 成立",
        [
            "gen04 把 task3 从 39.93 拉回到 45.22。",
            "它不是阶段性 trick，而是统一修 prototype classifier 的几何。",
            "因此它最适合被保留为“通用 continual classifier improvement”。",
        ],
        None,
    ),
    (
        "为什么 LwF 仍然值得保留",
        [
            "gen05 把 task3 进一步从 45.22 拉到 46.76。",
            "虽然 task2 没继续涨，但总体 score 仍然提高。",
            "说明 LwF 在这条严格通用链里依旧是有效的旧知识保持机制。",
        ],
        None,
    ),
    (
        "为什么 affine 和 hybrid 没成立",
        [
            "gen06 affineall: 84.49 / 55.86 / 43.09",
            "gen07 hybridall: 84.49 / 56.40 / 43.33",
            "这说明它们不能机械地从 stage1/早期统一启用。",
            "更准确的定位应当是：条件性有效，而不是通用主线组件。",
        ],
        None,
    ),
    (
        "通用链最终保留什么",
        [
            "去掉 contrastive",
            "normalized NME",
            "LwF",
            "这三项是本轮最稳、最适合写成通用 continual 方法的核心。",
        ],
        None,
    ),
    (
        "和当前更强主线的关系",
        [
            "更强 fixed-budget 主线：s2ace00_ptblendnew02_stage101212_confirm",
            "结果：84.18 / 60.70 / 49.40",
            "它更强，但包含明显更具阶段性的设计。",
            "因此论文里可以分层叙述：先讲通用方法链，再讲场景增强版主线。",
        ],
        None,
    ),
    (
        "推荐写法",
        [
            "基础 pipeline：no contrastive -> normNME -> LwF",
            "说明 adapter/affine/hybrid 在“全任务统一启用”的严格标准下没有成立。",
            "增强版方法再单独介绍：为什么需要 stage2-focused 机制。",
            "这样方法故事会更清晰，也更符合你的论文目标。",
        ],
        None,
    ),
]

for i, (title, bullets, subtitle) in enumerate(slides):
    add_slide(prs, title, bullets, subtitle)

prs.save(OUT)
print(str(OUT))
