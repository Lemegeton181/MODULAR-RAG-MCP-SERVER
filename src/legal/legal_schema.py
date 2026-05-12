"""Base legal field registry and evidence-keyword catalog.

This module is the *schema half* of the dynamic field system. It is a
pure-data module — no I/O, no regex execution at import time, no LLM.
Three dictionaries form the contract that the rest of the legal package
consumes:

* :data:`FIELD_DEFS` — structured fields where a single canonical value
  is meaningful (case_no, applicant, court, amounts, dates, ...). Each
  entry tells the field extractor *how* to mine values, and tells the
  query understanding layer *which* user phrasings hit this field.
* :data:`EVIDENCE_KEYWORDS` — non-structured terms that signal evidence
  presence (转账凭证 / 聊天记录 / 工资 ...). These do NOT have a canonical
  value; matches turn the query into ``evidence_search`` and feed
  expanded terms into hybrid retrieval.
* :data:`PARTY_ROLE_FIELDS` — the subset of FIELD_DEFS that count as
  「当事人」上位概念。 Asking 「当事人是谁」 expands to all of them.

Design notes (kept short on purpose):
- 「当事人」 is treated as an umbrella that *expands* into the role
  fields, not as a synonym for any single role.
- Extract patterns are deliberately conservative. We prefer to miss a
  value than to invent a wrong one — see ``_FIELD_VALUE_TAIL`` in
  :mod:`src.legal.field_extractor`.
- ``query_triggers`` are matched against the user query as substrings
  (no LLM). They live next to the field they belong to so that adding a
  new field updates query understanding automatically.
"""
from __future__ import annotations

from typing import Dict, List, TypedDict

CATEGORY_DOC_BASIC = "doc_basic"
CATEGORY_INSTITUTION = "institution"
CATEGORY_PARTY = "party"
CATEGORY_IDENTITY = "identity"
CATEGORY_REQUEST = "request"
CATEGORY_FACT = "fact"
CATEGORY_EVIDENCE = "evidence"
CATEGORY_AMOUNT = "amount"
CATEGORY_DATE = "date"
CATEGORY_LAW = "law"

ALL_CATEGORIES = [
    CATEGORY_DOC_BASIC,
    CATEGORY_INSTITUTION,
    CATEGORY_PARTY,
    CATEGORY_IDENTITY,
    CATEGORY_REQUEST,
    CATEGORY_FACT,
    CATEGORY_EVIDENCE,
    CATEGORY_AMOUNT,
    CATEGORY_DATE,
    CATEGORY_LAW,
]


class FieldDef(TypedDict, total=False):
    canonical_name: str
    category: str
    aliases: List[str]
    query_triggers: List[str]
    extract_patterns: List[str]
    is_extractable: bool


# ---------------------------------------------------------------------------
# Structured field definitions
# ---------------------------------------------------------------------------

# Conservative regex anchors. Most use a "label : value" shape with the
# value captured by a named group. The field_extractor module compiles
# these once and caps the captured value length to avoid swallowing
# whole paragraphs when the punctuation is missing.

_CASE_NO_PATTERNS = [
    # 哈劳人仲字（2025）1077号 / （2025）京01民初123号 / [2024]xxx号
    r"(?P<value>[一-鿿]{0,8}[字号]?\s*[（(]\s*\d{4}\s*[)）]\s*[一-鿿0-9A-Za-z]+\s*\d+\s*号)",
    r"(?P<value>[一-鿿]{1,12}\s*[（(]\s*\d{4}\s*[)）]\s*[一-鿿0-9A-Za-z]*\s*\d+\s*号)",
]

_DOC_NUMBER_LABELED = [
    r"(?:文书编号|案号|裁决书编号|判决书编号|调解书编号|仲裁文书编号)\s*[:：]\s*(?P<value>\S{3,80})",
]

_AMOUNT_TAIL = r"(?:人民币\s*)?[\d,，]+(?:\.\d+)?\s*(?:元|万元|万|圆)"

# field_id -> definition
FIELD_DEFS: Dict[str, FieldDef] = {
    # --- 文书基础信息 -----------------------------------------------------
    "case_no": {
        "canonical_name": "案号",
        "category": CATEGORY_DOC_BASIC,
        "aliases": ["案号", "文书编号", "案件编号", "裁决书编号", "判决书编号", "调解书编号", "仲裁文书编号"],
        "query_triggers": ["案号", "文书编号", "案件编号", "裁决书编号", "判决书编号", "调解书编号", "仲裁文书编号", "编号", "案件号"],
        "extract_patterns": _DOC_NUMBER_LABELED + _CASE_NO_PATTERNS,
        "is_extractable": True,
    },
    "doc_type": {
        "canonical_name": "文书类型",
        "category": CATEGORY_DOC_BASIC,
        "aliases": ["文书类型", "文书种类", "案件类型"],
        "query_triggers": ["文书类型", "文书种类", "案件类型", "什么文书", "哪种文书"],
        # 抽取从首页 / 标题命中：仲裁裁决书 / 民事判决书 / 调解书 / 起诉状 / 答辩状 / 仲裁申请书
        "extract_patterns": [
            r"(?P<value>(?:民事|刑事|行政)?(?:仲裁裁决书|裁决书|民事判决书|判决书|调解书|起诉状|答辩状|仲裁申请书|劳动仲裁裁决书|执行裁定书|裁定书))",
        ],
        "is_extractable": True,
    },
    # --- 机构信息 ---------------------------------------------------------
    "court": {
        "canonical_name": "法院",
        "category": CATEGORY_INSTITUTION,
        "aliases": ["法院", "审理法院", "受理法院", "管辖法院"],
        "query_triggers": ["法院", "哪个法院", "审理法院", "受理法院", "管辖法院"],
        "extract_patterns": [
            r"(?P<value>[一-鿿]{2,20}人民法院)",
        ],
        "is_extractable": True,
    },
    "arbitration_committee": {
        "canonical_name": "仲裁委员会",
        "category": CATEGORY_INSTITUTION,
        "aliases": ["仲裁委员会", "仲裁委", "受理机构", "审理机构"],
        "query_triggers": ["仲裁委员会", "仲裁委", "受理机构", "审理机构", "哪个仲裁", "受理单位"],
        "extract_patterns": [
            r"(?P<value>[一-鿿]{2,20}劳动(?:人事)?争议仲裁委员会)",
            r"(?P<value>[一-鿿]{2,20}仲裁委员会)",
        ],
        "is_extractable": True,
    },
    "procuratorate": {
        "canonical_name": "检察院",
        "category": CATEGORY_INSTITUTION,
        "aliases": ["检察院", "人民检察院"],
        "query_triggers": ["检察院", "人民检察院"],
        "extract_patterns": [
            r"(?P<value>[一-鿿]{2,20}人民检察院)",
        ],
        "is_extractable": True,
    },
    "police_authority": {
        "canonical_name": "公安机关",
        "category": CATEGORY_INSTITUTION,
        "aliases": ["公安机关", "公安局"],
        "query_triggers": ["公安机关", "公安局"],
        "extract_patterns": [
            r"(?P<value>[一-鿿]{2,20}公安(?:分)?局)",
        ],
        "is_extractable": True,
    },
    # --- 当事人 / 诉讼参与人 ----------------------------------------------
    "applicant": {
        "canonical_name": "申请人",
        "category": CATEGORY_PARTY,
        "aliases": ["申请人"],
        "query_triggers": ["申请人", "申请方"],
        "extract_patterns": [r"(?<![一-鿿])申请人\s*[:：]\s*(?P<value>\S[^\n]{0,80})"],
        "is_extractable": True,
    },
    "respondent": {
        "canonical_name": "被申请人",
        "category": CATEGORY_PARTY,
        "aliases": ["被申请人"],
        "query_triggers": ["被申请人", "被申请方"],
        "extract_patterns": [r"被申请人\s*[:：]\s*(?P<value>\S[^\n]{0,80})"],
        "is_extractable": True,
    },
    "plaintiff": {
        "canonical_name": "原告",
        "category": CATEGORY_PARTY,
        "aliases": ["原告"],
        "query_triggers": ["原告"],
        "extract_patterns": [r"(?<![一-鿿])原告\s*[:：]\s*(?P<value>\S[^\n]{0,80})"],
        "is_extractable": True,
    },
    "defendant": {
        "canonical_name": "被告",
        "category": CATEGORY_PARTY,
        "aliases": ["被告"],
        "query_triggers": ["被告"],
        "extract_patterns": [r"(?<![一-鿿])被告\s*[:：]\s*(?P<value>\S[^\n]{0,80})"],
        "is_extractable": True,
    },
    "third_party": {
        "canonical_name": "第三人",
        "category": CATEGORY_PARTY,
        "aliases": ["第三人"],
        "query_triggers": ["第三人"],
        "extract_patterns": [r"(?<![一-鿿])第三人\s*[:：]\s*(?P<value>\S[^\n]{0,80})"],
        "is_extractable": True,
    },
    "appellant": {
        "canonical_name": "上诉人",
        "category": CATEGORY_PARTY,
        "aliases": ["上诉人"],
        "query_triggers": ["上诉人"],
        "extract_patterns": [r"(?<![一-鿿])上诉人\s*[:：]\s*(?P<value>\S[^\n]{0,80})"],
        "is_extractable": True,
    },
    "appellee": {
        "canonical_name": "被上诉人",
        "category": CATEGORY_PARTY,
        "aliases": ["被上诉人"],
        "query_triggers": ["被上诉人"],
        "extract_patterns": [r"被上诉人\s*[:：]\s*(?P<value>\S[^\n]{0,80})"],
        "is_extractable": True,
    },
    "legal_representative": {
        "canonical_name": "法定代表人",
        "category": CATEGORY_PARTY,
        "aliases": ["法定代表人", "法人代表"],
        "query_triggers": ["法定代表人", "法人代表"],
        "extract_patterns": [r"法定代表人\s*[:：]\s*(?P<value>\S[^\n]{0,40})"],
        "is_extractable": True,
    },
    "agent_attorney": {
        "canonical_name": "委托代理人",
        "category": CATEGORY_PARTY,
        "aliases": ["委托代理人", "代理律师", "代理人"],
        "query_triggers": ["委托代理人", "代理律师", "代理人"],
        "extract_patterns": [r"委托代理人\s*[:：]\s*(?P<value>\S[^\n]{0,80})"],
        "is_extractable": True,
    },
    "defender": {
        "canonical_name": "辩护人",
        "category": CATEGORY_PARTY,
        "aliases": ["辩护人"],
        "query_triggers": ["辩护人"],
        "extract_patterns": [r"辩护人\s*[:：]\s*(?P<value>\S[^\n]{0,80})"],
        "is_extractable": True,
    },
    "witness": {
        "canonical_name": "证人",
        "category": CATEGORY_PARTY,
        "aliases": ["证人"],
        "query_triggers": ["证人", "证人是谁", "有没有证人"],
        "extract_patterns": [r"证人\s*[:：]\s*(?P<value>\S[^\n]{0,40})"],
        "is_extractable": True,
    },
    # --- 主体身份信息 -----------------------------------------------------
    "company_name": {
        "canonical_name": "公司名称",
        "category": CATEGORY_IDENTITY,
        "aliases": ["公司名称", "单位名称"],
        "query_triggers": ["公司名称", "单位名称", "什么公司"],
        "extract_patterns": [
            r"(?P<value>[一-鿿]{2,30}(?:有限公司|股份有限公司|有限责任公司|集团公司|分公司|事务所))",
        ],
        "is_extractable": True,
    },
    "uscc": {
        "canonical_name": "统一社会信用代码",
        "category": CATEGORY_IDENTITY,
        "aliases": ["统一社会信用代码", "信用代码", "USCC"],
        "query_triggers": ["统一社会信用代码", "信用代码", "USCC"],
        "extract_patterns": [
            r"(?:统一社会信用代码|信用代码)\s*[:：]?\s*(?P<value>[0-9A-HJ-NPQRTUWXY]{18})",
        ],
        "is_extractable": True,
    },
    "id_card": {
        "canonical_name": "身份证号",
        "category": CATEGORY_IDENTITY,
        "aliases": ["身份证号", "身份证号码", "公民身份号码"],
        "query_triggers": ["身份证号", "身份证号码", "公民身份号码"],
        "extract_patterns": [
            r"(?:身份证号(?:码)?|公民身份号码)\s*[:：]?\s*(?P<value>\d{17}[\dXx])",
        ],
        "is_extractable": True,
    },
    "address": {
        "canonical_name": "住所地",
        "category": CATEGORY_IDENTITY,
        "aliases": ["住所地", "地址", "住址", "经常居住地", "户籍地"],
        "query_triggers": ["住所地", "地址", "住址", "经常居住地", "户籍地"],
        "extract_patterns": [
            r"(?:住所地|住址|地址|经常居住地|户籍地)\s*[:：]\s*(?P<value>\S[^\n]{4,80})",
        ],
        "is_extractable": True,
    },
    # --- 请求事项 ---------------------------------------------------------
    "claim": {
        "canonical_name": "申请事项",
        "category": CATEGORY_REQUEST,
        "aliases": ["申请事项", "请求事项", "诉讼请求", "仲裁请求", "上诉请求", "执行请求"],
        "query_triggers": ["申请事项", "请求事项", "诉讼请求", "仲裁请求", "上诉请求", "执行请求", "请求", "诉求"],
        "extract_patterns": [
            r"(?:申请事项|请求事项|诉讼请求|仲裁请求|上诉请求|执行请求)\s*[:：]?\s*(?P<value>\S[^\n]{4,200})",
        ],
        "is_extractable": True,
    },
    # --- 金额（含二倍工资专项） -------------------------------------------
    "double_salary_amount": {
        "canonical_name": "二倍工资金额",
        "category": CATEGORY_AMOUNT,
        "aliases": ["二倍工资", "双倍工资", "二倍工资差额"],
        "query_triggers": ["二倍工资", "双倍工资", "二倍工资差额", "二倍工资金额"],
        "extract_patterns": [
            rf"(?:二倍工资|双倍工资|二倍工资差额)\s*(?:金额|差额)?\s*[:：]?\s*(?P<value>{_AMOUNT_TAIL})",
        ],
        "is_extractable": True,
    },
    "compensation_amount": {
        "canonical_name": "赔偿金额",
        "category": CATEGORY_AMOUNT,
        "aliases": ["赔偿金额", "经济补偿金", "惩罚性赔偿", "赔偿金"],
        "query_triggers": ["赔偿金额", "赔偿金", "经济补偿金", "惩罚性赔偿", "赔多少", "赔偿多少"],
        "extract_patterns": [
            rf"(?:赔偿金额|赔偿金|经济补偿金|惩罚性赔偿)\s*[:：]?\s*(?P<value>{_AMOUNT_TAIL})",
        ],
        "is_extractable": True,
    },
    "loan_amount": {
        "canonical_name": "借款金额",
        "category": CATEGORY_AMOUNT,
        "aliases": ["借款金额", "本金"],
        "query_triggers": ["借款金额", "借了多少", "本金"],
        "extract_patterns": [
            rf"(?:借款金额|借款本金|本金)\s*[:：]?\s*(?P<value>{_AMOUNT_TAIL})",
        ],
        "is_extractable": True,
    },
    "case_amount": {
        "canonical_name": "涉案金额",
        "category": CATEGORY_AMOUNT,
        "aliases": ["涉案金额", "案件标的"],
        "query_triggers": ["涉案金额", "案件标的"],
        "extract_patterns": [
            rf"(?:涉案金额|案件标的额?)\s*[:：]?\s*(?P<value>{_AMOUNT_TAIL})",
        ],
        "is_extractable": True,
    },
    # --- 日期 -------------------------------------------------------------
    "filing_date": {
        "canonical_name": "立案日期",
        "category": CATEGORY_DATE,
        "aliases": ["立案日期", "受理日期"],
        "query_triggers": ["立案日期", "受理日期", "什么时候立案"],
        "extract_patterns": [
            r"(?:立案日期|受理日期)\s*[:：]?\s*(?P<value>\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日)",
        ],
        "is_extractable": True,
    },
    "ruling_date": {
        "canonical_name": "裁决日期",
        "category": CATEGORY_DATE,
        "aliases": ["裁决日期", "判决日期"],
        "query_triggers": ["裁决日期", "判决日期", "什么时候判决", "什么时候裁决"],
        "extract_patterns": [
            r"(?:裁决日期|判决日期)\s*[:：]?\s*(?P<value>\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日)",
        ],
        "is_extractable": True,
    },
    "contract_date": {
        "canonical_name": "合同签订日期",
        "category": CATEGORY_DATE,
        "aliases": ["合同签订日期", "签订日期"],
        "query_triggers": ["合同签订日期", "签订日期", "什么时候签的合同"],
        "extract_patterns": [
            r"(?:合同签订日期|签订日期)\s*[:：]?\s*(?P<value>\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日)",
        ],
        "is_extractable": True,
    },
    # --- 法律依据和结果 ---------------------------------------------------
    "legal_basis": {
        "canonical_name": "法律依据",
        "category": CATEGORY_LAW,
        "aliases": ["法律依据", "法条", "适用法律"],
        "query_triggers": ["法律依据", "法条", "适用法律", "依据什么法律", "什么法条"],
        "extract_patterns": [
            r"(?:法律依据|适用法律|依据)\s*[:：]\s*(?P<value>\S[^\n]{4,120})",
        ],
        "is_extractable": True,
    },
    "ruling_result": {
        "canonical_name": "裁决结果",
        "category": CATEGORY_LAW,
        "aliases": ["裁决结果", "判决结果", "判决主文"],
        "query_triggers": ["裁决结果", "判决结果", "判决主文", "怎么判的", "怎么裁的"],
        "extract_patterns": [
            r"(?:裁决结果|判决结果|判决主文)\s*[:：]?\s*(?P<value>\S[^\n]{4,200})",
        ],
        "is_extractable": True,
    },
    # --- 事实争议（论述型，不强抽 value，但参与 query_triggers）---------
    "labor_relation": {
        "canonical_name": "劳动关系",
        "category": CATEGORY_FACT,
        "aliases": ["劳动关系"],
        "query_triggers": ["劳动关系", "是否存在劳动关系", "劳动关系认定"],
        "extract_patterns": [],
        "is_extractable": False,
    },
    "employer_liability": {
        "canonical_name": "用工主体责任",
        "category": CATEGORY_FACT,
        "aliases": ["用工主体责任"],
        "query_triggers": ["用工主体责任", "用工主体"],
        "extract_patterns": [],
        "is_extractable": False,
    },
    "law_application_error": {
        "canonical_name": "适用法律错误",
        "category": CATEGORY_FACT,
        "aliases": ["适用法律错误"],
        "query_triggers": ["适用法律错误", "适用法律是否错误", "法律适用错误"],
        "extract_patterns": [],
        "is_extractable": False,
    },
    "evidence_insufficient": {
        "canonical_name": "证据不足",
        "category": CATEGORY_FACT,
        "aliases": ["证据不足"],
        "query_triggers": ["证据不足", "举证不能"],
        "extract_patterns": [],
        "is_extractable": False,
    },
}


# Subset of FIELD_DEFS that count as 「当事人」 — asking 「当事人是谁」 expands
# into all of these.
PARTY_ROLE_FIELDS: List[str] = [
    "applicant",
    "respondent",
    "plaintiff",
    "defendant",
    "third_party",
    "appellant",
    "appellee",
    "legal_representative",
    "agent_attorney",
    "defender",
    "witness",
]


# ---------------------------------------------------------------------------
# Evidence keywords — non-structured, no canonical value
# ---------------------------------------------------------------------------

# Group -> keywords. Matching ANY of a group's keywords in user query
# turns it into evidence_search and feeds the keywords into hybrid as
# expanded terms. Matching keywords in document text marks page-level
# evidence presence (source_type='keyword').
EVIDENCE_KEYWORDS: Dict[str, List[str]] = {
    "contract": ["合同", "协议", "劳动合同", "购销合同", "服务协议"],
    "iou": ["借条", "欠条", "借据"],
    "receipt": ["收据", "发票"],
    "transfer": ["转账凭证", "转账记录", "银行流水", "汇款凭证", "转账"],
    "chat": ["微信聊天记录", "聊天记录", "短信记录", "微信记录"],
    "av": ["录音", "录像", "录音录像", "视频证据", "音频证据", "照片"],
    "attendance": ["考勤", "考勤记录", "打卡记录"],
    "salary": ["工资", "工资条", "工资表", "工资流水"],
    "social_security": ["社保", "社保记录", "社保缴纳记录", "公积金"],
    "delivery": ["快递单", "签收记录", "送达回证", "邮寄凭证"],
    "testimony": ["证人证言", "证言"],
    "appraisal": ["鉴定意见", "鉴定报告", "勘验笔录"],
}


# Flat alias → (group, normalized) for fast in-text scanning
def evidence_keyword_index() -> Dict[str, str]:
    """Return ``alias -> group`` for every evidence keyword."""
    out: Dict[str, str] = {}
    for group, kws in EVIDENCE_KEYWORDS.items():
        for kw in kws:
            out[kw] = group
    return out


# ---------------------------------------------------------------------------
# 「当事人」 umbrella expansion
# ---------------------------------------------------------------------------

PARTY_UMBRELLA_TRIGGERS = ["当事人", "案件当事人", "诉讼当事人"]


def is_party_umbrella_query(query: str) -> bool:
    """Return True iff the user query asks about parties as an umbrella."""
    q = query or ""
    return any(t in q for t in PARTY_UMBRELLA_TRIGGERS)


# ---------------------------------------------------------------------------
# Convenience getters
# ---------------------------------------------------------------------------

def fields_by_category(category: str) -> List[str]:
    return [fid for fid, d in FIELD_DEFS.items() if d.get("category") == category]


def all_query_triggers() -> Dict[str, List[str]]:
    """Return ``field_id -> query_triggers`` for every field."""
    return {fid: list(d.get("query_triggers", [])) for fid, d in FIELD_DEFS.items()}
