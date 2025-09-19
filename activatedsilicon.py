#!/usr/bin/env python3
"""
Activated Silicon — PyQt5
Dark theme • Lazy Tab Loading • Markdown Zoom/Pan • ASCII Sanitizer
AI Assist (OpenAI direct, no backend) • Review Drawer (Rendered / Diff / Table Changes)
Maturity levels: 0. Placeholder • 1. Immature • 2. Mature • 3. Locked

Toolbar:
- 🧩 New Entry / 🗂️ New Folder / ✏️ Rename / 🗑️ Delete
- 📦 Archive / 📂 Open Location
- 💾 Save (Ctrl+S)
- 🔎＋ / 🔎－ / 🎯 Reset Zoom (focused markdown editor)
- 🤖 AI Assist: Generate / Suggest Edits / Re-analyze / Set Model… / Set API Key…

Notes:
- Model default: gpt-5 (change from AI Assist menu or via env OPENAI_MODEL)
- API key: reads OPENAI_API_KEY (or set in the app for this session)
"""

import sys, shutil, os, datetime, json, re, tempfile, subprocess, platform, hashlib, difflib
from pathlib import Path

from PyQt5.QtCore import Qt, QSortFilterProxyModel, QModelIndex, QThread, pyqtSignal, QSize, QTimer
from PyQt5.QtGui import QKeySequence, QIcon, QPixmap, QPainter, QFont, QTextOption
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileSystemModel, QTreeView, QToolBar, QAction, QFileDialog,
    QInputDialog, QMessageBox, QLabel, QAbstractItemView, QFormLayout,
    QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QSpacerItem, QSizePolicy, QSplitter, QTabWidget, QTextEdit,
    QStyleFactory, QStyledItemDelegate, QScrollBar, QMenu, QComboBox
)

# --- OpenAI SDK (no backend needed) ------------------------------------------
import os
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # guarded at runtime

MODEL_NAME_DEFAULT = os.environ.get("OPENAI_MODEL", "gpt-5")  # latest & greatest

APP_TITLE = "Activated Silicon"
DEFAULT_CATALOG_DIR = Path.cwd() / "activated_silicon"

FIELD_ORDER = [
    ("Title", "Click or tap here to enter text."),
    ("Part Number", "PN-XXX"),
    ("Revision", "A1-00"),
    ("Interface", "I2C / SPI / UART / Analog / Digital"),
    ("Type of Design", "Specialized / Fundamental"),
    ("Level of Novelty", "Inventive / Common"),
    ("Performance Feedback", "Exists / Does Not Exist"),
    ("Verified Performance", "Yes / No"),
]

PIN_HEADERS = ["Pin", "Name", "Description", "Note"]
TEST_HEADERS = ["Test No.", "Name", "Description", "Note"]

SECTION_NAMES = ["Netlist", "Partlist", "Pin Interface", "Tests", "EPSA", "WCCA", "FMEA"]

# New maturity levels & helpers
MATURITY_LEVELS = ["0. Placeholder", "1. Immature", "2. Mature", "3. Locked"]
MATURITY_CODE_TO_NAME = {0: "Placeholder", 1: "Immature", 2: "Mature", 3: "Locked"}
MATURITY_NAME_TO_CODE = {v.lower(): k for k, v in MATURITY_CODE_TO_NAME.items()}

def maturity_label_to_code(label: str) -> int:
    if not label: return 0
    label = label.strip()
    m = re.match(r"^\s*([0-3])", label)
    if m:
        return int(m.group(1))
    # try by name
    return MATURITY_NAME_TO_CODE.get(label.lower(), 0)

def maturity_code_to_label(code: int) -> str:
    code = int(code) if str(code).isdigit() else 0
    return f"{code}. {MATURITY_CODE_TO_NAME.get(code,'Placeholder')}"

def maturity_comment(code: int) -> str:
    return f"<!-- maturity: {code} {MATURITY_CODE_TO_NAME.get(code,'Placeholder')} -->"

def maturity_read_from_text(md: str) -> int | None:
    if not md: return None
    # Prefer numeric: <!-- maturity: 2 -->
    m = re.search(r"<!--\s*maturity:\s*([0-3])\b.*?-->", md, re.I)
    if m:
        return int(m.group(1))
    # Fallback to words
    m = re.search(r"<!--\s*maturity:\s*(placeholder|immature|mature|locked)\s*-->", md, re.I)
    if m:
        return MATURITY_NAME_TO_CODE.get(m.group(1).lower(), 0)
    return None

def today_iso(): return datetime.date.today().isoformat()
def now_stamp(): return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
def folder_meta_path(folder: Path) -> Path: return folder / f"{folder.name}.json"

MD_ROW_RE = re.compile(r'^\|\s*(?P<field>[^|]+?)\s*\|\s*(?P<value>[^|]*?)\s*\|$')

def _divider_for(headers): return "| " + " | ".join("-" * len(h) for h in headers) + " |"

NEW_ENTRY_TEMPLATE = f"""# Circuit Metadata

**Last Updated:** {today_iso()}

| Field                  | Value                     |
| ---------------------- | ------------------------- |
| Title                  | Click or tap here to enter text. |
| Part Number            | PN-XXX                    |
| Revision               | A1-00                     |
| Interface              | I2C / SPI / UART / Analog / Digital |
| Type of Design         | Specialized / Fundamental |
| Level of Novelty       | Inventive / Common        |
| Performance Feedback   | Exists / Does Not Exist   |
| Verified Performance   | Yes / No                  |

## Used On

| PN         | Occurrences |
| ---------- | ----------- |
| (None)     | 0           |

## Netlist
<!-- section: Netlist -->
(paste or type your netlist here — raw markdown; tables/lists render)

## Partlist
<!-- section: Partlist -->
(paste or type your partlist here — raw markdown; tables/lists render)

## Pin Interface

| {" | ".join(PIN_HEADERS)} |
{_divider_for(PIN_HEADERS)}

## Tests

| {" | ".join(TEST_HEADERS)} |
{_divider_for(TEST_HEADERS)}

## EPSA
<!-- section: EPSA -->
{maturity_comment(0)}
(paste EPSA here — include mission profile, environments, parts stress tables, margins, references)

## WCCA
<!-- section: WCCA -->
{maturity_comment(0)}
(paste WCCA here — assumptions, equations, corners, tabulated results, conclusions)

## FMEA
<!-- section: FMEA -->
{maturity_comment(0)}
(paste FMEA here — failure modes/effects/causes, detection controls, severity/occurrence/detection, RPN table)
""".strip() + "\n"

# ---------- Emoji icon ----------
def make_emoji_icon(emoji: str, px: int = 256) -> QIcon:
    pm = QPixmap(px, px); pm.fill(Qt.transparent)
    painter = QPainter(pm)
    try:
        f = QFont("Segoe UI Emoji", int(px * 0.66))
        f.setStyleStrategy(QFont.PreferAntialias); painter.setFont(f)
        painter.drawText(pm.rect(), Qt.AlignCenter, emoji)
    finally:
        painter.end()
    return QIcon(pm)

# ---------- Proxy model ----------
class DescProxyModel(QSortFilterProxyModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs); self._desc_cache = {}
    def filterAcceptsRow(self, source_row, source_parent):
        sm = self.sourceModel(); idx = sm.index(source_row, 0, source_parent)
        if not idx.isValid(): return False
        if sm.isDir(idx): return True
        return sm.fileName(idx).lower().endswith(".md")
    def flags(self, index):
        base = super().flags(index)
        if not index.isValid(): return Qt.ItemIsDropEnabled
        sidx = self.mapToSource(index); sm = self.sourceModel()
        if sm.isDir(sidx): return base | Qt.ItemIsDragEnabled | Qt.ItemIsDropEnabled
        return (base | Qt.ItemIsDragEnabled) & ~Qt.ItemIsDropEnabled
    def supportedDropActions(self): return Qt.MoveAction
    def canDropMimeData(self, data, action, row, column, parent):
        if action != Qt.MoveAction: return False
        sp = self.mapToSource(parent); sm = self.sourceModel()
        return (not sp.isValid()) or sm.isDir(sp)
    def dropMimeData(self, data, action, row, column, parent):
        if action != Qt.MoveAction: return False
        smodel = self.sourceModel(); src_parent = self.mapToSource(parent)
        if src_parent.isValid() and not smodel.isDir(src_parent): src_parent = src_parent.parent()
        return smodel.dropMimeData(data, action, row, column, src_parent)
    def columnCount(self, parent): return max(2, super().columnCount(parent))
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid(): return None
        if index.column() == 0: return super().data(index, role)
        if index.column() == 1 and role in (Qt.DisplayRole, Qt.ToolTipRole):
            sidx = self.mapToSource(index.sibling(index.row(), 0))
            spath = Path(self.sourceModel().filePath(sidx)); key = str(spath)
            cached = self._desc_cache.get(key)
            if cached is None:
                if spath.is_dir(): cached = self._read_folder_title(spath)
                elif spath.is_file() and spath.suffix.lower() == ".md": cached = self._read_title_from_md(spath)
                else: cached = ""
                self._desc_cache[key] = cached
            return cached
        if index.column() >= 2 and role == Qt.DisplayRole: return ""
        return super().data(index, role)
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return ["Name", "Description"][section] if section in (0,1) else super().headerData(section, orientation, role)
        return super().headerData(section, orientation, role)
    def _read_title_from_md(self, path: Path) -> str:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
        for line in text.splitlines():
            s = line.strip()
            if not s.startswith("|"): continue
            parts = [p.strip() for p in s.strip("|").split("|")]
            if len(parts) >= 2 and parts[0].lower() == "title": return parts[1]
        return ""
    def _read_folder_title(self, folder: Path) -> str:
        try:
            meta_p = folder_meta_path(folder)
            if not meta_p.exists(): return ""
            with meta_p.open("r", encoding="utf-8") as f: meta = json.load(f)
            return meta.get("TITLE") or meta.get("title") or meta.get("description", "")
        except Exception:
            return ""
    def refresh_desc(self, path: Path):
        key = str(path); self._desc_cache.pop(key, None)
        sm = self.sourceModel(); sidx = sm.index(str(path))
        if sidx.isValid():
            pidx = self.mapFromSource(sidx)
            if pidx.isValid():
                left = pidx.sibling(pidx.row(), 1)
                self.dataChanged.emit(left, left, [Qt.DisplayRole, Qt.ToolTipRole])

# ---------- Dark titlebar (Windows) ----------
if platform.system() == "Windows":
    import ctypes
    from ctypes import wintypes
    def _set_win_dark_titlebar(hwnd: int, enabled: bool = True):
        try:
            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1 = 19
            attribute = wintypes.DWORD(DWMWA_USE_IMMERSIVE_DARK_MODE)
            pv = ctypes.c_int(1 if enabled else 0)
            dwm = ctypes.WinDLL("dwmapi")
            res = dwm.DwmSetWindowAttribute(wintypes.HWND(hwnd), attribute, ctypes.byref(pv), ctypes.sizeof(pv))
            if res != 0:
                attribute = wintypes.DWORD(DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1)
                dwm.DwmSetWindowAttribute(wintypes.HWND(hwnd), attribute, ctypes.byref(pv), ctypes.sizeof(pv))
        except Exception: pass
    def apply_windows_dark_titlebar(widget):
        try:
            hwnd = int(widget.winId()); _set_win_dark_titlebar(hwnd, True)
        except Exception: pass
else:
    def apply_windows_dark_titlebar(widget): pass

# ---------- Slim inline editor ----------
class SlimLineEditDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent); editor.setFrame(False)
        editor.setStyleSheet(
            "QLineEdit { background-color:#2A2D31; color:#E6E6E6; border:1px solid #3A3F44; padding:2px 4px; }"
        )
        return editor

# ---------- Zoomable Markdown Editor ----------
class ZoomableMarkdownEdit(QTextEdit):
    def __init__(self, parent=None, placeholder: str = ""):
        super().__init__(parent)
        self._supports_markdown = hasattr(self, "setMarkdown")
        self._zoom_steps = 0; self._panning = False; self._space_down = False; self._last_pos = None
        self.setAcceptRichText(True); self.setPlaceholderText(placeholder)
        self.setLineWrapMode(QTextEdit.NoWrap); self.document().setDocumentMargin(12)
        f = self.font()
        if f.pointSize() < 10: f.setPointSize(12); self.setFont(f)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded); self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        mono = QFont("Consolas" if platform.system()=="Windows" else "Menlo", max(11, f.pointSize()))
        self.setFont(mono)
        self.setTabStopDistance(self.fontMetrics().horizontalAdvance(' ') * 4)
        self.setWordWrapMode(QTextOption.NoWrap)
    def set_markdown_text(self, text: str):
        if self._supports_markdown: self.setMarkdown(text)
        else: self.setPlainText(text)
    def markdown_text(self) -> str:
        if self._supports_markdown and hasattr(self, "toMarkdown"): return self.toMarkdown()
        return self.toPlainText()
    def wheelEvent(self, e):
        if QApplication.keyboardModifiers() & Qt.ControlModifier:
            if e.angleDelta().y() > 0: self.zoomIn(1); self._zoom_steps += 1
            else: self.zoomOut(1); self._zoom_steps -= 1
            e.accept(); return
        super().wheelEvent(e)
    def reset_zoom(self):
        if self._zoom_steps>0: self.zoomOut(self._zoom_steps)
        elif self._zoom_steps<0: self.zoomIn(-self._zoom_steps)
        self._zoom_steps = 0

# ---------- ASCII sanitizer ----------
_SANITIZE_MAP = {
    "\u2018":"'","\u2019":"'","\u201A":"'","\u201B":"'",
    "\u201C":'"',"\u201D":'"',"\u201E":'"',"\u201F":'"',
    "\u00A0":" ","\u2007":" ","\u202F":" ","\u200B":"","\uFEFF":"",
    "\u2013":"-","\u2014":"--","\u2212":"-","\u2026":"...",
    "\u2192":"->","\u2190":"<-","\u2194":"<->","\u21D2":"=>","\u21D4":"<=>",
    "\u00D7":"x","\u00F7":"/","\u00B0":" deg ","\u03A9":" Ohm ","\u00B5":"u","\u03BC":"u",
    "\u2022":"-","\u25CF":"-",
}
_SANITIZE_RE = re.compile("|".join(map(re.escape, _SANITIZE_MAP.keys()))) if _SANITIZE_MAP else None

def ascii_sanitize(text: str) -> str:
    if not text: return text
    text = _SANITIZE_RE.sub(lambda m: _SANITIZE_MAP[m.group(0)], text) if _SANITIZE_RE else text
    text = text.replace("\r\n","\n").replace("\r","\n").replace("\t","    ")
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
    return text

# ---------- Table helpers (Markdown <-> rows) ----------
def md_table_to_rows(md: str) -> list[list[str]]:
    if not md: return []
    lines = md.splitlines()
    rows = []
    in_table = False
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("|") and s.endswith("|"):
            if not in_table:
                if i+1 < len(lines) and set(lines[i+1].replace("|","").strip()) <= set("-: "):
                    in_table = True
            if in_table:
                if set(s.replace("|","").strip()) <= set("-: "):
                    continue
                cells = [c.strip() for c in s.strip("|").split("|")]
                rows.append(cells)
        elif in_table:
            break
    return rows

def rows_to_md_table(rows: list[list[str]]) -> str:
    if not rows: return ""
    headers = rows[0]
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join("-"*max(3,len(h)) for h in headers) + " |")
    for r in rows[1:]:
        r_ = (r + [""]*len(headers))[:len(headers)]
        out.append("| " + " | ".join(r_) + " |")
    return "\n".join(out)

def first_table_in(md: str) -> tuple[str, int, int]:
    if not md: return "", -1, -1
    lines = md.splitlines(True)
    n = len(lines); i = 0
    while i < n:
        s = lines[i].strip()
        if s.startswith("|") and s.endswith("|"):
            if i+1 < n and set(lines[i+1].replace("|","").strip()) <= set("-: "):
                j = i+2
                while j < n and lines[j].strip().startswith("|"):
                    j += 1
                return "".join(lines[i:j]), i, j
        i += 1
    return "", -1, -1

# ---------- OpenAI Worker ----------
class OpenAIWorker(QThread):
    """
    Calls OpenAI Responses API off the GUI thread and emits a JSON dict
    or {"_error": "..."} on failure.
    """
    finished = pyqtSignal(dict)

    def __init__(self, api_key: str | None, model_name: str, payload: dict, timeout: int = 180):
        super().__init__()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model_name = model_name or MODEL_NAME_DEFAULT
        self.payload = payload
        self.timeout = timeout

    def _build_prompt(self, p: dict) -> tuple[str, str]:
        # Map numeric maturity to semantics for the model
        def map_maturity(mcode: int) -> str:
            return {0:"placeholder", 1:"immature", 2:"mature", 3:"locked"}.get(mcode, "placeholder")

        # Normalize maturity in inputs
        inputs = p.get("inputs", {})
        for key in ("epsa","wcca","fmea"):
            if key in inputs and isinstance(inputs[key], dict):
                cur = inputs[key].get("current_md","") or ""
                code = maturity_read_from_text(cur)
                if code is None:
                    # if not embedded, try the field value or assume placeholder
                    code = maturity_label_to_code(inputs[key].get("maturity","0"))
                inputs[key]["maturity"] = map_maturity(code)

        p["inputs"] = inputs

        instructions = (
            "You are an expert electronics reliability and analysis assistant. "
            "Given circuit metadata, netlist, partlist, and current EPSA/WCCA/FMEA sections with maturity levels, "
            "produce structured improvements for each analysis section.\n\n"
            "HARD REQUIREMENTS:\n"
            "1) For EACH of EPSA, WCCA, and FMEA, include a SINGLE 'Master Analysis Table' as the FIRST table in the section. "
            "   The table MUST have these columns (exact order):\n"
            "   | Item | Function/Block | Assumptions | Method | Key Equations | Inputs/Bounds | Worst-Case Case | Result/Value | Margin | Determination | Risk | Detection/Controls | Severity | Occurrence | Detection | RPN | Criticality Class | Conclusion | Actions/Recommendations | References |\n"
            "   - Populate every row with concise, engineering-accurate content.\n"
            "   - Use additional rows to cover all relevant components/blocks and stress cases.\n"
            "2) After the table, include well-structured narrative subsections as needed (e.g., Mission Profile, Environments, Parts Stress, "
            "   Corner Cases, Calculation Details, Conclusions).\n"
            "3) Respect maturity: if a section is 'locked', do NOT change its content—return the same text in 'proposed_md' and put your commentary in 'rationale'.\n"
            "4) Output must be STRICT JSON that validates the provided response schema. No markdown fences, no extra prose outside JSON.\n"
        )

        response_schema = {
            "type": "object",
            "properties": {
                "epsa": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["unchanged","suggested","generated"]},
                        "rationale": {"type": "string"},
                        "confidence": {"type": "number"},
                        "proposed_md": {"type": "string"},
                        "diff_hunks": {"type": "array", "items": {
                            "type":"object",
                            "properties":{
                                "type":{"type":"string","enum":["replace","insert","delete"]},
                                "locator":{"type":"object","properties":{"heading":{"type":"string"}}, "required":["heading"]},
                                "old":{"type":"string"},
                                "new":{"type":"string"}
                            },
                            "required":["type","locator"]
                        }}
                    },
                    "required": ["status","proposed_md"]
                },
                "wcca": {"$ref":"#/properties/epsa"},
                "fmea": {"$ref":"#/properties/epsa"}
            },
            "required": ["epsa","wcca","fmea"]
        }

        user_bundle = {
            "request_meta": p.get("request_meta", {}),
            "inputs": p.get("inputs", {}),
            "outputs_spec": p.get("outputs_spec", {"format":"json","fields":["epsa","wcca","fmea"]}),
            "response_schema": response_schema,
            "exemplar_response": {
                "epsa": {"status":"suggested","rationale":"example","confidence":0.6,"proposed_md":"<!-- maturity: 1 Immature -->\n## EPSA\n","diff_hunks":[]},
                "wcca": {"status":"unchanged","rationale":"example","confidence":0.5,"proposed_md":"","diff_hunks":[]},
                "fmea": {"status":"generated","rationale":"example","confidence":0.7,"proposed_md":"<!-- maturity: 0 Placeholder -->\n## FMEA\n","diff_hunks":[]}
            }
        }
        user_json = json.dumps(user_bundle, ensure_ascii=True)
        return instructions, user_json

    def run(self):
        if OpenAI is None:
            self.finished.emit({"_error": "OpenAI SDK not installed. Run: pip install --upgrade openai"})
            return
        if not self.api_key:
            self.finished.emit({"_error": "OPENAI_API_KEY not set. Use AI Assist ▸ Set API Key…"})
            return

        try:
            client = OpenAI(api_key=self.api_key)
            instructions, user_json = self._build_prompt(self.payload)

            def _extract_responses_text(resp_obj):
                # responses.create common extractors
                raw = getattr(resp_obj, "output_text", None)
                if raw:
                    return raw
                try:
                    return resp_obj.output[0].content[0].text
                except Exception:
                    return ""

            # --- Try Responses API, with temperature ---
            try:
                resp = client.responses.create(
                    model=self.model_name,
                    instructions=instructions,
                    input=user_json,
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    timeout=self.timeout,
                )
                raw = _extract_responses_text(resp)
            except TypeError:
                # Older SDK: retry Responses without response_format; still with temp
                try:
                    resp = client.responses.create(
                        model=self.model_name,
                        instructions=instructions + " ALWAYS return a single valid JSON object. No backticks.",
                        input=user_json,
                        temperature=0.2,
                        timeout=self.timeout,
                    )
                    raw = _extract_responses_text(resp)
                except Exception as e:
                    # Fallback to Chat Completions (with temp; may be rejected)
                    raw = self._call_chat_completions(client, instructions, user_json, use_temperature=True)
            except Exception as e:
                # If server rejected temperature specifically, retry WITHOUT temperature
                if "param" in str(e).lower() and "temperature" in str(e).lower():
                    try:
                        resp = client.responses.create(
                            model=self.model_name,
                            instructions=instructions,
                            input=user_json,
                            response_format={"type": "json_object"},
                            timeout=self.timeout,
                        )
                        raw = _extract_responses_text(resp)
                    except TypeError:
                        # Older SDK: again without response_format
                        resp = client.responses.create(
                            model=self.model_name,
                            instructions=instructions + " ALWAYS return a single valid JSON object. No backticks.",
                            input=user_json,
                            timeout=self.timeout,
                        )
                        raw = _extract_responses_text(resp)
                    except Exception:
                        # Final fallback: Chat Completions WITHOUT temperature
                        raw = self._call_chat_completions(client, instructions, user_json, use_temperature=False)
                else:
                    # Other error -> try Chat Completions path
                    raw = self._call_chat_completions(client, instructions, user_json, use_temperature=True)

            # Parse JSON (strict)
            try:
                data = json.loads(raw)
                self.finished.emit(data)
            except Exception as pe:
                self.finished.emit({
                    "_error": f"Model did not return valid JSON: {pe}\nRaw (first 800 chars): {raw[:800]}"
                })

        except Exception as e:
            self.finished.emit({"_error": str(e)})

    def _call_chat_completions(self, client, instructions: str, user_json: str, use_temperature: bool = True) -> str:
        """
        Fallback for older SDKs / models. If temperature is rejected, call again without it.
        """
        def _extract_chat_text(cc_obj):
            try:
                msg = cc_obj.choices[0].message
                if isinstance(msg.content, list):
                    parts = []
                    for p in msg.content:
                        if isinstance(p, dict) and "text" in p:
                            parts.append(p["text"])
                        elif isinstance(p, str):
                            parts.append(p)
                    return "".join(parts)
                return msg.content or ""
            except Exception:
                return ""

        # Try with response_format first
        try:
            kwargs = dict(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": user_json},
                ],
                response_format={"type": "json_object"},
            )
            if use_temperature:
                kwargs["temperature"] = 0.2
            cc = client.chat.completions.create(**kwargs)
            return _extract_chat_text(cc)
        except TypeError:
            # SDK that doesn't support response_format here
            try:
                kwargs = dict(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": instructions + " ALWAYS return a single valid JSON object. No backticks."},
                        {"role": "user", "content": user_json},
                    ],
                )
                if use_temperature:
                    kwargs["temperature"] = 0.2
                cc = client.chat.completions.create(**kwargs)
                return _extract_chat_text(cc)
            except Exception as e:
                # If temperature is rejected, retry once without it
                if "temperature" in str(e).lower():
                    cc = client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": instructions + " ALWAYS return a single valid JSON object. No backticks."},
                            {"role": "user", "content": user_json},
                        ],
                    )
                    return _extract_chat_text(cc)
                return ""
        except Exception as e:
            # If temperature is rejected here, retry once without it
            if "temperature" in str(e).lower():
                try:
                    kwargs = dict(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": instructions},
                            {"role": "user", "content": user_json},
                        ],
                        response_format={"type": "json_object"},
                    )
                    # no temperature
                    cc = client.chat.completions.create(**kwargs)
                    return _extract_chat_text(cc)
                except Exception:
                    pass
            return ""

# ---------- Review Drawer ----------
class ReviewDrawer(QWidget):
    apply_all_clicked = pyqtSignal()
    apply_selected_clicked = pyqtSignal()
    discard_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVisible(False)
        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0); outer.setSpacing(6)

        header = QHBoxLayout()
        self.lbl_title = QLabel("AI Suggestions"); header.addWidget(self.lbl_title)
        header.addItem(QSpacerItem(40,20,QSizePolicy.Expanding,QSizePolicy.Minimum))
        self.btn_apply_selected = QPushButton("Apply Selected")
        self.btn_apply_all = QPushButton("Apply All")
        self.btn_discard = QPushButton("Discard")
        for b in (self.btn_apply_selected, self.btn_apply_all, self.btn_discard):
            b.setMinimumHeight(28)
        header.addWidget(self.btn_apply_selected); header.addWidget(self.btn_apply_all); header.addWidget(self.btn_discard)
        outer.addLayout(header)

        self.tabs = QTabWidget(self)
        # Rendered
        self.rendered = ZoomableMarkdownEdit(self, placeholder="(Proposed content)")
        self.rendered.setReadOnly(True)
        t1 = QWidget(); v1 = QVBoxLayout(t1); v1.addWidget(self.rendered)
        self.tabs.addTab(t1, "Rendered")
        # Diff
        self.diff_left = ZoomableMarkdownEdit(self, placeholder="(Current)")
        self.diff_right = ZoomableMarkdownEdit(self, placeholder="(Proposed)")
        self.diff_left.setReadOnly(True); self.diff_right.setReadOnly(True)
        t2 = QWidget(); v2 = QHBoxLayout(t2); v2.addWidget(self.diff_left); v2.addWidget(self.diff_right)
        self.tabs.addTab(t2, "Diff")
        # Table Changes
        t3 = QWidget(); v3 = QVBoxLayout(t3)
        self.table_changes = QTableWidget(0, 6, t3)
        self.table_changes.setHorizontalHeaderLabels(["Apply","Key","Column","Before","After","Change"])
        for c in (0,1,2,5):
            self.table_changes.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        for c in (3,4):
            self.table_changes.horizontalHeader().setSectionResizeMode(c, QHeaderView.Stretch)
        self.table_changes.verticalHeader().setVisible(False)
        v3.addWidget(self.table_changes)
        # Quick selectors
        hs = QHBoxLayout()
        btn_all = QPushButton("Select All Modified"); btn_none = QPushButton("Clear Selection")
        hs.addWidget(btn_all); hs.addWidget(btn_none); hs.addItem(QSpacerItem(40,20,QSizePolicy.Expanding,QSizePolicy.Minimum))
        v3.addLayout(hs)
        self.tabs.addTab(t3, "Table Changes")
        outer.addWidget(self.tabs)

        btn_all.clicked.connect(self._select_all_modified)
        btn_none.clicked.connect(self._clear_selection)
        self.btn_apply_all.clicked.connect(self.apply_all_clicked.emit)
        self.btn_apply_selected.clicked.connect(self.apply_selected_clicked.emit)
        self.btn_discard.clicked.connect(self.discard_clicked.emit)

    def _select_all_modified(self):
        for r in range(self.table_changes.rowCount()):
            w = self.table_changes.cellWidget(r, 0)
            if hasattr(w, "setChecked"): w.setChecked(True)
    def _clear_selection(self):
        for r in range(self.table_changes.rowCount()):
            w = self.table_changes.cellWidget(r, 0)
            if hasattr(w, "setChecked"): w.setChecked(False)

# ---------- Main Window ----------
class CatalogWindow(QMainWindow):
    def __init__(self, catalog_root: Path, app_icon: QIcon):
        super().__init__()
        self.setWindowTitle(APP_TITLE); self.setWindowIcon(app_icon); self.resize(1420, 940)

        self.catalog_root = catalog_root
        self.current_path: Path | None = None
        self.current_folder: Path | None = None

        # Lazy caches
        self.section_texts: dict[str, str] = {}
        self.section_loaded: dict[str, bool] = {}
        self.pin_rows_cache: list[list[str]] = []
        self.test_rows_cache: list[list[str]] = []

        # AI state
        self.ai_result: dict | None = None
        self.ai_target_section: str | None = None
        # --- ETA / timing state ---
        self._ai_start_ts: datetime.datetime | None = None
        self._eta_timer: QTimer | None = None
        self._eta_target_sec: int | None = None
        
        # Maturity state (editor may drop HTML comments; keep a canonical value here)
        self.maturity_state = {"EPSA": 0, "WCCA": 0, "FMEA": 0}

        # Toolbar
        tb = QToolBar("Main", self); tb.setMovable(False); self.addToolBar(tb)
        act_new_entry = QAction("🧩 New Entry", self);  act_new_entry.triggered.connect(self.create_new_entry);  tb.addAction(act_new_entry)
        act_new_folder = QAction("🗂️ New Folder", self); act_new_folder.triggered.connect(self.create_new_folder); tb.addAction(act_new_folder)
        act_rename     = QAction("✏️ Rename", self);     act_rename.triggered.connect(self.rename_item);           tb.addAction(act_rename)
        act_delete     = QAction("🗑️ Delete", self);     act_delete.triggered.connect(self.delete_item);           tb.addAction(act_delete)
        tb.addSeparator()
        act_archive = QAction("📦 Archive", self); act_archive.triggered.connect(self.archive_script_folder); tb.addAction(act_archive)
        act_open_loc = QAction("📂 Open Location", self); act_open_loc.triggered.connect(self.open_file_location); tb.addAction(act_open_loc)
        tb.addSeparator()
        self.act_save = QAction("💾 Save (Ctrl+S)", self); self.act_save.setShortcut(QKeySequence.Save)
        self.act_save.triggered.connect(self.save_from_form); tb.addAction(self.act_save)
        tb.addSeparator()
        self.act_zoom_in = QAction("🔎＋ Zoom In", self); self.act_zoom_out = QAction("🔎－ Zoom Out", self); self.act_zoom_reset = QAction("🎯 Reset Zoom", self)
        self.act_zoom_in.setShortcut(QKeySequence("Ctrl++")); self.act_zoom_out.setShortcut(QKeySequence("Ctrl+-")); self.act_zoom_reset.setShortcut(QKeySequence("Ctrl+0"))
        tb.addAction(self.act_zoom_in); tb.addAction(self.act_zoom_out); tb.addAction(self.act_zoom_reset)
        def _focused_markdown():
            w = QApplication.focusWidget(); return w if isinstance(w, ZoomableMarkdownEdit) else None
        self.act_zoom_in.triggered.connect(lambda: (_focused_markdown() and _focused_markdown().zoomIn(1)))
        self.act_zoom_out.triggered.connect(lambda: (_focused_markdown() and _focused_markdown().zoomOut(1)))
        self.act_zoom_reset.triggered.connect(lambda: (_focused_markdown() and _focused_markdown().reset_zoom()))

        # 🤖 AI Assist (direct OpenAI)
        self.current_model_name = MODEL_NAME_DEFAULT
        self.current_api_key = os.environ.get("OPENAI_API_KEY", "")

        act_ai = QAction("🤖 AI Assist", self)
        m = QMenu(self)
        m.addAction("Generate", lambda: self.run_ai("generate"))
        m.addAction("Suggest Edits", lambda: self.run_ai("suggest"))
        m.addAction("Re-analyze", lambda: self.run_ai("reanalyze"))
        m.addSeparator()
        def _set_model():
            name, ok = self.ask_text("Model", "OpenAI model name:", default=self.current_model_name)
            if ok and name.strip(): self.current_model_name = name.strip()
        def _set_api_key():
            key, ok = self.ask_text("OpenAI API Key", "Paste your OPENAI_API_KEY:\n(Stored in memory for this session)", default=(self.current_api_key or "sk-"))
            if ok: self.current_api_key = key.strip()
        m.addAction("Set Model…", _set_model)
        m.addAction("Set API Key…", _set_api_key)
        act_ai.setMenu(m); tb.addAction(act_ai)

        # FS model & tree
        self.fs_model = QFileSystemModel(self); self.fs_model.setReadOnly(False)
        self.fs_model.setRootPath(str(self.catalog_root)); self.fs_model.setNameFilters(["*.md"]); self.fs_model.setNameFilterDisables(False)
        self.fs_model.fileRenamed.connect(self.on_fs_file_renamed)
        self.proxy = DescProxyModel(self); self.proxy.setSourceModel(self.fs_model)
        self.tree = QTreeView(self); self.tree.setModel(self.proxy)
        self.tree.setItemDelegate(SlimLineEditDelegate(self.tree))
        self.tree.setRootIndex(self.proxy.mapFromSource(self.fs_model.index(str(self.catalog_root))))
        self.tree.setHeaderHidden(False); self.tree.setSortingEnabled(True); self.tree.sortByColumn(0, Qt.AscendingOrder)
        for col in range(2, self.proxy.columnCount(self.tree.rootIndex())): self.tree.setColumnHidden(col, True)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents); self.tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tree.setDragEnabled(True); self.tree.setAcceptDrops(True); self.tree.setDropIndicatorShown(True)
        self.tree.setDefaultDropAction(Qt.MoveAction); self.tree.setDragDropMode(QAbstractItemView.DragDrop); self.tree.setDragDropOverwriteMode(False)
        self.tree.selectionModel().selectionChanged.connect(self.on_tree_selection)

        # Right panel
        right_container = QWidget(self); right_layout = QVBoxLayout(right_container); right_layout.setContentsMargins(0,0,0,0); right_layout.setSpacing(8)
        self.path_label = QLabel("", self); right_layout.addWidget(self.path_label)

        # Folder metadata
        self.folder_panel = QGroupBox("Folder Metadata", self); fp = QFormLayout(self.folder_panel)
        self.folder_title = QLineEdit(self.folder_panel); self.folder_desc  = QLineEdit(self.folder_panel)
        self.folder_summary = QTextEdit(self.folder_panel); self.folder_owner = QLineEdit(self.folder_panel); self.folder_tags  = QLineEdit(self.folder_panel)
        self.folder_created = QLineEdit(self.folder_panel); self.folder_updated = QLineEdit(self.folder_panel)
        for w in (self.folder_created, self.folder_updated): w.setReadOnly(True)
        self.folder_title.setPlaceholderText("TITLE (UPPERCASE)"); self.folder_desc.setPlaceholderText("DESCRIPTION (UPPERCASE)")
        self.folder_summary.setPlaceholderText("Summary / notes for this folder…"); self.folder_owner.setPlaceholderText("Owner name or team")
        self.folder_tags.setPlaceholderText("Comma-separated tags (e.g., power, digital, hv)")
        fp.addRow("TITLE:", self.folder_title); fp.addRow("DESCRIPTION:", self.folder_desc); fp.addRow("Summary:", self.folder_summary)
        fp.addRow("Owner:", self.folder_owner); fp.addRow("Tags:", self.folder_tags); fp.addRow("Created:", self.folder_created); fp.addRow("Last Updated:", self.folder_updated)
        right_layout.addWidget(self.folder_panel, 1)

        # Tabs
        self.tabs = QTabWidget(self)

        # Metadata tab
        meta_tab = QWidget(self); meta_v = QVBoxLayout(meta_tab); meta_v.setContentsMargins(0,0,0,0); meta_v.setSpacing(8)
        self.fields_group = QGroupBox("Circuit Metadata", meta_tab); self.fields_form = QFormLayout(self.fields_group)
        self.field_widgets: dict[str, QLineEdit] = {}
        for label, placeholder in FIELD_ORDER:
            le = QLineEdit(self.fields_group); le.setPlaceholderText(placeholder)
            self.field_widgets[label] = le; self.fields_form.addRow(label + ":", le)
        meta_v.addWidget(self.fields_group)
        self.tabs.addTab(meta_tab, "Metadata")

        # Used On tab
        used_tab = QWidget(self); used_v = QVBoxLayout(used_tab)
        self.used_table = QTableWidget(0, 2, used_tab); self.used_table.setHorizontalHeaderLabels(["PN", "Occurrences"])
        self.used_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch); self.used_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.used_table.verticalHeader().setVisible(False)
        used_v.addWidget(self.used_table)
        h = QHBoxLayout(); b1 = QPushButton("Add Row"); b2 = QPushButton("Remove Selected")
        b1.clicked.connect(self.add_used_row); b2.clicked.connect(self.remove_used_row)
        h.addWidget(b1); h.addWidget(b2); used_v.addLayout(h)
        self.tabs.addTab(used_tab, "Used On")

        # Netlist (markdown)
        net_tab = QWidget(self); net_v = QVBoxLayout(net_tab)
        self.netlist_edit = ZoomableMarkdownEdit(net_tab, "(paste or type your netlist here — tables/lists render)")
        net_v.addWidget(self.netlist_edit); self.tabs.addTab(net_tab, "Netlist")

        # Partlist (markdown)
        pl_tab = QWidget(self); pl_v = QVBoxLayout(pl_tab)
        self.partlist_edit = ZoomableMarkdownEdit(pl_tab, "(paste or type your partlist here — tables/lists render)")
        pl_v.addWidget(self.partlist_edit); self.tabs.addTab(pl_tab, "Partlist")

        # Pin Interface (table)
        pin_tab = QWidget(self); pin_v = QVBoxLayout(pin_tab)
        self.pin_table = QTableWidget(0, len(PIN_HEADERS), pin_tab); self.pin_table.setHorizontalHeaderLabels(PIN_HEADERS)
        for c in range(len(PIN_HEADERS)):
            self.pin_table.horizontalHeader().setSectionResizeMode(c, QHeaderView.Stretch if PIN_HEADERS[c]=="Description" else QHeaderView.ResizeToContents)
        self.pin_table.verticalHeader().setVisible(False)
        ph = QHBoxLayout(); pa = QPushButton("Add Row"); pd = QPushButton("Remove Selected")
        pa.clicked.connect(lambda: self.add_row(self.pin_table, default=["","","",""])); pd.clicked.connect(lambda: self.remove_selected_row(self.pin_table))
        pin_v.addWidget(self.pin_table); ph.addWidget(pa); ph.addWidget(pd); pin_v.addLayout(ph)
        self.tabs.addTab(pin_tab, "Pin Interface")

        # Tests (table)
        test_tab = QWidget(self); test_v = QVBoxLayout(test_tab)
        self.test_table = QTableWidget(0, len(TEST_HEADERS), test_tab); self.test_table.setHorizontalHeaderLabels(TEST_HEADERS)
        for c in range(len(TEST_HEADERS)):
            self.test_table.horizontalHeader().setSectionResizeMode(c, QHeaderView.Stretch if TEST_HEADERS[c] in ("Description","Note") else QHeaderView.ResizeToContents)
        self.test_table.verticalHeader().setVisible(False)
        th = QHBoxLayout(); ta = QPushButton("Add Row"); td = QPushButton("Remove Selected")
        ta.clicked.connect(lambda: self.add_row(self.test_table, default=["","","",""])); td.clicked.connect(lambda: self.remove_selected_row(self.test_table))
        test_v.addWidget(self.test_table); th.addWidget(ta); th.addWidget(td); test_v.addLayout(th)
        self.tabs.addTab(test_tab, "Tests")

        # EPSA/WCCA/FMEA (markdown + maturity)
        def build_markdown_tab(name: str):
            tab = QWidget(self); v = QVBoxLayout(tab)
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{name} Maturity:"))
            cmb = QComboBox(); cmb.addItems(MATURITY_LEVELS); row.addWidget(cmb)
            row.addItem(QSpacerItem(40,20,QSizePolicy.Expanding,QSizePolicy.Minimum)); v.addLayout(row)
            ed = ZoomableMarkdownEdit(tab, f"({name} markdown here)")
            v.addWidget(ed); return tab, ed, cmb

        epsa_tab, self.epsa_edit, self.epsa_cmb = build_markdown_tab("EPSA")
        wcca_tab, self.wcca_edit, self.wcca_cmb = build_markdown_tab("WCCA")
        fmea_tab, self.fmea_edit, self.fmea_cmb = build_markdown_tab("FMEA")
        self.tabs.addTab(epsa_tab, "EPSA"); self.tabs.addTab(wcca_tab, "WCCA"); self.tabs.addTab(fmea_tab, "FMEA")

        # AI Statistics tab (new)
        self.ai_stats_tab = self._make_ai_stats_tab()
        self.tabs.addTab(self.ai_stats_tab, "AI Statistics")

        self.tabs.currentChanged.connect(self.on_tab_changed)
        right_layout.addWidget(self.tabs, 1)

        # Review Drawer
        self.review = ReviewDrawer(self)
        self.review.apply_all_clicked.connect(self._apply_all_from_ai)
        self.review.apply_selected_clicked.connect(self._apply_selected_from_ai)
        self.review.discard_clicked.connect(self._discard_ai)
        right_layout.addWidget(self.review)

        # Splitter
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.addWidget(self.tree); splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 1); splitter.setStretchFactor(1, 2); splitter.setSizes([380, 1040])

        central = QWidget(self); outer = QHBoxLayout(central); outer.setContentsMargins(8,8,8,8); outer.setSpacing(8)
        outer.addWidget(splitter); self.setCentralWidget(central)

        self.apply_dark_styles(); self.show_file_ui(False)

    # ---------- Styling ----------
    def apply_dark_styles(self):
        self.setStyleSheet("""
            QWidget { background-color:#202225; color:#E6E6E6; }
            QToolBar { background:#1B1E20; spacing:6px; border:0; }
            QToolButton, QPushButton { color:#E6E6E6; }
            QLabel { color:#E6E6E6; }
            QGroupBox { border:1px solid #3A3F44; border-radius:6px; margin-top:12px; padding-top:8px; }
            QGroupBox::title { left:10px; padding:0 4px; color:#CFCFCF; }
            QLineEdit, QTextEdit { background:#2A2D31; color:#E6E6E6; border:1px solid #3A3F44; border-radius:6px; padding:6px; }
            QTreeView QLineEdit { background:#2A2D31; color:#E6E6E6; border:1px solid #3A3F44; padding:2px 4px; }
            QPushButton { background:#2F343A; border:1px solid #444; border-radius:6px; padding:6px 12px; }
            QPushButton:hover { background:#3A4047; } QPushButton:pressed { background:#2A2F35; }
            QTreeView { background:#1E2124; alternate-background-color:#24272B; border:1px solid #3A3F44; }
            QTreeView::item:selected { background:#3B4252; color:#E6E6E6; }
            QHeaderView::section { background:#2A2D31; color:#E6E6E6; border:0; padding:6px; font-weight:600; }
            QTableWidget { background:#1E2124; color:#E6E6E6; gridline-color:#3A3F44; border:1px solid #3A3F44; border-radius:6px; }
            QTabBar::tab { background:#2A2D31; color:#E6E6E6; padding:8px 12px; margin-right:2px; border-top-left-radius:6px; border-top-right-radius:6px; }
            QTabBar::tab:selected { background:#3A3F44; } QTabBar::tab:hover { background:#34383D; }
        """)

    # ---------- Dialog helpers ----------
    def _apply_dark(self, dlg):
        try: apply_windows_dark_titlebar(dlg)
        except Exception: pass
    def ask_text(self, title: str, label: str, default: str = "") -> tuple[str, bool]:
        dlg = QInputDialog(self); dlg.setWindowTitle(title); dlg.setLabelText(label); dlg.setTextValue(default)
        self._apply_dark(dlg); ok = dlg.exec_() == dlg.Accepted; return (dlg.textValue(), ok)
    def ask_yes_no(self, title: str, text: str) -> bool:
        mb = QMessageBox(self); mb.setWindowTitle(title); mb.setText(text)
        mb.setIcon(QMessageBox.Question); mb.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        self._apply_dark(mb); return mb.exec_() == QMessageBox.Yes
    def info(self, title: str, text: str):
        mb = QMessageBox(self); mb.setWindowTitle(title); mb.setText(text)
        mb.setIcon(QMessageBox.Information); mb.setStandardButtons(QMessageBox.Ok); self._apply_dark(mb); mb.exec_()
    def warn(self, title: str, text: str):
        mb = QMessageBox(self); mb.setWindowTitle(title); mb.setText(text)
        mb.setIcon(QMessageBox.Warning); mb.setStandardButtons(QMessageBox.Ok); self._apply_dark(mb); mb.exec_()
    def error(self, title: str, text: str):
        mb = QMessageBox(self); mb.setWindowTitle(title); mb.setText(text)
        mb.setIcon(QMessageBox.Critical); mb.setStandardButtons(QMessageBox.Ok); self._apply_dark(mb); mb.exec_()

    # ---------- UI helpers ----------
    def show_file_ui(self, file_selected: bool):
        self.tabs.setVisible(file_selected)
        self.folder_panel.setVisible(not file_selected)

    def selected_source_index(self) -> QModelIndex | None:
        sel = self.tree.selectionModel().selectedIndexes()
        if not sel: return None
        idx = sel[0]
        if idx.column()!=0: idx = self.proxy.index(idx.row(), 0, idx.parent())
        return self.proxy.mapToSource(idx)

    def selected_path(self) -> Path | None:
        sidx = self.selected_source_index()
        if not sidx or not sidx.isValid(): return None
        return Path(self.fs_model.filePath(sidx))

    def is_markdown(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower()==".md"

    def _ensure_maturity_marker(self, _section_name: str, md: str, cmb=None) -> str:
        """
        Back-compat shim. Old code calls:
            _ensure_maturity_marker("EPSA", md, self.epsa_cmb)
        New flow uses a canonical integer code; this adapter preserves old behavior.
        """
        if cmb is not None:
            code = maturity_label_to_code(cmb.currentText())
        else:
            # Fallback: read from text if no combobox is provided
            code = maturity_read_from_text(md) or 0
        return self._ensure_maturity_marker_code(md, int(code))

    # --- Section normalizers -------------------------------------------------
    def _strip_inner_heading_preserve_maturity(self, section_name: str, body_md: str) -> str:
        """
        If the body starts with an optional maturity comment and then a duplicate
        inner '## <section_name>' heading, remove ONLY that inner heading while
        preserving the maturity marker and spacing.
        """
        if not body_md:
            return body_md

        lines = body_md.splitlines()
        i = 0
        prefix = []

        # Preserve leading blanks
        while i < len(lines) and not lines[i].strip():
            prefix.append(lines[i])
            i += 1

        # Preserve maturity comment if present
        maturity_re = re.compile(
            r"<!--\s*maturity:\s*(?:[0-3]|placeholder|immature|mature|locked)\s*-->",
            re.I,
        )
        if i < len(lines) and maturity_re.search(lines[i] or ""):
            prefix.append(lines[i])
            i += 1
            # Optional blank after maturity comment
            if i < len(lines) and not lines[i].strip():
                prefix.append(lines[i])
                i += 1

        # Remove a duplicate inner heading if it appears next
        if i < len(lines) and lines[i].strip().lower() == f"## {section_name}".lower():
            i += 1
            if i < len(lines) and not lines[i].strip():
                i += 1
            return "\n".join(prefix + lines[i:]).lstrip("\n")

        return body_md

    # --- Backwards-compat alias (so any stray old callsites still resolve correctly)
    _strip_redundant_section_heading = _strip_inner_heading_preserve_maturity

    def _normalize_section_body(self, section_name: str, body_md: str) -> str:
        """
        Clean an analysis section body *without* adding outer '## <section>' heading.
        - ASCII sanitize
        - preserve a single maturity marker (if present) and keep it at the top
        - remove duplicate inner '## <section_name>' heading
        - trim leading/trailing blank lines
        """
        md = ascii_sanitize(body_md or "")

        # Remove duplicate inner heading but keep any maturity marker intact
        md = self._strip_inner_heading_preserve_maturity(section_name, md)

        # Collapse to a single maturity marker at the very top
        maturity_re = re.compile(
            r"<!--\s*maturity:\s*(?:[0-3]|placeholder|immature|mature|locked)\s*-->",
            re.I,
        )
        found = maturity_re.findall(md)
        marker = found[0] if found else None
        md_wo = maturity_re.sub("", md).lstrip("\n")

        return (f"{marker}\n{md_wo}" if marker else md_wo).strip("\n")

    # ---------- Section body normalization ----------
    @staticmethod
    def _strip_redundant_section_heading(self, section_name: str, body_md: str) -> str:
        """
        If the body starts with an optional maturity comment and then a duplicate
        '## <section_name>' heading, remove ONLY that inner heading while preserving
        the maturity marker and surrounding spacing.
        """
        if not body_md:
            return body_md

        lines = body_md.splitlines()
        i = 0
        prefix = []

        # Preserve leading blanks (harmless)
        while i < len(lines) and not lines[i].strip():
            prefix.append(lines[i])
            i += 1

        # Preserve maturity comment if present
        maturity_re = re.compile(
            r"<!--\s*maturity:\s*(?:[0-3]|placeholder|immature|mature|locked)\s*-->",
            re.I
        )
        if i < len(lines) and maturity_re.search(lines[i] or ""):
            prefix.append(lines[i])
            i += 1
            # Optional blank after maturity
            if i < len(lines) and not lines[i].strip():
                prefix.append(lines[i])
                i += 1

        # If next nonblank is a duplicate inner heading, drop it
        if i < len(lines) and lines[i].strip().lower() == f"## {section_name}".lower():
            i += 1
            # Skip one optional blank after the heading
            if i < len(lines) and not lines[i].strip():
                i += 1
            return "\n".join(prefix + lines[i:]).lstrip("\n")

        return body_md

    # ---------- Table helpers ----------
    def add_row(self, table: QTableWidget, default=None):
        r = table.rowCount(); table.insertRow(r)
        cols = table.columnCount(); default = default or ["" for _ in range(cols)]
        for c in range(cols):
            table.setItem(r, c, QTableWidgetItem(default[c]))
    def remove_selected_row(self, table: QTableWidget):
        r = table.currentRow()
        if r >= 0: table.removeRow(r)

    # ---------- Selection / Lazy load ----------
    def on_tree_selection(self, *_):
        path = self.selected_path()
        if not path: return
        self.section_texts = {name:"" for name in SECTION_NAMES}
        self.section_loaded = {name:False for name in SECTION_NAMES}
        self.pin_rows_cache = []; self.test_rows_cache = []
        if path.is_dir():
            self.current_path=None; self.current_folder=path; self.path_label.setText(f"Folder: {path}")
            self.load_folder_meta(path); self.tabs.setVisible(False); self.folder_panel.setVisible(True); self.review.setVisible(False)
            # Update AI stats view for current folder
            self.refresh_ai_stats()
            return
        if self.is_markdown(path):
            self.current_folder=None; self.load_file_lazy(path); self.tabs.setVisible(True); self.folder_panel.setVisible(False); self.review.setVisible(False)
            # Update AI stats view for parent folder
            self.refresh_ai_stats()

    def load_folder_meta(self, folder: Path):
        """
        Populate the right-hand 'Folder Metadata' panel from <folder>/<folder>.json.
        Uses read_folder_meta(...) and maps fields into the UI widgets.
        """
        meta = self.read_folder_meta(folder)

        # Title / Description (UPPERCASE in file, but show as-is)
        self.folder_title.setText(meta.get("TITLE", "") or "")
        self.folder_desc.setText(meta.get("DESCRIPTION", "") or "")

        # Summary / Owner
        self.folder_summary.setPlainText(meta.get("Summary", "") or "")
        self.folder_owner.setText(meta.get("Owner", "") or "")

        # Tags (stored as list; display as comma-separated text)
        tags_val = meta.get("Tags", [])
        if isinstance(tags_val, list):
            tags_text = ", ".join(str(t) for t in tags_val)
        else:
            # tolerate older schemas where Tags might be a string
            tags_text = str(tags_val or "")
        self.folder_tags.setText(tags_text)

        # Dates (read-only fields)
        self.folder_created.setText(meta.get("Created", "") or "")
        self.folder_updated.setText(meta.get("Last Updated", "") or "")

    def _refresh_maturity_dropdown_for_current_tab(self):
        if not self.tabs.count():
            return
        label = self.tabs.tabText(self.tabs.currentIndex())
        if label not in ("EPSA", "WCCA", "FMEA"):
            return
        cmb = {"EPSA": self.epsa_cmb, "WCCA": self.wcca_cmb, "FMEA": self.fmea_cmb}[label]
        code = int(self.maturity_state.get(label, 0))
        cmb.blockSignals(True)
        cmb.setCurrentText(maturity_code_to_label(code))
        cmb.blockSignals(False)

    def load_file_lazy(self, path: Path):
        """
        Load a Markdown entry file without rendering all sections immediately.
        Also seeds self.maturity_state (EPSA/WCCA/FMEA) from the raw file text,
        since the editor may drop HTML comments like <!-- maturity: ... -->.
        """
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            self.error("Error", f"Failed to read file:\n{e}")
            return

        self.current_path = path
        self.path_label.setText(f"File: {path}")

        # --- Parse top metadata table ---
        fields = {k: "" for k, _ in FIELD_ORDER}
        used_rows = []
        lines = [ln.rstrip("\n") for ln in text.splitlines()]
        n = len(lines)

        i = 0
        while i < n:
            if lines[i].strip().lower().startswith("| field") and "| value" in lines[i].lower():
                i += 2
                while i < n and lines[i].strip().startswith("|"):
                    m = MD_ROW_RE.match(lines[i].strip())
                    if m:
                        field = m.group("field").strip()
                        value = m.group("value").strip()
                        if field in fields:
                            fields[field] = value
                    i += 1
                break
            i += 1

        def find_header(title: str):
            target = f"## {title}".lower()
            for idx, ln in enumerate(lines):
                if ln.strip().lower() == target:
                    return idx
            return None

        # --- Parse "Used On" table (if present) ---
        uix = find_header("Used On")
        if uix is not None:
            j = uix + 1
            # advance to header row
            while j < n and not lines[j].strip().startswith("|"):
                j += 1
            # skip divider
            j += 2
            while j < n and lines[j].strip().startswith("|"):
                m = MD_ROW_RE.match(lines[j].strip())
                if m:
                    used_rows.append([m.group("field").strip(), m.group("value").strip()])
                j += 1

        def capture_block(title: str) -> str:
            idx = find_header(title)
            if idx is None:
                return ""
            j = idx + 1
            chunk = []
            while j < n and not lines[j].startswith("## "):
                chunk.append(lines[j])
                j += 1
            # trim leading/trailing blanks
            while chunk and not chunk[0].strip():
                chunk.pop(0)
            while chunk and not chunk[-1].strip():
                chunk.pop()
            return "\n".join(chunk)

        # --- Cache raw section texts; mark as not-yet-hydrated in editors ---
        self.section_texts = {name: "" for name in SECTION_NAMES}
        self.section_loaded = {name: False for name in SECTION_NAMES}
        self.pin_rows_cache = []
        self.test_rows_cache = []

        for name in SECTION_NAMES:
            self.section_texts[name] = capture_block(name)
            self.section_loaded[name] = False

        # --- NEW: seed canonical maturity from RAW text (editor may drop comments) ---
        # Requires: self.maturity_state = {"EPSA": 0, "WCCA": 0, "FMEA": 0} set in __init__
        for sec in ("EPSA", "WCCA", "FMEA"):
            raw = self.section_texts.get(sec, "") or ""
            code = maturity_read_from_text(raw)
            self.maturity_state[sec] = 0 if code is None else int(code)

        # --- Populate minimal UI (top fields + Used On) ---
        for key, _ in FIELD_ORDER:
            self.field_widgets[key].setText(fields.get(key, ""))

        self.used_table.setRowCount(0)
        for pn, occ in used_rows:
            r = self.used_table.rowCount()
            self.used_table.insertRow(r)
            self.used_table.setItem(r, 0, QTableWidgetItem(pn))
            self.used_table.setItem(r, 1, QTableWidgetItem(occ))

        # refresh description column in the tree
        self.proxy.refresh_desc(path)

    # ---------- Hydrate on demand ----------
    def on_tab_changed(self, index: int):
        label = self.tabs.tabText(index)

        if label in ("Netlist", "Partlist", "EPSA", "WCCA", "FMEA"):
            ed_map = {
                "Netlist": self.netlist_edit,
                "Partlist": self.partlist_edit,
                "EPSA": self.epsa_edit,
                "WCCA": self.wcca_edit,
                "FMEA": self.fmea_edit,
            }
            ed = ed_map[label]

            # Lazy hydrate editor content from raw cache when the tab is first shown
            if not self.section_loaded.get(label):
                ed.set_markdown_text(self.section_texts.get(label, "") or "")
                self.section_loaded[label] = True
                ed.textChanged.connect(lambda sn=label, e=ed: self._on_markdown_changed(sn, e))

            # Maturity sync for engineering tabs
            if label in ("EPSA", "WCCA", "FMEA"):
                cmb_map = {"EPSA": self.epsa_cmb, "WCCA": self.wcca_cmb, "FMEA": self.fmea_cmb}
                cmb = cmb_map[label]

                # 1) Drive dropdown from canonical state (do NOT read editor; it may drop comments)
                code = int(self.maturity_state.get(label, 0))
                cmb.blockSignals(True)
                cmb.setCurrentText(maturity_code_to_label(code))
                cmb.blockSignals(False)

                # 2) Connect once: when user changes dropdown, update state (and optionally show marker)
                if not getattr(ed, "_maturity_connected", False):
                    def _on_maturity_changed(_text=None, sn=label, c=cmb, e=ed):
                        self.maturity_state[sn] = maturity_label_to_code(c.currentText())
                        # Optional: keep marker visible in the editor (Qt may later drop it; harmless)
                        try:
                            self._write_maturity_marker(sn, e, self.maturity_state[sn])
                        except Exception:
                            pass

                    cmb.currentTextChanged.connect(_on_maturity_changed)
                    ed._maturity_connected = True

        elif label == "Pin Interface":
            self._hydrate_table_tab("Pin Interface", self.pin_table, PIN_HEADERS, cache_attr="pin_rows_cache")
        elif label == "Tests":
            self._hydrate_table_tab("Tests", self.test_table, TEST_HEADERS, cache_attr="test_rows_cache")

    def _on_markdown_changed(self, section_name: str, editor: ZoomableMarkdownEdit):
        self.section_texts[section_name] = editor.markdown_text()

    def _hydrate_table_tab(self, section_name: str, table: QTableWidget, headers: list[str], cache_attr: str):
        if self.section_loaded.get(section_name): return
        raw = self.section_texts.get(section_name,""); rows = self._parse_table_from_block(raw)
        setattr(self, cache_attr, rows); table.setRowCount(0)
        for row in rows: self.add_row(table, row + [""]*(len(headers)-len(row)))
        self.section_loaded[section_name]=True

    @staticmethod
    def _parse_table_from_block(block_text: str) -> list:
        rows=[]
        if not block_text: return rows
        for ln in block_text.splitlines():
            if not ln.strip().startswith("|"): continue
            if set(ln.replace("|","").strip()) <= set("-: "): continue
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            rows.append(cells)
        return rows

    # ---------- Save ----------
    def save_from_form(self):
        # --- Save a Markdown entry file ---
        if self.current_path and self.is_markdown(self.current_path):
            # Harvest top metadata fields
            fields = {k: w.text().strip() for k, w in self.field_widgets.items()}

            # Harvest "Used On" table
            used_rows = []
            for r in range(self.used_table.rowCount()):
                pn = self.used_table.item(r, 0).text().strip() if self.used_table.item(r, 0) else ""
                oc = self.used_table.item(r, 1).text().strip() if self.used_table.item(r, 1) else ""
                used_rows.append([pn, oc])

            # Helpers to pull markdown/tables, respecting lazy-load state
            def harvest_markdown(name: str, editor: ZoomableMarkdownEdit) -> str:
                txt = editor.markdown_text() if self.section_loaded.get(name) else self.section_texts.get(name, "")
                return ascii_sanitize(txt)

            def harvest_table(table: QTableWidget, cols: int, cache_attr: str, section_name: str) -> list[list[str]]:
                if self.section_loaded.get(section_name):
                    rows = []
                    for rr in range(table.rowCount()):
                        row = []
                        for cc in range(cols):
                            it = table.item(rr, cc)
                            row.append(it.text().strip() if it else "")
                        rows.append(row)
                    setattr(self, cache_attr, rows)
                    return rows
                else:
                    return getattr(self, cache_attr)

            # Section bodies
            netlist = harvest_markdown("Netlist", self.netlist_edit)
            partlist = harvest_markdown("Partlist", self.partlist_edit)

            epsa_txt = harvest_markdown("EPSA", self.epsa_edit)
            wcca_txt = harvest_markdown("WCCA", self.wcca_edit)
            fmea_txt = harvest_markdown("FMEA", self.fmea_edit)

            # Normalize + ensure maturity markers
            epsa_txt = self._ensure_maturity_marker("EPSA",
                        self._normalize_section_body("EPSA", epsa_txt), self.epsa_cmb)
            wcca_txt = self._ensure_maturity_marker("WCCA",
                        self._normalize_section_body("WCCA", wcca_txt), self.wcca_cmb)
            fmea_txt = self._ensure_maturity_marker("FMEA",
                        self._normalize_section_body("FMEA", fmea_txt), self.fmea_cmb)

            # Tables
            pin_rows  = harvest_table(self.pin_table,  len(PIN_HEADERS),  "pin_rows_cache",  "Pin Interface")
            test_rows = harvest_table(self.test_table, len(TEST_HEADERS), "test_rows_cache", "Tests")

            # Final Markdown
            text = self.build_markdown(fields, used_rows, netlist, partlist, pin_rows, test_rows, epsa_txt, wcca_txt, fmea_txt)

            # --- Atomic write (no backups created) ---
            try:
                tmp_path = self.current_path.with_suffix(self.current_path.suffix + f".tmp.{os.getpid()}.{now_stamp()}")
                tmp_path.write_text(text, encoding="utf-8")
                os.replace(str(tmp_path), str(self.current_path))  # atomic on same filesystem
            except Exception as e:
                try:
                    if 'tmp_path' in locals() and tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass
                self.error("Error", f"Failed to save file:\n{e}")
                return

            # Refresh UI + toast
            self.proxy.refresh_desc(self.current_path)
            self.info("Saved", "Catalog entry saved.")
            return

        # --- Save a folder's metadata JSON ---
        if self.current_folder and self.current_folder.exists() and self.current_folder.is_dir():
            meta_p = folder_meta_path(self.current_folder)
            created = today_iso()
            if meta_p.exists():
                try:
                    old = json.loads(meta_p.read_text(encoding="utf-8"))
                    created = old.get("Created", created)
                except Exception:
                    pass

            raw_tags = (self.folder_tags.text() or "").strip()
            tags_list = [t.strip() for t in raw_tags.split(",") if t.strip()]

            meta = {
                "TITLE":        (self.folder_title.text() or "").upper(),
                "DESCRIPTION":  (self.folder_desc.text()  or "").upper(),
                "Summary":      self.folder_summary.toPlainText().strip(),
                "Owner":        (self.folder_owner.text() or "").strip(),
                "Tags":         tags_list,
                "Created":      created,
                "Last Updated": today_iso()
            }

            try:
                meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            except Exception as e:
                self.error("Error", f"Failed to save folder metadata:\n{e}")
                return

            self.proxy.refresh_desc(self.current_folder)
            self.folder_created.setText(meta["Created"])
            self.folder_updated.setText(meta["Last Updated"])
            self.info("Saved", "Folder metadata saved.")
            return

        # --- Nothing selected ---
        self.info("Save", "Select a folder or a Markdown file to save.")

    def build_markdown(self, fields: dict, used_rows: list, netlist: str, partlist_text: str,
                    pin_rows: list, test_rows: list, epsa_text: str, wcca_text: str, fmea_text: str) -> str:
        meta_lines = ["| Field                  | Value                     |","| ---------------------- | ------------------------- |"]
        for key,_ in FIELD_ORDER: meta_lines.append(f"| {key:<22} | {fields.get(key,'').strip()} |")

        used = ["## Used On","","| PN         | Occurrences |","| ---------- | ----------- |"]
        if used_rows:
            for pn,occ in used_rows: used.append(f"| {pn or '(None)'} | {occ or '0'} |")
        else:
            used.append("| (None)     | 0           |")

        net_block  = netlist.strip() if netlist.strip() else "(paste or type your netlist here)"
        part_block = partlist_text.strip() if partlist_text.strip() else "(paste or type your partlist here — raw markdown; tables/lists render)"

        def build_table(headers, rows):
            out = ["| " + " | ".join(headers) + " |",
                "| " + " | ".join("-"*len(h) for h in headers) + " |"]
            for r in rows:
                rr = (r + [""]*len(headers))[:len(headers)]
                out.append("| " + " | ".join(rr) + " |")
            return out

        pin   = ["## Pin Interface",""] + build_table(PIN_HEADERS, pin_rows)
        tests = ["## Tests",""] + build_table(TEST_HEADERS, test_rows)

        # ⬇️ Use the ALREADY-NORMALIZED section bodies passed in
        epsa_body = epsa_text.strip() or maturity_comment(0) + "\n(paste EPSA here)"
        wcca_body = wcca_text.strip() or maturity_comment(0) + "\n(paste WCCA here)"
        fmea_body = fmea_text.strip() or maturity_comment(0) + "\n(paste FMEA here)"

        out = [
            "# Circuit Metadata","",f"**Last Updated:** {today_iso()}","",
            "\n".join(meta_lines),"", "\n".join(used),"",
            "## Netlist","", net_block,"",
            "## Partlist","", part_block,"",
            "\n".join(pin),"", "\n".join(tests),"",
            "## EPSA","",  epsa_body,"",
            "## WCCA","",  wcca_body,"",
            "## FMEA","",  fmea_body,"",
        ]
        return "\n".join(out)

    # ---------- Maturity helpers ----------
    def _write_maturity_marker(self, name: str, editor: ZoomableMarkdownEdit, code: int):
        md = editor.markdown_text() or ""
        cm = maturity_comment(code)
        if "<!-- maturity:" in md:
            md = re.sub(r"<!--\s*maturity:\s*(?:[0-3]|placeholder|immature|mature|locked).*?-->", cm, md, flags=re.I)
        else:
            md = cm + "\n" + md
        editor.set_markdown_text(md)  # refresh

    def _ensure_maturity_marker_code(self, md: str, code: int) -> str:
        """
        Guarantee exactly one <!-- maturity: X ... --> marker at the top of md,
        replacing any existing marker regardless of wording/case.
        """
        cm = maturity_comment(code)
        md = md or ""
        if "<!-- maturity:" in md.lower():
            return re.sub(
                r"<!--\s*maturity:\s*(?:[0-3]|placeholder|immature|mature|locked).*?-->",
                cm,
                md,
                flags=re.I,
            )
        return cm + "\n" + md

    # ---------- AI Assist ----------
    def run_ai(self, mode: str):
        if not (self.current_path and self.is_markdown(self.current_path)):
            self.info("AI Assist", "Open an entry first."); return

        meta = {k: self.field_widgets[k].text().strip() for k,_ in FIELD_ORDER}
        net = ascii_sanitize(self.netlist_edit.markdown_text() if self.section_loaded.get("Netlist") else self.section_texts.get("Netlist",""))
        pl  = ascii_sanitize(self.partlist_edit.markdown_text() if self.section_loaded.get("Partlist") else self.section_texts.get("Partlist",""))
        epsa_text = self.epsa_edit.markdown_text() if self.section_loaded.get("EPSA") else self.section_texts.get("EPSA","")
        wcca_text = self.wcca_edit.markdown_text() if self.section_loaded.get("WCCA") else self.section_texts.get("WCCA","")
        fmea_text = self.fmea_edit.markdown_text() if self.section_loaded.get("FMEA") else self.section_texts.get("FMEA","")

        payload = {
            "request_meta": { "client":"activated-silicon", "task": mode, "respect_locked": True },
            "inputs": {
                "metadata": meta,
                "netlist_md": net,
                "partlist_md": pl,
                "epsa": {"maturity": maturity_read_from_text(epsa_text) or 0, "current_md": ascii_sanitize(epsa_text)},
                "wcca": {"maturity": maturity_read_from_text(wcca_text) or 0, "current_md": ascii_sanitize(wcca_text)},
                "fmea": {"maturity": maturity_read_from_text(fmea_text) or 0, "current_md": ascii_sanitize(fmea_text)},
            },
            "outputs_spec": { "format":"json", "fields":["epsa","wcca","fmea"] }
        }

        # --- start timing + ETA prediction ---
        self._ai_start_ts = datetime.datetime.now()
        folder = self._folder_of_current_file()
        cur_tab = self.tabs.tabText(self.tabs.currentIndex()) if self.tabs.count() else "EPSA"
        doc_type = cur_tab if cur_tab in ("EPSA","WCCA","FMEA") else "EPSA"
        self._eta_target_sec = self._estimate_eta_sec(folder, doc_type) if folder else 75

        self._open_review_drawer_busy()
        self.worker = OpenAIWorker(
            api_key=self.current_api_key or os.environ.get("OPENAI_API_KEY"),
            model_name=self.current_model_name,
            payload=payload,
            timeout=180
        )
        self.worker.finished.connect(self._on_ai_finished)
        self.worker.start()

    def _open_review_drawer_busy(self):
        self.review.setVisible(True)
        # Start/update live header timer immediately
        self._start_eta_ui_timer()
        self.review.rendered.set_markdown_text("_Awaiting response..._")
        self.review.diff_left.set_markdown_text("")
        self.review.diff_right.set_markdown_text("")
        self.review.table_changes.setRowCount(0)

    # ---------- ETA / Timer helpers ----------
    def _folder_of_current_file(self) -> Path | None:
        if self.current_path: return self.current_path.parent
        if self.current_folder: return self.current_folder
        return None

    def _start_eta_ui_timer(self):
        # Refresh running header every 250ms
        if self._eta_timer:
            try: self._eta_timer.stop()
            except Exception: pass
        self._eta_timer = QTimer(self)
        self._eta_timer.timeout.connect(self._tick_eta_ui)
        self._eta_timer.start(250)
        self._tick_eta_ui()

    def _tick_eta_ui(self):
        if not self._ai_start_ts:
            self.review.lbl_title.setText("AI Suggestions (running...)"); return
        elapsed = int((datetime.datetime.now() - self._ai_start_ts).total_seconds())
        eta = int(self._eta_target_sec or 75)
        # Prefer the engineering tab if focused; else fall back to current AI target
        tab = self.tabs.tabText(self.tabs.currentIndex()) if self.tabs.count() else (self.ai_target_section or "EPSA")
        section = tab if tab in ("EPSA","WCCA","FMEA") else (self.ai_target_section or "EPSA")
        def _fmt(sec: int) -> str:
            m, s = divmod(max(0, sec), 60)
            return f"{m:02d}:{s:02d}"
        self.review.lbl_title.setText(f"AI Suggestions — {section} (running… {_fmt(elapsed)} / ETA ≈ {_fmt(eta)})")

    # ---------- Timer stats (folder JSON) ----------
    def _load_timer_stats(self, folder: Path) -> dict:
        meta = self.read_folder_meta(folder)
        ts = meta.get("TimerStats")
        if not ts:
            ts = {
                "version": 1,
                "bin_edges_sec": [30,60,120,300,600,1200,1800,3600,7200],
                "overall": {"runs":0,"hist":[0]*8,"ewma_sec":None,"last_durations_sec":[]},
                "by_doc_type": {k: {"runs":0,"hist":[0]*8,"ewma_sec":None,"last_durations_sec":[]} for k in ["EPSA","WCCA","FMEA"]},
                "updated_at": today_iso()
            }
            meta["TimerStats"] = ts
            (folder / f"{folder.name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return ts

    def _save_timer_stats(self, folder: Path, ts: dict):
        meta = self.read_folder_meta(folder)
        meta["TimerStats"] = ts
        meta["TimerStats"]["updated_at"] = today_iso()
        (folder / f"{folder.name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        # If the stats tab is visible, refresh it live
        try:
            if self.tabs.currentWidget() is self.ai_stats_tab:
                self.refresh_ai_stats()
        except Exception:
            pass

    @staticmethod
    def _bin_index(edges, x):
        for i, edge in enumerate(edges):
            if x <= edge: return i
        return len(edges) - 1

    @staticmethod
    def _percentile(data, p):
        if not data: return None
        d = sorted(data); k = (len(d)-1) * p
        f = int(k); c = min(f+1, len(d)-1)
        if f == c: return d[f]
        return d[f] + (d[c]-d[f]) * (k-f)

    def _estimate_eta_sec(self, folder: Path | None, doc_type: str) -> int:
        if folder is None: return 75
        ts = self._load_timer_stats(folder)
        slot = ts["by_doc_type"].setdefault(doc_type, {"runs":0,"hist":[0]*8,"ewma_sec":None,"last_durations_sec":[]})
        recent = slot.get("last_durations_sec", [])
        if len(recent) >= 3:
            med = self._percentile(recent, 0.5)
            if med is not None: return int(round(med))
        ewma = slot.get("ewma_sec")
        if ewma: return int(round(ewma))
        overall_recent = ts["overall"]["last_durations_sec"]
        if len(overall_recent) >= 3:
            med = self._percentile(overall_recent, 0.5)
            if med is not None: return int(round(med))
        return 75

    def _record_run(self, folder: Path | None, doc_type: str, duration_sec: float, ok: bool = True):
        if folder is None: return
        ts = self._load_timer_stats(folder)
        edges = ts["bin_edges_sec"]
        idx = self._bin_index(edges, duration_sec)
        # ensure slot
        slot = ts["by_doc_type"].setdefault(doc_type, {"runs":0,"hist":[0]*len(edges),"ewma_sec":None,"last_durations_sec":[]})
        # update per-type
        slot["runs"] += 1
        slot["hist"][idx] += 1
        lst = slot["last_durations_sec"]; lst.append(int(duration_sec))
        if len(lst) > 50: del lst[:len(lst)-50]
        # EWMA (α=0.25)
        a = 0.25
        slot["ewma_sec"] = (a * duration_sec + (1-a) * slot["ewma_sec"]) if slot["ewma_sec"] else duration_sec
        # update overall
        ov = ts["overall"]
        ov["runs"] += 1
        ov["hist"][idx] += 1
        ov_lst = ov["last_durations_sec"]; ov_lst.append(int(duration_sec))
        if len(ov_lst) > 50: del ov_lst[:len(ov_lst)-50]
        ov["ewma_sec"] = (a * duration_sec + (1-a) * ov["ewma_sec"]) if ov["ewma_sec"] else duration_sec
        self._save_timer_stats(folder, ts)

    def _on_ai_finished(self, result: dict):
        if "_error" in result:
            self.review.lbl_title.setText("AI Suggestions (error)")
            self.review.rendered.set_markdown_text(f"**Error:** {result['_error']}")
            # still record duration
            try:
                folder = self._folder_of_current_file()
                section = self.ai_target_section or "EPSA"
                if folder and self._ai_start_ts:
                    duration = (datetime.datetime.now() - self._ai_start_ts).total_seconds()
                    self._record_run(folder, section, duration_sec=duration, ok=False)
            finally:
                if self._eta_timer: self._eta_timer.stop()
                self._eta_timer = None; self._ai_start_ts = None
            return
        self.ai_result = result
        # prefer focused tab if EPSA/WCCA/FMEA; else EPSA
        idx = self.tabs.currentIndex(); label = self.tabs.tabText(idx)
        if label not in ("EPSA","WCCA","FMEA"): label = "EPSA"
        self.ai_target_section = label
        self._render_ai_result_for(label)
        # record success + stop timer
        try:
            folder = self._folder_of_current_file()
            section = self.ai_target_section or label or "EPSA"
            if folder and self._ai_start_ts:
                duration = (datetime.datetime.now() - self._ai_start_ts).total_seconds()
                self._record_run(folder, section, duration_sec=duration, ok=True)
        finally:
            if self._eta_timer: self._eta_timer.stop()
            self._eta_timer = None; self._ai_start_ts = None

    def _render_ai_result_for(self, section: str):
        sec = section.lower()
        if not self.ai_result or sec not in self.ai_result:
            self.review.lbl_title.setText("AI Suggestions (empty)")
            self.review.rendered.set_markdown_text("_No suggestions returned._"); return
        data = self.ai_result[sec]
        proposed = data.get("proposed_md","")
        rationale = data.get("rationale","")
        status = data.get("status","suggested"); conf = data.get("confidence", 0.0)

        self.review.lbl_title.setText(f"AI Suggestions — {section} [{status}, confidence {conf:.2f}]")
        md = (proposed or "").strip()
        if rationale: md = f"> **Rationale:** {rationale}\n\n{md}"
        self.review.rendered.set_markdown_text(md or "_(No proposed content)_")

        current = {"EPSA": self.epsa_edit, "WCCA": self.wcca_edit, "FMEA": self.fmea_edit}[section]
        cur_text = current.markdown_text() if self.section_loaded.get(section) else self.section_texts.get(section,"")
        self.review.diff_left.set_markdown_text(cur_text or "")
        self.review.diff_right.set_markdown_text(proposed or "")

        cur_table, cs, ce = first_table_in(cur_text)
        prop_table, ps, pe = first_table_in(proposed)
        self.review.table_changes.setRowCount(0)
        if cur_table and prop_table:
            cur_rows = md_table_to_rows(cur_table)
            prop_rows = md_table_to_rows(prop_table)
            if cur_rows and prop_rows and len(cur_rows[0])==len(prop_rows[0]):
                headers = prop_rows[0]
                cur_map = {tuple(r[:1]): r for r in cur_rows[1:]}
                prop_map = {tuple(r[:1]): r for r in prop_rows[1:]}
                keys = sorted(set(cur_map.keys()) | set(prop_map.keys()), key=lambda k: (str(k[0]).lower(),))
                for key in keys:
                    cr = cur_map.get(key); pr = prop_map.get(key)
                    if cr is None:
                        for ci in range(len(headers)):
                            r = self.review.table_changes.rowCount(); self.review.table_changes.insertRow(r)
                            chk = QTableWidgetItem(); chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled); chk.setCheckState(Qt.Checked)
                            self.review.table_changes.setItem(r,0,chk)
                            self.review.table_changes.setItem(r,1,QTableWidgetItem(key[0]))
                            self.review.table_changes.setItem(r,2,QTableWidgetItem(headers[ci]))
                            self.review.table_changes.setItem(r,3,QTableWidgetItem(""))
                            self.review.table_changes.setItem(r,4,QTableWidgetItem(pr[ci] if pr and ci<len(pr) else ""))
                            self.review.table_changes.setItem(r,5,QTableWidgetItem("added"))
                    elif pr is None:
                        r = self.review.table_changes.rowCount(); self.review.table_changes.insertRow(r)
                        chk = QTableWidgetItem(); chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled); chk.setCheckState(Qt.Unchecked)
                        self.review.table_changes.setItem(r,0,chk)
                        self.review.table_changes.setItem(r,1,QTableWidgetItem(key[0]))
                        self.review.table_changes.setItem(r,2,QTableWidgetItem("(row)"))
                        self.review.table_changes.setItem(r,3,QTableWidgetItem(" | ".join(cr)))
                        self.review.table_changes.setItem(r,4,QTableWidgetItem(""))
                        self.review.table_changes.setItem(r,5,QTableWidgetItem("removed"))
                    else:
                        for ci in range(len(headers)):
                            before = cr[ci] if ci<len(cr) else ""
                            after  = pr[ci] if ci<len(pr) else ""
                            if before != after:
                                r = self.review.table_changes.rowCount(); self.review.table_changes.insertRow(r)
                                chk = QTableWidgetItem(); chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled); chk.setCheckState(Qt.Checked)
                                self.review.table_changes.setItem(r,0,chk)
                                self.review.table_changes.setItem(r,1,QTableWidgetItem(key[0]))
                                self.review.table_changes.setItem(r,2,QTableWidgetItem(headers[ci]))
                                self.review.table_changes.setItem(r,3,QTableWidgetItem(before))
                                self.review.table_changes.setItem(r,4,QTableWidgetItem(after))
                                self.review.table_changes.setItem(r,5,QTableWidgetItem("modified"))

        self.review.setVisible(True); self.review.tabs.setCurrentIndex(0)

    def _discard_ai(self):
        self.review.setVisible(False); self.ai_result=None; self.ai_target_section=None

    def _apply_all_from_ai(self):
        if not self.ai_result or not self.ai_target_section: return
        sec = self.ai_target_section.lower()
        proposed = self.ai_result.get(sec, {}).get("proposed_md","") or ""
        self._apply_section_replace(self.ai_target_section, proposed)

    def _apply_selected_from_ai(self):
        if not self.ai_result or not self.ai_target_section: return
        section = self.ai_target_section
        ed = {"EPSA": self.epsa_edit, "WCCA": self.wcca_edit, "FMEA": self.fmea_edit}[section]
        cur_text = ed.markdown_text() if self.section_loaded.get(section) else self.section_texts.get(section,"")
        proposed = self.ai_result[section.lower()].get("proposed_md","") or ""

        cur_table, cs, ce = first_table_in(cur_text)
        prop_table, ps, pe = first_table_in(proposed)
        if not (cur_table and prop_table and cs>=0 and ce>=0 and ps>=0 and pe>=0):
            self._apply_section_replace(section, proposed); return

        cur_rows = md_table_to_rows(cur_table); prop_rows = md_table_to_rows(prop_table)
        if not(cur_rows and prop_rows) or len(cur_rows[0]) != len(prop_rows[0]):
            self._apply_section_replace(section, proposed); return

        headers = prop_rows[0]
        cur_map = {tuple(r[:1]): r for r in cur_rows[1:]}
        prop_map = {tuple(r[:1]): r for r in prop_rows[1:]}
        accepted = {k: v[:] for k,v in cur_map.items()}

        for r in range(self.review.table_changes.rowCount()):
            chk_item = self.review.table_changes.item(r,0)
            if not chk_item or chk_item.checkState() != Qt.Checked: continue
            key = self.review.table_changes.item(r,1).text()
            col = self.review.table_changes.item(r,2).text()
            before = self.review.table_changes.item(r,3).text()
            after  = self.review.table_changes.item(r,4).text()
            change = self.review.table_changes.item(r,5).text()

            if change == "removed":
                accepted.pop((key,), None); continue
            if change == "added" and col != "(row)":
                if (key,) not in accepted:
                    pr = prop_map.get((key,))
                    if pr: accepted[(key,)] = pr[:]
                    else: accepted[(key,)] = ["" for _ in headers]
                try: ci = headers.index(col)
                except ValueError: continue
                accepted[(key,)][ci] = after
            elif change == "modified":
                if (key,) not in accepted:
                    accepted[(key,)] = ["" for _ in headers]
                try: ci = headers.index(col)
                except ValueError: continue
                accepted[(key,)][ci] = after

        new_rows = [headers] + [accepted[k] for k in sorted(accepted.keys(), key=lambda x: str(x[0]).lower())]
        new_table_md = rows_to_md_table(new_rows)
        lines = cur_text.splitlines(True)
        new_lines = lines[:cs] + [new_table_md + ("\n" if not new_table_md.endswith("\n") else "")] + lines[ce:]
        spliced = "".join(new_lines)
        self._apply_section_replace(section, spliced)

    def _apply_section_replace(self, section: str, new_md: str):
        ed = {"EPSA": self.epsa_edit, "WCCA": self.wcca_edit, "FMEA": self.fmea_edit}[section]
        # Normalize to avoid inner duplicate headings (## EPSA/WCCA/FMEA)
        new_md = self._normalize_section_body(section, new_md)

        cur = ed.markdown_text() if self.section_loaded.get(section) else self.section_texts.get(section,"")
        code = maturity_read_from_text(cur)
        if code is None:
            # sync from combobox if available
            cmb = {"EPSA":self.epsa_cmb,"WCCA":self.wcca_cmb,"FMEA":self.fmea_cmb}[section]
            code = maturity_label_to_code(cmb.currentText())

        # Ensure a single maturity marker at the very top
        cm = maturity_comment(code)
        if "<!-- maturity:" in new_md:
            new = re.sub(r"<!--\s*maturity:\s*(?:[0-3]|placeholder|immature|mature|locked).*?-->", cm, new_md, flags=re.I)
        else:
            new = cm + "\n" + new_md

        ed.set_markdown_text(new)
        self.section_texts[section] = new
        self.section_loaded[section] = True
        self.review.setVisible(False)
        self.info("Applied", f"{section} updated from AI suggestions. Remember to Save.")

    # ---------- Folder meta / FS ops ----------
    def read_folder_meta(self, folder: Path) -> dict:
        meta_p = folder_meta_path(folder)
        if not meta_p.exists():
            meta = {"TITLE":"","DESCRIPTION":"","Summary":"","Owner":"","Tags":[],"Created":today_iso(),"Last Updated":today_iso()}
            try: meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            except Exception: pass
            return meta
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            meta.setdefault("TITLE", meta.get("title","").upper() if meta.get("title") else meta.get("TITLE",""))
            meta.setdefault("DESCRIPTION", meta.get("description","").upper() if meta.get("description") else meta.get("DESCRIPTION",""))
            meta.setdefault("Summary", meta.get("Summary","")); meta.setdefault("Owner", meta.get("Owner",""))
            if "Tags" in meta and isinstance(meta["Tags"], str):
                meta["Tags"] = [t.strip() for t in meta["Tags"].split(",") if t.strip()]
            meta.setdefault("Tags", meta.get("Tags",[])); meta.setdefault("Created", meta.get("Created", today_iso()))
            meta.setdefault("Last Updated", meta.get("Last Updated", today_iso()))
            return meta
        except Exception:
            return {"TITLE":"","DESCRIPTION":"","Summary":"","Owner":"","Tags":[],"Created":today_iso(),"Last Updated":today_iso()}

    def archive_script_folder(self):
        try:
            script_dir = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
        except Exception:
            self.error("Archive","Could not determine script directory."); return
        ts = now_stamp(); temp_base = Path(tempfile.gettempdir()) / ts
        try: shutil.make_archive(str(temp_base),'zip', root_dir=str(script_dir.parent), base_dir=script_dir.name)
        except Exception as e:
            self.error("Archive", f"Failed to create archive:\n{e}"); return
        temp_zip = Path(str(temp_base)+".zip");
        if not temp_zip.exists(): self.error("Archive","Archive creation failed."); return
        dest_zip = script_dir / f"{ts}.zip"
        try:
            if dest_zip.exists(): dest_zip = script_dir / f"{ts}_1.zip"
            shutil.move(str(temp_zip), str(dest_zip))
        except Exception as e:
            self.error("Archive", f"Failed to move archive:\n{e}"); return
        self.info("Archive", f"Created: {dest_zip}")

    def on_fs_file_renamed(self, dir_path_str: str, old_name: str, new_name: str):
        try:
            dir_path = Path(dir_path_str); old_path = dir_path / old_name; new_path = dir_path / new_name
            if new_path.is_dir():
                old_meta = new_path / f"{old_name}.json"; new_meta = new_path / f"{new_name}.json"
                if old_meta.exists():
                    if new_meta.exists():
                        try: old_meta.unlink()
                        except Exception: pass
                    else:
                        try: old_meta.rename(new_meta)
                        except Exception:
                            try:
                                data = old_meta.read_text(encoding="utf-8"); new_meta.write_text(data, encoding="utf-8"); old_meta.unlink()
                            except Exception: pass
                self.proxy.refresh_desc(new_path)
                if self.current_folder and (self.current_folder == old_path or self.current_folder.name == old_name and self.current_folder.parent == dir_path):
                    self.current_folder = new_path; self.path_label.setText(f"Folder: {new_path}")
            else:
                if new_path.suffix.lower()==".md": self.proxy.refresh_desc(new_path)
                if self.current_path and (self.current_path == old_path or (self.current_path.name==old_name and self.current_path.parent==dir_path)):
                    self.current_path = new_path; self.path_label.setText(f"File: {new_path}")
        except Exception:
            pass

    def open_file_location(self):
        path = self.selected_path()
        if not path: self.info("Open Location","Select a folder or file first."); return
        try:
            if platform.system()=="Windows":
                subprocess.run(["explorer", "/select,", str(path.resolve())] if path.is_file() else ["explorer", str(path.resolve())])
            elif platform.system()=="Darwin":
                subprocess.run(["open", "-R", str(path.resolve())] if path.is_file() else ["open", str(path.resolve())])
            else:
                target = str(path.parent.resolve() if path.is_file() else path.resolve()); subprocess.run(["xdg-open", target])
        except Exception as e:
            self.error("Open Location", f"Failed to open location:\n{e}")

    def create_new_folder(self):
        base = self.selected_path() or self.catalog_root
        if base.is_file(): base = base.parent
        name, ok = self.ask_text("New Folder","Folder name:")
        if not ok or not name.strip(): return
        target = base / name.strip()
        try:
            target.mkdir(parents=True, exist_ok=False); self.read_folder_meta(target); self.proxy.refresh_desc(target)
        except FileExistsError:
            self.warn("Exists", "A file/folder with that name already exists.")
        except Exception as e:
            self.error("Error", f"Failed to create folder:\n{e}")

    def create_new_entry(self):
        base = self.selected_path() or self.catalog_root
        if base.is_file(): base = base.parent
        name, ok = self.ask_text("New Entry","File name (without extension):")
        if not ok or not name.strip(): return
        safe = name.strip();
        if not safe.lower().endswith(".md"): safe += ".md"
        target = base / safe
        if target.exists(): self.warn("Exists","A file with that name already exists."); return
        try:
            target.write_text(NEW_ENTRY_TEMPLATE, encoding="utf-8")
        except Exception as e:
            self.error("Error", f"Failed to create file:\n{e}"); return
        self.load_file_lazy(target)
        sidx = self.fs_model.index(str(target))
        if sidx.isValid():
            pidx = self.proxy.mapFromSource(sidx)
            if pidx.isValid(): self.tree.setCurrentIndex(pidx)

    def rename_item(self):
        path = self.selected_path()
        if not path: self.info("Rename","Select a file or folder to rename."); return
        new_name, ok = self.ask_text("Rename","New name:", default=path.name)
        if not ok or not new_name.strip(): return
        new_path = path.parent / new_name.strip()
        if new_path.exists(): self.warn("Exists","Target name already exists."); return
        try:
            if path.is_dir():
                old_folder_name = path.name; path.rename(new_path)
                old_meta = new_path / f"{old_folder_name}.json"; new_meta = new_path / f"{new_path.name}.json"
                if old_meta.exists():
                    if new_meta.exists():
                        try: old_meta.unlink()
                        except Exception: pass
                    else:
                        try: old_meta.rename(new_meta)
                        except Exception:
                            try:
                                data = old_meta.read_text(encoding="utf-8"); new_meta.write_text(data, encoding="utf-8"); old_meta.unlink()
                            except Exception: pass
                self.proxy.refresh_desc(new_path)
                if self.current_folder and self.current_folder==path: self.current_folder=new_path; self.path_label.setText(f"Folder: {new_path}")
            else:
                path.rename(new_path); self.proxy.refresh_desc(new_path)
                if self.current_path and self.current_path==path: self.current_path=new_path; self.path_label.setText(f"File: {new_path}")
        except Exception as e:
            self.error("Error", f"Failed to rename:\n{e}")

    def delete_item(self):
        path = self.selected_path()
        if not path: return
        typ = "folder" if path.is_dir() else "file"
        if not self.ask_yes_no("Delete", f"Delete this {typ}?\n{path}"): return
        try:
            if path.is_dir(): shutil.rmtree(path)
            else: path.unlink()
        except Exception as e:
            self.error("Error", f"Failed to delete:\n{e}"); return
        if self.current_path and self.current_path==path: self.current_path=None; self.path_label.setText("")
        if self.current_folder and self.current_folder==path: self.current_folder=None; self.path_label.setText("")

    def add_used_row(self):
        r=self.used_table.rowCount(); self.used_table.insertRow(r)
        self.used_table.setItem(r,0,QTableWidgetItem("")); self.used_table.setItem(r,1,QTableWidgetItem("1"))
    def remove_used_row(self):
        r=self.used_table.currentRow()
        if r>=0: self.used_table.removeRow(r)

    # ---------- AI Statistics Tab ----------
    def _make_ai_stats_tab(self):
        tab = QWidget(self)
        v = QVBoxLayout(tab); v.setContentsMargins(0,0,0,0); v.setSpacing(8)

        top = QHBoxLayout()
        self.ai_stats_folder_lbl = QLabel("(no folder selected)")
        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self.refresh_ai_stats)
        top.addWidget(QLabel("Folder:"))
        top.addWidget(self.ai_stats_folder_lbl)
        top.addStretch(1)
        top.addWidget(btn_refresh)
        v.addLayout(top)

        # Overall summary table
        self.ai_overall_tbl = QTableWidget(0, 5, tab)
        self.ai_overall_tbl.setHorizontalHeaderLabels(["Metric", "Value", "P50 (s)", "P90 (s)", "Recent (last 10)"])
        self.ai_overall_tbl.verticalHeader().setVisible(False)
        self.ai_overall_tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.ai_overall_tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.ai_overall_tbl.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.ai_overall_tbl.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.ai_overall_tbl.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        v.addWidget(QLabel("Overall"))
        v.addWidget(self.ai_overall_tbl)

        # By-type table
        self.ai_bytype_tbl = QTableWidget(0, 6, tab)
        self.ai_bytype_tbl.setHorizontalHeaderLabels(["Doc Type", "Runs", "EWMA (s)", "P50 (s)", "P90 (s)", "Recent (last 10)"])
        self.ai_bytype_tbl.verticalHeader().setVisible(False)
        for c in range(5):
            self.ai_bytype_tbl.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.ai_bytype_tbl.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        v.addWidget(QLabel("By Document Type"))
        v.addWidget(self.ai_bytype_tbl)

        # Histogram table
        self.ai_hist_tbl = QTableWidget(0, 3, tab)
        self.ai_hist_tbl.setHorizontalHeaderLabels(["Bin (s)", "Overall Count", "Notes"])
        self.ai_hist_tbl.verticalHeader().setVisible(False)
        self.ai_hist_tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.ai_hist_tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.ai_hist_tbl.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        v.addWidget(QLabel("Histogram (Overall)"))
        v.addWidget(self.ai_hist_tbl)

        return tab

    def _fmt_recent(self, arr):
        if not arr: return ""
        return ", ".join(str(x) for x in arr[-10:])

    def _fmt_float(self, x, nd=1):
        return "" if x is None else f"{float(x):.{nd}f}"

    def _fmt_bins(self, edges):
        # Build inclusive bins like <=30, 31-60, 61-120, ..., >last
        if not edges:
            return ["(no bins)"]
        out = []
        prev = 0
        for e in edges[:-1]:
            out.append(f"{prev+1}-{e}")
            prev = e
        out.insert(0, f"<= {edges[0]}")
        out.append(f">{edges[-1]}")
        return out

    def refresh_ai_stats(self):
        folder = self._folder_of_current_file()
        if not folder:
            self.ai_stats_folder_lbl.setText("(no folder selected)")
            self.ai_overall_tbl.setRowCount(0)
            self.ai_bytype_tbl.setRowCount(0)
            self.ai_hist_tbl.setRowCount(0)
            return

        self.ai_stats_folder_lbl.setText(str(folder))
        ts = self._load_timer_stats(folder)  # creates skeleton if missing

        # Overall table
        ov = ts.get("overall", {})
        edges = ts.get("bin_edges_sec", [30,60,120,300,600,1200,1800,3600,7200])
        p50 = self._percentile(ov.get("last_durations_sec", []), 0.5)
        p90 = self._percentile(ov.get("last_durations_sec", []), 0.9)
        self.ai_overall_tbl.setRowCount(0)
        def add_overall(metric, value, p50, p90, recent):
            r = self.ai_overall_tbl.rowCount(); self.ai_overall_tbl.insertRow(r)
            self.ai_overall_tbl.setItem(r, 0, QTableWidgetItem(metric))
            self.ai_overall_tbl.setItem(r, 1, QTableWidgetItem(str(value)))
            self.ai_overall_tbl.setItem(r, 2, QTableWidgetItem("" if p50 is None else str(int(p50))))
            self.ai_overall_tbl.setItem(r, 3, QTableWidgetItem("" if p90 is None else str(int(p90))))
            self.ai_overall_tbl.setItem(r, 4, QTableWidgetItem(self._fmt_recent(recent)))

        add_overall("Runs", ov.get("runs", 0), p50, p90, ov.get("last_durations_sec", []))
        add_overall("EWMA (s)", self._fmt_float(ov.get("ewma_sec"), 1), p50, p90, ov.get("last_durations_sec", []))

        # By type
        self.ai_bytype_tbl.setRowCount(0)
        by = ts.get("by_doc_type", {})
        for doc in ("EPSA", "WCCA", "FMEA"):
            slot = by.get(doc, {})
            rp50 = self._percentile(slot.get("last_durations_sec", []), 0.5)
            rp90 = self._percentile(slot.get("last_durations_sec", []), 0.9)
            r = self.ai_bytype_tbl.rowCount(); self.ai_bytype_tbl.insertRow(r)
            self.ai_bytype_tbl.setItem(r, 0, QTableWidgetItem(doc))
            self.ai_bytype_tbl.setItem(r, 1, QTableWidgetItem(str(slot.get("runs", 0))))
            self.ai_bytype_tbl.setItem(r, 2, QTableWidgetItem(self._fmt_float(slot.get("ewma_sec"), 1)))
            self.ai_bytype_tbl.setItem(r, 3, QTableWidgetItem("" if rp50 is None else str(int(rp50))))
            self.ai_bytype_tbl.setItem(r, 4, QTableWidgetItem("" if rp90 is None else str(int(rp90))))
            self.ai_bytype_tbl.setItem(r, 5, QTableWidgetItem(self._fmt_recent(slot.get("last_durations_sec", []))))

        # Histogram
        self.ai_hist_tbl.setRowCount(0)
        labels = self._fmt_bins(edges)
        hist = (ov.get("hist") or [0] * (len(labels)))[:len(labels)]
        # If hist shorter (older file), pad
        if len(hist) < len(labels):
            hist = hist + [0] * (len(labels) - len(hist))
        for i, lab in enumerate(labels):
            r = self.ai_hist_tbl.rowCount(); self.ai_hist_tbl.insertRow(r)
            self.ai_hist_tbl.setItem(r, 0, QTableWidgetItem(lab))
            self.ai_hist_tbl.setItem(r, 1, QTableWidgetItem(str(hist[i])))
            note = ""
            if i == 0:
                note = f"≤ {edges[0]}s"
            elif i == len(labels) - 1:
                note = f"> {edges[-1]}s"
            else:
                note = f"{lab} s"
            self.ai_hist_tbl.setItem(r, 2, QTableWidgetItem(note))

# ---------- App boot ----------
def ensure_catalog_root(start_dir: Path | None = None) -> Path:
    root = DEFAULT_CATALOG_DIR if start_dir is None else start_dir
    if not root.exists():
        try: root.mkdir(parents=True, exist_ok=True)
        except Exception: pass
    if not root.exists() or not root.is_dir():
        dlg = QFileDialog(); dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.ShowDirsOnly, True); dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        dlg.setWindowTitle("Select Root Folder"); apply_windows_dark_titlebar(dlg)
        if dlg.exec_():
            sel = dlg.selectedFiles()
            if sel:
                root = Path(sel[0]); root.mkdir(parents=True, exist_ok=True)
    return root

def main():
    app = QApplication(sys.argv); app.setStyle(QStyleFactory.create("Fusion"))
    icon = make_emoji_icon("💠", px=256); app.setWindowIcon(icon)
    root = ensure_catalog_root(); win = CatalogWindow(root, icon)
    win.show(); apply_windows_dark_titlebar(win)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
