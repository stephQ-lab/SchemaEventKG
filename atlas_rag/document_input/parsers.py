import hashlib
import json
import re
from importlib.metadata import version
from pathlib import Path

from atlas_rag.document_input.structure import finalize_document


def parser_versions():
    return {name: version(name) for name in ("docling", "docling-core", "python-docx", "python-pptx")}


def native_parser_signature():
    signature = hashlib.sha256()
    for name in ("parsers.py", "office_charts.py", "structure.py", "table_processing.py"):
        signature.update(Path(__file__).with_name(name).read_bytes())
    for name in ("python-docx", "python-pptx", "ssf"):
        signature.update((name + version(name)).encode("utf-8"))
    return signature.hexdigest()


def new_document(doc_id, title, source, source_hash):
    return {"id": doc_id, "title": title, "source_file": str(source),
            "source_sha256": source_hash, "parser_versions": parser_versions(),
            "blocks": [], "issues": []}


def append_block(document, kind, text="", **kwargs):
    block = {"id": "%s:b%d" % (document["id"], len(document["blocks"]) + 1),
             "kind": kind, "text": text, **kwargs}
    document["blocks"].append(block)
    return block


def native_table(document, rows, section, **kwargs):
    from atlas_rag.document_input.table_processing import is_header

    rows = [[str(cell or "").strip() for cell in row] for row in rows]
    if not rows or not any(any(row) for row in rows):
        return
    block = append_block(document, "table", rows=rows, header_rows=int(is_header(rows[0])),
                         section=section, caption=section, **kwargs)
    document["issues"].append({"kind": "native_table_header_inferred" if block["header_rows"] else "native_table_header_unverified",
                               "block_id": block["id"], "reason": "原生表格保留完整行列，表头按内容规则识别或标记为待确认"})
    return block


def parse_docx(path, doc_id, title, source_hash, media_dir):
    from docx import Document
    from docx.opc.constants import CONTENT_TYPE as CT
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    from pptx.chart.chart import Chart
    from pptx.oxml import parse_xml
    from atlas_rag.document_input.office_charts import chart_tables

    document = new_document(doc_id, title, path, source_hash)
    document["native_parser_version"] = 5
    document["native_parser_sha256"] = native_parser_signature()
    native = Document(path)
    section = ""
    parsed_charts = set()

    def read_items(items, parent=""):
        nonlocal section
        items = list(items)
        for number, item in enumerate(items, 1):
            locator = "%s/%d" % (parent, number)
            if isinstance(item, Paragraph):
                text = item.text.strip()
                for relationship in item._p.xpath(".//*[local-name()='chart']/@r:id"):
                    part = native.part.related_parts[relationship]
                    chart = Chart(parse_xml(part.blob), part)
                    descriptions = [neighbor.text.strip() for neighbor in items[max(0, number - 3):number + 2]
                                    if isinstance(neighbor, Paragraph) and re.match(r"^(?:图表?|表|Figure)\s*\d+", neighbor.text.strip())]
                    caption = chart.chart_title.text_frame.text if chart.has_title and chart.chart_title.has_text_frame else "；".join(descriptions) or section
                    for payload in chart_tables(chart):
                        block = append_block(document, "table", caption=caption, section=section, page=None, locator=locator,
                                             chart_part=str(part.partname), caption_sources=descriptions, **payload)
                        if payload.get("category_verified") is False:
                            document["issues"].append({"kind": "chart_category_unverified", "block_id": block["id"],
                                                       "reason": "图表原生缓存没有横轴标签，数值序列保留，时间与统计范围需要核对"})
                    parsed_charts.add(str(part.partname))
                if not text:
                    continue
                style = item.style.name if item.style else ""
                heading = style.startswith(("Heading", "标题")) or style == "Title"
                if heading:
                    section = text
                append_block(document, "section_header" if heading else "text", text,
                             section=section, locator=locator, page=None)
            elif isinstance(item, Table):
                block = native_table(document, [[cell.text for cell in row.cells] for row in item.rows],
                                     section, locator=locator, page=None)
                if block is not None:
                    first = item.rows[0]
                    # python-docx提供原生XML对象及XPath，保留文件中明确的表头标记。
                    marked = bool(first._tr.xpath("./w:trPr/w:tblHeader"))
                    if marked:
                        block["header_rows"] = 1
                        document["issues"] = [issue for issue in document["issues"] if issue.get("block_id") != block["id"]]
                    block["cells"] = [{"row": r + 1, "column": c + 1, "text": cell.text,
                                       "grid_span": cell.grid_span, "is_merged": cell.grid_span > 1}
                                      for r, row in enumerate(item.rows) for c, cell in enumerate(row.cells)]
                seen = set()
                for row in item.rows:
                    for cell in row.cells:
                        if cell._tc in seen:
                            continue
                        seen.add(cell._tc)
                        for nested in cell.tables:
                            read_items([nested], locator + "/nested")

    read_items(native.iter_inner_content())
    media_dir.mkdir(parents=True, exist_ok=True)
    for part in native.part.package.parts:
        if str(part.partname).startswith("/word/media/"):
            media_hash = hashlib.sha256(part.blob).hexdigest()
            target = media_dir / (media_hash[:20] + Path(str(part.partname)).suffix)
            target.write_bytes(part.blob)
            document["issues"].append({"kind": "embedded_image_requires_review", "asset": target.name,
                                       "part": str(part.partname), "reason": "图片内容未作为结构化数值输入"})
        elif part.content_type == CT.DML_CHART and str(part.partname) not in parsed_charts:
            document["issues"].append({"kind": "word_chart_requires_review", "part": str(part.partname),
                                       "reason": "Word图表数据需单独核对，正文与表格已保留"})
    return finalize_document(document)


def parse_pptx(path, doc_id, title, source_hash, media_dir):
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    from atlas_rag.document_input.office_charts import chart_tables

    document = new_document(doc_id, title, path, source_hash)
    document["native_parser_version"] = 5
    document["native_parser_sha256"] = native_parser_signature()
    deck = Presentation(path)
    media_dir.mkdir(parents=True, exist_ok=True)

    def nearby_descriptions(shape, shapes):
        candidates = []
        for other in shapes:
            if other is shape or not other.has_text_frame or not other.text.strip() or len(other.text) > 150:
                continue
            overlap = max(0, min(shape.left + shape.width, other.left + other.width) - max(shape.left, other.left))
            gap = max(0, shape.top - other.top - other.height)
            if (other.top <= shape.top + 254000 and other.top + other.height <= shape.top + 381000
                    and gap <= min(914400, max(254000, shape.height // 4))
                    and overlap >= min(shape.width, other.width) * 0.5):
                candidates.append((gap, other))
        selected = [other for _, other in sorted(candidates, key=lambda item: (item[0], item[1].top))[:2]]
        return [{"text": other.text.strip(), "shape_id": other.shape_id,
                 "bbox": [other.left, other.top, other.left + other.width, other.top + other.height]}
                for other in sorted(selected, key=lambda item: (item.top, item.left))]

    def read_shapes(shapes, page, section, parent_locator=""):
        for shape in sorted(shapes, key=lambda item: (item.top, item.left)):
            bbox = [shape.left, shape.top, shape.left + shape.width, shape.top + shape.height]
            locator = "%s/shape%d" % (parent_locator or "slide%d" % page, shape.shape_id)
            info = {"page": page, "bbox": bbox, "coordinate_system": "EMU_GROUP" if parent_locator else "EMU", "section": section, "locator": locator}
            descriptions = nearby_descriptions(shape, shapes) if (shape.shape_type == MSO_SHAPE_TYPE.GROUP or shape.has_table or shape.has_chart) else []
            description = "；".join(item["text"] for item in descriptions)
            if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                read_shapes(shape.shapes, page, description or section, locator)
            elif shape.has_table:
                block = native_table(document, [[cell.text for cell in row.cells] for row in shape.table.rows], section,
                                     **{key: value for key, value in info.items() if key != "section"})
                if block is not None:
                    if shape.table.first_row:
                        block["header_rows"] = 1
                        document["issues"] = [issue for issue in document["issues"] if issue.get("block_id") != block["id"]]
                    block["caption"] = description or section
                    block["caption_sources"] = descriptions
            elif shape.has_chart:
                chart = shape.chart
                caption = chart.chart_title.text_frame.text if chart.has_title and chart.chart_title.has_text_frame else description or section
                for payload in chart_tables(chart):
                    block = append_block(document, "table", caption=caption, caption_sources=descriptions, **payload, **info)
                    if payload.get("category_verified") is False:
                        document["issues"].append({"kind": "chart_category_unverified", "block_id": block["id"],
                                                   "reason": "图表原生缓存没有横轴标签，数值序列保留，时间与统计范围需要核对"})
            elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                image = shape.image
                target = media_dir / (hashlib.sha256(image.blob).hexdigest()[:20] + "." + image.ext)
                target.write_bytes(image.blob)
                append_block(document, "picture", asset=target.name, **info)
                document["issues"].append({"kind": "embedded_image_requires_review", "asset": target.name, **info,
                                           "reason": "图片图表保留原件，图片内数值没有自动转写"})
            elif shape.has_text_frame and shape.text.strip():
                append_block(document, "text", shape.text.strip(), **info)

    for number, slide in enumerate(deck.slides, 1):
        section = slide.shapes.title.text.strip() if slide.shapes.title is not None else ""
        read_shapes(slide.shapes, number, section)
        if slide.has_notes_slide:
            note = slide.notes_slide.notes_text_frame.text.strip()
            if note:
                append_block(document, "speaker_note", note, page=number, section=section)
    document["page_count"] = len(deck.slides)
    return finalize_document(document)


class LayoutParser:
    def __init__(self, model_dir=None, threads=4):
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.object_detection_engine_options import OnnxRuntimeObjectDetectionEngineOptions
        from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions, TableFormerMode
        from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

        options = PdfPipelineOptions()
        options.layout_options.engine_options = OnnxRuntimeObjectDetectionEngineOptions()
        options.accelerator_options = AcceleratorOptions(num_threads=threads, device=AcceleratorDevice.CPU)
        options.do_ocr = False
        options.ocr_options = RapidOcrOptions(lang=["ch"], backend="onnxruntime")
        options.do_table_structure = False
        options.table_structure_options.mode = TableFormerMode.FAST
        options.table_structure_options.do_cell_matching = True
        options.generate_picture_images = True
        options.images_scale = 1.5
        if model_dir:
            options.artifacts_path = Path(model_dir).resolve()
        image_options = options.model_copy(deep=True)
        image_options.do_ocr = True
        image_options.do_table_structure = True
        self.converter = DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=image_options),
        })
        self.scan_converter = DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=image_options),
        })

    def parse(self, path, doc_id, title, source_hash, artifact_dir):
        from docling.datamodel.base_models import ConversionStatus
        from docling_core.types.doc import ContentLayer, PictureItem, TableItem

        document = new_document(doc_id, title, path, source_hash)
        document["table_mode"] = "fast"
        document["layout_engine"] = "onnxruntime"
        converter = self.converter
        if Path(path).suffix.lower() == ".pdf":
            import pdfplumber

            with pdfplumber.open(path) as pdf:
                scanned = sum(len((page.extract_text() or "").strip()) < 50 for page in pdf.pages) > len(pdf.pages) / 2
            if scanned:
                converter = self.scan_converter
            else:
                document["table_mode"] = "native_pdf"
        result = converter.convert(Path(path), raises_on_error=True)
        if result.status != ConversionStatus.SUCCESS:
            raise RuntimeError("Docling未完整解析文档：%s；%s" % (path, result.errors))
        native = result.document
        artifact_dir.mkdir(parents=True, exist_ok=True)
        native.save_as_json(artifact_dir / "docling.json")
        section = ""
        for item, _ in native.iterate_items(included_content_layers=set(ContentLayer)):
            kind = item.label.value
            prov = [{"page": entry.page_no, "bbox": entry.bbox.model_dump(mode="json"),
                     "charspan": entry.charspan} for entry in item.prov]
            first = prov[0] if prov else {}
            info = {"page": first.get("page"), "bbox": first.get("bbox"), "provenance": prov,
                    "section": section, "native_ref": item.self_ref}
            if isinstance(item, TableItem):
                grid = item.data.grid
                rows = [[cell.text for cell in row] for row in grid]
                header_count = 0
                for row in grid:
                    if any(cell.column_header for cell in row):
                        header_count += 1
                    else:
                        break
                block = append_block(document, "table", rows=rows, header_rows=header_count,
                                     cells=[cell.model_dump(mode="json") for cell in item.data.table_cells],
                                     caption=item.caption_text(native) or section, **info)
                if not header_count:
                    document["issues"].append({"kind": "table_header_unverified", "block_id": block["id"], **info})
            elif isinstance(item, PictureItem):
                image = item.get_image(native)
                filename = "picture_%d.png" % (len(document["blocks"]) + 1)
                if image is not None:
                    image.save(artifact_dir / filename)
                append_block(document, "picture", item.caption_text(native), asset=filename, **info)
                document["issues"].append({"kind": "picture_requires_review", "asset": filename, **info,
                                           "reason": "图表图片保留坐标和原件，没有推算曲线数值"})
            elif hasattr(item, "text"):
                text = item.text.strip()
                if kind in ("section_header", "title"):
                    section = text
                    info["section"] = section
                append_block(document, kind, text, **info)
        document["page_count"] = len(native.pages)
        document["confidence"] = json.loads(result.confidence.model_dump_json())
        return finalize_document(document, allow_empty=True)
