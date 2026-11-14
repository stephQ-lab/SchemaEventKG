#!/usr/bin/env python3
"""
Knowledge Graph Extraction Pipeline
Extracts entities, relations, and events from text data using transformer models.
"""
import re
import json
import os
import argparse
from datetime import datetime
import time
from typing import List, Dict, Any, Tuple
from pathlib import Path
import torch
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
import json_repair
import copy

from atlas_rag.llm_generator import LLMGenerator
from atlas_rag.kg_construction.utils.json_processing.json_to_csv import json2csv
from atlas_rag.kg_construction.concept_generation import generate_concept
from atlas_rag.kg_construction.utils.csv_processing.merge_csv import merge_csv_files
from atlas_rag.kg_construction.utils.csv_processing.csv_to_graphml import csvs_to_graphml, csvs_to_temp_graphml
from atlas_rag.kg_construction.concept_to_csv import all_concept_triples_csv_to_csv
from atlas_rag.kg_construction.utils.csv_processing.csv_add_numeric_id import add_csv_columns
from atlas_rag.kg_construction.triple_config import ProcessingConfig
from atlas_rag.llm_generator.prompt.triple_extraction_prompt import TRIPLE_INSTRUCTIONS
from atlas_rag.llm_generator.format.validate_json_schema import ATLAS_SCHEMA
from atlas_rag.kg_construction.document_view import document_title, prompt_view, split_paragraphs
from atlas_rag.document_input.structure import document_prompt_context, load_structured_sample, structured_prompt
 
# Prompt instructions and schema, will be modified during the initialization of KnowledgeGraphExtractor according to the config if provided

class TextChunker:
    """Handles text chunking based on character limits."""
    
    def __init__(self, config: ProcessingConfig):
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
    
    def split_text(self, text: str) -> List[str]:
        if not text:
            return []
        if getattr(self, "preserve_document_context", False):
            return split_paragraphs(text, self.chunk_size)

        # 1. Split text into words first to avoid cutting inside a word
        words = text.split(' ')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            # +1 for the space we removed during split
            word_len = len(word) + 1 
            
            # If adding this word exceeds chunk size, save the current chunk
            if current_length + word_len > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    
                    # LOGIC FOR OVERLAP:
                    # Keep the last N words that fit within the overlap limit
                    # This is harder to do perfectly with simple loops.
                    # Usually, we start a new chunk with the current word.
                    
                    # Simplified overlap approach (Keep last X words roughly):
                    overlap_len = 0
                    new_start_index = len(current_chunk)
                    
                    # Walk backwards to find how many words fit in the overlap
                    while new_start_index > 0 and overlap_len < self.chunk_overlap:
                        new_start_index -= 1
                        overlap_len += len(current_chunk[new_start_index]) + 1
                    
                    current_chunk = current_chunk[new_start_index:]
                    current_length = overlap_len
            
            current_chunk.append(word)
            current_length += word_len
            
        # Add the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

class DatasetProcessor:
    """Processes and prepares dataset for knowledge graph extraction."""
    
    def __init__(self, config: ProcessingConfig, prompt_instructions: Dict, result_schema: Dict):
        self.config = config
        self.chunker = TextChunker(config)
        self.chunker.preserve_document_context = config.preserve_document_context
        self.benchmark = config.benchmark
        self.deduplicate_text = config.deduplicate_text
        self.prompt_instructions = prompt_instructions
        self.result_schema = result_schema

    def filter_language_content(self, sample: Dict[str, Any]) -> bool:
        """Check if content is in English."""
        metadata = sample.get("metadata", {})
        language = metadata.get("lang", "en")  # Default to English if not specified
        supported_languages = list(self.prompt_instructions.keys())
        return language in supported_languages
    
    
    def create_sample_chunks(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks from a single sample."""
        structured = load_structured_sample(sample, self.config.data_directory, self.config.chunk_size)
        if structured is not None:
            if self.config.remove_doc_spaces:
                raise ValueError("结构化输入需要保留原文字符位置")
            document, pieces = structured
            context = document_prompt_context(document)
            result = []
            for piece in pieces:
                metadata = dict(sample["metadata"])
                metadata["structure_units"] = json.dumps(piece["units"], ensure_ascii=False)
                metadata["structure_chunk_index"] = piece["index"]
                metadata["document_title"] = document["title"]
                result.append({"id": sample["id"], "text": piece["text"],
                               "prompt_text": structured_prompt(document, piece, len(pieces), context),
                               "source_text": document["text"], "chunk_id": piece["index"],
                               "metadata": metadata})
            return result
        original_text = sample.get("text", "")
        if self.config.remove_doc_spaces:
            original_text = re.sub(r'\s+', ' ',original_text).strip()
        text_chunks = self.chunker.split_text(original_text)
        chunks = []
        self.text_set = set()  # For deduplication
        total = len(text_chunks)
        for chunk_idx, chunk_text in enumerate(text_chunks):
            if self.deduplicate_text:
                if chunk_text in self.text_set:
                    continue
                self.text_set.add(chunk_text)
            shown = chunk_text
            if self.config.preserve_document_context:
                shown = prompt_view(original_text, chunk_text, chunk_idx + 1, total)
            chunk_data = {
                "id": sample["id"],
                "text": chunk_text,
                "prompt_text": shown,
                "source_text": original_text,
                "chunk_id": chunk_idx,
                "metadata": sample["metadata"]
            }
            chunks.append(chunk_data)
            
        return chunks
    
    def prepare_dataset(self, raw_dataset) -> List[Dict[str, Any]]:
        """Process raw dataset into chunks suitable for processing with generalized slicing."""
        
        processed_samples = []
        if self.benchmark:
            raw_dataset = raw_dataset.select(range(min(100, len(raw_dataset))))
        total_texts = len(raw_dataset)
        # Handle edge cases
        if total_texts == 0:
            print(f"No texts found for shard {self.config.current_shard_triple+1}/{self.config.total_shards_triple}")
            return processed_samples
        
        # Calculate base and remainder for fair distribution
        base_texts_per_shard = total_texts // self.config.total_shards_triple
        remainder = total_texts % self.config.total_shards_triple
        
        # Calculate start index
        if self.config.current_shard_triple < remainder:
            start_idx = self.config.current_shard_triple * (base_texts_per_shard + 1)
        else:
            start_idx = remainder * (base_texts_per_shard + 1) + (self.config.current_shard_triple - remainder) * base_texts_per_shard
        
        # Calculate end index
        if self.config.current_shard_triple < remainder:
            end_idx = start_idx + (base_texts_per_shard + 1)
        else:
            end_idx = start_idx + base_texts_per_shard
        
        # Ensure indices are within bounds
        start_idx = min(start_idx, total_texts)
        end_idx = min(end_idx, total_texts)
        
        print(f"Processing shard {self.config.current_shard_triple+1}/{self.config.total_shards_triple} "
            f"(texts {start_idx}-{end_idx-1} of {total_texts}, {end_idx - start_idx} documents)")
        
        # Process documents in assigned shard
        for idx in range(start_idx, end_idx):
            sample = raw_dataset[idx]
            
            # Filter by language
            if not self.filter_language_content(sample):
                print(f"Unsupported language in sample {idx}, skipping.")
                continue
                
            # Create chunks
            chunks = self.create_sample_chunks(sample)
            processed_samples.extend(chunks)
            
            # Debug mode early termination
            if self.config.debug_mode and len(processed_samples) >= 20:
                print("Debug mode: Stopping at 20 chunks")
                break
        
        print(f"Generated {len(processed_samples)} chunks for shard {self.config.current_shard_triple+1}/{self.config.total_shards_triple}")
        return processed_samples

class CustomDataLoader:
    """Custom data loader for knowledge graph extraction."""
    
    def __init__(self, dataset, processor: DatasetProcessor, prompt_instructions: Dict, result_schema: Dict):
        self.raw_dataset = dataset
        self.processor = processor
        self.processed_data = processor.prepare_dataset(dataset)
        self.prompt_instructions = prompt_instructions
        self.result_schema = result_schema
        self.stage_to_prompt_dict = {
            "stage_1": "entity_relation",
            "stage_2": "event_entity",
            "stage_3": "event_relation"
        }

    def __len__(self) -> int:
        return len(self.processed_data)
    
    def create_batch_instructions(self, batch_data: List[Dict[str, Any]]) -> List[str]:
        messages_dict = {key: [] for key in self.result_schema.keys()}
        for item in batch_data:
            # get language
            language = item.get("metadata",{}).get("lang", "en")
            system_msg = self.prompt_instructions.get(language, self.prompt_instructions["en"])['system'] 

            for key in self.result_schema.keys():
                passage = item.get("prompt_text") or item["text"]
                stage_msg = self.prompt_instructions.get(language, self.prompt_instructions["en"])[key] + '\n' + passage
                messages_dict[key].append([
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": stage_msg}
                ])
        
        return messages_dict
    
    def __iter__(self):
        """Iterate through batches."""
        batch_size = self.processor.config.batch_size_triple
        start_idx = self.processor.config.resume_from * batch_size
        
        for i in tqdm(range(start_idx, len(self.processed_data), batch_size)):
            batch_data = self.processed_data[i:i + batch_size]
            
            # Prepare instructions
            instructions = self.create_batch_instructions(batch_data)
            
            # Extract batch information
            batch_ids = [item["id"] for item in batch_data]
            batch_metadata = [item["metadata"] for item in batch_data]
            batch_texts = [item["text"] for item in batch_data]
            
            yield instructions, batch_ids, batch_texts, batch_metadata

class OutputParser:
    """Parses model outputs and extracts structured data."""
    def __init__(self):
        pass

    def extract_structured_data(self, outputs: List[str]) -> List[List[Dict[str, Any]]]:
        """Extract structured data from model outputs."""
        results = []
        for output in outputs:
            try:
                parsed_data = json_repair.loads(output) if isinstance(output, str) and output.strip() else []
            except Exception as exc:
                print(f"Could not parse model output: {exc}")
                parsed_data = []
            if not isinstance(parsed_data, list):
                parsed_data = []
            results.append(parsed_data)
            
        return results

class KnowledgeGraphExtractor:
    """Main class for knowledge graph extraction pipeline."""
    
    def __init__(self, model:LLMGenerator, config: ProcessingConfig):
        self.config = config
        self.model = model
        self.model_name = model.model_name
        self.parser = OutputParser()        

        # Create instance-specific copies of defaults
        self.prompt_instructions = copy.deepcopy(TRIPLE_INSTRUCTIONS)
        self.result_schema = copy.deepcopy(ATLAS_SCHEMA)
        
        # check both prompt and schema is provided if one of them is provided
        self.custom_prompt = None
        self.custom_schema = None
        if self.config.triple_extraction_prompt_path is not None:
            assert self.config.triple_extraction_schema_path is not None, "Schema path must be provided if prompt path is provided"
            with open(self.config.triple_extraction_prompt_path, 'r') as f:
                self.custom_prompt = json_repair.loads(f.read())
            self.prompt_instructions = self.custom_prompt
            print(f"Using custom kg extraction prompt:\n{self.custom_prompt}")
        if self.config.triple_extraction_schema_path is not None:
            assert self.config.triple_extraction_prompt_path is not None, "Prompt path must be provided if schema path is provided"
            with open(self.config.triple_extraction_schema_path, 'r') as f:
                self.custom_schema = json_repair.loads(f.read())
            self.result_schema = self.custom_schema 
            print(f"Using custom kg extraction schema:\n{self.custom_schema}")

            # validate one-to-one and one-to-many triple schema
            for triple_schema in self.custom_schema.values():
                print(triple_schema)
                triples = triple_schema['items']['properties']
                # check for one-to-one or one-to-many
                one_to_many = any(prop.get('type') == "array" for prop in triples.values())
                one_to_one = not one_to_many
                if one_to_many:
                    continue
                # check relation exist
                assert "Relation" in list(triples.keys()) or "relation" in list(triples.keys()), "Relation field must be present in triple schema for one-to-one triple"
                if one_to_one:
                    assert len(triples) == 3, "One-to-one triple schema must have exactly three fields: subject, relation, object"
        if self.config.preserve_document_context:
            self._add_document_constraints()

    def _add_document_constraints(self):
        """Keep the original prompts, and add the commodity-task constraints."""
        zh = self.prompt_instructions.get("zh-CN")
        if not zh:
            return
        zh["entity_relation"] += (
            "\n补充约束：实体要能在原文中找到，并且是具体的事物：品种、企业、地点、装置、指标。不要把具体数字写成实体。"
            "例如不要出现「108.1美元/吨」「17114.72万吨」「环比上涨1.75美元」。价格、库存、产量的对比写成趋势：上涨、下跌、累库、去库、增加、减少。"
            "时间可以保留：本周、本期、下期、昨日、5月、截至5月21日。"
            "同一事物的不同说法、不同断句、从长句截出的短词，只留一个实体，用原文里最完整的那个说法。"
            "不要把不同规格写成同一个实体。"
            "企业写全称；原文只有简称时，简称和全称分两条，不要猜全称。"
            "「属于」只用于品种到品类，例如 PB粉属于铁矿石、电石法PVC属于PVC。"
            "电石、乙烯、螺纹钢若原文写的是原料、产出或消耗，关系就用「原料」「产出」或「消耗」，不要写成属于品类。"
            "地点保持原文，不要把「华南」写成「广东」。"
            "全文品类列表只用于消解指代，不能把列表里有、但本段原文没有的名称写成事实。"
        )
        zh["event_entity"] += (
            "\n补充约束：事件句写时间、对象和趋势，不写具体数值和单位。例如「本周47港铁矿石港口库存下降」，不要写成「环比去库75万吨至17114万吨」。"
            "不要把数值改写成「高位」「持稳」这类原文没有的程度词；趋势用上涨、下跌、增加、减少、累库、去库。"
            "地点、企业、装置名称保持原文。相对时间保留「昨日」「本周」「下周」，不要改成研报发布日期。"
            "原文是预计、将要发生或未来的事，事件文字里保留「预计」或「将」，不要写成已经发生。"
            "同一事实只写一次。措辞不同、断句不同、从长句截出一截，都不要再写一条。"
            "逐项识别供应、需求、库存、成本、利润、价格、政策与预测事件；一条事件只表达一个主体的一项变化。"
            "原因与结果分别写成事件，主体或指标不同就分别保留；禁止把多项变化压成一个笼统句子。"
            "表格里的一行变化写成一个带趋势的事件，不要把单元格里的数字单独写成实体。"
        )
        zh["event_relation"] += (
            "\n补充约束：事件和事件只用这些关系，方向按下面的头和尾。"
            "「导致」：头是原因，尾是结果，原文同一句里必须有「导致」「由于」「因为」「因此」「所以」「使得」「造成」「引发」「促使」「带动」或「受……影响」。"
            "「促进」：头推动尾。「抑制」：头压制尾。「必要条件」：头是尾发生之前必须满足的条件，例如「仍需」。"
            "「之前」「之后」：同一句原文里必须同时看得到两件事，并且有「之前、之后、此前、随后、然后、接着」这类词。没有这句话就不要写。"
            "「同时」：同一句里并列发生。"
            "「对比」：头与尾比较。「组成」：头包含尾。"
            "写关系前，这一句原文必须同时能找到头和尾。没有触发词的并列分句不要写成「导致」。"
            "不要把「受阻」里的「受」当成因果。不要把「仍需」写成原因导致结果。"
            "检查整段中所有已出现的事件；可连接不同句子的明确逻辑，不要只检查相邻分句。"
            "不要为了延长路径创造中间事件；跨片段和跨文章的关系由后续文档事件阶段核对。"
        )

    def set_model(self, model:LLMGenerator):
        """Set or update the LLM model."""
        self.model = model

    def load_dataset(self) -> Any:
        """Load and prepare dataset."""
        data_path = Path(self.config.data_directory)
        all_files = os.listdir(data_path)
        
        valid_files = [
            filename for filename in all_files
            if filename.startswith(self.config.filename_pattern) and
            (filename.endswith(".json.gz") or filename.endswith(".json") or filename.endswith(".jsonl") or filename.endswith(".jsonl.gz"))
        ]
        
        print(f"Found data files: {valid_files}")
        if not valid_files:
            raise FileNotFoundError("没有匹配的输入文件：" + str(data_path))
        data_files = [str(data_path / name) for name in sorted(valid_files)]
        if self.config.load_local_json:
            if len(data_files) != 1 or not data_files[0].endswith(".json"):
                raise ValueError("当前入口需要单个JSON数组文件")
            with open(data_files[0], encoding="utf-8") as handle:
                rows = json.load(handle)
            if not isinstance(rows, list):
                raise ValueError("输入文件需要JSON数组")
            return {"train": rows}
        return load_dataset("json", data_files={"train": data_files})
    
    def process_stage(self, instructions, result_schema: Dict) -> Tuple[List[str], List[List[Dict[str, Any]]]]:
        """Process one stage. `instructions` is the batch for that stage, a list of messages."""
        n = len(instructions) if instructions else 0
        pad = ("[]", {"completion_tokens": 0, "time": 0}) if self.config.record else "[]"
        try:
            outputs = self.model.triple_extraction(messages=instructions,
                                                    max_tokens=self.config.max_new_tokens if self.config.max_new_tokens else self.model.config.max_tokens, 
                                                    repetition_penalty=self.config.repetition_penalty if self.config.repetition_penalty else self.model.config.repetition_penalty,
                                                    record=self.config.record,
                                                    result_schema=result_schema, 
                                                    allow_empty=self.config.allow_empty)
        except Exception as exc:
            if self.config.strict_stage_errors:
                raise
            print(f"Stage failed, filling {n} empty results: {exc}")
            outputs = [pad] * n
        if self.config.strict_stage_errors and (not isinstance(outputs, list) or len(outputs) != n):
            raise ValueError("模型返回数量与输入数量不一致")
        if not isinstance(outputs, list):
            outputs = []
        if len(outputs) != n:
            print(f"Stage returned {len(outputs)} results for {n} inputs; empty rows fill the gap")
            outputs = (list(outputs) + [pad] * n)[:n]
        if self.config.record:
            text_outputs = []
            for output in outputs:
                if isinstance(output, (list, tuple)) and output and isinstance(output[0], str):
                    text_outputs.append(output[0])
                elif isinstance(output, str):
                    text_outputs.append(output)
                else:
                    text_outputs.append("[]")
        else:
            text_outputs = [item if isinstance(item, str) else "[]" for item in outputs]
        if self.config.strict_stage_errors:
            for output in text_outputs:
                if not isinstance(output, str) or not output.strip():
                    raise ValueError("模型返回空内容")
                parsed = json_repair.loads(output)
                if not isinstance(parsed, list):
                    raise ValueError("模型返回内容需要是JSON数组")
        structured_data = self.parser.extract_structured_data(text_outputs)
        if len(structured_data) != n:
            structured_data = (list(structured_data) + [[] for _ in range(n)])[:n]
        return outputs, structured_data
    
    def create_output_filename(self) -> str:
        """Create output filename with timestamp and shard info."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        model_name_safe = self.config.model_path.replace("/", "_")
        
        filename = (f"{model_name_safe}_{self.config.filename_pattern}_output_"
                   f"{timestamp}_{self.config.current_shard_triple + 1}_in_{self.config.total_shards_triple}.json")
        
        extraction_dir = os.path.join(self.config.output_directory, "kg_extraction")
        os.makedirs(extraction_dir, exist_ok=True)
        
        return os.path.join(extraction_dir, filename)

    def prepare_result_dict(self, id: str, text: str, metadata: Dict[str, Any], stage_outputs: List[Any], result_keys: List[str]) -> Dict[str, Any]:
        """Prepare result dictionary for a single sample."""
        sample_dict = {}
        sample_dict['id'] = id
        sample_dict['original_text'] = text
        try:
            metadata = json.loads(json.dumps(metadata, ensure_ascii=False, default=str))
        except Exception:
            metadata = {"lang": (metadata or {}).get("lang", "zh-CN") if isinstance(metadata, dict) else "zh-CN"}
        sample_dict['metadata'] = metadata
        llm_outputs, result_dicts = stage_outputs
        if self.config.record:
            llm_usage = [output[1] for output in llm_outputs]
            llm_outputs = [output[0] for output in llm_outputs]

        for key, llm_output, result_dict in zip(result_keys, llm_outputs, result_dicts):
            if not isinstance(result_dict, list):
                result_dict = []
            sample_dict[f'{key}_dict'] = result_dict
            sample_dict[f'{key}_output'] = llm_output
        if self.config.record:
            for key, llm_usage in zip(result_keys, llm_usage):
                sample_dict[f'{key}_usage'] = llm_usage

        if 'date_download' in sample_dict['metadata']:
            sample_dict['metadata']['date_download'] = str(sample_dict['metadata']['date_download'])
        return sample_dict

    def debug_print_result(self, result: Dict[str, Any]):
        """Print result for debugging."""
        for key, value in result.items():
            print(f"{key}: {value}")
            print("-" * 100)
    
    def run_extraction(self):
        """Run the complete knowledge graph extraction pipeline."""
        # Setup
        os.makedirs(self.config.output_directory+'/kg_extraction', exist_ok=True)
        dataset = self.load_dataset()
        
        if self.config.debug_mode:
            print("Debug mode: Processing only 20 samples")
        
        # Create data processor and loader
        processor = DatasetProcessor(self.config, self.prompt_instructions, self.result_schema)
        data_loader = CustomDataLoader(dataset["train"], processor, self.prompt_instructions, self.result_schema)
        if self.config.preserve_document_context:
            self._write_document_index(data_loader.processed_data)

        output_file = self.create_output_filename()
        completed = 0
        if self.config.resume_extraction:
            output_file = os.path.join(self.config.output_directory, "kg_extraction",
                                       self.config.filename_pattern + "_extraction.json")
            if os.path.isfile(output_file):
                with open(output_file, encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            raise ValueError("抽取结果包含空行")
                        row = json.loads(line)
                        if completed >= len(data_loader.processed_data):
                            raise ValueError("抽取结果多于输入片段")
                        expected = data_loader.processed_data[completed]
                        if row.get("id") != expected["id"] or row.get("original_text") != expected["text"]:
                            raise ValueError("已有抽取结果与当前输入顺序不一致")
                        completed += 1
            if completed == len(data_loader.processed_data):
                print("extraction already complete", completed, flush=True)
                return
            self.config.resume_from = completed // self.config.batch_size_triple
            print("extraction completed", completed, "of", len(data_loader), flush=True)
        print(f"Model: {self.config.model_path}")
        
        batch_counter = 0
        if self.config.record:
            extraction_start_time = time.time()
        with torch.no_grad():
            with open(output_file, "a" if self.config.resume_extraction else "w", encoding="utf-8") as output_stream:
                for batch in data_loader:
                    batch_counter += 1
                    messages_dict, batch_ids, batch_texts, batch_metadata = batch
                    
                    stage_outputs = {}
                    result_keys = list(messages_dict.keys())
                    for key in result_keys:
                        batch_result = self.process_stage(messages_dict[key], result_schema = self.result_schema[key])
                        # each batch_result is (llm_rawoutput, triple-dict), and at the same time llm_rawoutput can be (llm_rawoutput, llm_usage)
                        
                        stage_outputs[key] = batch_result


                    # Write results
                    print(f"Processed {batch_counter} batches ({batch_counter * self.config.batch_size_triple} chunks)")
                    for i in range(len(batch_ids)):
                        if ((self.config.resume_from + batch_counter - 1) * self.config.batch_size_triple
                                + i < completed):
                            continue
                        id = batch_ids[i]
                        text = batch_texts[i]
                        metadata = batch_metadata[i]

                        triple_dicts = []
                        llm_outputs = []
                        for key in result_keys:
                            temp_batch_result = stage_outputs[key]
                            llm_output, triple_dict = temp_batch_result
    
                            triple_dicts.append(triple_dict[i])
                            llm_outputs.append(llm_output[i])

                        result = self.prepare_result_dict(id, text, metadata, (llm_outputs, triple_dicts), result_keys)

                        if self.config.debug_mode:
                            self.debug_print_result(result)
   
                        output_stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                        output_stream.flush()
            if self.config.record and self.config.benchmark:
                # add the total time taken for extraction at the last json object
                total_extraction_time = time.time() - extraction_start_time
                # read the json and modify the last json object
                with open(output_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                if len(lines) > 0:
                    last_line = lines[-1]
                    try:
                        last_json = json.loads(last_line)
                        last_json['total_extraction_time_seconds'] = total_extraction_time
                        lines[-1] = json.dumps(last_json, ensure_ascii=False) + "\n"
                        with open(output_file, "w", encoding="utf-8") as f:
                            f.writelines(lines)
                    except:
                        print("Failed to add extraction time to the last json object")

    def _write_document_index(self, chunks):
        """One row per source document. Later stages read this instead of a chunk."""
        path = os.path.join(self.config.output_directory, "kg_extraction", "documents.jsonl")
        seen = set()
        with open(path, "w", encoding="utf-8") as handle:
            for chunk in chunks:
                doc_id = chunk["id"]
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                source = chunk.get("source_text") or chunk["text"]
                handle.write(json.dumps({
                    "id": doc_id,
                    "title": chunk.get("metadata", {}).get("document_title") or document_title(source),
                    "text": source,
                    "metadata": {key: value for key, value in chunk.get("metadata", {}).items()
                                 if key != "structure_units"},
                }, ensure_ascii=False) + "\n")
        source_path = os.path.join(self.config.output_directory, "kg_extraction", "source_index.jsonl")
        with open(source_path, "w", encoding="utf-8") as handle:
            for chunk in chunks:
                metadata = chunk.get("metadata", {})
                if "structure_units" not in metadata:
                    continue
                handle.write(json.dumps({
                    "document_id": chunk["id"], "text": chunk["text"],
                    "units": json.loads(metadata["structure_units"]),
                    "metadata": {key: value for key, value in metadata.items() if key != "structure_units"},
                }, ensure_ascii=False) + "\n")
        registry = Path(self.config.data_directory) / "document_registry.json"
        if registry.is_file():
            with registry.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            with open(os.path.join(self.config.output_directory, "kg_extraction", "document_registry.json"),
                      "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
        print("document index", len(seen), flush=True)

    def link_documents(self):
        from atlas_rag.kg_construction.document_linking import link_documents
        link_documents(self.model, self.config)

    def convert_json_to_csv(self):
        json2csv(
            dataset = self.config.filename_pattern, 
            output_dir=f"{self.config.output_directory}/triples_csv",
            data_dir=f"{self.config.output_directory}/kg_extraction",
            schema = self.result_schema,
            custom = self.custom_schema != None,
            scope_by_document = self.config.document_scope_nodes,
        )
        csvs_to_temp_graphml(
            triple_node_file=f"{self.config.output_directory}/triples_csv/triple_nodes_{self.config.filename_pattern}_from_json_without_emb.csv",
            triple_edge_file=f"{self.config.output_directory}/triples_csv/triple_edges_{self.config.filename_pattern}_from_json_without_emb.csv",
            config = self.config
        )
    
    def generate_concept_csv_temp(self, batch_size: int = None, **kwargs):
        if self.config.benchmark and self.config.record:
            concept_start_time = time.time()
        generate_concept(
            model=self.model,
            input_file=f"{self.config.output_directory}/triples_csv/missing_concepts_{self.config.filename_pattern}_from_json.csv",
            output_folder=f"{self.config.output_directory}/concepts",
            output_file="concept.json",
            logging_file=f"{self.config.output_directory}/concepts/logging.txt",
            config=self.config,
            batch_size=batch_size if batch_size else self.config.batch_size_concept,
            shard=self.config.current_shard_concept,
            num_shards=self.config.total_shards_concept,
            record = self.config.record,
            **kwargs
        )
        if self.config.benchmark and self.config.record:  
            # add line to the logging file
            total_concept_time = time.time() - concept_start_time
            with open(f"{self.config.output_directory}/concepts/logging.txt", "a") as f:
                f.write(f"Total concept generation time: {total_concept_time:.2f} seconds\n")
    
    def create_concept_csv(self):
        merge_csv_files(
            output_file=f"{self.config.output_directory}/triples_csv/{self.config.filename_pattern}_from_json_with_concept.csv",
            input_dir=f"{self.config.output_directory}/concepts",
        )
        all_concept_triples_csv_to_csv(
            node_file=f'{self.config.output_directory}/triples_csv/triple_nodes_{self.config.filename_pattern}_from_json_without_emb.csv',
            edge_file=f'{self.config.output_directory}/triples_csv/triple_edges_{self.config.filename_pattern}_from_json_without_emb.csv',
            concepts_file=f'{self.config.output_directory}/triples_csv/{self.config.filename_pattern}_from_json_with_concept.csv',
            output_node_file=f'{self.config.output_directory}/concept_csv/concept_nodes_{self.config.filename_pattern}_from_json_with_concept.csv',
            output_edge_file=f'{self.config.output_directory}/concept_csv/concept_edges_{self.config.filename_pattern}_from_json_with_concept.csv',
            output_full_concept_triple_edges=f'{self.config.output_directory}/concept_csv/triple_edges_{self.config.filename_pattern}_from_json_with_concept.csv',
        )
        
    def convert_to_neo4j(self):
        """Write a Cypher file. Load it when NEO4J_URI is set."""
        from atlas_rag.kg_construction.neo4j_export import csv_to_cypher, load_cypher

        pattern = self.config.filename_pattern
        root = self.config.output_directory
        edge_file = (
            f"{root}/concept_csv/triple_edges_{pattern}_from_json_with_concept.csv"
            if self.config.include_concept else
            f"{root}/triples_csv/triple_edges_{pattern}_from_json_without_emb.csv"
        )
        output = f"{root}/neo4j/{pattern}.cypher"
        csv_to_cypher(
            node_file=f"{root}/triples_csv/triple_nodes_{pattern}_from_json_without_emb.csv",
            edge_file=edge_file,
            text_node_file=f"{root}/triples_csv/text_nodes_{pattern}_from_json.csv",
            text_edge_file=f"{root}/triples_csv/text_edges_{pattern}_from_json.csv",
            output_file=output,
            concept_node_file=f"{root}/concept_csv/concept_nodes_{pattern}_from_json_with_concept.csv" if self.config.include_concept else None,
            concept_edge_file=f"{root}/concept_csv/concept_edges_{pattern}_from_json_with_concept.csv" if self.config.include_concept else None,
            source_index_file=f"{root}/kg_extraction/source_index.jsonl",
        )
        uri = os.environ.get("NEO4J_URI", "")
        if uri:
            load_cypher(
                output,
                uri,
                os.environ.get("NEO4J_USER", "neo4j"),
                os.environ.get("NEO4J_PASSWORD", ""),
            )
        else:
            print("NEO4J_URI is unset. Cypher file is ready:", output, flush=True)
        return output

    def convert_to_graphml(self):
        csvs_to_graphml(
            triple_node_file=f"{self.config.output_directory}/triples_csv/triple_nodes_{self.config.filename_pattern}_from_json_without_emb.csv",
            text_node_file=f"{self.config.output_directory}/triples_csv/text_nodes_{self.config.filename_pattern}_from_json.csv",
            triple_edge_file=f"{self.config.output_directory}/concept_csv/triple_edges_{self.config.filename_pattern}_from_json_with_concept.csv" if self.config.include_concept else \
                f"{self.config.output_directory}/triples_csv/triple_edges_{self.config.filename_pattern}_from_json_without_emb.csv",
            text_edge_file=f"{self.config.output_directory}/triples_csv/text_edges_{self.config.filename_pattern}_from_json.csv",
            concept_node_file=f"{self.config.output_directory}/concept_csv/concept_nodes_{self.config.filename_pattern}_from_json_with_concept.csv" if self.config.include_concept else None,
            concept_edge_file=f"{self.config.output_directory}/concept_csv/concept_edges_{self.config.filename_pattern}_from_json_with_concept.csv" if self.config.include_concept else None,
            output_file=f"{self.config.output_directory}/kg_graphml/{self.config.filename_pattern}_graph.graphml",
            include_concept = self.config.include_concept
        )
    
    def add_numeric_id(self):
        add_csv_columns(
            node_csv=f"{self.config.output_directory}/triples_csv/triple_nodes_{self.config.filename_pattern}_from_json_without_emb.csv",
            edge_csv=f"{self.config.output_directory}/concept_csv/triple_edges_{self.config.filename_pattern}_from_json_with_concept.csv",
            text_csv=f"{self.config.output_directory}/triples_csv/text_nodes_{self.config.filename_pattern}_from_json.csv",
            node_with_numeric_id=f"{self.config.output_directory}/triples_csv/triple_nodes_{self.config.filename_pattern}_from_json_without_emb_with_numeric_id.csv",
            edge_with_numeric_id=f"{self.config.output_directory}/triples_csv/triple_edges_{self.config.filename_pattern}_from_json_without_emb_with_numeric_id.csv",
            text_with_numeric_id=f"{self.config.output_directory}/triples_csv/text_nodes_{self.config.filename_pattern}_from_json_with_numeric_id.csv",
        )

    def compute_kg_embedding(self, encoder_model, batch_size: int = 2048):
        encoder_model.compute_kg_embedding(
            node_csv_without_emb=f"{self.config.output_directory}/triples_csv/triple_nodes_{self.config.filename_pattern}_from_json_without_emb.csv",
            node_csv_file=f"{self.config.output_directory}/triples_csv/triple_nodes_{self.config.filename_pattern}_from_json_with_emb.csv",
            edge_csv_without_emb=f"{self.config.output_directory}/concept_csv/triple_edges_{self.config.filename_pattern}_from_json_with_concept.csv",
            edge_csv_file=f"{self.config.output_directory}/triples_csv/triple_edges_{self.config.filename_pattern}_from_json_with_concept_with_emb.csv",
            text_node_csv_without_emb=f"{self.config.output_directory}/triples_csv/text_nodes_{self.config.filename_pattern}_from_json.csv",
            text_node_csv=f"{self.config.output_directory}/triples_csv/text_nodes_{self.config.filename_pattern}_from_json_with_emb.csv",
            batch_size = 2048
        )

    def create_faiss_index(self, index_type="HNSW,Flat", faiss_gpu=True):
        from atlas_rag.vectorstore.create_neo4j_index import create_faiss_index
        create_faiss_index(self.config.output_directory, self.config.filename_pattern, index_type, faiss_gpu)

def parse_command_line_arguments() -> ProcessingConfig:
    """Parse command line arguments and return configuration."""
    parser = argparse.ArgumentParser(description="Knowledge Graph Extraction Pipeline")
    
    parser.add_argument("-m", "--model", type=str, required=True,
                       default="meta-llama/Meta-Llama-3-8B-Instruct",
                       help="Model path for knowledge extraction")
    parser.add_argument("--data_dir", type=str, default="your_data_dir",
                       help="Directory containing input data")
    parser.add_argument("--file_name", type=str, default="en_simple_wiki_v0",
                       help="Filename pattern to match")
    parser.add_argument("-b", "--batch_size", type=int, default=16,
                       help="Batch size for processing")
    parser.add_argument("--output_dir", type=str, default="./generation_result_debug",
                       help="Output directory for results")
    parser.add_argument("--total_shards_triple", type=int, default=1,
                       help="Total number of data shards")
    parser.add_argument("--shard", type=int, default=0,
                       help="Current shard index")
    parser.add_argument("--bit8", action="store_true",
                       help="Use 8-bit quantization")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--resume", type=int, default=0,
                       help="Resume from specific batch")
    
    args = parser.parse_args()
    
    return ProcessingConfig(
        model_path=args.model,
        data_directory=args.data_dir,
        filename_pattern=args.file_name,
        batch_size=args.batch_size,
        output_directory=args.output_dir,
        total_shards_triple=args.total_shards_triple,
        current_shard_triple=args.shard,
        use_8bit=args.bit8,
        debug_mode=args.debug,
        resume_from=args.resume
    )

def main():
    """Main entry point for the knowledge graph extraction pipeline."""
    config = parse_command_line_arguments()
    extractor = KnowledgeGraphExtractor(config)
    extractor.run_extraction()

if __name__ == "__main__":
    main() 
