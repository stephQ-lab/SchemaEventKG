from tqdm import tqdm
import argparse
import os
import csv 
import json
import re
import hashlib

from atlas_rag.kg_construction.commodity import (
    PARENT,
    belongs_edge,
    canonical_mention,
    document_categories,
    member_targets,
)
from atlas_rag.kg_construction.event_ontology import normalize_event_relation

# Increase the field size limit
csv.field_size_limit(10 * 1024 * 1024)  # 10 MB limit


# Function to compute a hash ID from text
def compute_hash_id(text):
    # Use SHA-256 to generate a hash
    text = text + '_text'
    hash_object = hashlib.sha256(text.encode('utf-8'))
    return hash_object.hexdigest()  # Return hash as a hex string

def node_label(text):
    """Entity and event names keep the thing, the time and the trend, not the figure."""
    return canonical_mention(clean_text(text))


def clean_text(text):
    # remove NUL as well
    
    new_text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("\v", " ").replace("\f", " ").replace("\b", " ").replace("\a", " ").replace("\\e", " ").replace(";", ",")
    new_text = new_text.replace("\x00", "")
    new_text = re.sub(r'\s+', ' ', new_text).strip()

    return new_text

def remove_NUL(text):
    return text.replace("\x00", "")

def custom_schema_json_2_csv(dataset, data_dir, output_dir, schema):
    visited_nodes = set()
    visited_hashes = set()

    all_entities = set()
    all_events = set()
    all_relations = set()

    file_dir_list = [f for f in os.listdir(data_dir) if dataset in f]
    file_dir_list = sorted(file_dir_list)
    print("Loading data from the json files")
    print("Number of files: ", len(file_dir_list))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Define output file paths
    node_csv_without_emb = os.path.join(output_dir, f"triple_nodes_{dataset}_from_json_without_emb.csv")
    edge_csv_without_emb = os.path.join(output_dir, f"triple_edges_{dataset}_from_json_without_emb.csv")
    node_text_file = os.path.join(output_dir, f"text_nodes_{dataset}_from_json.csv")
    edge_text_file = os.path.join(output_dir, f"text_edges_{dataset}_from_json.csv")
    missing_concepts_file = os.path.join(output_dir, f"missing_concepts_{dataset}_from_json.csv")

    # Open CSV files for writing
    with open(node_text_file, "w", newline='', encoding='utf-8', errors='ignore') as csvfile_node_text, \
         open(edge_text_file, "w", newline='', encoding='utf-8', errors='ignore') as csvfile_edge_text, \
         open(node_csv_without_emb, "w", newline='', encoding='utf-8', errors='ignore') as csvfile_node, \
         open(edge_csv_without_emb, "w", newline='', encoding='utf-8', errors='ignore') as csvfile_edge:

        csv_writer_node_text = csv.writer(csvfile_node_text)
        csv_writer_edge_text = csv.writer(csvfile_edge_text)
        writer_node = csv.writer(csvfile_node)
        writer_edge = csv.writer(csvfile_edge)

        # Write headers
        csv_writer_node_text.writerow(["text_id:ID", "original_text", ":LABEL"])
        csv_writer_edge_text.writerow([":START_ID", ":END_ID", ":TYPE"])
        writer_node.writerow(["name:ID", "type", "concepts", "synsets", ":LABEL"])
        writer_edge.writerow([":START_ID", ":END_ID", "relation", "concepts", "synsets", ":TYPE"])
        
        # Process each file
        for file_dir in tqdm(file_dir_list):
            print("Processing file for file ids: ", file_dir)
            with open(os.path.join(data_dir, file_dir), "r", encoding="utf-8") as jsonfile:
                for line in jsonfile:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        print(f"Skip bad json line in {file_dir}: {exc}")
                        continue
                    if not isinstance(data, dict) or "original_text" not in data:
                        print(f"Skip record without original_text in {file_dir}")
                        continue
                    original_text = data["original_text"]
                    original_text = remove_NUL(original_text)
                    if "Here is the passage." in original_text:
                        original_text = original_text.split("Here is the passage.")[-1]
                    eot_token = "<|eot_id|>"
                    original_text = original_text.split(eot_token)[0]

                    text_hash_id = compute_hash_id(original_text)

                    # Write the original text as nodes
                    if text_hash_id not in visited_hashes:
                        visited_hashes.add(text_hash_id)
                        csv_writer_node_text.writerow([text_hash_id, original_text, "Text"])

                    file_id = str(data["id"])
                   
                    triple_name_keys = list(schema.keys())
                    for key in triple_name_keys:
                        triple_dict = data[f'{key}_dict']
                        schema_items = schema[key]['items']['properties']
                        # check if one to many or one to one
                        one_to_many = False
                        one_to_one = False
                        for item in schema_items.values():
                            if item['type'] == "array":
                                one_to_many = True
                                break
                        if not one_to_many:
                            one_to_one = True
                        
                        # process for case one to one
                        if one_to_one:
                            triple_keys = list(schema_items.keys())
                            entity_keys = []
                            node_types = []
                            for key in triple_keys:
                                if key not in ["Relation", "relation"]:
                                    entity_keys.append(key)
                                    node_types.append(schema_items[key]['node_type'] if 'node_type' in schema_items[key] else 'entity')
                            assert len(entity_keys) == 2, f"One-to-one triple schema must have exactly two entities: {entity_keys}"    
                            for triple in triple_dict:
                               # get the three keys
                               # We can only ensure either "Relation" or "relation" is always exist for edge name for one-to-one triple
                                assert not ("Relation" not in triple and "relation" not in triple), f"Missing relation key for {triple}"
                                
                                # check if it is relation or Relation for relation key
                                head_entity, relation, tail_entity = (triple[key] for key in triple_keys)
                                if head_entity is None or tail_entity is None or relation is None:
                                    continue
                                if head_entity.isspace() or tail_entity.isspace() or relation.isspace():
                                    continue
                                if len(head_entity) == 0 or len(tail_entity) == 0 or len(relation) == 0:
                                    continue

                                # Add nodes to files
                                if head_entity not in visited_nodes:
                                    visited_nodes.add(head_entity)
                                    all_entities.add(head_entity)
                                    writer_node.writerow([head_entity, node_types[0], [], [], "Node"])
                                    csv_writer_edge_text.writerow([head_entity, text_hash_id, "Source"])

                                if tail_entity not in visited_nodes:
                                    visited_nodes.add(tail_entity)
                                    all_entities.add(tail_entity)
                                    writer_node.writerow([tail_entity, node_types[1], [], [], "Node"])
                                    csv_writer_edge_text.writerow([tail_entity, text_hash_id, "Source"])

                                all_relations.add(relation)
                                writer_edge.writerow([head_entity, tail_entity, relation, [], [], "Relation"])
                        
                        if one_to_many:
                            triple_keys = list(schema_items.keys())
                            entity_keys = []
                            relation = None
                            # get the three keys
                            # For one-to-many triple, We cannot ensure either "Relation" or "relation" is always exist for edge name, but we can check if it exist
                            
                            for key in triple_keys:
                                # check if it is relation or Relation for relation key
                                if key not in ["Relation", "relation"]:
                                    entity_keys.append(key)
                                else:
                                    relation = triple[key]
                            if relation is None:
                                relation = "is participated by"
                            assert len(entity_keys) == 2, f"One-to-many triple schema must have exactly two entities: {entity_keys}"  
                            for triple in triple_dict:
                                many_entity = None
                                many_entity_key = None
                                many_entity_node_type = None
                                for key in entity_keys:
                                    if type(triple[key]) == list:
                                        many_entity = triple[key]
                                        many_entity_key = key
                                        many_entity_node_type = schema_items[key]['items']['node_type'] if 'node_type' in schema_items[key]['items'] else 'entity'
                                        break
                                one_entity = None
                                one_entity_node_type = None
                                for key in entity_keys:
                                    if key != many_entity_key:
                                        one_entity = triple[key]
                                        one_entity_node_type = schema_items[key]['node_type'] if 'node_type' in schema_items[key] else 'event'
                                if many_entity is None or one_entity is None or relation is None:
                                    continue
                                if one_entity.isspace() or relation.isspace():
                                    continue
                                if len(many_entity) == 0 or len(one_entity) == 0 or len(relation) == 0:
                                    continue
                                # Add nodes to files
                                if one_entity not in visited_nodes:
                                    visited_nodes.add(one_entity)
                                    all_events.add(one_entity)
                                    writer_node.writerow([one_entity, one_entity_node_type, [], [], "Node"])
                                    csv_writer_edge_text.writerow([head_entity, text_hash_id, "Source"])

                                for entity in many_entity:
                                    if entity is None or entity.isspace():
                                        continue
                                    if entity not in visited_nodes:
                                        visited_nodes.add(entity)
                                        all_entities.add(entity)
                                        writer_node.writerow([entity, many_entity_node_type, [], [], "Node"])
                                        csv_writer_edge_text.writerow([entity, text_hash_id, "Source"])

                                    all_relations.add(relation)
                                    writer_edge.writerow([head_entity, entity, relation, [], [], "Relation"])
                                    
    with open(missing_concepts_file, "w", newline='', encoding='utf-8', errors='ignore') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Name", "Type"])
        for entity in all_entities:
            writer.writerow([entity, "Entity"])
        for event in all_events:
            writer.writerow([event, "Event"])
        for relation in all_relations:
            writer.writerow([relation, "Relation"])

def json2csv(dataset, data_dir, output_dir, schema, custom, test=False, scope_by_document=False):
    """
    Convert JSON files to CSV files for nodes, edges, and missing concepts.

    Args:
        dataset (str): Name of the dataset.
        data_dir (str): Directory containing the JSON files.
        output_dir (str): Directory to save the output CSV files.
        test (bool): If True, run in test mode (process only 3 files).
    """
    if custom:
        return custom_schema_json_2_csv(dataset, data_dir, output_dir, schema)
    visited_nodes = set()
    visited_sources = set()
    visited_hashes = set()

    all_entities = set()
    all_events = set()
    all_relations = set()

    file_dir_list = [
        f for f in os.listdir(data_dir)
        if dataset in f and f.endswith(".json") and os.path.isfile(os.path.join(data_dir, f))
    ]
    file_dir_list = sorted(file_dir_list)
    from eventkg.trends import rewrite_market_label
    if test:
        file_dir_list = file_dir_list[:3]
    print("Loading data from the json files")
    print("Number of files: ", len(file_dir_list))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Define output file paths
    node_csv_without_emb = os.path.join(output_dir, f"triple_nodes_{dataset}_from_json_without_emb.csv")
    edge_csv_without_emb = os.path.join(output_dir, f"triple_edges_{dataset}_from_json_without_emb.csv")
    node_text_file = os.path.join(output_dir, f"text_nodes_{dataset}_from_json.csv")
    edge_text_file = os.path.join(output_dir, f"text_edges_{dataset}_from_json.csv")
    missing_concepts_file = os.path.join(output_dir, f"missing_concepts_{dataset}_from_json.csv")

    if test:
        node_text_file = os.path.join(output_dir, f"text_nodes_{dataset}_from_json_test.csv")
        edge_text_file = os.path.join(output_dir, f"text_edges_{dataset}_from_json_test.csv")
        node_csv_without_emb = os.path.join(output_dir, f"triple_nodes_{dataset}_from_json_without_emb_test.csv")
        edge_csv_without_emb = os.path.join(output_dir, f"triple_edges_{dataset}_from_json_without_emb_test.csv")
        missing_concepts_file = os.path.join(output_dir, f"missing_concepts_{dataset}_from_json_test.csv")

    # Open CSV files for writing
    with open(node_text_file, "w", newline='', encoding='utf-8', errors='ignore') as csvfile_node_text, \
         open(edge_text_file, "w", newline='', encoding='utf-8', errors='ignore') as csvfile_edge_text, \
         open(node_csv_without_emb, "w", newline='', encoding='utf-8', errors='ignore') as csvfile_node, \
         open(edge_csv_without_emb, "w", newline='', encoding='utf-8', errors='ignore') as csvfile_edge:

        csv_writer_node_text = csv.writer(csvfile_node_text)
        csv_writer_edge_text = csv.writer(csvfile_edge_text)
        writer_node = csv.writer(csvfile_node)
        writer_edge = csv.writer(csvfile_edge)

        # Write headers
        node_header = ["name:ID", "type", "concepts", "synsets", ":LABEL"]
        edge_header = [":START_ID", ":END_ID", "relation", "concepts", "synsets", ":TYPE"]
        text_edge_header = [":START_ID", ":END_ID", ":TYPE"]
        if scope_by_document:
            node_header.append("scope")
            edge_header.append("scope")
            text_edge_header.append("scope")
        csv_writer_node_text.writerow(["text_id:ID", "original_text", ":LABEL"])
        csv_writer_edge_text.writerow(text_edge_header)
        writer_node.writerow(node_header)
        writer_edge.writerow(edge_header)

        written_edges = set()

        def ensure_node(name, ntype, text_hash_id, file_id, categories=()):
            if categories and name in categories:
                ntype = "commodity"
            key = (file_id, name) if scope_by_document else name
            first = key not in visited_nodes
            if first:
                visited_nodes.add(key)
                row = [name, ntype, [], [], "Node"]
                if scope_by_document:
                    row.append(file_id)
                writer_node.writerow(row)
                if ntype == "event":
                    all_events.add(name)
                elif ntype == "entity":
                    all_entities.add(name)
            # Unscoped path keeps the original rule: one source edge, written
            # with the first node. Scoped path keeps one source edge per passage.
            if not scope_by_document and not first:
                return
            source_key = (file_id, name, text_hash_id) if scope_by_document else (name, text_hash_id)
            if source_key not in visited_sources:
                visited_sources.add(source_key)
                row = [name, text_hash_id, "Source"]
                if scope_by_document:
                    row.append(file_id)
                csv_writer_edge_text.writerow(row)

        def ensure_edge(head, relation, tail, file_id):
            if relation in ("涉及", "属于"):
                key = (file_id, head, relation, tail)
                if key in written_edges:
                    return
                written_edges.add(key)
            row = [head, tail, relation, [], [], "Relation"]
            if scope_by_document:
                row.append(file_id)
            writer_edge.writerow(row)
        
        # Process each file
        for file_dir in tqdm(file_dir_list):
            print("Processing file for file ids: ", file_dir)
            with open(os.path.join(data_dir, file_dir), "r", encoding="utf-8") as jsonfile:
                for line in jsonfile:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        print(f"Skip bad json line in {file_dir}: {exc}")
                        continue
                    if not isinstance(data, dict) or "original_text" not in data:
                        print(f"Skip record without original_text in {file_dir}")
                        continue
                    original_text = data["original_text"]
                    original_text = remove_NUL(original_text)
                    if "Here is the passage." in original_text:
                        original_text = original_text.split("Here is the passage.")[-1]
                    eot_token = "<|eot_id|>"
                    original_text = original_text.split(eot_token)[0]

                    text_hash_id = compute_hash_id(original_text)

                    # Write the original text as nodes
                    if text_hash_id not in visited_hashes:
                        visited_hashes.add(text_hash_id)
                        csv_writer_node_text.writerow([text_hash_id, original_text, "Text"])

                    file_id = str(data["id"])
                    categories = data.get("document_categories")
                    if not isinstance(categories, list):
                        meta = data.get("metadata") or {}
                        if not isinstance(meta, dict):
                            meta = {}
                        categories = document_categories(
                            doc_id=file_id,
                            file_name=str(meta.get("file") or ""),
                            text=original_text,
                        )
                    for category in categories:
                        ensure_node(category, "commodity", text_hash_id, file_id, categories)
                    entity_relation_dict = data.get("entity_relation_dict") or []
                    event_entity_relation_dict = data.get("event_entity_dict") or []
                    event_relation_dict = data.get("event_relation_dict") or []
                    if not isinstance(entity_relation_dict, list):
                        entity_relation_dict = []
                    if not isinstance(event_entity_relation_dict, list):
                        event_entity_relation_dict = []
                    if not isinstance(event_relation_dict, list):
                        event_relation_dict = []

                    # Process entity triples
                    entity_triples = []
                    for entity_triple in entity_relation_dict:
                        try:
                            assert isinstance(entity_triple["Head"], str)
                            assert isinstance(entity_triple["Relation"], str)
                            assert isinstance(entity_triple["Tail"], str)

                            head_entity = entity_triple["Head"]
                            relation = entity_triple["Relation"]
                            tail_entity = entity_triple["Tail"]

                            # Clean the text
                            head_entity = node_label(head_entity)
                            relation = clean_text(relation)
                            tail_entity = node_label(tail_entity)

                            if not head_entity or not tail_entity or head_entity == tail_entity:
                                continue
                            if categories and relation == "属于" and not belongs_edge(head_entity, tail_entity):
                                continue

                            entity_triples.append((head_entity, relation, tail_entity))
                        except:
                            print(f"Error processing entity triple: {entity_triple}")
                            continue

                    # Process event triples
                    event_triples = []
                    for event_triple in event_relation_dict:
                        try:
                            assert isinstance(event_triple["Head"], str)
                            assert isinstance(event_triple["Relation"], str)
                            assert isinstance(event_triple["Tail"], str)
                            head_event = event_triple["Head"]
                            relation = event_triple["Relation"]
                            tail_event = event_triple["Tail"]
                            
                            # Clean the text
                            head_event = node_label(head_event)
                            relation = clean_text(relation)
                            tail_event = node_label(tail_event)

                            if not head_event or not tail_event or head_event == tail_event:
                                continue
                            if head_event == tail_event and categories:
                                continue
                            if categories:
                                relation = normalize_event_relation(head_event, relation, tail_event, original_text)
                                if not relation:
                                    continue

                            event_triples.append((head_event, relation, tail_event))
                        except:
                            print(f"Error processing event triple: {event_triple}")

                    # Process event-entity triples
                    event_entity_triples = []
                    commodity_triples = []
                    member_names = []
                    for event_entity_participations in event_entity_relation_dict:
                        if "Event" not in event_entity_participations or "Entity" not in event_entity_participations:
                            continue
                        if not isinstance(event_entity_participations["Event"], str) or not isinstance(event_entity_participations["Entity"], list):
                            continue

                        for entity in event_entity_participations["Entity"]:
                            if not isinstance(entity, str):
                                continue

                            entity = node_label(entity)
                            event = node_label(event_entity_participations["Event"])

                            if not event or not entity or event == entity:
                                continue

                            event_entity_triples.append((event, "is participated by", entity))

                        for commodity in event_entity_participations.get("Commodity") or []:
                            if not isinstance(commodity, str):
                                continue
                            commodity = clean_text(commodity)
                            event = clean_text(event_entity_participations["Event"])
                            if not commodity or not event:
                                continue
                            if categories and commodity in categories:
                                commodity_triples.append((event, "涉及", commodity))
                            elif categories and commodity in PARENT:
                                member_names.append(commodity)
                        for member in event_entity_participations.get("Member") or []:
                            member = clean_text(member) if isinstance(member, str) else ""
                            if member and member in PARENT:
                                member_names.append(member)

                    # Write nodes and edges to CSV files
                    for entity_triple in entity_triples:
                        head_entity, relation, tail_entity = entity_triple
                        head_entity, tail_entity = rewrite_market_label(head_entity), rewrite_market_label(tail_entity)
                        if not head_entity or not tail_entity or not relation:
                            continue
                        if head_entity.isspace() or tail_entity.isspace() or relation.isspace():
                            continue
                        ensure_node(head_entity, "entity", text_hash_id, file_id, categories)
                        ensure_node(tail_entity, "entity", text_hash_id, file_id, categories)
                        all_relations.add(relation)
                        ensure_edge(head_entity, relation, tail_entity, file_id)

                    for event_triple in event_triples:
                        head_event, relation, tail_event = event_triple
                        head_event, tail_event = rewrite_market_label(head_event), rewrite_market_label(tail_event)
                        if not head_event or not tail_event or not relation:
                            continue
                        if head_event.isspace() or tail_event.isspace() or relation.isspace():
                            continue
                        ensure_node(head_event, "event", text_hash_id, file_id, categories)
                        ensure_node(tail_event, "event", text_hash_id, file_id, categories)
                        all_relations.add(relation)
                        ensure_edge(head_event, relation, tail_event, file_id)

                    for event_entity_triple in event_entity_triples:
                        head_event, relation, tail_entity = event_entity_triple
                        head_event, tail_entity = rewrite_market_label(head_event), rewrite_market_label(tail_entity)
                        if not head_event or not tail_entity or not relation:
                            continue
                        if head_event.isspace() or tail_entity.isspace() or relation.isspace():
                            continue
                        ensure_node(head_event, "event", text_hash_id, file_id, categories)
                        ensure_node(tail_entity, "entity", text_hash_id, file_id, categories)
                        all_relations.add(relation)
                        ensure_edge(head_event, relation, tail_entity, file_id)

                    for commodity_triple in commodity_triples:
                        head_event, relation, commodity = commodity_triple
                        head_event = rewrite_market_label(head_event)
                        commodity = rewrite_market_label(commodity)
                        if not head_event or not commodity or not relation:
                            continue
                        ensure_node(head_event, "event", text_hash_id, file_id, categories)
                        ensure_node(commodity, "commodity", text_hash_id, file_id, categories)
                        all_relations.add(relation)
                        ensure_edge(head_event, relation, commodity, file_id)

                    for member in member_names:
                        ensure_node(member, "entity", text_hash_id, file_id, categories)
                        for category in member_targets(member, categories):
                            ensure_node(category, "commodity", text_hash_id, file_id, categories)
                            all_relations.add("属于")
                            ensure_edge(member, "属于", category, file_id)

    # Write missing concepts to CSV
    with open(missing_concepts_file, "w", newline='', encoding='utf-8', errors='ignore') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Name", "Type"])
        for entity in all_entities:
            writer.writerow([entity, "Entity"])
        for event in all_events:
            writer.writerow([event, "Event"])
        for relation in all_relations:
            writer.writerow([relation, "Relation"])

    print("Data to CSV completed successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="[pes2o_abstract, en_simple_wiki_v0, cc_en]")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the graph raw JSON files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output CSV files")
    parser.add_argument("--test", action="store_true", help="Test the script")
    args = parser.parse_args()
    json2csv(dataset=args.dataset, data_dir=args.data_dir, output_dir=args.output_dir, test=args.test)
